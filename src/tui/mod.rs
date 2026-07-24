//! Ratatui-based monitor mode: a full-screen dashboard with header tabs,
//! the three-axis usage table, token charts, and a command prompt.

pub mod commands;
pub mod data;
pub mod input;
mod palette;
pub mod render;
mod table;

use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

use chrono::Local;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::layout::Alignment;
use ratatui::prelude::CrosstermBackend;
use ratatui::widgets::Paragraph;

use crate::sync::worker::SyncWorker;
use crate::{AppState, IntegrityStatus, IntervalSlideDirection, updater};
use commands::{Effect, HelpView};
use input::{CommandHistory, InputLine};

pub struct MonitorConfig {
    pub auto_update: bool,
    pub auto_update_interval_seconds: u64,
}

const NOTICE_TTL: Duration = Duration::from_secs(6);

struct Notice {
    text: String,
    expires_at: Instant,
}

/// Tracks the host that the in-flight source refresh was started for. Source
/// scans are intentionally single-flight, but a host change must not let an
/// older scan replace the newly selected host's cache.
#[derive(Default)]
struct RawRefreshTracker {
    running: bool,
    active_host: Option<String>,
    refresh_again: bool,
}

impl RawRefreshTracker {
    /// Returns true when the caller should start a refresh immediately. A
    /// different host selected during an existing refresh is coalesced into
    /// one follow-up request instead of launching another source scan.
    fn request(&mut self, host: &Option<String>) -> bool {
        if self.running {
            self.refresh_again = self.active_host.as_deref() != host.as_deref();
            return false;
        }
        self.running = true;
        self.active_host = host.clone();
        true
    }

    /// Returns true if the completed data belongs to an obsolete host or a
    /// newer host request was coalesced while it ran.
    fn complete(&mut self, current_host: &Option<String>) -> bool {
        let stale = self.active_host.as_deref() != current_host.as_deref();
        self.running = false;
        self.active_host = None;
        let refresh_again = stale || self.refresh_again;
        self.refresh_again = false;
        refresh_again
    }

    fn abandon(&mut self) {
        self.running = false;
        self.active_host = None;
        self.refresh_again = false;
    }
}

fn request_background_refresh(state: &mut AppState, tracker: &mut RawRefreshTracker) {
    if tracker.request(&state.host) {
        crate::start_background_raw_refresh(state);
        if state.raw_refresh.is_none() {
            tracker.abandon();
        }
    }
}

fn make_notice(messages: Vec<String>) -> Option<Notice> {
    if messages.is_empty() {
        return None;
    }
    Some(Notice {
        text: messages.join(" | "),
        expires_at: Instant::now() + NOTICE_TTL,
    })
}

type Term = Terminal<CrosstermBackend<io::Stdout>>;

fn setup_terminal() -> io::Result<Term> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(io::stdout()))
}

fn restore_terminal() {
    disable_raw_mode().ok();
    execute!(io::stdout(), LeaveAlternateScreen).ok();
}

fn install_panic_hook() {
    let original = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        original(info);
    }));
}

/// Leave the alternate screen, run `body` with normal stdout, then re-enter
/// and force a full redraw. Used for the self-updater, whose progress prints
/// straight to the terminal.
fn suspended<T>(terminal: &mut Term, body: impl FnOnce() -> T) -> T {
    disable_raw_mode().ok();
    execute!(io::stdout(), LeaveAlternateScreen).ok();
    let result = body();
    enable_raw_mode().ok();
    execute!(io::stdout(), EnterAlternateScreen).ok();
    terminal.clear().ok();
    result
}

fn run_updater(terminal: &mut Term, prefix: &'static str) -> Vec<String> {
    suspended(terminal, || {
        println!();
        match updater::run_update(|message| println!("{prefix}{message}")) {
            Ok(updater::UpdateOutcome::AlreadyLatest { current, latest }) => vec![format!(
                "Already on latest version: v{current} (remote: v{latest})."
            )],
            Err(err) => vec![format!("Update failed: {err}")],
        }
    })
}

fn load_duration_text(elapsed: Duration) -> String {
    if elapsed.as_secs() >= 1 {
        format!("Data loaded in {:.1}s", elapsed.as_secs_f64())
    } else {
        format!("Data loaded in {}ms", elapsed.as_millis())
    }
}

pub fn run_monitor(state: &mut AppState, sync_worker: Option<SyncWorker>, config: MonitorConfig) {
    install_panic_hook();
    let mut terminal = match setup_terminal() {
        Ok(terminal) => terminal,
        Err(err) => {
            eprintln!("failed to initialize terminal: {err}");
            return;
        }
    };

    let load_started = Instant::now();
    terminal
        .draw(|frame| {
            frame.render_widget(
                Paragraph::new("Loading usage data...").alignment(Alignment::Center),
                frame.area(),
            );
        })
        .ok();
    let now = Local::now();
    let required_horizon = crate::compute_required_horizon(&state.time_window, now);
    state.raw_cache = Some(crate::read_cached_raw_data_for_window(
        state.host.as_deref(),
        state.local_host_id.as_deref(),
        required_horizon,
        now,
    ));

    let mut dashboard = data::build(state);
    let mut input = InputLine::new();
    let mut history = CommandHistory::new();
    let mut pending_events: VecDeque<Event> = VecDeque::new();
    let mut notice = make_notice(vec![load_duration_text(load_started.elapsed())]);
    let mut help: Option<HelpView> = None;
    let mut observed_sync_revision = 0_u64;
    let mut sync_status = crate::current_sync_status(sync_worker.as_ref());
    let mut refresh_tracker = RawRefreshTracker::default();

    let monitor_interval = |state: &AppState| Duration::from_secs(state.monitor_interval);
    let mut next_refresh = Instant::now() + monitor_interval(state);
    let machine_id = state.local_host_id.clone().unwrap_or_default();
    let mut next_sync =
        Instant::now() + crate::monitor_sync_delay(monitor_interval(state), &machine_id);
    let mut next_auto_update = config.auto_update.then(Instant::now);
    let mut initial_refresh_pending = true;

    'monitor: loop {
        if crate::poll_background_raw_refresh(state) {
            let needs_replacement = refresh_tracker.complete(&state.host);
            if needs_replacement {
                // Do not retain a result collected for an old host. The
                // current host can be rebuilt from its cache while the single
                // replacement source scan runs in the background.
                state.raw_cache = None;
                request_background_refresh(state, &mut refresh_tracker);
            }
            if sync_worker.is_some() {
                next_sync = crate::monitor_sync_deadline_after_refresh(
                    Instant::now(),
                    next_sync,
                    monitor_interval(state),
                    &machine_id,
                );
            }
            dashboard = data::build(state);
        } else if refresh_tracker.running && state.raw_refresh.is_none() {
            // A worker can disconnect without sending its snapshot. Let the
            // next refresh request start normally instead of keeping the
            // single-flight guard stuck forever.
            refresh_tracker.abandon();
        }

        if let Some(stats) =
            crate::poll_sync_worker_status(sync_worker.as_ref(), &mut observed_sync_revision)
        {
            if stats.running {
                state.integrity_status = IntegrityStatus::Checking;
                state.integrity_started_at.get_or_insert_with(Instant::now);
            }
            if let Some(verification) = stats.integrity_verification.as_ref() {
                let duration = state
                    .integrity_started_at
                    .take()
                    .map(|started_at| started_at.elapsed())
                    .unwrap_or_default();
                state.integrity_status =
                    crate::integrity_status_from_verification(verification, duration);
            }
            if !stats.running && stats.last_error.is_none() && stats.success_count > 0 {
                state.raw_cache = Some(crate::read_full_cached_raw_data_for_hosts(
                    state.host.as_deref(),
                    state.local_host_id.as_deref(),
                    Local::now(),
                ));
                dashboard = data::build(state);
            }
            sync_status = crate::format_monitor_sync_status(&stats);
        }

        if notice
            .as_ref()
            .is_some_and(|n| Instant::now() >= n.expires_at)
        {
            notice = None;
        }

        if let Some(deadline) = next_auto_update
            && Instant::now() >= deadline
        {
            let messages = run_updater(&mut terminal, "auto-update: ");
            notice = make_notice(messages);
            next_auto_update = Some(crate::auto_update_deadline_after(
                Instant::now(),
                config.auto_update_interval_seconds,
            ));
        }

        if Instant::now() >= next_refresh {
            request_background_refresh(state, &mut refresh_tracker);
            dashboard = data::build(state);
            next_refresh = Instant::now() + monitor_interval(state);
        }

        if let Some(worker) = sync_worker.as_ref()
            && Instant::now() >= next_sync
        {
            worker.request_sync();
            next_sync = Instant::now() + crate::monitor_sync_interval(monitor_interval(state));
        }

        let refresh_in = next_refresh.saturating_duration_since(Instant::now());
        {
            let ui = render::Ui {
                dash: &dashboard,
                state,
                input: &input,
                notice: notice.as_ref().map(|n| n.text.as_str()),
                sync_status: sync_status.as_deref(),
                refresh_in,
                help,
            };
            if terminal.draw(|frame| render::draw(frame, &ui)).is_err() {
                break 'monitor;
            }
        }
        if initial_refresh_pending {
            // Let the first dashboard reach the terminal before the expensive
            // source scan competes for CPU, memory, and filesystem bandwidth.
            request_background_refresh(state, &mut refresh_tracker);
            initial_refresh_pending = false;
        }

        let mut timeout =
            Duration::from_millis(500).min(next_refresh.saturating_duration_since(Instant::now()));
        if sync_worker.is_some() {
            timeout = timeout.min(next_sync.saturating_duration_since(Instant::now()));
        }
        if let Some(deadline) = next_auto_update {
            timeout = timeout.min(deadline.saturating_duration_since(Instant::now()));
        }

        let next_event = if let Some(event) = pending_events.pop_front() {
            Some(event)
        } else if crossterm::event::poll(timeout).unwrap_or(false) {
            crossterm::event::read().ok()
        } else {
            None
        };
        let Some(event) = next_event else {
            continue;
        };

        match event {
            Event::Key(KeyEvent {
                code: KeyCode::Char('c') | KeyCode::Char('d'),
                modifiers,
                ..
            }) if modifiers.contains(KeyModifiers::CONTROL) => {
                break 'monitor;
            }
            Event::Key(KeyEvent {
                code: KeyCode::Enter,
                ..
            }) => {
                let command = input.snapshot().trim().to_string();
                history.record(&command);
                input.clear();
                notice = None;
                if help.is_some() && command.is_empty() {
                    help = None;
                    continue;
                }
                let outcome = commands::execute(state, &command);
                match outcome.effect {
                    Effect::Exit => break 'monitor,
                    Effect::Help => {
                        help = match help {
                            None => Some(HelpView::Index),
                            Some(_) => None,
                        };
                    }
                    Effect::HelpTopic(idx) => help = Some(HelpView::Topic(idx)),
                    Effect::Update => {
                        let messages = run_updater(&mut terminal, "");
                        notice = make_notice(messages);
                    }
                    Effect::Refresh => {
                        dashboard = data::build(state);
                        next_refresh = Instant::now() + monitor_interval(state);
                    }
                    Effect::ViewChanged => {
                        data::rebuild_view(&mut dashboard, state.table_view);
                    }
                    Effect::ReloadRefresh => {
                        request_background_refresh(state, &mut refresh_tracker);
                        dashboard = data::build(state);
                        next_refresh = Instant::now() + monitor_interval(state);
                    }
                    Effect::IntervalChanged => {
                        let (refresh_at, sync_at) = crate::monitor_deadlines_after_interval_change(
                            Instant::now(),
                            state.monitor_interval,
                            &machine_id,
                        );
                        next_refresh = refresh_at;
                        next_sync = sync_at;
                    }
                    Effect::None => {}
                }
                if !outcome.messages.is_empty() {
                    notice = make_notice(outcome.messages);
                }
            }
            Event::Key(KeyEvent {
                code: code @ (KeyCode::Tab | KeyCode::BackTab),
                ..
            }) => {
                let step = if code == KeyCode::Tab { 1 } else { -1 };
                let outcome = commands::rotate_tool(state, step);
                if outcome.effect == Effect::Refresh {
                    dashboard = data::build(state);
                    next_refresh = Instant::now() + monitor_interval(state);
                }
                notice = make_notice(outcome.messages);
            }
            Event::Key(KeyEvent {
                code: KeyCode::Up, ..
            }) => {
                if let Some(recalled) = history.navigate_up(input.snapshot()) {
                    notice = None;
                    input.replace(recalled);
                }
            }
            Event::Key(KeyEvent {
                code: KeyCode::Down,
                ..
            }) => {
                if let Some(recalled) = history.navigate_down() {
                    notice = None;
                    input.replace(recalled);
                }
            }
            Event::Key(KeyEvent {
                code: KeyCode::PageUp,
                ..
            }) => {
                let now = Local::now();
                if let Some(new_window) = state.time_window.slide_back(now) {
                    state.time_window = new_window;
                    dashboard = data::build(state);
                    next_refresh = Instant::now() + monitor_interval(state);
                }
            }
            Event::Key(KeyEvent {
                code: KeyCode::PageDown,
                ..
            }) => {
                let now = Local::now();
                if let Some(new_window) = state.time_window.slide_forward(now) {
                    state.time_window = new_window;
                    dashboard = data::build(state);
                    next_refresh = Instant::now() + monitor_interval(state);
                }
            }
            Event::Key(KeyEvent {
                code: code @ (KeyCode::Left | KeyCode::Right),
                ..
            }) => {
                if input.is_empty() {
                    let now = Local::now();
                    let first = if code == KeyCode::Left {
                        IntervalSlideDirection::Newer
                    } else {
                        IntervalSlideDirection::Older
                    };
                    let directions =
                        crate::collect_interval_slide_directions(&mut pending_events, first);
                    if let Some(new_window) = crate::apply_interval_slide_directions(
                        &state.time_window,
                        now,
                        crate::get_chart_target_width(),
                        directions,
                    ) {
                        state.time_window = new_window;
                        dashboard = data::build(state);
                        next_refresh = Instant::now() + monitor_interval(state);
                    }
                } else if code == KeyCode::Left {
                    input.move_left();
                } else {
                    input.move_right();
                }
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char(c @ ('+' | '-')),
                modifiers,
                ..
            }) if input.is_empty() && !modifiers.contains(KeyModifiers::CONTROL) => {
                let now = Local::now();
                let new_window = if c == '+' {
                    state.time_window.zoom_in(now)
                } else {
                    state.time_window.zoom_out(now)
                };
                if let Some(new_window) = new_window {
                    state.time_window = new_window;
                    dashboard = data::build(state);
                    next_refresh = Instant::now() + monitor_interval(state);
                }
            }
            Event::Key(KeyEvent {
                code: KeyCode::Esc, ..
            }) => match help {
                Some(HelpView::Topic(_)) => help = Some(HelpView::Index),
                Some(HelpView::Index) => help = None,
                None => {
                    input.clear();
                    notice = None;
                }
            },
            Event::Key(KeyEvent {
                code: KeyCode::Backspace,
                ..
            }) => {
                if input.backspace() {
                    notice = None;
                }
            }
            Event::Key(KeyEvent {
                code: KeyCode::Char(c),
                modifiers,
                ..
            }) if !modifiers.contains(KeyModifiers::CONTROL) => {
                notice = None;
                input.insert_char(c);
            }
            Event::Resize(_, _) => {
                dashboard = data::build(state);
            }
            _ => {}
        }
    }

    restore_terminal();
    println!("Monitoring stopped.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changed_host_queues_one_replacement_refresh_after_the_old_request_finishes() {
        let mut tracker = RawRefreshTracker::default();
        let all_hosts = None;
        let remote_host = Some("build-host".to_string());

        assert!(tracker.request(&all_hosts));
        assert!(!tracker.request(&remote_host));
        assert!(tracker.complete(&remote_host));

        assert!(tracker.request(&remote_host));
        assert!(!tracker.complete(&remote_host));
    }

    #[test]
    fn returning_to_the_active_host_does_not_queue_an_unneeded_refresh() {
        let mut tracker = RawRefreshTracker::default();
        let all_hosts = None;
        let remote_host = Some("build-host".to_string());

        assert!(tracker.request(&all_hosts));
        assert!(!tracker.request(&remote_host));
        assert!(!tracker.request(&all_hosts));
        assert!(!tracker.complete(&all_hosts));
    }
}
