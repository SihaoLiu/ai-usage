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
use std::sync::mpsc;
use std::time::{Duration, Instant};

use chrono::Local;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::prelude::CrosstermBackend;

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

#[derive(Default)]
struct SourceRefreshTracker {
    receiver: Option<mpsc::Receiver<crate::BackgroundSourceRefresh>>,
    pending: bool,
}

struct SourceRefreshCompletion {
    refresh: Option<crate::BackgroundSourceRefresh>,
    follow_up: bool,
}

impl SourceRefreshTracker {
    fn request(&mut self, state: &AppState) {
        if self.receiver.is_some() {
            self.pending = true;
            return;
        }
        let loaded_generation = state
            .raw_cache
            .as_ref()
            .map(|cache| cache.persistent_generation.clone());
        self.start(loaded_generation);
    }

    fn start(&mut self, loaded_generation: Option<String>) {
        self.receiver = Some(crate::start_background_source_refresh(loaded_generation));
    }

    fn poll(&mut self) -> Option<SourceRefreshCompletion> {
        let receiver = self.receiver.take()?;
        match receiver.try_recv() {
            Ok(refresh) => Some(SourceRefreshCompletion {
                refresh: Some(refresh),
                follow_up: std::mem::take(&mut self.pending),
            }),
            Err(mpsc::TryRecvError::Empty) => {
                self.receiver = Some(receiver);
                None
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                let follow_up = std::mem::take(&mut self.pending);
                follow_up.then_some(SourceRefreshCompletion {
                    refresh: None,
                    follow_up,
                })
            }
        }
    }

    #[cfg(test)]
    fn is_running(&self) -> bool {
        self.receiver.is_some()
    }
}

/// Tracks the host, kind, and range of the in-flight cache load. Loads stay
/// single-flight while newer requests coalesce to the latest requested band.
#[derive(Default)]
struct RawRefreshTracker {
    active_host: Option<String>,
    active: Option<RawLoadRequest>,
    follow_up: Option<RawLoadRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RawLoadKind {
    Prefetch,
    Cached,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RawLoadRequest {
    kind: RawLoadKind,
    range: crate::RawDataRange,
}

impl RawLoadRequest {
    fn prefetch(range: crate::RawDataRange) -> Self {
        Self {
            kind: RawLoadKind::Prefetch,
            range,
        }
    }

    fn cached(range: crate::RawDataRange) -> Self {
        Self {
            kind: RawLoadKind::Cached,
            range,
        }
    }

    fn coalesce(self, newer: Self) -> Self {
        let priority = |kind| match kind {
            RawLoadKind::Prefetch => 0,
            RawLoadKind::Cached => 1,
        };
        Self {
            kind: if priority(self.kind) >= priority(newer.kind) {
                self.kind
            } else {
                newer.kind
            },
            range: newer.range,
        }
    }
}

impl RawRefreshTracker {
    fn request_prefetch(&mut self, host: &Option<String>, range: crate::RawDataRange) -> bool {
        let requested = RawLoadRequest::prefetch(range);
        let Some(active) = self.active else {
            self.active = Some(requested);
            self.active_host = host.clone();
            return true;
        };

        if self.active_host.as_deref() != host.as_deref() {
            self.queue_follow_up(requested);
            return false;
        }
        if let Some(pending) = self.follow_up {
            if pending.kind == RawLoadKind::Prefetch && active.range.covers(range) {
                self.follow_up = None;
            } else {
                self.follow_up = Some(pending.coalesce(requested));
            }
        } else if !active.range.covers(range) {
            self.queue_follow_up(requested);
        }
        false
    }

    fn request_reload(&mut self, host: &Option<String>, range: crate::RawDataRange) -> bool {
        let requested = RawLoadRequest::cached(range);
        if self.active.is_none() {
            self.active = Some(requested);
            self.active_host = host.clone();
            return true;
        }

        if self.active_host.as_deref() != host.as_deref() {
            self.queue_follow_up(requested);
            return false;
        }
        self.queue_follow_up(requested);
        false
    }

    fn queue_follow_up(&mut self, request: RawLoadRequest) {
        self.follow_up = Some(
            self.follow_up
                .map_or(request, |pending| pending.coalesce(request)),
        );
    }

    fn is_running(&self) -> bool {
        self.active.is_some()
    }

    fn cancel_pending_prefetch(&mut self, host: &Option<String>) {
        if self.active_host.as_deref() == host.as_deref()
            && self
                .follow_up
                .is_some_and(|request| request.kind == RawLoadKind::Prefetch)
        {
            self.follow_up = None;
        }
    }

    fn complete(&mut self, current_host: &Option<String>) -> Option<RawLoadRequest> {
        let active = self.active.take()?;
        let stale = self.active_host.as_deref() != current_host.as_deref();
        self.active_host = None;
        if stale && self.follow_up.is_none() {
            self.queue_follow_up(RawLoadRequest::cached(active.range));
        }
        self.follow_up.take()
    }

    fn abandon(&mut self) {
        self.active_host = None;
        self.active = None;
        self.follow_up = None;
    }
}

fn request_background_reload(state: &mut AppState, tracker: &mut RawRefreshTracker) {
    let range = crate::raw_cache_target_range(&state.time_window, Local::now());
    request_background_reload_to(state, tracker, range);
}

fn request_background_prefetch(state: &mut AppState, tracker: &mut RawRefreshTracker) {
    let range = crate::raw_cache_target_range(&state.time_window, Local::now());
    request_background_prefetch_to(state, tracker, range);
}

fn request_background_prefetch_to(
    state: &mut AppState,
    tracker: &mut RawRefreshTracker,
    range: crate::RawDataRange,
) {
    if tracker.request_prefetch(&state.host, range) {
        crate::start_background_raw_prefetch(state, range);
        if state.raw_refresh.is_none() {
            tracker.abandon();
        }
    }
}

fn request_background_reload_to(
    state: &mut AppState,
    tracker: &mut RawRefreshTracker,
    range: crate::RawDataRange,
) {
    if tracker.request_reload(&state.host, range) {
        crate::start_background_raw_reload(state, range);
        if state.raw_refresh.is_none() {
            tracker.abandon();
        }
    }
}

fn request_follow_up(
    state: &mut AppState,
    tracker: &mut RawRefreshTracker,
    request: RawLoadRequest,
) {
    let current_range = crate::raw_cache_target_range(&state.time_window, Local::now());
    match request.kind {
        RawLoadKind::Prefetch => request_background_prefetch_to(state, tracker, current_range),
        RawLoadKind::Cached => request_background_reload_to(state, tracker, current_range),
    }
}

fn rebuild_or_reload_window(
    state: &mut AppState,
    dashboard: &mut data::Dashboard,
    tracker: &mut RawRefreshTracker,
) {
    let now = Local::now();
    *dashboard = data::build(state);
    if crate::raw_cache_needs_prefetch(state, now) {
        request_background_prefetch(state, tracker);
    } else {
        tracker.cancel_pending_prefetch(&state.host);
    }
}

fn apply_source_refresh_result(
    state: &mut AppState,
    dashboard: &mut data::Dashboard,
    tracker: &mut RawRefreshTracker,
    changed: bool,
) {
    if changed {
        *dashboard = data::build(state);
        request_background_reload(state, tracker);
    } else {
        rebuild_or_reload_window(state, dashboard, tracker);
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

fn apply_integrity_stats(
    state: &mut AppState,
    stats: &crate::sync::worker::SyncStats,
    observed_integrity_revision: &mut u64,
) {
    if stats.integrity_unavailable {
        state.integrity_status = IntegrityStatus::Unavailable;
        state.integrity_started_at = None;
        return;
    }

    if stats.integrity_revision > *observed_integrity_revision {
        *observed_integrity_revision = stats.integrity_revision;
        if let Some(verification) = stats.integrity_verification.as_ref() {
            let duration = state
                .integrity_started_at
                .take()
                .map(|started_at| started_at.elapsed())
                .unwrap_or_default();
            state.integrity_status =
                crate::integrity_status_from_verification(verification, duration);
            return;
        }
    }

    if stats.integrity_verification.is_some() {
        return;
    }

    if stats.running {
        state.integrity_status = IntegrityStatus::Checking;
        state.integrity_started_at.get_or_insert_with(Instant::now);
    } else {
        state.integrity_status = IntegrityStatus::Pending;
        state.integrity_started_at = None;
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
    let mut dashboard = data::build(state);
    let mut input = InputLine::new();
    let mut history = CommandHistory::new();
    let mut pending_events: VecDeque<Event> = VecDeque::new();
    let mut notice = None;
    let mut help: Option<HelpView> = None;
    let mut observed_sync_revision = 0_u64;
    let mut observed_integrity_revision = 0_u64;
    let mut observed_remote_cache_revision = 0_u64;
    let mut sync_status = crate::current_sync_status(sync_worker.as_ref());
    let mut refresh_tracker = RawRefreshTracker::default();
    let mut source_refresh_tracker = SourceRefreshTracker::default();
    let initial_range = crate::raw_cache_visible_range(&state.time_window, Local::now());
    request_background_reload_to(state, &mut refresh_tracker, initial_range);

    let monitor_interval = |state: &AppState| Duration::from_secs(state.monitor_interval);
    let mut next_refresh = Instant::now() + monitor_interval(state);
    let machine_id = state.local_host_id.clone().unwrap_or_default();
    let mut next_sync =
        Instant::now() + crate::monitor_sync_delay(monitor_interval(state), &machine_id);
    let mut next_auto_update = config.auto_update.then(Instant::now);
    let mut initial_load_pending = true;
    let mut initial_source_refresh_pending = true;

    'monitor: loop {
        if crate::poll_version_cache(state) {
            dashboard = data::build(state);
        }

        if crate::poll_background_raw_refresh(state) {
            let stale_host = refresh_tracker.active_host.as_deref() != state.host.as_deref();
            let follow_up = refresh_tracker.complete(&state.host);
            if stale_host {
                if let Some(stale) = state.raw_cache.take() {
                    crate::retire_raw_cache(stale);
                }
            } else {
                dashboard = data::build(state);
                if initial_load_pending {
                    notice = make_notice(vec![load_duration_text(load_started.elapsed())]);
                    initial_load_pending = false;
                }
            }
            if let Some(kind) = follow_up {
                request_follow_up(state, &mut refresh_tracker, kind);
            } else if !stale_host && crate::raw_cache_needs_prefetch(state, Local::now()) {
                request_background_prefetch(state, &mut refresh_tracker);
            } else if !stale_host && initial_source_refresh_pending {
                source_refresh_tracker.request(state);
                initial_source_refresh_pending = false;
            }
        } else if refresh_tracker.is_running() && state.raw_refresh.is_none() {
            // A worker can disconnect without sending its snapshot. Let the
            // next refresh request start normally instead of keeping the
            // single-flight guard stuck forever.
            refresh_tracker.abandon();
            let range = crate::raw_cache_visible_range(&state.time_window, Local::now());
            request_background_reload_to(state, &mut refresh_tracker, range);
        }

        if let Some(completion) = source_refresh_tracker.poll() {
            let follow_up_generation = completion
                .refresh
                .as_ref()
                .map(|refresh| refresh.generation.clone())
                .or_else(|| {
                    state
                        .raw_cache
                        .as_ref()
                        .map(|cache| cache.persistent_generation.clone())
                });
            let changed = completion
                .refresh
                .as_ref()
                .is_some_and(|refresh| refresh.changed);
            if completion.refresh.is_some() {
                apply_source_refresh_result(state, &mut dashboard, &mut refresh_tracker, changed);
                if changed && sync_worker.is_some() {
                    next_sync = crate::monitor_sync_deadline_after_refresh(
                        Instant::now(),
                        next_sync,
                        monitor_interval(state),
                        &machine_id,
                    );
                }
            }
            if completion.follow_up {
                source_refresh_tracker.start(follow_up_generation);
            }
        }

        if let Some(stats) =
            crate::poll_sync_worker_status(sync_worker.as_ref(), &mut observed_sync_revision)
        {
            apply_integrity_stats(state, &stats, &mut observed_integrity_revision);
            if !stats.running
                && stats.last_error.is_none()
                && stats.remote_cache_revision > observed_remote_cache_revision
            {
                observed_remote_cache_revision = stats.remote_cache_revision;
                request_background_reload(state, &mut refresh_tracker);
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
            source_refresh_tracker.request(state);
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
                        rebuild_or_reload_window(state, &mut dashboard, &mut refresh_tracker);
                        next_refresh = Instant::now() + monitor_interval(state);
                    }
                    Effect::ViewChanged => {
                        data::rebuild_view(&mut dashboard, state.table_view);
                    }
                    Effect::ReloadRefresh => {
                        rebuild_or_reload_window(state, &mut dashboard, &mut refresh_tracker);
                        source_refresh_tracker.request(state);
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
                    rebuild_or_reload_window(state, &mut dashboard, &mut refresh_tracker);
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
                    rebuild_or_reload_window(state, &mut dashboard, &mut refresh_tracker);
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
                    rebuild_or_reload_window(state, &mut dashboard, &mut refresh_tracker);
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
                        rebuild_or_reload_window(state, &mut dashboard, &mut refresh_tracker);
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
                    rebuild_or_reload_window(state, &mut dashboard, &mut refresh_tracker);
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
                rebuild_or_reload_window(state, &mut dashboard, &mut refresh_tracker);
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
    use std::collections::HashMap;

    use crate::constants::{AllPricing, SubscriptionFees};
    use crate::table_view::TableView;
    use crate::time_utils::TimeWindow;
    use chrono::Duration as ChronoDuration;

    fn range(start_day: i64, end_day: i64) -> crate::RawDataRange {
        let origin =
            crate::time_utils::parse_timestamp("2026-01-01T00:00:00Z").expect("range origin");
        crate::RawDataRange::from_bounds(
            origin + ChronoDuration::days(start_day),
            origin + ChronoDuration::days(end_day),
        )
    }

    fn state_with_hot_cache() -> AppState {
        let now = Local::now();
        AppState {
            tool: "all".to_string(),
            table_view: TableView::Flat,
            host: None,
            session_id: None,
            local_host_id: None,
            days: 3,
            time_window: TimeWindow::rolling_days(3),
            monitor_interval: 3600,
            pricing: AllPricing::load_raw().finalize(),
            subscription_fees: SubscriptionFees::default(),
            version_cache: HashMap::new(),
            all_tool_prompt: None,
            raw_cache: Some(crate::RawDataCache {
                claude: Vec::new(),
                codex: Vec::new(),
                gemini: Vec::new(),
                kimi: Vec::new(),
                omp: Vec::new(),
                range: crate::RawDataRange::from_bounds(
                    now - ChronoDuration::days(8),
                    now + ChronoDuration::days(8),
                ),
                has_source_data: false,
                local_host_id: None,
                local_record_keys: HashMap::new(),
                persistent_generation: String::new(),
                local_session_metadata_current: true,
            }),
            raw_refresh: None,
            integrity_status: IntegrityStatus::Checked {
                duration: Duration::ZERO,
            },
            integrity_started_at: None,
        }
    }

    #[test]
    fn uncovered_window_rebuilds_immediately_and_queues_cache_expansion() {
        let mut state = state_with_hot_cache();
        let mut dashboard = data::build(&mut state);
        assert!(dashboard.window_label.contains("last 3 days"));

        state.days = 30;
        state.time_window = TimeWindow::rolling_days(30);
        let target_range = crate::raw_cache_target_range(&state.time_window, Local::now());
        let mut tracker = RawRefreshTracker {
            active_host: None,
            active: Some(RawLoadRequest::cached(range(0, 1))),
            follow_up: None,
        };

        rebuild_or_reload_window(&mut state, &mut dashboard, &mut tracker);

        assert!(dashboard.window_label.contains("last 30 days"));
        assert_eq!(
            tracker.follow_up,
            Some(RawLoadRequest::prefetch(target_range))
        );
    }

    #[test]
    fn covered_window_near_cache_edge_queues_prefetch() {
        let mut state = state_with_hot_cache();
        let now = Local::now();
        state.time_window = state
            .time_window
            .slide_back_by(now, ChronoDuration::days(2))
            .expect("historical window");
        assert!(crate::raw_cache_covers_window(&state, now));

        let mut dashboard = data::build(&mut state);
        let target_range = crate::raw_cache_target_range(&state.time_window, Local::now());
        let mut tracker = RawRefreshTracker {
            active_host: None,
            active: Some(RawLoadRequest::cached(range(0, 1))),
            follow_up: None,
        };

        rebuild_or_reload_window(&mut state, &mut dashboard, &mut tracker);

        assert_eq!(
            tracker.follow_up,
            Some(RawLoadRequest::prefetch(target_range))
        );
    }

    #[test]
    fn active_prefetch_covering_the_window_suppresses_duplicate_reload() {
        let mut state = state_with_hot_cache();
        state.days = 7;
        state.time_window = TimeWindow::rolling_days(7);
        let target_range = crate::raw_cache_target_range(&state.time_window, Local::now());
        let mut dashboard = data::build(&mut state);
        let mut tracker = RawRefreshTracker {
            active_host: None,
            active: Some(RawLoadRequest {
                kind: RawLoadKind::Prefetch,
                range: target_range,
            }),
            follow_up: None,
        };

        rebuild_or_reload_window(&mut state, &mut dashboard, &mut tracker);

        assert_eq!(tracker.follow_up, None);
    }

    #[test]
    fn continuous_prefetch_requests_coalesce_to_the_latest_edge() {
        let host = None;
        let mut tracker = RawRefreshTracker::default();
        let first = range(0, 10);
        let second = range(-10, 0);
        let latest = range(-20, -10);

        assert!(tracker.request_prefetch(&host, first));
        assert!(!tracker.request_prefetch(&host, second));
        assert!(!tracker.request_prefetch(&host, latest));

        assert_eq!(
            tracker.complete(&host),
            Some(RawLoadRequest::prefetch(latest))
        );
    }

    #[test]
    fn reversed_navigation_replaces_an_obsolete_prefetch_edge() {
        let host = None;
        let mut tracker = RawRefreshTracker::default();
        let first = range(0, 10);
        let older = range(-20, -10);
        let reversed = range(-10, 0);

        assert!(tracker.request_prefetch(&host, first));
        assert!(!tracker.request_prefetch(&host, older));
        assert!(!tracker.request_prefetch(&host, reversed));

        assert_eq!(
            tracker.complete(&host),
            Some(RawLoadRequest::prefetch(reversed))
        );
    }

    #[test]
    fn returning_to_resident_history_cancels_an_obsolete_prefetch_edge() {
        let mut state = state_with_hot_cache();
        let current = crate::raw_cache_target_range(&state.time_window, Local::now());
        state.raw_cache.as_mut().unwrap().range = current;
        let mut dashboard = data::build(&mut state);
        let mut tracker = RawRefreshTracker::default();
        let historical = range(-100, -50);

        assert!(tracker.request_prefetch(&state.host, current));
        assert!(!tracker.request_prefetch(&state.host, historical));
        rebuild_or_reload_window(&mut state, &mut dashboard, &mut tracker);

        assert_eq!(tracker.follow_up, None);
    }

    #[test]
    fn returning_to_resident_does_not_follow_a_distant_active_prefetch() {
        let mut state = state_with_hot_cache();
        let current = crate::raw_cache_target_range(&state.time_window, Local::now());
        state.raw_cache.as_mut().unwrap().range = current;
        let mut dashboard = data::build(&mut state);
        let mut tracker = RawRefreshTracker {
            active_host: None,
            active: Some(RawLoadRequest::prefetch(range(-120, -80))),
            follow_up: Some(RawLoadRequest::prefetch(range(-160, -120))),
        };

        rebuild_or_reload_window(&mut state, &mut dashboard, &mut tracker);

        assert_eq!(tracker.follow_up, None);
    }

    #[test]
    fn navigation_does_not_compact_the_resident_cache_on_the_input_thread() {
        let mut state = state_with_hot_cache();
        let resident_range = state.raw_cache.as_ref().unwrap().range;
        let mut dashboard = data::build(&mut state);
        let mut tracker = RawRefreshTracker {
            active_host: None,
            active: Some(RawLoadRequest::cached(range(0, 1))),
            follow_up: None,
        };

        rebuild_or_reload_window(&mut state, &mut dashboard, &mut tracker);

        assert_eq!(state.raw_cache.as_ref().unwrap().range, resident_range);
    }

    #[test]
    fn navigation_queues_a_new_target_ahead_of_an_undersized_active_load() {
        let mut state = state_with_hot_cache();
        state.days = 7;
        state.time_window = TimeWindow::rolling_days(7);
        let mut dashboard = data::build(&mut state);
        let target_range = crate::raw_cache_target_range(&state.time_window, Local::now());
        let mut tracker = RawRefreshTracker {
            active_host: None,
            active: Some(RawLoadRequest::cached(range(0, 1))),
            follow_up: None,
        };

        rebuild_or_reload_window(&mut state, &mut dashboard, &mut tracker);

        assert_eq!(
            tracker.follow_up,
            Some(RawLoadRequest::prefetch(target_range))
        );
    }

    #[test]
    fn changed_host_queues_one_replacement_refresh_after_the_old_request_finishes() {
        let mut tracker = RawRefreshTracker::default();
        let all_hosts = None;
        let remote_host = Some("build-host".to_string());
        let first = range(0, 8);
        let second = range(5, 13);

        assert!(tracker.request_prefetch(&all_hosts, first));
        assert!(!tracker.request_prefetch(&remote_host, second));
        assert_eq!(
            tracker.complete(&remote_host),
            Some(RawLoadRequest::prefetch(second))
        );

        assert!(tracker.request_prefetch(&remote_host, second));
        assert_eq!(tracker.complete(&remote_host), None);
    }

    #[test]
    fn returning_to_the_active_host_does_not_queue_an_unneeded_refresh() {
        let mut tracker = RawRefreshTracker::default();
        let all_hosts = None;
        let remote_host = Some("build-host".to_string());
        let first = range(0, 8);
        let second = range(5, 13);

        assert!(tracker.request_prefetch(&all_hosts, first));
        assert!(!tracker.request_prefetch(&remote_host, second));
        assert!(!tracker.request_prefetch(&all_hosts, first));
        assert_eq!(tracker.complete(&all_hosts), None);
    }

    #[test]
    fn cache_generation_change_queues_a_reload_behind_prefetch() {
        let mut tracker = RawRefreshTracker::default();
        let host = None;
        let requested = range(0, 11);

        assert!(tracker.request_prefetch(&host, requested));
        assert!(!tracker.request_reload(&host, requested));
        assert_eq!(
            tracker.complete(&host),
            Some(RawLoadRequest::cached(requested))
        );
    }

    #[test]
    fn source_refresh_does_not_occupy_the_history_prefetch_slot() {
        let (_source_tx, source_rx) = mpsc::channel();
        let source_tracker = SourceRefreshTracker {
            receiver: Some(source_rx),
            pending: false,
        };
        let mut tracker = RawRefreshTracker::default();
        let host = None;

        assert!(source_tracker.is_running());
        assert!(tracker.request_prefetch(&host, range(0, 23)));
    }

    #[test]
    fn overlapping_source_refresh_requests_coalesce_into_one_follow_up() {
        let (source_tx, source_rx) = mpsc::channel();
        let mut source_tracker = SourceRefreshTracker {
            receiver: Some(source_rx),
            pending: false,
        };
        let state = state_with_hot_cache();

        source_tracker.request(&state);
        source_tracker.request(&state);
        assert!(source_tracker.pending);
        source_tx
            .send(crate::BackgroundSourceRefresh {
                changed: false,
                generation: "refreshed-generation".to_string(),
            })
            .expect("finish source refresh");

        let completion = source_tracker.poll().expect("completed source refresh");

        assert!(completion.follow_up);
        let refresh = completion.refresh.expect("refresh result");
        assert!(!refresh.changed);
        assert_eq!(refresh.generation, "refreshed-generation");
        assert!(!source_tracker.pending);
        assert!(!source_tracker.is_running());
        assert!(source_tracker.poll().is_none());
    }

    #[test]
    fn disconnected_source_refresh_preserves_a_pending_request() {
        let (source_tx, source_rx) = mpsc::channel();
        let mut source_tracker = SourceRefreshTracker {
            receiver: Some(source_rx),
            pending: false,
        };
        let state = state_with_hot_cache();

        source_tracker.request(&state);
        drop(source_tx);

        let completion = source_tracker
            .poll()
            .expect("disconnected refresh completion");
        assert!(completion.refresh.is_none());
        assert!(completion.follow_up);
        assert!(!source_tracker.pending);
        assert!(!source_tracker.is_running());
    }

    #[test]
    fn unchanged_source_refresh_rebuilds_the_live_window() {
        let mut state = state_with_hot_cache();
        state.raw_cache.as_mut().unwrap().range =
            crate::raw_cache_target_range(&state.time_window, Local::now());
        let mut dashboard = data::build(&mut state);
        dashboard.window_label = "stale window".to_string();
        let mut tracker = RawRefreshTracker::default();

        apply_source_refresh_result(&mut state, &mut dashboard, &mut tracker, false);

        assert_ne!(dashboard.window_label, "stale window");
        assert!(!tracker.is_running());
    }

    #[test]
    fn changed_source_refresh_starts_one_cached_load() {
        let mut state = state_with_hot_cache();
        state.raw_cache.as_mut().unwrap().range =
            crate::raw_cache_visible_range(&state.time_window, Local::now());
        let (_raw_tx, raw_rx) = mpsc::channel();
        state.raw_refresh = Some(raw_rx);
        let mut dashboard = data::build(&mut state);
        let mut tracker = RawRefreshTracker::default();

        apply_source_refresh_result(&mut state, &mut dashboard, &mut tracker, true);

        assert_eq!(tracker.active.unwrap().kind, RawLoadKind::Cached);
        assert_eq!(tracker.follow_up, None);
    }

    #[test]
    fn completed_cycle_without_integrity_result_becomes_pending() {
        let mut state = state_with_hot_cache();
        state.integrity_status = IntegrityStatus::Checking;
        state.integrity_started_at = Some(Instant::now());
        let stats = crate::sync::worker::SyncStats {
            running: false,
            last_error: Some("network unavailable".to_string()),
            ..Default::default()
        };
        let mut observed = 0;

        apply_integrity_stats(&mut state, &stats, &mut observed);

        assert_eq!(state.integrity_status, IntegrityStatus::Pending);
        assert!(state.integrity_started_at.is_none());
    }

    #[test]
    fn missed_integrity_start_without_result_clears_previous_success() {
        let mut state = state_with_hot_cache();
        state.integrity_status = IntegrityStatus::Checked {
            duration: Duration::from_millis(20),
        };
        let stats = crate::sync::worker::SyncStats {
            running: false,
            last_error: Some("network unavailable".to_string()),
            ..Default::default()
        };
        let mut observed = 0;

        apply_integrity_stats(&mut state, &stats, &mut observed);

        assert_eq!(state.integrity_status, IntegrityStatus::Pending);
        assert!(state.integrity_started_at.is_none());
    }

    #[test]
    fn later_progress_preserves_an_observed_integrity_result() {
        let mut state = state_with_hot_cache();
        let checked = IntegrityStatus::Checked {
            duration: Duration::from_millis(20),
        };
        state.integrity_status = checked;
        let mut observed = 1;
        let mut stats = crate::sync::worker::SyncStats {
            running: true,
            integrity_revision: 1,
            integrity_verification: Some(crate::sync::integrity::IntegrityVerification::Checked {
                checked_hosts: 1,
            }),
            ..Default::default()
        };

        apply_integrity_stats(&mut state, &stats, &mut observed);
        assert_eq!(state.integrity_status, checked);

        stats.running = false;
        apply_integrity_stats(&mut state, &stats, &mut observed);
        assert_eq!(state.integrity_status, checked);
    }

    #[test]
    fn unsupported_integrity_check_is_not_left_checking() {
        let mut state = state_with_hot_cache();
        state.integrity_status = IntegrityStatus::Checking;
        let stats = crate::sync::worker::SyncStats {
            running: true,
            integrity_unavailable: true,
            ..Default::default()
        };
        let mut observed = 0;

        apply_integrity_stats(&mut state, &stats, &mut observed);

        assert_eq!(state.integrity_status, IntegrityStatus::Unavailable);
    }

    #[test]
    fn completed_unsupported_integrity_cycle_remains_unavailable() {
        let mut state = state_with_hot_cache();
        state.integrity_status = IntegrityStatus::Checking;
        state.integrity_started_at = Some(Instant::now());
        let stats = crate::sync::worker::SyncStats {
            running: false,
            integrity_unavailable: true,
            ..Default::default()
        };
        let mut observed = 0;

        apply_integrity_stats(&mut state, &stats, &mut observed);

        assert_eq!(state.integrity_status, IntegrityStatus::Unavailable);
        assert!(state.integrity_started_at.is_none());
    }
}
