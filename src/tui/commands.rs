//! Monitor prompt commands: parsing and application to `AppState`.
//!
//! Commands mutate state and report what happened; the event loop decides
//! how to react (rebuild the dashboard, reload raw data, exit, ...). This
//! keeps the command surface testable without a terminal.

use chrono::Local;

use crate::table_view::TableView;
use crate::time_utils::TimeWindow;
use crate::tool::Tool;
use crate::{
    AppState, DAY_PRESET_DAYS, MONTH_PRESET_DAYS, WEEK_PRESET_DAYS, YEAR_PRESET_DAYS,
    get_tool_data_dir, host_label, is_current_rolling_days_preset, known_host_ids,
    parse_host_selection, parse_time_window_command,
};

/// What the event loop must do after a command ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// Nothing changed; show the messages only.
    None,
    /// State changed; rebuild the dashboard from the cached raw data.
    Refresh,
    /// State changed and the raw cache is stale; kick a background reload
    /// and rebuild.
    ReloadRefresh,
    /// Only the table view changed; reshape the table from cached data.
    ViewChanged,
    /// The refresh interval changed; reschedule timers.
    IntervalChanged,
    /// Toggle the help index overlay.
    Help,
    /// Open the detail page for one help topic.
    HelpTopic(usize),
    /// Leave monitor mode.
    Exit,
    /// Run the self-updater (the loop suspends the terminal around it).
    Update,
}

/// Which page of the help overlay is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpView {
    /// Command index: every command with a one-line summary.
    Index,
    /// Detail page for `help_topics()[i]`.
    Topic(usize),
}

pub struct Outcome {
    pub messages: Vec<String>,
    pub effect: Effect,
}

impl Outcome {
    fn new(effect: Effect) -> Self {
        Outcome {
            messages: Vec::new(),
            effect,
        }
    }

    fn message(effect: Effect, message: impl Into<String>) -> Self {
        Outcome {
            messages: vec![message.into()],
            effect,
        }
    }
}

fn switch_days_preset(state: &mut AppState, days: i64) -> Outcome {
    if is_current_rolling_days_preset(&state.time_window, days) {
        return Outcome::message(
            Effect::None,
            format!(
                "Already showing last {} day{}.",
                days,
                if days == 1 { "" } else { "s" }
            ),
        );
    }
    state.days = days;
    state.time_window = TimeWindow::rolling_days(days);
    Outcome::message(
        Effect::Refresh,
        format!(
            "Changed to {} day{}",
            days,
            if days == 1 { "" } else { "s" }
        ),
    )
}

/// Rotate the active tool through `Tool::ROTATION` by `step` (+1 forward,
/// -1 backward), skipping tools whose data directory does not exist. Drives
/// the `n` command and the Tab / Shift+Tab tab-bar keys.
pub fn rotate_tool(state: &mut AppState, step: isize) -> Outcome {
    let rotation = Tool::ROTATION;
    let len = rotation.len() as isize;
    let current = Tool::from_key(&state.tool).unwrap_or(Tool::All);
    let mut idx = rotation
        .iter()
        .position(|tool| *tool == current)
        .unwrap_or(0) as isize;
    let mut messages = Vec::new();
    for _ in 0..rotation.len() {
        idx = (idx + step).rem_euclid(len);
        let candidate = rotation[idx as usize];
        if let Some(dir) = get_tool_data_dir(candidate.key())
            && !dir.exists()
        {
            messages.push(format!(
                "Skipping {} (no data dir)",
                candidate.display_name()
            ));
            continue;
        }
        state.tool = candidate.key().to_string();
        messages.push(format!("Switched to {}", candidate.display_name()));
        return Outcome {
            messages,
            effect: Effect::Refresh,
        };
    }
    Outcome::message(Effect::None, "No tool with a data directory found.")
}

fn switch_tool(state: &mut AppState, requested: &str) -> Outcome {
    match Tool::from_key(requested) {
        Some(new_tool) => {
            if let Some(dir) = get_tool_data_dir(new_tool.key())
                && !dir.exists()
            {
                return Outcome::message(
                    Effect::None,
                    format!("Error: Data directory not found at {}", dir.display()),
                );
            }
            state.tool = new_tool.key().to_string();
            Outcome::message(
                Effect::Refresh,
                format!("Switched to {}", new_tool.display_name()),
            )
        }
        None => Outcome::message(
            Effect::None,
            "Usage: t, tool [claude|codex|gemini|kimi|omp|all]",
        ),
    }
}

fn switch_host(state: &mut AppState, selection: &str) -> Outcome {
    match parse_host_selection(selection) {
        Ok(new_host) => {
            if state.host == new_host {
                return Outcome::message(
                    Effect::None,
                    format!(
                        "Already showing host {}.",
                        host_label(state.host.as_deref())
                    ),
                );
            }
            state.host = new_host;
            if let Some(cache) = state.raw_cache.take() {
                crate::retire_raw_cache(cache);
            }
            state.raw_cache_last_used_at = None;
            Outcome::message(
                Effect::ReloadRefresh,
                format!("Switched to host {}", host_label(state.host.as_deref())),
            )
        }
        Err(err) => Outcome::message(Effect::None, err),
    }
}

fn set_view(state: &mut AppState, view: TableView) -> Outcome {
    state.table_view = view;
    Outcome::message(
        Effect::ViewChanged,
        format!("Table view: {}", view.description()),
    )
}

fn set_session(state: &mut AppState, session_id: Option<&str>) -> Outcome {
    let Some(session_id) = session_id else {
        return Outcome {
            messages: vec![
                format!(
                    "Current session: {}",
                    state.session_id.as_deref().unwrap_or("all")
                ),
                "Usage: session <ID> | session clear".to_string(),
            ],
            effect: Effect::None,
        };
    };
    if session_id.eq_ignore_ascii_case("clear") || session_id.eq_ignore_ascii_case("all") {
        if state.session_id.take().is_some() {
            return Outcome::message(Effect::Refresh, "Showing all sessions.");
        }
        return Outcome::message(Effect::None, "Already showing all sessions.");
    }
    if session_id.is_empty() {
        return Outcome::message(Effect::None, "Usage: session <ID> | session clear");
    }
    if state.session_id.as_deref() == Some(session_id) {
        return Outcome::message(
            Effect::None,
            format!("Already tracking session {session_id}."),
        );
    }
    state.session_id = Some(session_id.to_string());
    Outcome::message(
        Effect::ReloadRefresh,
        format!("Tracking session {session_id}; refreshing source logs."),
    )
}

/// Execute one prompt command against the app state.
pub fn execute(state: &mut AppState, raw: &str) -> Outcome {
    let command = raw.trim();
    match command {
        "" | "r" | "refresh" => Outcome::message(Effect::ReloadRefresh, "Refreshing"),
        "n" => rotate_tool(state, 1),
        "a" => {
            if state.tool != "all" {
                state.tool = "all".to_string();
                Outcome::message(Effect::Refresh, "Switched to All Tools")
            } else {
                Outcome::message(Effect::None, "Already monitoring all tools.")
            }
        }
        "v" | "view" => set_view(state, state.table_view.next()),
        "d" | "day" | "days" => switch_days_preset(state, DAY_PRESET_DAYS),
        "w" | "week" => switch_days_preset(state, WEEK_PRESET_DAYS),
        "m" | "month" => switch_days_preset(state, MONTH_PRESET_DAYS),
        "y" | "year" => switch_days_preset(state, YEAR_PRESET_DAYS),
        "h" | "help" => Outcome::new(Effect::Help),
        "e" | "exit" => Outcome::new(Effect::Exit),
        "update" | "upgrade" => Outcome::new(Effect::Update),
        "t" | "tool" => Outcome {
            messages: vec![
                format!("Current tool: {}", state.tool),
                "Usage: t, tool [claude|codex|gemini|kimi|omp|all]".to_string(),
            ],
            effect: Effect::None,
        },
        "host" => {
            let known_hosts = known_host_ids(state.local_host_id.as_deref());
            let mut messages = vec![
                format!("Current host: {}", host_label(state.host.as_deref())),
                "Usage: host [all|HOST]".to_string(),
            ];
            if !known_hosts.is_empty() {
                messages.push(format!("Known hosts: {}", known_hosts.join(", ")));
            }
            Outcome {
                messages,
                effect: Effect::None,
            }
        }
        "i" | "interval" => Outcome {
            messages: vec![
                format!("Current interval: {} seconds", state.monitor_interval),
                "Usage: i <N> or interval <N>".to_string(),
            ],
            effect: Effect::None,
        },
        "session" => set_session(state, None),
        _ => {
            let now = Local::now();
            if let Some(parsed) = parse_time_window_command(command, &state.time_window, now) {
                return match parsed {
                    Ok(window) => {
                        state.time_window = window;
                        Outcome::message(
                            Effect::Refresh,
                            format!("Time window: {}", state.time_window.display_label(now)),
                        )
                    }
                    Err(err) => Outcome::message(Effect::None, err),
                };
            }

            let parts: Vec<&str> = command.splitn(2, ' ').collect();
            if parts.len() != 2 {
                return Outcome::message(
                    Effect::None,
                    format!("Unknown command: '{}'. Type h for help.", command),
                );
            }
            let arg = parts[1].trim();
            match parts[0] {
                "h" | "help" => match find_help_topic(arg) {
                    Some(idx) => Outcome::new(Effect::HelpTopic(idx)),
                    None => Outcome::message(
                        Effect::None,
                        format!("No help topic '{}'. Type h for the index.", arg),
                    ),
                },
                "v" | "view" => match TableView::from_key(arg) {
                    Some(view) => set_view(state, view),
                    None => Outcome::message(Effect::None, "Usage: v, view [flat|vendor|model]"),
                },
                "t" | "tool" => switch_tool(state, arg),
                "host" => switch_host(state, arg),
                "session" => set_session(state, Some(arg)),
                "d" | "day" | "days" => match arg.parse::<i64>() {
                    Ok(n) if n >= 1 => {
                        state.days = n;
                        state.time_window = TimeWindow::rolling_days(n);
                        Outcome::message(Effect::Refresh, format!("Changed to {} days", n))
                    }
                    Ok(_) => Outcome::message(Effect::None, "Days must be at least 1."),
                    Err(_) => Outcome::message(Effect::None, "Invalid days value."),
                },
                "i" | "interval" => match arg.parse::<u64>() {
                    Ok(n) if n >= 1 => {
                        state.monitor_interval = n;
                        Outcome::message(
                            Effect::IntervalChanged,
                            format!("Refresh interval changed to {} seconds.", n),
                        )
                    }
                    Ok(_) => Outcome::message(Effect::None, "Interval must be at least 1 second."),
                    Err(_) => Outcome::message(Effect::None, "Invalid interval value."),
                },
                _ => Outcome::message(
                    Effect::None,
                    format!("Unknown command: '{}'. Type h for help.", command),
                ),
            }
        }
    }
}

/// One help topic: index summary plus a detail page.
pub struct HelpTopic {
    /// Canonical topic name (what `h <name>` matches).
    pub name: &'static str,
    /// Display form of the command with aliases/arguments.
    pub invocation: &'static str,
    /// Extra lookup keys (aliases) that `h <key>` also matches.
    pub keys: &'static [&'static str],
    pub summary: &'static str,
    pub detail: &'static [&'static str],
}

static HELP_TOPICS: &[HelpTopic] = &[
    HelpTopic {
        name: "session",
        invocation: "session <ID> | session clear",
        keys: &[],
        summary: "Track usage from one conversation",
        detail: &[
            "Usage: session <ID>    show only one harness session",
            "       session clear   return to every session",
            "       session         show the active session filter",
            "",
            "The selected id is applied together with the tool, host, and",
            "time-window filters. Setting an id starts a background source",
            "refresh so newly parsed session metadata is available.",
        ],
    },
    HelpTopic {
        name: "refresh",
        invocation: "r, refresh, empty Enter",
        keys: &["r"],
        summary: "Reload raw data and redraw",
        detail: &[
            "Usage: r | refresh | press Enter on an empty prompt",
            "",
            "Rescans every tool's session logs in the background and redraws",
            "the dashboard from the freshest cache. The auto-refresh countdown",
            "restarts. Sync (when configured) is requested right after the",
            "rescan completes.",
        ],
    },
    HelpTopic {
        name: "tool",
        invocation: "t <X> | n | a | Tab",
        keys: &["t", "n", "a", "tab"],
        summary: "Switch harness (claude|codex|gemini|kimi|omp|all)",
        detail: &[
            "Usage: t <claude|codex|gemini|kimi|omp|all>",
            "       n              rotate to the next tool",
            "       Tab / Shift+Tab cycle the header tabs forward / backward",
            "       a              jump straight to tool=all",
            "",
            "Rotation skips tools whose data directory does not exist.",
            "tool=all shows every harness combined: the table gains a Harness",
            "column and the chart compares total consumption per harness.",
        ],
    },
    HelpTopic {
        name: "view",
        invocation: "v | view <X>",
        keys: &["v"],
        summary: "Cycle or set table shape (flat|vendor|model)",
        detail: &[
            "Usage: v                cycle flat -> vendor -> model",
            "       view <flat|vendor|model>",
            "",
            "flat:   one row per model x harness; Vendor / Model / Harness as",
            "        columns, vendor shown once per group.",
            "vendor: rows grouped under a vendor heading with per-vendor",
            "        subtotal rows.",
            "model:  one row per model, merged across harnesses; the Harness",
            "        column lists tags like CC,OMP.",
            "",
            "Also available at startup: --view <flat|vendor|model>.",
        ],
    },
    HelpTopic {
        name: "host",
        invocation: "host [all|ID]",
        keys: &[],
        summary: "Filter usage to one machine (needs sync)",
        detail: &[
            "Usage: host           show current host and known machine ids",
            "       host all       include every machine",
            "       host <ID>      only records from machine <ID>",
            "",
            "Host filtering relies on the sync cache; remote machines appear",
            "after their records have been pulled.",
        ],
    },
    HelpTopic {
        name: "window",
        invocation: "d | w | m | y | days N",
        keys: &["d", "w", "m", "y", "day", "days", "week", "month", "year"],
        summary: "Rolling window presets: 1 / 7 / 30 / 365 days",
        detail: &[
            "Usage: d | day        last 1 day",
            "       w | week       last 7 days",
            "       m | month      last 30 days",
            "       y | year       last 365 days",
            "       days <N>       rolling window of N days",
            "",
            "Cost projections (daily/weekly/monthly) scale from the data",
            "inside the selected window.",
        ],
    },
    HelpTopic {
        name: "date",
        invocation: "date YYYY-MM-DD",
        keys: &[],
        summary: "Show one complete local day",
        detail: &[
            "Usage: date 2026-07-21",
            "",
            "Pins the window to that local calendar day (midnight to",
            "midnight). Use `latest` to return to a rolling window.",
        ],
    },
    HelpTopic {
        name: "range",
        invocation: "range A B",
        keys: &[],
        summary: "Show an inclusive local date span",
        detail: &[
            "Usage: range 2026-07-01 2026-07-15",
            "",
            "Both endpoints are included; the order does not matter.",
            "Use `latest` to return to a rolling window.",
        ],
    },
    HelpTopic {
        name: "latest",
        invocation: "latest",
        keys: &[],
        summary: "Follow the present with the current span",
        detail: &[
            "Usage: latest",
            "",
            "Keeps the current window width and anchors its newest edge at",
            "now. Each refresh advances both bounds by the elapsed time.",
        ],
    },
    HelpTopic {
        name: "interval",
        invocation: "i, interval <N>",
        keys: &["i"],
        summary: "Change the auto-refresh interval (seconds)",
        detail: &[
            "Usage: i <seconds> | interval <seconds>",
            "",
            "Sets how often the dashboard reloads data automatically; the",
            "countdown is shown in the header. Sync (when configured) runs at",
            "one third of this interval. Bare `i` prints the current value.",
        ],
    },
    HelpTopic {
        name: "keys",
        invocation: "PgUp/PgDn, arrows, +/-",
        keys: &["navigation", "pgup", "pgdn"],
        summary: "Keyboard time-window navigation",
        detail: &[
            "Tab / Shift+Tab cycle the tool tabs forward / backward",
            "PgUp / PgDn    slide the window back / forward by its own width",
            "               (PgDn snaps to the present at the newest edge)",
            "Left / Right   empty prompt: step newer / older by one chart",
            "               interval; while typing: move the cursor",
            "+ / -          empty prompt: zoom the window in / out",
            "Up / Down      walk the command history",
            "Esc            close help, or clear the prompt",
        ],
    },
    HelpTopic {
        name: "update",
        invocation: "update, upgrade",
        keys: &["upgrade"],
        summary: "Download the latest release and restart",
        detail: &[
            "Usage: update",
            "",
            "Checks GitHub releases, downloads a newer binary when available,",
            "and restarts into it. The dashboard is suspended while the",
            "updater prints its progress.",
        ],
    },
    HelpTopic {
        name: "exit",
        invocation: "e, exit, Ctrl+C/D",
        keys: &["e"],
        summary: "Leave monitor mode",
        detail: &[
            "Usage: e | exit, or press Ctrl+C / Ctrl+D",
            "",
            "Restores the terminal and quits.",
        ],
    },
    HelpTopic {
        name: "help",
        invocation: "h [topic]",
        keys: &["h"],
        summary: "Toggle this help; h <topic> for details",
        detail: &[
            "Usage: h | help          toggle the command index",
            "       h <topic>         open a topic's detail page",
            "",
            "Topics match the command name or any alias, e.g. `h v`,",
            "`h view`, `h days`, `h keys`. Esc goes back to the index.",
        ],
    },
];

pub fn help_topics() -> &'static [HelpTopic] {
    HELP_TOPICS
}

pub fn find_help_topic(query: &str) -> Option<usize> {
    let q = query.trim().to_ascii_lowercase();
    HELP_TOPICS
        .iter()
        .position(|t| t.name == q || t.keys.contains(&q.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{AllPricing, SubscriptionFees};
    use std::collections::HashMap;

    fn test_state() -> AppState {
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
            raw_cache: None,
            raw_cache_last_used_at: None,
            raw_refresh: None,
            integrity_status: crate::IntegrityStatus::Checking { percent: 0 },
            integrity_started_at: None,
        }
    }

    #[test]
    fn view_command_cycles_and_sets() {
        let mut state = test_state();
        let outcome = execute(&mut state, "v");
        assert_eq!(outcome.effect, Effect::ViewChanged);
        assert_eq!(state.table_view, TableView::Vendor);

        let outcome = execute(&mut state, "view model");
        assert_eq!(outcome.effect, Effect::ViewChanged);
        assert_eq!(state.table_view, TableView::Model);

        let outcome = execute(&mut state, "view bogus");
        assert_eq!(outcome.effect, Effect::None);
        assert_eq!(state.table_view, TableView::Model);
    }

    #[test]
    fn interval_command_validates_and_reschedules() {
        let mut state = test_state();
        let outcome = execute(&mut state, "i 30");
        assert_eq!(outcome.effect, Effect::IntervalChanged);
        assert_eq!(state.monitor_interval, 30);

        let outcome = execute(&mut state, "i 0");
        assert_eq!(outcome.effect, Effect::None);
        assert_eq!(state.monitor_interval, 30);

        let outcome = execute(&mut state, "i abc");
        assert_eq!(outcome.effect, Effect::None);
    }

    #[test]
    fn host_switch_keeps_the_in_flight_refresh_receiver() {
        let mut state = test_state();
        let (_tx, rx) = std::sync::mpsc::channel();
        state.raw_refresh = Some(rx);

        let outcome = execute(&mut state, "host workstation");

        assert_eq!(outcome.effect, Effect::ReloadRefresh);
        assert!(state.raw_refresh.is_some());
    }

    #[test]
    fn day_presets_and_custom_days_change_window() {
        let mut state = test_state();
        let outcome = execute(&mut state, "w");
        assert_eq!(outcome.effect, Effect::Refresh);
        assert_eq!(state.days, WEEK_PRESET_DAYS);

        // Same preset again: no refresh.
        let outcome = execute(&mut state, "w");
        assert_eq!(outcome.effect, Effect::None);

        let outcome = execute(&mut state, "days 14");
        assert_eq!(outcome.effect, Effect::Refresh);
        assert_eq!(state.days, 14);
    }

    #[test]
    fn latest_preserves_the_current_window_span() {
        let mut state = test_state();
        state.time_window =
            TimeWindow::from_range("2026-07-27T08:00", "2026-07-27T20:00").expect("range");
        let now = Local::now();
        let (before_start, before_end) = state.time_window.bounds(now);

        let outcome = execute(&mut state, "latest");
        let later = Local::now() + chrono::Duration::minutes(5);
        let (after_start, after_end) = state.time_window.bounds(later);

        assert_eq!(outcome.effect, Effect::Refresh);
        assert_eq!(after_end - after_start, before_end - before_start);
        assert_eq!(after_end, later);
    }

    #[test]
    fn exit_help_update_and_unknown_have_expected_effects() {
        let mut state = test_state();
        assert_eq!(execute(&mut state, "e").effect, Effect::Exit);
        assert_eq!(execute(&mut state, "help").effect, Effect::Help);
        assert_eq!(execute(&mut state, "update").effect, Effect::Update);
        let outcome = execute(&mut state, "wat");
        assert_eq!(outcome.effect, Effect::None);
        assert!(outcome.messages[0].contains("Unknown command"));
    }

    #[test]
    fn help_topics_resolve_by_name_and_alias() {
        let mut state = test_state();

        let view_idx = find_help_topic("view").expect("view topic");
        assert_eq!(find_help_topic("v"), Some(view_idx));
        assert_eq!(
            execute(&mut state, "h view").effect,
            Effect::HelpTopic(view_idx)
        );
        assert_eq!(
            execute(&mut state, "help days").effect,
            Effect::HelpTopic(find_help_topic("window").expect("window topic"))
        );

        let outcome = execute(&mut state, "h nonsense");
        assert_eq!(outcome.effect, Effect::None);
        assert!(outcome.messages[0].contains("No help topic"));

        // Every topic has non-empty detail lines and a summary.
        for topic in help_topics() {
            assert!(!topic.summary.is_empty());
            assert!(!topic.detail.is_empty(), "topic {}", topic.name);
        }
    }

    #[test]
    fn refresh_reloads_raw_data() {
        let mut state = test_state();
        assert_eq!(execute(&mut state, "").effect, Effect::ReloadRefresh);
        assert_eq!(execute(&mut state, "r").effect, Effect::ReloadRefresh);
    }

    #[test]
    fn session_command_sets_and_clears_the_conversation_filter() {
        let mut state = test_state();
        let outcome = execute(&mut state, "session convo-123");
        assert_eq!(outcome.effect, Effect::ReloadRefresh);
        assert_eq!(state.session_id.as_deref(), Some("convo-123"));

        let outcome = execute(&mut state, "session clear");
        assert_eq!(outcome.effect, Effect::Refresh);
        assert_eq!(state.session_id, None);
    }

    #[test]
    fn tool_rotation_round_trips_and_lands_on_valid_tools() {
        let mut state = test_state();
        let start = state.tool.clone();

        // Forward then backward returns to the starting tool: the set of
        // skipped (missing-dir) tools is the same in both directions.
        let fwd = rotate_tool(&mut state, 1);
        assert!(Tool::from_key(&state.tool).is_some());
        if fwd.effect == Effect::Refresh {
            let back = rotate_tool(&mut state, -1);
            assert_eq!(back.effect, Effect::Refresh);
            assert_eq!(state.tool, start);
        }

        // A full forward cycle visits `all` again (it is always valid).
        for _ in 0..Tool::ROTATION.len() {
            rotate_tool(&mut state, 1);
            if state.tool == "all" {
                break;
            }
        }
        assert_eq!(state.tool, "all");
    }
}
