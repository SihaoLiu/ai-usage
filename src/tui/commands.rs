//! Monitor prompt commands: parsing and application to `AppState`.
//!
//! Commands mutate state and report what happened; the event loop decides
//! how to react (rebuild the dashboard, reload raw data, exit, ...). This
//! keeps the command surface testable without a terminal.

use chrono::Local;

use crate::constants::save_subscription_fees;
use crate::refresh::RefreshInterval;
use crate::table_view::{TableMetric, TableView};
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
    /// Only table presentation changed; reshape it from cached data.
    TableChanged,
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

fn known_host_usage_labels(
    state: &AppState,
    known_hosts: Vec<String>,
    now: chrono::DateTime<Local>,
) -> Vec<String> {
    let selected_tool = Tool::from_key(&state.tool).unwrap_or(Tool::All);
    let local_host_id = state.local_host_id.as_deref();
    let all_data = crate::load_resident_all_tool_data(state, now);
    let buckets: [(&[crate::data::UsageEntry], Tool); 5] = [
        (&all_data.claude, Tool::Claude),
        (&all_data.codex, Tool::Codex),
        (&all_data.gemini, Tool::Gemini),
        (&all_data.kimi, Tool::Kimi),
        (&all_data.omp, Tool::Omp),
    ];
    let mut totals: std::collections::HashMap<String, u128> =
        known_hosts.into_iter().map(|host| (host, 0)).collect();
    for (entries, tool) in buckets {
        if !selected_tool.is_all() && selected_tool != tool {
            continue;
        }
        for entry in entries {
            if entry.model.contains("<synthetic>") {
                continue;
            }
            let Some(host) = entry.host_id.as_deref().or(local_host_id) else {
                continue;
            };
            *totals.entry(host.to_string()).or_default() +=
                crate::stats::entry_total_with_cache(entry, tool.key());
        }
    }

    format_host_usage_totals(totals)
}

fn format_host_usage_totals(totals: std::collections::HashMap<String, u128>) -> Vec<String> {
    let grand_total: u128 = totals.values().copied().sum();
    let percentage_tenths = |tokens: u128| {
        if grand_total > 0 {
            (tokens * 1_000 + grand_total / 2) / grand_total
        } else {
            0
        }
    };
    let mut totals = totals
        .into_iter()
        .map(|(host, tokens)| (host, percentage_tenths(tokens)))
        .collect::<Vec<_>>();
    totals.sort_by(|(left_host, left_tenths), (right_host, right_tenths)| {
        right_tenths
            .cmp(left_tenths)
            .then_with(|| left_host.cmp(right_host))
    });
    totals
        .into_iter()
        .map(|(host, tenths)| {
            let percentage = if tenths % 10 == 0 {
                format!("{}", tenths / 10)
            } else {
                format!("{}.{:01}", tenths / 10, tenths % 10)
            };
            format!("{host}({percentage}%)")
        })
        .collect()
}

fn set_view(state: &mut AppState, view: TableView) -> Outcome {
    state.table_view = view;
    Outcome::message(
        Effect::TableChanged,
        format!("Table view: {}", view.description()),
    )
}

fn set_sort(state: &mut AppState, sort_metric: TableMetric) -> Outcome {
    state.sort_metric = sort_metric;
    Outcome::message(
        Effect::TableChanged,
        format!("Sort: {} (descending)", sort_metric.label()),
    )
}

fn set_refresh_interval(state: &mut AppState, arg: &str) -> Outcome {
    match RefreshInterval::parse(arg) {
        Ok(interval) => {
            state.refresh_interval = interval;
            let described = crate::describe_refresh_interval(state, Local::now());
            Outcome::message(
                Effect::IntervalChanged,
                match interval {
                    RefreshInterval::Auto => {
                        format!("Refresh interval follows the chart interval: {described}.")
                    }
                    RefreshInterval::Manual(_) => {
                        format!("Refresh interval changed to {described}.")
                    }
                },
            )
        }
        Err(message) => Outcome::message(Effect::None, message),
    }
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

const COST_USAGE: &str =
    "Usage: cost [all|claude|codex|gemini|kimi] [non-negative integer or decimal]";

fn cost_vendor(vendor: &str) -> Option<&'static str> {
    match vendor {
        "claude" => Some("Claude"),
        "codex" => Some("Codex"),
        "gemini" => Some("Gemini"),
        "kimi" => Some("Kimi"),
        _ => None,
    }
}

fn cost_summary(state: &AppState) -> Outcome {
    Outcome::message(
        Effect::None,
        format!(
            "Monthly costs: Claude ${:.2} | Codex ${:.2} | Gemini ${:.2} | Kimi ${:.2}",
            state.subscription_fees.claude,
            state.subscription_fees.codex,
            state.subscription_fees.gemini,
            state.subscription_fees.kimi
        ),
    )
}

fn parse_monthly_cost(raw: &str) -> Option<f64> {
    let valid = match raw.split_once('.') {
        Some((whole, fraction)) => {
            !whole.is_empty()
                && !fraction.is_empty()
                && whole.bytes().all(|byte| byte.is_ascii_digit())
                && fraction.bytes().all(|byte| byte.is_ascii_digit())
        }
        None => !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit()),
    };
    if !valid {
        return None;
    }
    raw.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn set_monthly_cost(state: &mut AppState, vendor: &str, raw_value: &str) -> Outcome {
    let Some(label) = cost_vendor(vendor) else {
        return Outcome::message(Effect::None, COST_USAGE);
    };
    let Some(value) = parse_monthly_cost(raw_value) else {
        return Outcome::message(Effect::None, COST_USAGE);
    };

    let mut updated = state.subscription_fees.clone();
    match vendor {
        "claude" => updated.claude = value,
        "codex" => updated.codex = value,
        "gemini" => updated.gemini = value,
        "kimi" => updated.kimi = value,
        _ => unreachable!("validated cost vendor"),
    }

    if let Err(error) = save_subscription_fees(&state.fee_env_path, &updated) {
        return Outcome::message(
            Effect::None,
            format!("Could not save monthly costs: {error}"),
        );
    }
    state.subscription_fees = updated;
    Outcome::message(
        Effect::Refresh,
        format!("{label} monthly cost set to ${value:.2} and saved to .fee.env."),
    )
}

fn cost_command(state: &mut AppState, args: &str) -> Outcome {
    let parts: Vec<&str> = args.split_whitespace().collect();
    match parts.as_slice() {
        [] | ["all"] => cost_summary(state),
        [vendor] => match cost_vendor(vendor) {
            Some(label) => Outcome::message(
                Effect::None,
                format!(
                    "{label} monthly cost: ${:.2}",
                    state.subscription_fees.get(vendor)
                ),
            ),
            None => Outcome::message(Effect::None, COST_USAGE),
        },
        [vendor, value] => set_monthly_cost(state, vendor, value),
        _ => Outcome::message(Effect::None, COST_USAGE),
    }
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
        "s" | "sort" => set_sort(state, state.sort_metric.next()),
        "d" | "day" | "days" => switch_days_preset(state, DAY_PRESET_DAYS),
        "w" | "week" => switch_days_preset(state, WEEK_PRESET_DAYS),
        "m" | "month" => switch_days_preset(state, MONTH_PRESET_DAYS),
        "y" | "year" => switch_days_preset(state, YEAR_PRESET_DAYS),
        "h" | "help" => Outcome::new(Effect::Help),
        "e" | "exit" => Outcome::new(Effect::Exit),
        "update" | "upgrade" => Outcome::new(Effect::Update),
        "cost" => cost_command(state, ""),
        "t" | "tool" => Outcome {
            messages: vec![
                format!("Current tool: {}", state.tool),
                "Usage: t, tool [claude|codex|gemini|kimi|omp|all]".to_string(),
            ],
            effect: Effect::None,
        },
        "host" => {
            let known_hosts = known_host_ids(state.local_host_id.as_deref());
            let known_hosts = known_host_usage_labels(state, known_hosts, Local::now());
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
                format!(
                    "Current interval: {}",
                    crate::describe_refresh_interval(state, Local::now())
                ),
                "Usage: i <N|auto> or interval <N|auto>".to_string(),
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
                    None => Outcome::message(Effect::None, "Usage: v, view [flat|vendor]"),
                },
                "s" | "sort" => match TableMetric::from_key(arg) {
                    Some(metric) => set_sort(state, metric),
                    None => Outcome::message(
                        Effect::None,
                        "Usage: s, sort [msgs|cache|prefill|decode|total|cost|rate]",
                    ),
                },
                "t" | "tool" => switch_tool(state, arg),
                "host" => switch_host(state, arg),
                "session" => set_session(state, Some(arg)),
                "cost" => cost_command(state, arg),
                "d" | "day" | "days" => match arg.parse::<i64>() {
                    Ok(n) if n >= 1 => {
                        state.days = n;
                        state.time_window = TimeWindow::rolling_days(n);
                        Outcome::message(Effect::Refresh, format!("Changed to {} days", n))
                    }
                    Ok(_) => Outcome::message(Effect::None, "Days must be at least 1."),
                    Err(_) => Outcome::message(Effect::None, "Invalid days value."),
                },
                "i" | "interval" => set_refresh_interval(state, arg),
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
        name: "cost",
        invocation: "cost [all|VENDOR [AMOUNT]]",
        keys: &[],
        summary: "Show or change monthly fixed costs",
        detail: &[
            "Usage: cost | cost all      show every monthly fixed cost",
            "       cost <vendor>        show one vendor's monthly cost",
            "       cost <vendor> <fee>  save a new fee and redraw",
            "",
            "Harnesses: claude, codex, gemini, kimi",
            "Fees must be non-negative integers or decimals. Successful",
            "changes are persisted to the active .fee.env file.",
        ],
    },
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
        summary: "Cycle or set table shape (flat|vendor)",
        detail: &[
            "Usage: v                     toggle flat <-> vendor",
            "       view <flat|vendor>",
            "",
            "flat:   one row per model, merged across harnesses; the Harness",
            "        column lists tags like CC,OMP.",
            "vendor: the same model rows grouped under vendor headings, with",
            "        subtotal rows for vendors that have multiple models.",
            "",
            "Also available at startup: --view <flat|vendor>.",
        ],
    },
    HelpTopic {
        name: "sort",
        invocation: "s | sort <KEY>",
        keys: &["s"],
        summary: "Sort the table by a numeric column",
        detail: &[
            "Usage: s                     select the next sort key",
            "       sort <KEY>            select one sort key directly",
            "",
            "Keys: msgs, cache, prefill, decode, total, cost, rate",
            "All sorting is descending. Vendor view sorts groups by their",
            "aggregate value, then models within each group by the same key.",
            "",
            "Also available at startup: --sort <KEY>.",
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
        invocation: "i, interval <N|auto>",
        keys: &["i"],
        summary: "Change the auto-refresh interval (seconds)",
        detail: &[
            "Usage: i <seconds> | interval <seconds>   1 to 86400 seconds",
            "       i auto | interval auto",
            "",
            "Sets how often the dashboard reloads data automatically; the",
            "countdown is shown in the header. Sync (when configured) runs at",
            "one third of this interval but never faster than once a minute,",
            "offset by a per-machine delay so machines do not all sync at the",
            "same instant. Bare `i` prints the current value.",
            "",
            "auto is the startup mode: the cadence follows the chart interval",
            "shown in the span line, clamped to between 1 minute and 1 hour.",
            "It changes with the window, so zooming to a 5m chart refreshes",
            "every minute while a 1d chart refreshes hourly. An explicit value",
            "is kept until `i auto` restores automatic pacing.",
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
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_state() -> AppState {
        AppState {
            tool: "all".to_string(),
            table_view: TableView::Flat,
            sort_metric: crate::table_view::TableMetric::Messages,
            host: None,
            session_id: None,
            local_host_id: None,
            days: 3,
            time_window: TimeWindow::rolling_days(3),
            refresh_interval: crate::refresh::RefreshInterval::Manual(3600),
            pricing: AllPricing::load_raw().finalize(),
            subscription_fees: SubscriptionFees::default(),
            fee_env_path: PathBuf::from(".fee.env"),
            version_cache: HashMap::new(),
            all_tool_prompt: None,
            raw_cache: None,
            raw_cache_last_used_at: None,
            raw_refresh: None,
            integrity_status: crate::IntegrityStatus::Checking { percent: 0 },
            integrity_started_at: None,
        }
    }

    fn unique_fee_path(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ai-usage-cost-test-{}-{name}-{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create cost test directory");
        dir.join(".fee.env")
    }

    fn cost_test_state(name: &str) -> AppState {
        let mut state = test_state();
        state.fee_env_path = unique_fee_path(name);
        state.subscription_fees = SubscriptionFees {
            claude: 200.0,
            codex: 100.5,
            gemini: 19.99,
            kimi: 0.0,
        };
        state
    }

    fn remove_fee_test_dir(path: &std::path::Path) {
        if let Some(parent) = path.parent() {
            fs::remove_dir_all(parent).expect("remove cost test directory");
        }
    }

    #[test]
    fn cost_command_shows_all_or_one_monthly_fee() {
        let mut state = cost_test_state("show");

        let all = execute(&mut state, "cost");
        assert_eq!(all.effect, Effect::None);
        assert_eq!(
            all.messages,
            ["Monthly costs: Claude $200.00 | Codex $100.50 | Gemini $19.99 | Kimi $0.00"]
        );
        assert_eq!(execute(&mut state, "cost all").messages, all.messages);

        let one = execute(&mut state, "cost claude");
        assert_eq!(one.effect, Effect::None);
        assert_eq!(one.messages, ["Claude monthly cost: $200.00"]);

        let fee_path = state.fee_env_path.clone();
        remove_fee_test_dir(&fee_path);
    }

    #[test]
    fn cost_command_persists_one_fee_and_refreshes_the_dashboard() {
        let mut state = cost_test_state("persist");
        let fee_path = state.fee_env_path.clone();

        let outcome = execute(&mut state, "cost claude 125.50");

        assert_eq!(outcome.effect, Effect::Refresh);
        assert_eq!(state.subscription_fees.claude, 125.5);
        assert_eq!(
            outcome.messages,
            ["Claude monthly cost set to $125.50 and saved to .fee.env."]
        );
        assert_eq!(
            fs::read_to_string(&fee_path).expect("read saved fees"),
            "CLAUDE_MONTHLY_FEE=125.5\nCODEX_MONTHLY_FEE=100.5\nGEMINI_MONTHLY_FEE=19.99\nKIMI_MONTHLY_FEE=0\n"
        );

        remove_fee_test_dir(&fee_path);
    }

    #[test]
    fn cost_command_rejects_non_decimal_or_negative_values() {
        let mut state = cost_test_state("invalid");

        for input in [
            "cost claude -1",
            "cost claude +1",
            "cost claude .5",
            "cost claude 1.",
            "cost claude 1e3",
            "cost claude NaN",
            "cost claude inf",
            "cost claude 10 extra",
        ] {
            let outcome = execute(&mut state, input);
            assert_eq!(outcome.effect, Effect::None, "{input}");
            assert!(
                outcome.messages[0].contains("non-negative integer or decimal"),
                "{input}: {:?}",
                outcome.messages
            );
            assert_eq!(state.subscription_fees.claude, 200.0, "{input}");
        }
        assert!(!state.fee_env_path.exists());

        let fee_path = state.fee_env_path.clone();
        remove_fee_test_dir(&fee_path);
    }

    #[test]
    fn cost_command_keeps_memory_unchanged_when_persistence_fails() {
        let mut state = cost_test_state("write-failure");
        let test_root = state
            .fee_env_path
            .parent()
            .expect("cost test parent")
            .to_path_buf();
        state.fee_env_path = test_root.join("missing").join(".fee.env");

        let outcome = execute(&mut state, "cost kimi 12.25");

        assert_eq!(outcome.effect, Effect::None);
        assert_eq!(state.subscription_fees.kimi, 0.0);
        assert!(outcome.messages[0].starts_with("Could not save monthly costs:"));
        assert!(!state.fee_env_path.exists());

        fs::remove_dir_all(test_root).expect("remove cost test directory");
    }

    #[test]
    fn view_command_cycles_and_sets() {
        let mut state = test_state();
        let outcome = execute(&mut state, "v");
        assert_eq!(outcome.effect, Effect::TableChanged);
        assert_eq!(state.table_view, TableView::Vendor);

        let outcome = execute(&mut state, "v");
        assert_eq!(outcome.effect, Effect::TableChanged);
        assert_eq!(state.table_view, TableView::Flat);

        let outcome = execute(&mut state, "view model");
        assert_eq!(outcome.effect, Effect::TableChanged);
        assert_eq!(state.table_view, TableView::Flat);

        let outcome = execute(&mut state, "view bogus");
        assert_eq!(outcome.effect, Effect::None);
        assert_eq!(state.table_view, TableView::Flat);
    }

    #[test]
    fn sort_command_cycles_selects_and_rejects_unknown_keys() {
        use crate::table_view::TableMetric;

        let mut state = test_state();
        let outcome = execute(&mut state, "s");
        assert_eq!(outcome.effect, Effect::TableChanged);
        assert_eq!(state.sort_metric, TableMetric::CacheHit);
        assert_eq!(outcome.messages, ["Sort: Cache Hit (descending)"]);

        let outcome = execute(&mut state, "sort cost");
        assert_eq!(outcome.effect, Effect::TableChanged);
        assert_eq!(state.sort_metric, TableMetric::Cost);
        assert_eq!(outcome.messages, ["Sort: Cost (descending)"]);

        let outcome = execute(&mut state, "sort RATE");
        assert_eq!(outcome.effect, Effect::TableChanged);
        assert_eq!(state.sort_metric, TableMetric::Rate);

        let outcome = execute(&mut state, "sort bogus");
        assert_eq!(outcome.effect, Effect::None);
        assert_eq!(state.sort_metric, TableMetric::Rate);
        assert_eq!(
            outcome.messages,
            ["Usage: s, sort [msgs|cache|prefill|decode|total|cost|rate]"]
        );
    }

    #[test]
    fn interval_command_validates_and_reschedules() {
        let mut state = test_state();
        let outcome = execute(&mut state, "i 30");
        assert_eq!(outcome.effect, Effect::IntervalChanged);
        assert_eq!(state.refresh_interval, RefreshInterval::Manual(30));

        let outcome = execute(&mut state, "i 0");
        assert_eq!(outcome.effect, Effect::None);
        assert_eq!(state.refresh_interval, RefreshInterval::Manual(30));

        let outcome = execute(&mut state, "i abc");
        assert_eq!(outcome.effect, Effect::None);
        assert_eq!(state.refresh_interval, RefreshInterval::Manual(30));

        let outcome = execute(&mut state, &format!("i {}", u64::MAX));
        assert_eq!(outcome.effect, Effect::None);
        assert_eq!(state.refresh_interval, RefreshInterval::Manual(30));
    }

    #[test]
    fn interval_auto_restores_the_window_derived_cadence() {
        let mut state = test_state();
        state.refresh_interval = RefreshInterval::Manual(30);

        let outcome = execute(&mut state, "interval auto");

        assert_eq!(outcome.effect, Effect::IntervalChanged);
        assert_eq!(state.refresh_interval, RefreshInterval::Auto);
        assert!(
            outcome.messages[0].contains("chart interval"),
            "{:?}",
            outcome.messages
        );
    }

    #[test]
    fn bare_interval_reports_the_active_mode() {
        let mut state = test_state();
        state.refresh_interval = RefreshInterval::Auto;

        let auto = execute(&mut state, "i");
        assert_eq!(auto.effect, Effect::None);
        assert!(auto.messages[0].starts_with("Current interval: auto ("));

        state.refresh_interval = RefreshInterval::Manual(45);
        let manual = execute(&mut state, "i");
        assert_eq!(manual.messages[0], "Current interval: 45 seconds");
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
    fn host_command_labels_an_unused_known_host_with_zero_percent() {
        let mut state = test_state();
        state.local_host_id = Some("unused-test-host".to_string());

        let outcome = execute(&mut state, "host");
        let known_hosts = outcome
            .messages
            .iter()
            .find(|message| message.starts_with("Known hosts:"))
            .expect("known hosts message");

        assert!(
            known_hosts.contains("unused-test-host(0%)"),
            "{known_hosts}"
        );
    }

    #[test]
    fn known_hosts_are_sorted_by_visible_token_share() {
        let now = Local::now();
        let entry = |host: Option<&str>, usage: crate::data::TokenUsage| crate::data::UsageEntry {
            host_id: host.map(str::to_string),
            session_id: None,
            timestamp: now.to_rfc3339(),
            parsed_timestamp: Some(now),
            session_start_time: String::new(),
            session_end_time: String::new(),
            model: "test-model".to_string(),
            effort: None,
            fast_tier: -1,
            usage,
            costs: None,
        };
        let mut state = test_state();
        state.local_host_id = Some("alpha".to_string());
        let mut synthetic = entry(
            Some("foxtrot"),
            crate::data::TokenUsage {
                input_tokens: 900,
                ..Default::default()
            },
        );
        synthetic.model = "<synthetic>".to_string();
        state.raw_cache = Some(crate::RawDataCache {
            claude: vec![
                entry(
                    None,
                    crate::data::TokenUsage {
                        input_tokens: 679,
                        ..Default::default()
                    },
                ),
                synthetic,
            ],
            codex: vec![entry(
                Some("bravo"),
                crate::data::TokenUsage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cache_read_input_tokens: 25,
                    cache_creation_input_tokens: 999,
                    reasoning_output_tokens: 25,
                },
            )],
            gemini: vec![entry(
                Some("charlie"),
                crate::data::TokenUsage {
                    input_tokens: 40,
                    output_tokens: 10,
                    cache_read_input_tokens: 9,
                    cache_creation_input_tokens: 20,
                    reasoning_output_tokens: 999,
                },
            )],
            kimi: vec![entry(
                Some("delta"),
                crate::data::TokenUsage {
                    input_tokens: 32,
                    ..Default::default()
                },
            )],
            omp: vec![entry(
                Some("echo"),
                crate::data::TokenUsage {
                    input_tokens: 10,
                    ..Default::default()
                },
            )],
            range: crate::RawDataRange::from_bounds(
                now - chrono::Duration::days(3),
                now + chrono::Duration::days(1),
            ),
            has_source_data: true,
            local_host_id: state.local_host_id.clone(),
            local_record_keys: HashMap::new(),
            persistent_generation: String::new(),
            local_parser_revision_current: true,
        });

        let labels = known_host_usage_labels(
            &state,
            [
                "alpha", "bravo", "charlie", "delta", "echo", "golf", "foxtrot",
            ]
            .map(str::to_string)
            .to_vec(),
            now,
        );

        assert_eq!(
            labels,
            [
                "alpha(67.9%)",
                "bravo(20%)",
                "charlie(7.9%)",
                "delta(3.2%)",
                "echo(1%)",
                "foxtrot(0%)",
                "golf(0%)",
            ]
        );
    }

    #[test]
    fn hosts_with_the_same_displayed_percentage_use_alphabetical_order() {
        let labels = format_host_usage_totals(HashMap::from([
            ("beta".to_string(), 9),
            ("dominant".to_string(), 9_985),
            ("alpha".to_string(), 6),
        ]));

        assert_eq!(labels, ["dominant(99.9%)", "alpha(0.1%)", "beta(0.1%)"]);
    }

    #[test]
    fn host_percentages_handle_totals_larger_than_i64() {
        let labels = format_host_usage_totals(HashMap::from([
            ("beta".to_string(), i64::MAX as u128),
            ("alpha".to_string(), i64::MAX as u128),
        ]));

        assert_eq!(labels, ["alpha(50%)", "beta(50%)"]);
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
        let cost_idx = find_help_topic("cost").expect("cost topic");
        assert_eq!(find_help_topic("v"), Some(view_idx));
        assert_eq!(
            execute(&mut state, "h view").effect,
            Effect::HelpTopic(view_idx)
        );
        assert_eq!(
            execute(&mut state, "help days").effect,
            Effect::HelpTopic(find_help_topic("window").expect("window topic"))
        );
        assert_eq!(
            execute(&mut state, "h cost").effect,
            Effect::HelpTopic(cost_idx)
        );
        assert!(
            help_topics()[cost_idx]
                .detail
                .iter()
                .any(|line| line.contains(".fee.env"))
        );
        assert!(
            help_topics()[cost_idx]
                .detail
                .iter()
                .any(|line| line.contains("cost all"))
        );
        assert!(
            help_topics()[cost_idx]
                .detail
                .iter()
                .any(|line| line.starts_with("Harnesses:"))
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
    fn sort_help_is_discoverable_by_name_and_alias() {
        let index = find_help_topic("s").expect("sort help alias");
        let topic = &help_topics()[index];

        assert_eq!(topic.name, "sort");
        assert!(topic.detail.iter().any(|line| line.contains("descending")));
        assert!(topic.detail.iter().any(|line| line.contains("--sort")));
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
