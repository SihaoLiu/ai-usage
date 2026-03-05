use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{DateTime, Datelike, Duration, Local, Timelike};

use crate::formatting::{format_y_axis_value, format_total_value};
use crate::stats::{ModelTimeSeries, VendorTimeSeries};

const RESET_COLOR: &str = "\x1b[0m";

// Model display configuration for Claude (order matters)
fn model_config() -> Vec<(&'static str, &'static str, usize)> {
    vec![
        ("claude-opus-4-6", "Opus 4.6", 0),
        ("claude-opus-4-5-20251101", "Opus 4.5", 1),
        ("claude-opus-4-1-20250805", "Opus 4.1", 2),
        ("claude-sonnet-4-6", "Sonnet 4.6", 3),
        ("claude-sonnet-4-5-20250929", "Sonnet 4.5", 4),
        ("claude-sonnet-4-20250514", "Sonnet 4", 5),
        ("claude-haiku-4-5-20251001", "Haiku 4.5", 6),
    ]
}

fn model_order(model: &str) -> Option<usize> {
    model_config().iter().find(|(m, _, _)| *m == model).map(|(_, _, o)| *o)
}

fn model_short_name(model: &str) -> Option<&'static str> {
    model_config().iter().find(|(m, _, _)| *m == model).map(|(_, s, _)| *s)
}

// Line color configuration (ANSI 256-color)
fn line_color(key: &str) -> &'static str {
    match key {
        "opus_input" => "\x1b[38;5;196m",
        "opus_output" => "\x1b[38;5;203m",
        "sonnet_input" => "\x1b[38;5;33m",
        "sonnet_output" => "\x1b[38;5;75m",
        "haiku_input" => "\x1b[38;5;40m",
        "haiku_output" => "\x1b[38;5;120m",
        "model0_input" => "\x1b[38;5;208m",
        "model0_output" => "\x1b[38;5;215m",
        "model1_input" => "\x1b[38;5;135m",
        "model1_output" => "\x1b[38;5;177m",
        "model2_input" => "\x1b[38;5;37m",
        "model2_output" => "\x1b[38;5;80m",
        "model3_input" => "\x1b[38;5;197m",
        "model3_output" => "\x1b[38;5;218m",
        "model4_input" => "\x1b[38;5;226m",
        "model4_output" => "\x1b[38;5;228m",
        "model5_input" => "\x1b[38;5;51m",
        "model5_output" => "\x1b[38;5;87m",
        _ => "",
    }
}

fn vendor_color(vendor: &str) -> &'static str {
    match vendor {
        "Claude" => "\x1b[38;5;173m",
        "Codex" => "\x1b[38;5;255m",
        "Gemini" => "\x1b[38;5;33m",
        "All" => "\x1b[38;5;226m",
        _ => "\x1b[38;5;135m",
    }
}

fn get_short_model_name_for_chart(model: &str) -> String {
    if let Some(short) = model_short_name(model) {
        return short.to_string();
    }
    if model.contains(" (") && model.ends_with(')') {
        if let Some(idx) = model.rfind(" (") {
            let base = &model[..idx];
            let effort = &model[idx + 2..model.len() - 1];
            let effort_short = match effort {
                "low" => "L",
                "medium" => "M",
                "high" => "H",
                "xhigh" => "XH",
                _ => &effort[..1],
            };
            return format!("{}({})", base, effort_short);
        }
    }
    if model.len() > 12 {
        model[..12].to_string()
    } else {
        model.to_string()
    }
}

fn round_to_nice(value: f64, round_up: bool) -> i64 {
    let unit = if value >= 5_000_000_000.0 {
        5_000_000_000i64
    } else if value >= 5_000_000.0 {
        5_000_000
    } else if value >= 5_000.0 {
        5_000
    } else {
        5
    };
    if round_up {
        ((value as i64 + unit - 1) / unit) * unit
    } else {
        (value as i64 / unit) * unit
    }
}

#[derive(Clone)]
enum ChartColumn {
    Separator,
    Data { data_idx: usize, sub_col: usize },
}

struct ChartLayout {
    columns: Vec<ChartColumn>,
    data_to_col: HashMap<usize, usize>,
    sorted_times: Vec<DateTime<Local>>,
}

fn build_chart_layout(
    days_back: i64,
    interval_minutes: i64,
    target_width: Option<usize>,
) -> ChartLayout {
    let now = Local::now();
    let start_time = now - Duration::days(days_back);

    let total_minutes = start_time.hour() as i64 * 60 + start_time.minute() as i64;
    let interval_start = (total_minutes / interval_minutes) * interval_minutes;
    let start_rounded = start_time
        .with_hour((interval_start / 60) as u32).unwrap()
        .with_minute((interval_start % 60) as u32).unwrap()
        .with_second(0).unwrap()
        .with_nanosecond(0).unwrap();

    let mut sorted_times = Vec::new();
    let mut current = start_rounded;
    while current <= now {
        sorted_times.push(current);
        current = current + Duration::minutes(interval_minutes);
    }

    if sorted_times.len() > 500 {
        let step = sorted_times.len() / 500;
        sorted_times = sorted_times.into_iter().step_by(step.max(1)).collect();
    }

    // Reverse: most recent on left
    sorted_times.reverse();

    let num_data_points = sorted_times.len();
    let separator_count = sorted_times.iter().enumerate()
        .filter(|(i, t)| t.hour() == 0 && t.minute() == 0 && *i > 0)
        .count();

    let y_axis_width = 7usize;
    let x_scale = if let Some(tw) = target_width {
        let available = tw as f64 - y_axis_width as f64 - separator_count as f64;
        (available / num_data_points as f64).max(1.0)
    } else {
        2.4
    };

    let mut columns = Vec::new();
    let mut data_to_col = HashMap::new();
    let mut col_idx = 0usize;
    let mut accumulated = 0.0f64;

    for i in 0..num_data_points {
        let time = sorted_times[i];
        if time.hour() == 0 && time.minute() == 0 && i > 0 {
            columns.push(ChartColumn::Separator);
            col_idx += 1;
        }

        accumulated += x_scale;
        let mut cols = accumulated as usize;
        accumulated -= cols as f64;
        if cols < 1 {
            cols = 1;
        }

        data_to_col.insert(i, col_idx);

        for sub_col in 0..cols {
            columns.push(ChartColumn::Data { data_idx: i, sub_col });
            col_idx += 1;
        }
    }

    ChartLayout { columns, data_to_col, sorted_times }
}

fn print_daily_header(
    layout: &ChartLayout,
    line_values_fn: &dyn Fn(usize) -> f64,
) {
    // Group by calendar date
    let mut daily_data: BTreeMap<chrono::NaiveDate, (f64, Vec<usize>, DateTime<Local>)> = BTreeMap::new();

    for (col_idx, col) in layout.columns.iter().enumerate() {
        if let ChartColumn::Data { data_idx, sub_col, .. } = col {
            let time = layout.sorted_times[*data_idx];
            let date_key = time.date_naive();

            let entry = daily_data.entry(date_key).or_insert((0.0, Vec::new(), time));
            entry.1.push(col_idx);

            if *sub_col == 0 {
                entry.0 += line_values_fn(*data_idx);
            }
        }
    }

    let mut daily_totals: Vec<(usize, f64, DateTime<Local>)> = daily_data.values()
        .filter(|(_, cols, _)| !cols.is_empty())
        .map(|(total, cols, time)| {
            let mid = (cols.iter().min().unwrap() + cols.iter().max().unwrap()) / 2;
            (mid, *total, *time)
        })
        .collect();
    // Sort by column position (left to right) to match rendering order
    daily_totals.sort_by_key(|(mid_col, _, _)| *mid_col);

    let weekday_abbr = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

    let mut weekday_line = " ".repeat(7);
    let mut date_line = " ".repeat(7);
    let mut prev_end = 0usize;

    for (mid_col, total, day_start) in &daily_totals {
        let total_str = format_total_value(*total);
        let weekday = weekday_abbr[day_start.weekday().num_days_from_monday() as usize];
        let mut weekday_total = format!("{} : {}", weekday, total_str);
        let mut date_str = day_start.format(" %m / %d").to_string();

        let colon_idx = weekday_total.find(':').unwrap_or(0);
        let slash_idx = date_str.find('/').unwrap_or(0);

        if colon_idx > slash_idx {
            date_str = format!("{}{}", " ".repeat(colon_idx - slash_idx), date_str);
        } else if slash_idx > colon_idx {
            weekday_total = format!("{}{}", " ".repeat(slash_idx - colon_idx), weekday_total);
        }

        let max_len = weekday_total.len().max(date_str.len());
        weekday_total = format!("{:<width$}", weekday_total, width = max_len);
        date_str = format!("{:<width$}", date_str, width = max_len);

        let actual_colon = weekday_total.find(':').unwrap_or(0);
        let start_pos = if *mid_col > actual_colon {
            mid_col - actual_colon
        } else {
            0
        };
        let padding = if start_pos > prev_end {
            start_pos - prev_end
        } else {
            0
        };

        weekday_line.push_str(&" ".repeat(padding));
        date_line.push_str(&" ".repeat(padding));
        weekday_line.push_str(&weekday_total);
        date_line.push_str(&date_str);
        prev_end = start_pos + max_len;
    }

    println!("{}", weekday_line);
    println!("{}", date_line);
}

fn print_x_axis_labels(layout: &ChartLayout, _interval_minutes: i64) {
    println!();

    let first_time = layout.sorted_times.last().unwrap(); // oldest (reversed)
    let last_time = Local::now();
    let time_span_minutes = (last_time - *first_time).num_seconds() as f64 / 60.0;
    let target_tick = time_span_minutes * 0.05;

    let standard_intervals = [15, 30, 60, 120, 180, 240, 360, 480, 720, 1440];
    let tick_interval = *standard_intervals.iter()
        .min_by_key(|&&x| ((x as f64 - target_tick).abs() * 1000.0) as i64)
        .unwrap_or(&60);

    let mut current_tick = first_time
        .with_hour(0).unwrap()
        .with_minute(0).unwrap()
        .with_second(0).unwrap()
        .with_nanosecond(0).unwrap();

    let minutes_since_midnight = first_time.hour() as i64 * 60 + first_time.minute() as i64;
    let ticks_since = (minutes_since_midnight + tick_interval - 1) / tick_interval;
    current_tick = current_tick + Duration::minutes(ticks_since * tick_interval);

    let mut labels: Vec<String> = Vec::new();
    let mut positions: Vec<usize> = Vec::new();
    let mut used_positions: HashSet<usize> = HashSet::new();

    while current_tick <= last_time {
        let mut closest_idx = None;
        let mut min_diff = i64::MAX;

        for (i, time) in layout.sorted_times.iter().enumerate() {
            let diff = (*time - current_tick).num_seconds().abs();
            if diff < min_diff {
                min_diff = diff;
                closest_idx = Some(i);
            }
        }

        if let Some(idx) = closest_idx {
            if let Some(&pos) = layout.data_to_col.get(&idx) {
                if !used_positions.contains(&pos) {
                    let label = if tick_interval < 60 {
                        current_tick.format("%H:%M").to_string()
                    } else {
                        current_tick.format("%H").to_string()
                    };
                    labels.push(label);
                    positions.push(pos);
                    used_positions.insert(pos);
                }
            }
        }

        current_tick = current_tick + Duration::minutes(tick_interval);
    }

    let max_label_len = labels.iter().map(|l| l.len()).max().unwrap_or(0);

    for char_idx in 0..max_label_len {
        let mut line = "       ".to_string();
        for (col_idx, col) in layout.columns.iter().enumerate() {
            match col {
                ChartColumn::Separator => line.push('|'),
                ChartColumn::Data { .. } => {
                    let mut ch = ' ';
                    for (label_idx, &pos) in positions.iter().enumerate() {
                        if col_idx == pos && char_idx < labels[label_idx].len() {
                            ch = labels[label_idx].as_bytes()[char_idx] as char;
                            break;
                        }
                    }
                    line.push(ch);
                }
            }
        }
        println!("{}", line);
    }
}

fn render_grid(
    layout: &ChartLayout,
    chart_height: usize,
    max_value: i64,
    min_value: i64,
    lines: &[LineConfig],
    line_values: &HashMap<usize, Vec<f64>>,
    use_bold_for_last: bool,
) {
    let value_to_row = |value: f64| -> usize {
        if max_value == min_value {
            return 0;
        }
        ((value - min_value as f64) / (max_value - min_value) as f64 * (chart_height - 1) as f64) as usize
    };

    let line_rows: HashMap<usize, Vec<usize>> = line_values.iter()
        .map(|(&i, values)| (i, values.iter().map(|v| value_to_row(*v)).collect()))
        .collect();

    // grid[row][col] = (line_idx, char_type)
    let num_cols = layout.columns.len();
    let mut grid: Vec<Vec<Option<(usize, &'static str)>>> =
        vec![vec![None; num_cols]; chart_height];

    // Process lines in reverse order (first line draws on top)
    for line_idx in (0..lines.len()).rev() {
        let rows = &line_rows[&line_idx];

        for (col_idx, col) in layout.columns.iter().enumerate() {
            if let ChartColumn::Data { data_idx, sub_col, .. } = col {
                let curr_row = rows[*data_idx];
                let prev_row = if *data_idx > 0 { rows[*data_idx - 1] } else { curr_row };

                if *sub_col > 0 {
                    grid[curr_row][col_idx] = Some((line_idx, "flat"));
                } else if prev_row == curr_row {
                    grid[curr_row][col_idx] = Some((line_idx, "flat"));
                } else if prev_row < curr_row {
                    grid[prev_row][col_idx] = Some((line_idx, "up_to_right"));
                    for row in (prev_row + 1)..curr_row {
                        grid[row][col_idx] = Some((line_idx, "vertical"));
                    }
                    grid[curr_row][col_idx] = Some((line_idx, "up_from_left"));
                } else {
                    grid[prev_row][col_idx] = Some((line_idx, "down_to_right"));
                    for row in (curr_row + 1)..prev_row {
                        grid[row][col_idx] = Some((line_idx, "vertical"));
                    }
                    grid[curr_row][col_idx] = Some((line_idx, "down_from_left"));
                }
            }
        }
    }

    let char_map = |char_type: &str| -> char {
        match char_type {
            "flat" => '\u{2500}',           // ─
            "up_from_left" => '\u{256D}',   // ╭
            "down_from_left" => '\u{2570}', // ╰
            "down_to_right" => '\u{256E}',  // ╮
            "up_to_right" => '\u{256F}',    // ╯
            "vertical" => '\u{2502}',       // │
            _ => '\u{2500}',
        }
    };

    let char_map_bold = |char_type: &str| -> char {
        match char_type {
            "flat" => '\u{2501}',           // ━
            "up_from_left" => '\u{250F}',   // ┏
            "down_from_left" => '\u{2517}', // ┗
            "down_to_right" => '\u{2513}',  // ┓
            "up_to_right" => '\u{251B}',    // ┛
            "vertical" => '\u{2503}',       // ┃
            _ => '\u{2501}',
        }
    };

    // Draw chart from top to bottom
    for row in (0..chart_height).rev() {
        let y_val = min_value as f64 + (max_value - min_value) as f64 * row as f64 / (chart_height - 1) as f64;
        let y_label = format!("{} |", format_y_axis_value(y_val));

        let mut line_str = String::new();
        for (col_idx, col) in layout.columns.iter().enumerate() {
            match col {
                ChartColumn::Separator => line_str.push('|'),
                ChartColumn::Data { .. } => {
                    if let Some((line_idx, char_type)) = grid[row][col_idx] {
                        let is_bold = use_bold_for_last && line_idx == lines.len() - 1;
                        let ch = if is_bold {
                            char_map_bold(char_type)
                        } else {
                            char_map(char_type)
                        };
                        let color = &lines[line_idx].color;
                        line_str.push_str(color);
                        line_str.push(ch);
                        line_str.push_str(RESET_COLOR);
                    } else {
                        line_str.push(' ');
                    }
                }
            }
        }
        println!("{}{}", y_label, line_str);
    }

    // X-axis
    let mut x_axis = String::new();
    for col in &layout.columns {
        match col {
            ChartColumn::Separator => x_axis.push('\u{2534}'), // ┴
            ChartColumn::Data { .. } => x_axis.push('\u{2500}'), // ─
        }
    }
    println!("      \u{2514}{}", x_axis); // └
}

struct LineConfig {
    model: String,
    token_type: String,
    color: String,
    label: String,
}

/// Print a multi-line chart with multiple lines (models x token types).
pub fn print_multi_line_chart(
    time_series: &ModelTimeSeries,
    height: usize,
    days_back: i64,
    chart_type: &str,
    show_x_axis: bool,
    target_width: Option<usize>,
    interval_minutes: i64,
    vendor: &str,
    included_models: Option<&HashSet<String>>,
    show_legend: bool,
) {
    if time_series.is_empty() {
        println!("No time series data available.");
        return;
    }

    let layout = build_chart_layout(days_back, interval_minutes, target_width);

    if layout.sorted_times.len() < 2 {
        println!("Not enough data points for chart.");
        return;
    }

    let (token_types, type_labels, chart_title) = if chart_type == "io" {
        (
            vec!["input", "output"],
            HashMap::from([("input", "Input"), ("output", "Output")]),
            "Models Input / Output Token Consumption".to_string(),
        )
    } else {
        let labels = match vendor {
            "codex" => HashMap::from([("cache_read", "Cache Read In"), ("cache_creation", "Reasoning Out")]),
            "gemini" => HashMap::from([("cache_read", "Cache Read In"), ("cache_creation", "Thinking Out")]),
            _ => HashMap::from([("cache_read", "Cache Read In"), ("cache_creation", "Cache Create In")]),
        };
        let title = match vendor {
            "codex" => "Models Cache Read Input / Reasoning Output Token Consumption",
            "gemini" => "Models Cache Read Input / Thinking Output Token Consumption",
            _ => "Models Cache Read Input / Cache Creation Input Token Consumption",
        };
        (vec!["cache_read", "cache_creation"], labels, title.to_string())
    };

    // Collect all models
    let mut all_models: HashSet<String> = HashSet::new();
    for time in &layout.sorted_times {
        if let Some(model_map) = time_series.get(time) {
            all_models.extend(model_map.keys().cloned());
        }
    }
    if let Some(included) = included_models {
        all_models.retain(|m| included.contains(m));
    }

    let config = model_config();
    let known_set: HashSet<&str> = config.iter().map(|(m, _, _)| *m).collect();

    let mut known_models: Vec<String> = all_models.iter().filter(|m| known_set.contains(m.as_str())).cloned().collect();
    let mut other_models: Vec<String> = all_models.iter().filter(|m| !known_set.contains(m.as_str())).cloned().collect();

    known_models.sort_by_key(|m| model_order(m).unwrap_or(99));
    other_models.sort();

    let all_models_sorted: Vec<String> = known_models.into_iter().chain(other_models.into_iter()).collect();

    if all_models_sorted.is_empty() {
        println!("No models found in data.");
        return;
    }

    // Build line configurations
    let mut lines: Vec<LineConfig> = Vec::new();
    for (model_idx, model) in all_models_sorted.iter().enumerate() {
        if known_set.contains(model.as_str()) {
            let short_label = model_short_name(model).unwrap_or(model);
            let model_short = short_label.to_lowercase();
            let model_prefix = model_short.split_whitespace().next().unwrap_or("unknown");

            for token_type in &token_types {
                let color_suffix = if *token_type == "input" || *token_type == "cache_read" {
                    "input"
                } else {
                    "output"
                };
                let color_key = format!("{}_{}", model_prefix, color_suffix);
                lines.push(LineConfig {
                    model: model.clone(),
                    token_type: token_type.to_string(),
                    color: line_color(&color_key).to_string(),
                    label: format!("{} {}", short_label, type_labels[token_type]),
                });
            }
        } else {
            let short_label = get_short_model_name_for_chart(model);
            let color_idx = model_idx % 6;
            for token_type in &token_types {
                let color_suffix = if *token_type == "input" || *token_type == "cache_read" {
                    "input"
                } else {
                    "output"
                };
                let color_key = format!("model{}_{}", color_idx, color_suffix);
                lines.push(LineConfig {
                    model: model.clone(),
                    token_type: token_type.to_string(),
                    color: line_color(&color_key).to_string(),
                    label: format!("{} {}", short_label, type_labels[token_type]),
                });
            }
        }
    }

    // Calculate values for each line
    let mut line_values: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut all_values: Vec<f64> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let values: Vec<f64> = layout.sorted_times.iter().map(|time| {
            time_series.get(time)
                .and_then(|model_map| model_map.get(&line.model))
                .map(|breakdown| {
                    match line.token_type.as_str() {
                        "input" => breakdown.input,
                        "output" => breakdown.output,
                        "cache_creation" => breakdown.cache_creation,
                        "cache_read" => breakdown.cache_read,
                        _ => 0.0,
                    }
                })
                .unwrap_or(0.0)
        }).collect();
        all_values.extend_from_slice(&values);
        line_values.insert(i, values);
    }

    let max_value_raw = all_values.iter().cloned().fold(0.0f64, f64::max);
    let max_value_raw = if max_value_raw == 0.0 { 1.0 } else { max_value_raw };
    let max_value = round_to_nice(max_value_raw, true);
    let min_value = 0i64;
    let max_value = if max_value == min_value { min_value + 5000 } else { max_value };

    // Print title
    let total_width = layout.columns.len() + 7;
    if !show_x_axis {
        println!();
    }
    println!("{:^width$}", chart_title, width = total_width);
    println!("{}", "=".repeat(total_width));

    // Daily header - sum all lines for daily totals
    print_daily_header(&layout, &|data_idx| {
        lines.iter().enumerate().map(|(i, _)| line_values[&i][data_idx]).sum::<f64>()
    });

    render_grid(&layout, height, max_value, min_value, &lines, &line_values, false);

    if show_x_axis {
        print_x_axis_labels(&layout, interval_minutes);
    }

    if show_legend {
        let legend_parts: Vec<String> = lines.iter()
            .map(|l| format!("{}\u{2500}{} {}", l.color, RESET_COLOR, l.label))
            .collect();
        println!("Legend: {}", legend_parts.join("  "));
    }
}

/// Print a vendor comparison chart.
pub fn print_vendor_comparison_chart(
    time_series: &VendorTimeSeries,
    height: usize,
    days_back: i64,
    target_width: Option<usize>,
    interval_minutes: i64,
    show_legend: bool,
) {
    if time_series.is_empty() {
        println!("No time series data available.");
        return;
    }

    let layout = build_chart_layout(days_back, interval_minutes, target_width);

    if layout.sorted_times.len() < 2 {
        println!("Not enough data points for chart.");
        return;
    }

    // Collect all vendors
    let mut all_vendors: HashSet<String> = HashSet::new();
    for time in &layout.sorted_times {
        if let Some(vendor_map) = time_series.get(time) {
            all_vendors.extend(vendor_map.keys().cloned());
        }
    }

    let vendor_order = ["Claude", "Codex", "Gemini"];
    let mut vendors_sorted: Vec<String> = vendor_order.iter()
        .filter(|v| all_vendors.contains(**v))
        .map(|v| v.to_string())
        .collect();
    let mut remaining: Vec<&String> = all_vendors.iter()
        .filter(|v| !vendor_order.contains(&v.as_str()))
        .collect();
    remaining.sort_unstable();
    for v in remaining {
        vendors_sorted.push(v.clone());
    }

    if vendors_sorted.is_empty() {
        println!("No vendor data available.");
        return;
    }

    // Add "All" as last entry
    vendors_sorted.push("All".to_string());

    // Build vendor data
    let mut vendor_data: HashMap<String, Vec<f64>> = HashMap::new();
    for vendor in &vendors_sorted {
        if vendor == "All" {
            continue;
        }
        let values: Vec<f64> = layout.sorted_times.iter().map(|time| {
            time_series.get(time)
                .and_then(|vm| vm.get(vendor))
                .copied()
                .unwrap_or(0.0)
        }).collect();
        vendor_data.insert(vendor.clone(), values);
    }

    // Calculate "All" as sum
    let all_values: Vec<f64> = (0..layout.sorted_times.len()).map(|i| {
        vendors_sorted.iter()
            .filter(|v| *v != "All")
            .map(|v| vendor_data.get(v).map(|vals| vals[i]).unwrap_or(0.0))
            .sum()
    }).collect();
    vendor_data.insert("All".to_string(), all_values);

    // Find max value
    let max_value_raw = vendor_data.values()
        .flat_map(|vals| vals.iter())
        .cloned()
        .fold(0.0f64, f64::max);
    let max_value_raw = if max_value_raw == 0.0 { 1.0 } else { max_value_raw };
    let max_value = round_to_nice(max_value_raw, true);
    let min_value = 0i64;
    let max_value = if max_value == min_value { min_value + 5000 } else { max_value };

    // Build line configs
    let lines: Vec<LineConfig> = vendors_sorted.iter().map(|v| {
        LineConfig {
            model: v.clone(),
            token_type: String::new(),
            color: vendor_color(v).to_string(),
            label: v.clone(),
        }
    }).collect();

    let line_values: HashMap<usize, Vec<f64>> = vendors_sorted.iter().enumerate().map(|(i, v)| {
        (i, vendor_data[v].clone())
    }).collect();

    let chart_title = "Total Token Consumption by Vendor";
    let total_width = layout.columns.len() + 7;
    println!();
    println!("{:^width$}", chart_title, width = total_width);
    println!("{}", "=".repeat(total_width));

    // Daily header using "All" totals
    print_daily_header(&layout, &|data_idx| {
        vendor_data["All"][data_idx]
    });

    render_grid(&layout, height, max_value, min_value, &lines, &line_values, true);

    print_x_axis_labels(&layout, interval_minutes);

    if show_legend {
        // Calculate vendor totals and percentages
        let vendor_totals: HashMap<String, f64> = vendors_sorted.iter()
            .filter(|v| *v != "All")
            .map(|v| (v.clone(), vendor_data[v].iter().sum::<f64>()))
            .collect();
        let grand_total: f64 = vendor_totals.values().sum();

        let mut legend_items: Vec<String> = Vec::new();
        for v in vendors_sorted.iter().filter(|v| *v != "All") {
            let color = vendor_color(v);
            let pct = if grand_total > 0.0 {
                vendor_totals[v] / grand_total * 100.0
            } else {
                0.0
            };
            legend_items.push(format!("{}\u{2500}{} {}({:.1}%)", color, RESET_COLOR, v, pct));
        }
        let all_color = vendor_color("All");
        legend_items.push(format!("{}\u{2501}{} All(100%)", all_color, RESET_COLOR));
        println!("Legend: {}", legend_items.join("  "));
    }
}
