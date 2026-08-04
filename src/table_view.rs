//! Renderer-agnostic view model for the usage breakdown table.
//!
//! Aggregation produces one `ModelBreakdownRow` per (harness, model) pair.
//! This module merges those rows by model, then either presents the flat list
//! or groups it by model vendor so every renderer shares one source of truth
//! for grouping, ordering, and labels.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::model_id::{ModelIdentity, Vendor, parse_model_identity, short_label, sort_key};
use crate::model_overrides;
use crate::stats::ModelBreakdownRow;
use crate::tool::Tool;

/// The shape of the breakdown table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TableView {
    /// One row per model, merged across harnesses.
    #[default]
    Flat,
    /// Rows grouped under a vendor header, with per-vendor subtotals.
    Vendor,
}

impl TableView {
    pub fn from_key(value: &str) -> Option<Self> {
        match value {
            "flat" => Some(TableView::Flat),
            "vendor" => Some(TableView::Vendor),
            "model" => Some(TableView::Flat),
            _ => None,
        }
    }

    /// The next view in the `v` toggle cycle.
    pub fn next(self) -> Self {
        match self {
            TableView::Flat => TableView::Vendor,
            TableView::Vendor => TableView::Flat,
        }
    }

    /// Human description used in table titles and command feedback.
    pub fn description(self) -> &'static str {
        match self {
            TableView::Flat => "Flat",
            TableView::Vendor => "grouped by Vendor",
        }
    }
}

impl std::str::FromStr for TableView {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::from_key(value).ok_or_else(|| format!("unknown table view: {value}"))
    }
}

/// Numeric table column used as the descending sort key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TableMetric {
    #[default]
    Messages,
    CacheHit,
    Prefill,
    Decode,
    Total,
    Cost,
    Rate,
}

impl TableMetric {
    pub fn label(self) -> &'static str {
        match self {
            Self::Messages => "Msgs",
            Self::CacheHit => "Cache Hit",
            Self::Prefill => "Prefill",
            Self::Decode => "Decode",
            Self::Total => "Total",
            Self::Cost => "Cost",
            Self::Rate => "$/MTok",
        }
    }

    pub fn from_key(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "msg" | "msgs" | "message" | "messages" => Some(Self::Messages),
            "cache" | "cache-hit" | "cache_hit" | "cachehit" => Some(Self::CacheHit),
            "prefill" => Some(Self::Prefill),
            "decode" | "decoding" => Some(Self::Decode),
            "total" | "tokens" => Some(Self::Total),
            "cost" => Some(Self::Cost),
            "rate" | "$/mtok" => Some(Self::Rate),
            _ => None,
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Messages => Self::CacheHit,
            Self::CacheHit => Self::Prefill,
            Self::Prefill => Self::Decode,
            Self::Decode => Self::Total,
            Self::Total => Self::Cost,
            Self::Cost => Self::Rate,
            Self::Rate => Self::Messages,
        }
    }

    fn compare_desc(self, left: &RowMetrics, right: &RowMetrics) -> Ordering {
        match self {
            Self::Messages => right.count.cmp(&left.count),
            Self::CacheHit => right.cache_hit.cmp(&left.cache_hit),
            Self::Prefill => right.prefill.cmp(&left.prefill),
            Self::Decode => right.decoding.cmp(&left.decoding),
            Self::Total => right.tokens().cmp(&left.tokens()),
            Self::Cost => right.cost().total_cmp(&left.cost()),
            Self::Rate => right.cost_per_mtok().total_cmp(&left.cost_per_mtok()),
        }
    }
}

impl std::str::FromStr for TableMetric {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::from_key(value).ok_or_else(|| format!("unknown sort metric: {value}"))
    }
}

/// Token and cost totals bucketed by inference strategy. The bucketing rules
/// differ per harness (codex reasoning and gemini thinking tokens are decode
/// output; claude-style cache creation is prefill), so metrics are derived
/// per source row and only then summed, which keeps cross-harness merges
/// semantically correct.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RowMetrics {
    pub count: i64,
    pub cache_hit: i64,
    pub prefill: i64,
    pub decoding: i64,
    pub cache_hit_cost: f64,
    pub prefill_cost: f64,
    pub decoding_cost: f64,
}

impl RowMetrics {
    pub fn from_breakdown(row: &ModelBreakdownRow) -> Self {
        let (prefill, decoding) = match row.tool.as_str() {
            "codex" => (row.input, row.output + row.reasoning),
            "gemini" => (row.input, row.output + row.thinking),
            _ => (row.input + row.cache_creation, row.output),
        };
        let (prefill_cost, decoding_cost) = match row.tool.as_str() {
            "claude" | "kimi" | "omp" => {
                (row.input_cost + row.cache_creation_cost, row.output_cost)
            }
            _ => (row.input_cost, row.output_cost + row.cache_creation_cost),
        };
        RowMetrics {
            count: row.count,
            cache_hit: row.cache_read,
            prefill,
            decoding,
            cache_hit_cost: row.cache_read_cost,
            prefill_cost,
            decoding_cost,
        }
    }

    pub fn add(&mut self, other: &RowMetrics) {
        self.count += other.count;
        self.cache_hit += other.cache_hit;
        self.prefill += other.prefill;
        self.decoding += other.decoding;
        self.cache_hit_cost += other.cache_hit_cost;
        self.prefill_cost += other.prefill_cost;
        self.decoding_cost += other.decoding_cost;
    }

    pub fn tokens(&self) -> i64 {
        self.cache_hit + self.prefill + self.decoding
    }

    pub fn cost(&self) -> f64 {
        self.cache_hit_cost + self.prefill_cost + self.decoding_cost
    }

    pub fn cost_per_mtok(&self) -> f64 {
        let tokens = self.tokens();
        if tokens > 0 {
            self.cost() / (tokens as f64 / 1_000_000.0)
        } else {
            0.0
        }
    }
}

/// Sum of per-row metrics across the whole data set.
pub fn table_totals(rows: &[ModelBreakdownRow]) -> RowMetrics {
    let mut totals = RowMetrics::default();
    for row in rows {
        totals.add(&RowMetrics::from_breakdown(row));
    }
    totals
}

/// Projected cost figures for the summary line under the table.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CostSummary {
    pub total_cost: f64,
    pub daily: f64,
    pub weekly: f64,
    pub monthly: f64,
    /// Monthly API cost minus the subscription price.
    pub savings: f64,
    /// What the subscription costs per MTok at the observed volume.
    pub subscription_rate: f64,
}

pub fn cost_summary(
    totals: &RowMetrics,
    days_in_data: f64,
    subscription_price: f64,
) -> CostSummary {
    let total_cost = totals.cost();
    let daily = if days_in_data > 0.0 {
        total_cost / days_in_data
    } else {
        0.0
    };
    let monthly = daily * 30.0;
    let monthly_tokens = if days_in_data > 0.0 {
        (totals.tokens() as f64 / days_in_data) * 30.0
    } else {
        0.0
    };
    let subscription_rate = if monthly_tokens > 0.0 {
        subscription_price / (monthly_tokens / 1_000_000.0)
    } else {
        0.0
    };
    CostSummary {
        total_cost,
        daily,
        weekly: daily * 7.0,
        monthly,
        savings: monthly - subscription_price,
        subscription_rate,
    }
}

/// One data line of the table, ready for either renderer.
#[derive(Debug, Clone, PartialEq)]
pub struct DataRow {
    pub vendor: Vendor,
    /// Vendor cell text; empty when suppressed (repeat in flat view, or
    /// redundant under a vendor group header).
    pub vendor_label: String,
    /// Short display label (user overrides win over the algorithmic label).
    pub model_label: String,
    /// Raw model id; for merged rows with differing raw ids, the normalized id.
    pub model_raw: String,
    /// Harness cell text: a full display name, or a compact tag list when the
    /// row merges several harnesses (e.g. `CC,OMP`).
    pub harness_label: String,
    /// Compact harness tag(s) for narrow layouts (e.g. `CC` or `CC,OMP`).
    pub harness_short: String,
    pub metrics: RowMetrics,
}

/// One rendered line of the breakdown table.
#[derive(Debug, Clone, PartialEq)]
pub enum DisplayRow {
    /// Vendor group heading (vendor view only).
    GroupHeader {
        vendor: String,
    },
    Data(Box<DataRow>),
    /// Per-vendor subtotal (vendor view, groups with at least two rows).
    Subtotal {
        vendor: String,
        metrics: RowMetrics,
    },
}

/// Resolve the short display name for a model id: the user override file wins,
/// otherwise the label is derived algorithmically from the id.
pub fn display_model_name(model: &str) -> String {
    if let Some(label) = model_overrides::load().display.get(model) {
        return label.clone();
    }
    short_label(&parse_model_identity(model))
}

struct Entry {
    identity: ModelIdentity,
    harness_rank: usize,
    harness_name: String,
    harness_tag: String,
    model_raw: String,
    model_label: String,
    metrics: RowMetrics,
}

fn harness_rank(key: &str) -> usize {
    Tool::ROTATION
        .iter()
        .position(|t| t.key() == key)
        .unwrap_or(usize::MAX)
}

fn entry_for(row: &ModelBreakdownRow) -> Entry {
    let identity = parse_model_identity(&row.model);
    let (harness_name, harness_tag) = match Tool::from_key(&row.tool) {
        Some(tool) => (
            tool.display_name().to_string(),
            tool.short_label().to_string(),
        ),
        None => (row.tool.clone(), row.tool.clone()),
    };
    Entry {
        harness_rank: harness_rank(&row.tool),
        harness_name,
        harness_tag,
        model_raw: row.model.clone(),
        model_label: display_model_name(&row.model),
        metrics: RowMetrics::from_breakdown(row),
        identity,
    }
}

/// Entries ordered vendor-contiguously: model family rank leads the model
/// sort key, so all rows of a vendor stay adjacent, newest version first,
/// then harnesses in rotation order.
fn sorted_entries(rows: &[ModelBreakdownRow]) -> Vec<Entry> {
    let mut entries: Vec<Entry> = rows.iter().map(entry_for).collect();
    entries.sort_by(|a, b| {
        sort_key(&a.identity)
            .cmp(&sort_key(&b.identity))
            .then(a.harness_rank.cmp(&b.harness_rank))
            .then_with(|| a.harness_tag.cmp(&b.harness_tag))
            .then_with(|| a.identity.effort.cmp(&b.identity.effort))
            .then_with(|| a.model_raw.cmp(&b.model_raw))
    });
    entries
}

/// Build the display rows for the requested view.
pub fn build_table(
    rows: &[ModelBreakdownRow],
    view: TableView,
    sort_metric: TableMetric,
) -> Vec<DisplayRow> {
    match view {
        TableView::Flat => build_flat(rows, sort_metric),
        TableView::Vendor => build_vendor(rows, sort_metric),
    }
}

fn build_vendor(rows: &[ModelBreakdownRow], sort_metric: TableMetric) -> Vec<DisplayRow> {
    struct Group {
        vendor: Vendor,
        rows: Vec<DataRow>,
        metrics: RowMetrics,
    }

    let mut groups: Vec<Group> = Vec::new();
    for mut row in merged_model_rows(rows, sort_metric) {
        row.vendor_label.clear();
        match groups.iter_mut().find(|group| group.vendor == row.vendor) {
            Some(group) => {
                group.metrics.add(&row.metrics);
                group.rows.push(row);
            }
            None => groups.push(Group {
                vendor: row.vendor,
                metrics: row.metrics,
                rows: vec![row],
            }),
        }
    }
    groups.sort_by(|left, right| {
        sort_metric
            .compare_desc(&left.metrics, &right.metrics)
            .then_with(|| left.vendor.cmp(&right.vendor))
    });

    let mut out = Vec::new();
    for group in groups {
        out.push(DisplayRow::GroupHeader {
            vendor: group.vendor.display_name().to_string(),
        });
        let row_count = group.rows.len();
        for row in group.rows {
            out.push(DisplayRow::Data(Box::new(row)));
        }
        if row_count >= 2 {
            out.push(DisplayRow::Subtotal {
                vendor: group.vendor.display_name().to_string(),
                metrics: group.metrics,
            });
        }
    }
    out
}

fn build_flat(rows: &[ModelBreakdownRow], sort_metric: TableMetric) -> Vec<DisplayRow> {
    merged_model_rows(rows, sort_metric)
        .into_iter()
        .map(|row| DisplayRow::Data(Box::new(row)))
        .collect()
}

fn merged_model_rows(rows: &[ModelBreakdownRow], sort_metric: TableMetric) -> Vec<DataRow> {
    struct Merged {
        first: usize,
        sources: Vec<usize>,
        metrics: RowMetrics,
    }

    let entries = sorted_entries(rows);
    let mut merged: Vec<Merged> = Vec::new();
    let mut by_model: HashMap<(String, Option<String>), usize> = HashMap::new();

    for (idx, entry) in entries.iter().enumerate() {
        let key = (
            entry.identity.normalized_id.clone(),
            entry.identity.effort.clone(),
        );
        match by_model.get(&key) {
            Some(&slot) => {
                merged[slot].sources.push(idx);
                merged[slot].metrics.add(&entry.metrics);
            }
            None => {
                by_model.insert(key, merged.len());
                merged.push(Merged {
                    first: idx,
                    sources: vec![idx],
                    metrics: entry.metrics,
                });
            }
        }
    }

    merged.sort_by(|a, b| {
        sort_metric
            .compare_desc(&a.metrics, &b.metrics)
            .then_with(|| a.first.cmp(&b.first))
    });

    merged
        .into_iter()
        .map(|m| {
            let first = &entries[m.first];
            let same_raw = m
                .sources
                .iter()
                .all(|&i| entries[i].model_raw == first.model_raw);
            let model_raw = if same_raw {
                first.model_raw.clone()
            } else {
                first.identity.normalized_id.clone()
            };
            let mut harnesses: Vec<&Entry> = Vec::new();
            for &source in &m.sources {
                let entry = &entries[source];
                if !harnesses.iter().any(|seen| {
                    seen.harness_rank == entry.harness_rank && seen.harness_tag == entry.harness_tag
                }) {
                    harnesses.push(entry);
                }
            }
            let (harness_label, harness_short) = if harnesses.len() == 1 {
                (
                    harnesses[0].harness_name.clone(),
                    harnesses[0].harness_tag.clone(),
                )
            } else {
                let joined = harnesses
                    .iter()
                    .map(|entry| entry.harness_tag.as_str())
                    .collect::<Vec<_>>()
                    .join(",");
                (joined.clone(), joined)
            };
            DataRow {
                vendor: first.identity.vendor,
                vendor_label: first.identity.vendor.display_name().to_string(),
                model_label: first.model_label.clone(),
                model_raw,
                harness_label,
                harness_short,
                metrics: m.metrics,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(tool: &str, model: &str, count: i64) -> ModelBreakdownRow {
        ModelBreakdownRow {
            model: model.to_string(),
            tool: tool.to_string(),
            count,
            input: 100,
            output: 10,
            cache_creation: 5,
            cache_read: 50,
            reasoning: 0,
            thinking: 0,
            total: 110,
            total_with_cache: 165,
            input_cost: 1.0,
            output_cost: 2.0,
            cache_read_cost: 0.5,
            cache_creation_cost: 0.25,
        }
    }

    fn metric_row(
        model: &str,
        count: i64,
        cache_hit: i64,
        prefill: i64,
        decoding: i64,
        cost: f64,
    ) -> ModelBreakdownRow {
        let mut item = row("codex", model, count);
        item.input = prefill;
        item.output = decoding;
        item.cache_creation = 0;
        item.cache_read = cache_hit;
        item.reasoning = 0;
        item.thinking = 0;
        item.total = prefill + decoding;
        item.total_with_cache = cache_hit + prefill + decoding;
        item.input_cost = cost;
        item.output_cost = 0.0;
        item.cache_read_cost = 0.0;
        item.cache_creation_cost = 0.0;
        item
    }

    fn data_rows(rows: &[DisplayRow]) -> Vec<&DataRow> {
        rows.iter()
            .filter_map(|r| match r {
                DisplayRow::Data(d) => Some(d.as_ref()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn view_cycle_has_two_forms_and_keeps_model_as_a_flat_alias() {
        assert_eq!(TableView::Flat.next(), TableView::Vendor);
        assert_eq!(TableView::Vendor.next(), TableView::Flat);
        assert_eq!(TableView::from_key("model"), Some(TableView::Flat));
        assert_eq!(TableView::from_key("vendor"), Some(TableView::Vendor));
        assert_eq!(TableView::from_key("bogus"), None);
        assert_eq!(TableView::default(), TableView::Flat);
    }

    #[test]
    fn table_metric_parses_and_cycles_all_sortable_columns() {
        assert_eq!(TableMetric::default(), TableMetric::Messages);
        assert_eq!(TableMetric::from_key("MSGS"), Some(TableMetric::Messages));
        assert_eq!(
            TableMetric::from_key("cache-hit"),
            Some(TableMetric::CacheHit)
        );
        assert_eq!(TableMetric::from_key("prefill"), Some(TableMetric::Prefill));
        assert_eq!(TableMetric::from_key("decoding"), Some(TableMetric::Decode));
        assert_eq!(TableMetric::from_key("tokens"), Some(TableMetric::Total));
        assert_eq!(TableMetric::from_key("cost"), Some(TableMetric::Cost));
        assert_eq!(TableMetric::from_key("$/MTok"), Some(TableMetric::Rate));
        assert_eq!(TableMetric::from_key("bogus"), None);

        let mut metric = TableMetric::default();
        let mut visited = Vec::new();
        for _ in 0..7 {
            visited.push(metric);
            metric = metric.next();
        }
        assert_eq!(
            visited,
            [
                TableMetric::Messages,
                TableMetric::CacheHit,
                TableMetric::Prefill,
                TableMetric::Decode,
                TableMetric::Total,
                TableMetric::Cost,
                TableMetric::Rate,
            ]
        );
        assert_eq!(metric, TableMetric::Messages);
    }

    #[test]
    fn flat_view_sorts_descending_by_every_table_metric() {
        let rows = vec![
            metric_row("gpt-msgs", 100, 1, 1, 1, 1.0),
            metric_row("claude-cache", 1, 10_000, 1, 1, 1.0),
            metric_row("gemini-prefill", 1, 1, 10_000, 1, 1.0),
            metric_row("glm-decode", 1, 1, 1, 10_000, 1.0),
            metric_row("deepseek-total", 1, 4_000, 4_000, 4_000, 1.0),
            metric_row("grok-cost", 1, 300, 300, 300, 10_000.0),
            metric_row("k3-rate", 1, 0, 1, 0, 1_000.0),
        ];

        for (metric, expected) in [
            (TableMetric::Messages, "gpt-msgs"),
            (TableMetric::CacheHit, "claude-cache"),
            (TableMetric::Prefill, "gemini-prefill"),
            (TableMetric::Decode, "glm-decode"),
            (TableMetric::Total, "deepseek-total"),
            (TableMetric::Cost, "grok-cost"),
            (TableMetric::Rate, "k3-rate"),
        ] {
            let table = build_table(&rows, TableView::Flat, metric);
            assert_eq!(data_rows(&table)[0].model_raw, expected, "{metric:?}");
        }
    }

    #[test]
    fn metrics_bucket_strategy_tokens_per_harness_semantics() {
        let mut codex = row("codex", "gpt-5.5", 1);
        codex.reasoning = 7;
        let m = RowMetrics::from_breakdown(&codex);
        assert_eq!(m.prefill, 100);
        assert_eq!(m.decoding, 17);
        assert_eq!(m.prefill_cost, 1.0);
        assert_eq!(m.decoding_cost, 2.25);

        let mut gemini = row("gemini", "gemini-2.5-pro", 1);
        gemini.thinking = 4;
        let m = RowMetrics::from_breakdown(&gemini);
        assert_eq!(m.prefill, 100);
        assert_eq!(m.decoding, 14);

        let claude = row("claude", "claude-opus-4-8", 1);
        let m = RowMetrics::from_breakdown(&claude);
        assert_eq!(m.prefill, 105);
        assert_eq!(m.decoding, 10);
        assert_eq!(m.prefill_cost, 1.25);
        assert_eq!(m.decoding_cost, 2.0);
        assert_eq!(m.cache_hit, 50);
        assert_eq!(m.cache_hit_cost, 0.5);
        assert_eq!(m.tokens(), 165);
        assert_eq!(m.cost(), 3.75);
    }

    #[test]
    fn flat_view_merges_harnesses_and_orders_models_by_messages() {
        let rows = vec![
            row("codex", "gpt-5.5", 9),
            row("claude", "claude-opus-4-8", 5),
            row("omp", "anthropic/claude-opus-4-8", 3),
            row("claude", "glm-5.2", 2),
        ];
        let table = build_table(&rows, TableView::Flat, TableMetric::Messages);
        let data = data_rows(&table);
        assert_eq!(table.len(), 3);

        let summary: Vec<(String, String, String)> = data
            .iter()
            .map(|d| {
                (
                    d.vendor_label.clone(),
                    d.model_label.clone(),
                    d.harness_label.clone(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                (
                    "OpenAI".to_string(),
                    "GPT-5.5".to_string(),
                    "Codex".to_string()
                ),
                (
                    "Anthropic".to_string(),
                    "Opus 4.8".to_string(),
                    "CC,OMP".to_string()
                ),
                (
                    "Zhipu".to_string(),
                    "GLM-5.2".to_string(),
                    "Claude Code".to_string()
                ),
            ]
        );
        assert_eq!(data[1].metrics.count, 8);
    }

    #[test]
    fn vendor_view_groups_the_same_merged_model_rows_as_flat() {
        let rows = vec![
            row("codex", "gpt-5.5", 9),
            row("claude", "claude-opus-4-8", 5),
            row("omp", "anthropic/claude-opus-4-8", 3),
        ];
        let table = build_table(&rows, TableView::Vendor, TableMetric::Messages);
        let data = data_rows(&table);

        assert_eq!(data.len(), 2);
        let opus = data
            .iter()
            .find(|row| row.model_label == "Opus 4.8")
            .expect("merged Anthropic row");
        assert_eq!(opus.harness_label, "CC,OMP");
        assert_eq!(opus.metrics.count, 8);
        assert_eq!(opus.metrics.prefill, 210);
        assert!(data.iter().all(|row| row.vendor_label.is_empty()));
        assert!(
            !table
                .iter()
                .any(|row| matches!(row, DisplayRow::Subtotal { .. }))
        );
    }

    #[test]
    fn vendor_view_sorts_groups_by_aggregate_and_models_by_the_same_metric() {
        let rows = vec![
            row("codex", "gpt-5.4", 5),
            row("codex", "gpt-5.5", 6),
            row("claude", "claude-opus-4-8", 10),
        ];

        let table = build_table(&rows, TableView::Vendor, TableMetric::Messages);

        assert!(matches!(
            &table[0],
            DisplayRow::GroupHeader { vendor } if vendor == "OpenAI"
        ));
        assert!(matches!(&table[1], DisplayRow::Data(row) if row.metrics.count == 6));
        assert!(matches!(&table[2], DisplayRow::Data(row) if row.metrics.count == 5));
        assert!(matches!(
            &table[3],
            DisplayRow::Subtotal { metrics, .. } if metrics.count == 11
        ));
        assert!(matches!(
            &table[4],
            DisplayRow::GroupHeader { vendor } if vendor == "Anthropic"
        ));
    }

    #[test]
    fn vendor_rate_sort_uses_aggregate_cost_over_aggregate_tokens() {
        let rows = vec![
            metric_row("gpt-high-rate", 1, 0, 1, 0, 10.0),
            metric_row("gpt-volume", 1, 0, 999, 0, 0.0),
            metric_row("claude-balanced", 1, 0, 100, 0, 2.0),
        ];

        let table = build_table(&rows, TableView::Vendor, TableMetric::Rate);

        assert!(matches!(
            &table[0],
            DisplayRow::GroupHeader { vendor } if vendor == "Anthropic"
        ));
        assert!(matches!(&table[1], DisplayRow::Data(row) if row.model_raw == "claude-balanced"));
        assert!(matches!(
            &table[2],
            DisplayRow::GroupHeader { vendor } if vendor == "OpenAI"
        ));
        assert!(matches!(&table[3], DisplayRow::Data(row) if row.model_raw == "gpt-high-rate"));
    }

    #[test]
    fn normalized_model_ids_merge_across_harnesses_with_tag_list() {
        let mut codex = row("codex", "gpt-5.5", 9);
        codex.reasoning = 7;
        let rows = vec![
            codex,
            row("claude", "claude-opus-4-8", 5),
            row("omp", "anthropic/claude-opus-4-8", 3),
        ];
        let table = build_table(&rows, TableView::Flat, TableMetric::Messages);
        let data = data_rows(&table);
        assert_eq!(data.len(), 2);

        // gpt-5.5 has the higher message count, so it leads.
        assert_eq!(data[0].model_label, "GPT-5.5");
        assert_eq!(data[0].harness_label, "Codex");
        assert_eq!(data[0].metrics.decoding, 17);

        let opus = data[1];
        assert_eq!(opus.model_label, "Opus 4.8");
        assert_eq!(opus.vendor_label, "Anthropic");
        assert_eq!(opus.harness_label, "CC,OMP");
        assert_eq!(opus.harness_short, "CC,OMP");
        // Raw ids differ across sources, so the normalized id is shown.
        assert_eq!(opus.model_raw, "claude-opus-4-8");
        assert_eq!(opus.metrics.count, 8);
        // Both sources are claude-style: prefill = input + cache creation.
        assert_eq!(opus.metrics.prefill, 210);
        assert_eq!(opus.metrics.cache_hit, 100);
        assert_eq!(opus.metrics.decoding, 20);
    }

    #[test]
    fn normalized_aliases_from_one_harness_show_one_harness_label() {
        let rows = vec![
            row("omp", "claude-opus-4-8", 5),
            row("omp", "anthropic/claude-opus-4-8", 3),
        ];

        let table = build_table(&rows, TableView::Flat, TableMetric::Messages);
        let data = data_rows(&table);

        assert_eq!(data.len(), 1);
        assert_eq!(data[0].harness_label, "Oh My Pi");
        assert_eq!(data[0].harness_short, "OMP");
        assert_eq!(data[0].metrics.count, 8);
    }

    #[test]
    fn tied_rows_have_stable_order_across_effort_and_raw_aliases() {
        let rows = vec![
            row("codex", "gpt-5.5 (low)", 1),
            row("codex", "openai/gpt-5.5 (high)", 1),
            row("codex", "gpt-5.5 (high)", 1),
            row("codex", "openai/gpt-5.5 (low)", 1),
        ];
        let shuffled = vec![
            rows[1].clone(),
            rows[0].clone(),
            rows[3].clone(),
            rows[2].clone(),
        ];

        let ordered = build_table(&rows, TableView::Flat, TableMetric::Messages);
        let reordered = build_table(&shuffled, TableView::Flat, TableMetric::Messages);

        assert_eq!(ordered, reordered);
        assert_eq!(
            data_rows(&ordered)
                .iter()
                .map(|row| row.model_label.as_str())
                .collect::<Vec<_>>(),
            ["GPT-5.5(H)", "GPT-5.5(L)"]
        );
    }

    #[test]
    fn totals_sum_per_row_semantics() {
        let mut codex = row("codex", "gpt-5.5", 9);
        codex.reasoning = 7;
        let rows = vec![codex, row("claude", "claude-opus-4-8", 5)];
        let totals = table_totals(&rows);
        assert_eq!(totals.count, 14);
        // codex prefill 100 + claude prefill 105
        assert_eq!(totals.prefill, 205);
        // codex decoding 17 + claude decoding 10
        assert_eq!(totals.decoding, 27);
        assert_eq!(totals.tokens(), 332);
    }
}
