//! Renderer-agnostic view model for the usage breakdown table.
//!
//! The breakdown has three axes: Vendor (the model maker, derived from the
//! model id by `model_id`), Harness (the CLI tool that logged the usage), and
//! Model. Aggregation produces one `ModelBreakdownRow` per (harness, model)
//! pair; this module reshapes those rows into one of three user-selectable
//! table forms so the plain-text and ratatui renderers share a single source
//! of truth for grouping, merging, ordering, and labels.

use std::collections::HashMap;

use crate::model_id::{ModelIdentity, Vendor, parse_model_identity, short_label, sort_key};
use crate::model_overrides;
use crate::stats::ModelBreakdownRow;
use crate::tool::Tool;

/// The shape of the breakdown table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TableView {
    /// One row per (model, harness); Vendor / Model / Harness as columns,
    /// vendor label suppressed on consecutive rows of the same vendor.
    #[default]
    Flat,
    /// Rows grouped under a vendor header, with per-vendor subtotals.
    Vendor,
    /// One row per model, merged across harnesses; harnesses listed as tags.
    Model,
}

impl TableView {
    pub fn from_key(value: &str) -> Option<Self> {
        match value {
            "flat" => Some(TableView::Flat),
            "vendor" => Some(TableView::Vendor),
            "model" => Some(TableView::Model),
            _ => None,
        }
    }

    /// The next view in the `v` toggle cycle.
    pub fn next(self) -> Self {
        match self {
            TableView::Flat => TableView::Vendor,
            TableView::Vendor => TableView::Model,
            TableView::Model => TableView::Flat,
        }
    }

    /// Human description used in table titles and command feedback.
    pub fn description(self) -> &'static str {
        match self {
            TableView::Flat => "Vendor / Model / Harness",
            TableView::Vendor => "grouped by Vendor",
            TableView::Model => "by Model (harnesses merged)",
        }
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

pub fn cost_summary(totals: &RowMetrics, days_in_data: f64, subscription_price: f64) -> CostSummary {
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
    GroupHeader { vendor: String },
    Data(Box<DataRow>),
    /// Per-vendor subtotal (vendor view, groups with at least two rows).
    Subtotal { vendor: String, metrics: RowMetrics },
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
        Some(tool) => (tool.display_name().to_string(), tool.short_label().to_string()),
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
    });
    entries
}

fn data_row(entry: &Entry, vendor_label: String) -> DisplayRow {
    DisplayRow::Data(Box::new(DataRow {
        vendor: entry.identity.vendor,
        vendor_label,
        model_label: entry.model_label.clone(),
        model_raw: entry.model_raw.clone(),
        harness_label: entry.harness_name.clone(),
        harness_short: entry.harness_tag.clone(),
        metrics: entry.metrics,
    }))
}

/// Build the display rows for the requested view.
pub fn build_table(rows: &[ModelBreakdownRow], view: TableView) -> Vec<DisplayRow> {
    match view {
        TableView::Flat => build_flat(rows),
        TableView::Vendor => build_vendor(rows),
        TableView::Model => build_model(rows),
    }
}

fn build_flat(rows: &[ModelBreakdownRow]) -> Vec<DisplayRow> {
    let mut out = Vec::new();
    let mut prev_vendor: Option<Vendor> = None;
    for entry in sorted_entries(rows) {
        let vendor = entry.identity.vendor;
        let label = if prev_vendor == Some(vendor) {
            String::new()
        } else {
            vendor.display_name().to_string()
        };
        prev_vendor = Some(vendor);
        out.push(data_row(&entry, label));
    }
    out
}

fn build_vendor(rows: &[ModelBreakdownRow]) -> Vec<DisplayRow> {
    let entries = sorted_entries(rows);
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < entries.len() {
        let vendor = entries[idx].identity.vendor;
        let group_end = entries[idx..]
            .iter()
            .position(|e| e.identity.vendor != vendor)
            .map_or(entries.len(), |off| idx + off);
        let group = &entries[idx..group_end];

        out.push(DisplayRow::GroupHeader {
            vendor: vendor.display_name().to_string(),
        });
        let mut subtotal = RowMetrics::default();
        for entry in group {
            subtotal.add(&entry.metrics);
            out.push(data_row(entry, String::new()));
        }
        if group.len() >= 2 {
            out.push(DisplayRow::Subtotal {
                vendor: vendor.display_name().to_string(),
                metrics: subtotal,
            });
        }
        idx = group_end;
    }
    out
}

fn build_model(rows: &[ModelBreakdownRow]) -> Vec<DisplayRow> {
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

    // Busiest models first, model order as tie-breaker for stability.
    merged.sort_by(|a, b| {
        b.metrics
            .count
            .cmp(&a.metrics.count)
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
            let tags: Vec<&str> = m
                .sources
                .iter()
                .map(|&i| entries[i].harness_tag.as_str())
                .collect();
            let (harness_label, harness_short) = if m.sources.len() == 1 {
                (first.harness_name.clone(), first.harness_tag.clone())
            } else {
                let joined = tags.join(",");
                (joined.clone(), joined)
            };
            DisplayRow::Data(Box::new(DataRow {
                vendor: first.identity.vendor,
                vendor_label: first.identity.vendor.display_name().to_string(),
                model_label: first.model_label.clone(),
                model_raw,
                harness_label,
                harness_short,
                metrics: m.metrics,
            }))
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

    fn data_rows(rows: &[DisplayRow]) -> Vec<&DataRow> {
        rows.iter()
            .filter_map(|r| match r {
                DisplayRow::Data(d) => Some(d.as_ref()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn view_cycle_covers_all_forms() {
        assert_eq!(TableView::Flat.next(), TableView::Vendor);
        assert_eq!(TableView::Vendor.next(), TableView::Model);
        assert_eq!(TableView::Model.next(), TableView::Flat);
        assert_eq!(TableView::from_key("vendor"), Some(TableView::Vendor));
        assert_eq!(TableView::from_key("bogus"), None);
        assert_eq!(TableView::default(), TableView::Flat);
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
    fn flat_view_orders_vendor_contiguously_and_suppresses_repeats() {
        let rows = vec![
            row("codex", "gpt-5.5", 9),
            row("claude", "claude-opus-4-8", 5),
            row("omp", "anthropic/claude-opus-4-8", 3),
            row("claude", "glm-5.2", 2),
        ];
        let table = build_table(&rows, TableView::Flat);
        let data = data_rows(&table);
        assert_eq!(table.len(), 4);

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
                    "Anthropic".to_string(),
                    "Opus 4.8".to_string(),
                    "Claude Code".to_string()
                ),
                (String::new(), "Opus 4.8".to_string(), "Oh My Pi".to_string()),
                (
                    "OpenAI".to_string(),
                    "GPT-5.5".to_string(),
                    "Codex".to_string()
                ),
                (
                    "Zhipu".to_string(),
                    "GLM-5.2".to_string(),
                    "Claude Code".to_string()
                ),
            ]
        );
    }

    #[test]
    fn vendor_view_emits_headers_and_multi_row_subtotals() {
        let rows = vec![
            row("codex", "gpt-5.5", 9),
            row("claude", "claude-opus-4-8", 5),
            row("omp", "anthropic/claude-opus-4-8", 3),
        ];
        let table = build_table(&rows, TableView::Vendor);

        match &table[0] {
            DisplayRow::GroupHeader { vendor } => assert_eq!(vendor, "Anthropic"),
            other => panic!("expected header, got {:?}", other),
        }
        assert!(matches!(&table[1], DisplayRow::Data(d) if d.vendor_label.is_empty()));
        assert!(matches!(&table[2], DisplayRow::Data(_)));
        match &table[3] {
            DisplayRow::Subtotal { vendor, metrics } => {
                assert_eq!(vendor, "Anthropic");
                assert_eq!(metrics.count, 8);
                assert_eq!(metrics.prefill, 210);
            }
            other => panic!("expected subtotal, got {:?}", other),
        }
        match &table[4] {
            DisplayRow::GroupHeader { vendor } => assert_eq!(vendor, "OpenAI"),
            other => panic!("expected header, got {:?}", other),
        }
        assert!(matches!(&table[5], DisplayRow::Data(_)));
        // Single-row group: no subtotal.
        assert_eq!(table.len(), 6);
    }

    #[test]
    fn model_view_merges_across_harnesses_with_tag_list() {
        let mut codex = row("codex", "gpt-5.5", 9);
        codex.reasoning = 7;
        let rows = vec![
            codex,
            row("claude", "claude-opus-4-8", 5),
            row("omp", "anthropic/claude-opus-4-8", 3),
        ];
        let table = build_table(&rows, TableView::Model);
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
