//! Algorithmic model-id parsing.
//!
//! Turns a raw model identifier (as it appears in a vendor's session logs)
//! into a structured [`ModelIdentity`] from which presentation policy is
//! derived: a short display label, a chart sort key, and a chart color family.
//!
//! The point is to avoid hand-maintained per-model tables. A newly released
//! model (e.g. `claude-opus-4-8`, `gpt-5.6`, `gemini-3.2-pro`) parses into a
//! sensible label with zero code changes. Genuinely irregular cases are the
//! job of the external user override file, not this parser.

/// The model maker (vendor) a model belongs to, inferred from the id itself
/// (not from whichever harness happened to log it). This is the "Vendor" axis
/// of the Vendor / Harness / Model display model: a model always belongs to
/// exactly one vendor, while any harness can in principle drive any model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Vendor {
    Anthropic,
    OpenAI,
    Google,
    Moonshot,
    Zhipu,
    Unknown,
}

impl Vendor {
    /// Human-readable vendor name for table columns and group headers.
    pub fn display_name(self) -> &'static str {
        match self {
            Vendor::Anthropic => "Anthropic",
            Vendor::OpenAI => "OpenAI",
            Vendor::Google => "Google",
            Vendor::Moonshot => "Moonshot",
            Vendor::Zhipu => "Zhipu",
            Vendor::Unknown => "Other",
        }
    }
}

/// Structured view of a model id. Parsing strips provider prefixes, date and
/// preview snapshot suffixes, and an optional ` (effort)` annotation, then
/// classifies the remainder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    pub vendor: Vendor,
    /// Sub-family used for grouping/pricing: `opus`/`sonnet`/`haiku` for
    /// Claude, `gpt`/`o`/`codex` for OpenAI, `gemini` for Google, otherwise the
    /// first token of the normalized id.
    pub family: String,
    /// Human version string exactly as it should render, e.g. `4.8`, `5.5`,
    /// `2.0`, `3`, or empty when the id has no version component.
    pub version: String,
    /// `(major, minor)` parsed from `version`, used for ordering.
    pub version_key: (u32, u32),
    /// Recognized price/size-class tokens (e.g. `mini`, `codex`, `flash`,
    /// `lite`), in the order they appear. Empty for a base model.
    pub modifiers: Vec<String>,
    /// Reasoning-effort annotation when the id carried a trailing effort tag.
    pub effort: Option<String>,
    /// Provider/date/effort-stripped, lowercased id.
    pub normalized_id: String,
    /// The untouched input.
    pub raw: String,
}

/// Snapshot/preview markers stripped from the tail of an id.
const TAIL_MARKERS: &[&str] = &["preview", "latest", "exp", "beta", "snapshot"];

/// Claude sub-family tokens. `fable`/`mythos` are the Mythos-class tier above
/// opus (e.g. `claude-fable-5`).
const CLAUDE_FAMILIES: &[&str] = &["fable", "mythos", "opus", "sonnet", "haiku"];

fn is_all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Recognized reasoning-effort annotations.
const EFFORTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max"];

pub fn normalize_reasoning_effort(value: &str) -> Option<String> {
    let effort = value.trim().to_ascii_lowercase();
    EFFORTS.contains(&effort.as_str()).then_some(effort)
}

pub fn is_reasoning_effort(value: &str) -> bool {
    normalize_reasoning_effort(value).is_some()
}

/// Split off a trailing effort annotation, returning `(base, effort)`. Handles
/// both the parenthetical form (`gpt-5.5 (high)`) and the colon form
/// (`gpt-5.5:xhigh`).
fn split_effort(s: &str) -> (&str, Option<String>) {
    let s = s.trim();
    if s.ends_with(')')
        && let Some(idx) = s.rfind(" (")
    {
        return (
            &s[..idx],
            normalize_reasoning_effort(&s[idx + 2..s.len() - 1]),
        );
    }
    if let Some(idx) = s.rfind(':')
        && let Some(effort) = normalize_reasoning_effort(&s[idx + 1..])
    {
        return (&s[..idx], Some(effort));
    }
    (s, None)
}

/// Strip provider prefixes: anything before the last `/`, plus a leading
/// `anthropic.` (Bedrock) segment.
fn strip_provider_prefix(s: &str) -> &str {
    let after_slash = s.rsplit('/').next().unwrap_or(s);
    after_slash
        .strip_prefix("anthropic.")
        .unwrap_or(after_slash)
}

/// Drop trailing date and preview tokens without eating single-digit version
/// components. Handles `-YYYYMMDD`, a `<marker>-MM-YYYY` style run, and bare
/// trailing markers like `-preview` / `-latest`.
fn strip_tail_tokens(tokens: &mut Vec<String>) {
    // 8-digit YYYYMMDD date stamp.
    if tokens
        .last()
        .is_some_and(|t| t.len() == 8 && is_all_digits(t))
    {
        tokens.pop();
    }

    // A numeric run (e.g. `09-2025`) is only a date if it directly follows a
    // marker word; otherwise it is a real version component (`opus-4-8`).
    let mut first_num = tokens.len();
    while first_num > 0 && is_all_digits(&tokens[first_num - 1]) {
        first_num -= 1;
    }
    if first_num < tokens.len()
        && first_num > 0
        && TAIL_MARKERS.contains(&tokens[first_num - 1].as_str())
    {
        tokens.truncate(first_num - 1);
    }

    // Bare trailing markers.
    while tokens
        .last()
        .is_some_and(|t| TAIL_MARKERS.contains(&t.as_str()))
    {
        tokens.pop();
    }
}

/// Parse a raw model id into a [`ModelIdentity`].
pub fn parse_model_identity(raw: &str) -> ModelIdentity {
    let (base, effort) = split_effort(raw);

    if base == "<synthetic>" {
        return ModelIdentity {
            vendor: Vendor::Unknown,
            family: "synthetic".to_string(),
            version: String::new(),
            version_key: (0, 0),
            modifiers: Vec::new(),
            effort,
            normalized_id: "<synthetic>".to_string(),
            raw: raw.to_string(),
        };
    }

    let stripped = strip_provider_prefix(base).to_ascii_lowercase();
    let mut tokens: Vec<String> = stripped.split('-').map(|s| s.to_string()).collect();
    strip_tail_tokens(&mut tokens);
    let normalized_id = tokens.join("-");

    let has_claude_family = tokens.iter().any(|t| CLAUDE_FAMILIES.contains(&t.as_str()));
    let is_o_series = tokens
        .first()
        .is_some_and(|t| t.starts_with('o') && t[1..].starts_with(|c: char| c.is_ascii_digit()));

    let (vendor, family, version, version_key, modifiers) =
        if stripped.starts_with("claude") || has_claude_family {
            parse_claude(&tokens)
        } else if is_o_series {
            // o-series ids are rendered verbatim, but we still parse the series
            // number and any size suffix so ordering and pricing fallback work.
            let major = tokens[0][1..].parse::<u32>().unwrap_or(0);
            let modifiers = collect_modifiers(tokens.get(1..).unwrap_or(&[]), OPENAI_MODIFIERS);
            (
                Vendor::OpenAI,
                "o".to_string(),
                String::new(),
                (major, 0),
                modifiers,
            )
        } else if tokens.first().is_some_and(|t| t == "gpt") {
            parse_versioned(Vendor::OpenAI, "gpt", &tokens, OPENAI_MODIFIERS)
        } else if tokens.first().is_some_and(|t| t == "codex") {
            // `codex-mini-latest` etc.: no numeric version, modifiers follow.
            let modifiers = collect_modifiers(&tokens[1..], OPENAI_MODIFIERS);
            (
                Vendor::OpenAI,
                "codex".to_string(),
                String::new(),
                (0, 0),
                modifiers,
            )
        } else if tokens.first().is_some_and(|t| t == "gemini") {
            parse_versioned(Vendor::Google, "gemini", &tokens, GEMINI_MODIFIERS)
        } else if tokens.first().is_some_and(|t| t == "glm") {
            parse_versioned(Vendor::Zhipu, "glm", &tokens, ZHIPU_MODIFIERS)
        } else if tokens
            .first()
            .is_some_and(|t| t == "kimi" || is_k_version_token(t))
        {
            parse_kimi(&tokens)
        } else {
            let family = tokens.first().cloned().unwrap_or_default();
            (Vendor::Unknown, family, String::new(), (0, 0), Vec::new())
        };

    ModelIdentity {
        vendor,
        family,
        version,
        version_key,
        modifiers,
        effort,
        normalized_id,
        raw: raw.to_string(),
    }
}

const OPENAI_MODIFIERS: &[&str] = &[
    "mini", "nano", "max", "spark", "codex", "pro", "sol", "luna",
];
const GEMINI_MODIFIERS: &[&str] = &["pro", "flash", "lite", "image", "ultra"];
const KIMI_MODIFIERS: &[&str] = &["coding", "highspeed", "turbo"];
const ZHIPU_MODIFIERS: &[&str] = &["air", "airx", "x", "flash", "flashx"];

/// A bare `k<version>` token like `k3` or `k2.5` (the Kimi flagship line).
fn is_k_version_token(t: &str) -> bool {
    t.strip_prefix('k').is_some_and(|rest| {
        rest.starts_with(|c: char| c.is_ascii_digit())
            && rest.chars().all(|c| c.is_ascii_digit() || c == '.')
    })
}

/// Parse Kimi ids: `k3`, `kimi-k2.5`, `kimi-for-coding`,
/// `kimi-for-coding-highspeed`, `kimi-latest`.
fn parse_kimi(tokens: &[String]) -> (Vendor, String, String, (u32, u32), Vec<String>) {
    let (version, version_key) =
        tokens
            .iter()
            .find(|t| is_k_version_token(t))
            .map_or((String::new(), (0, 0)), |t| {
                let v = &t[1..];
                (v.to_string(), parse_dotted_version(v))
            });
    let modifiers = collect_modifiers(tokens, KIMI_MODIFIERS);
    (
        Vendor::Moonshot,
        "kimi".to_string(),
        version,
        version_key,
        modifiers,
    )
}

fn collect_modifiers(tokens: &[String], recognized: &[&str]) -> Vec<String> {
    tokens
        .iter()
        .filter(|t| recognized.contains(&t.as_str()))
        .cloned()
        .collect()
}

/// Parse Claude ids in either ordering: `claude-opus-4-8` (family first) or
/// the older `claude-3-7-sonnet` (numbers first).
fn parse_claude(tokens: &[String]) -> (Vendor, String, String, (u32, u32), Vec<String>) {
    let family = tokens
        .iter()
        .find(|t| CLAUDE_FAMILIES.contains(&t.as_str()))
        .cloned()
        .unwrap_or_else(|| "claude".to_string());

    let nums: Vec<u32> = tokens
        .iter()
        .filter(|t| is_all_digits(t))
        .filter_map(|t| t.parse::<u32>().ok())
        .collect();
    let (version, version_key) = format_version_from_parts(&nums);
    (Vendor::Anthropic, family, version, version_key, Vec::new())
}

/// Parse a versioned id whose version is the first dotted/numeric token after
/// the family (`gpt-5.5-codex`, `gemini-2.5-flash-lite`).
fn parse_versioned(
    vendor: Vendor,
    family: &str,
    tokens: &[String],
    recognized: &[&str],
) -> (Vendor, String, String, (u32, u32), Vec<String>) {
    let version_tok = tokens.get(1).cloned().unwrap_or_default();
    let (version, version_key) = if version_tok
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit())
    {
        let key = parse_dotted_version(&version_tok);
        (version_tok.clone(), key)
    } else {
        (String::new(), (0, 0))
    };
    let rest_start = if version.is_empty() { 1 } else { 2 };
    let modifiers = collect_modifiers(tokens.get(rest_start..).unwrap_or(&[]), recognized);
    (vendor, family.to_string(), version, version_key, modifiers)
}

/// `5.5` -> `(5, 5)`, `2.0` -> `(2, 0)`, `3` -> `(3, 0)`.
fn parse_dotted_version(s: &str) -> (u32, u32) {
    let mut parts = s.split('.');
    let major = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    let minor = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    (major, minor)
}

/// Hyphen-separated version parts -> `(string, key)`.
/// `[4, 8]` -> `("4.8", (4, 8))`, `[4]` -> `("4", (4, 0))`.
fn format_version_from_parts(nums: &[u32]) -> (String, (u32, u32)) {
    match nums {
        [] => (String::new(), (0, 0)),
        [major] => (major.to_string(), (*major, 0)),
        [major, minor, ..] => (format!("{}.{}", major, minor), (*major, *minor)),
    }
}

/// Capitalize the first ASCII letter of a token.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

/// Title-case an unknown id at word boundaries (`foo-bar_9` -> `Foo Bar 9`)
/// instead of slicing it mid-token.
fn title_case(s: &str) -> String {
    s.split(['-', '_', '.', '/'])
        .filter(|t| !t.is_empty())
        .map(capitalize)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Abbreviation for the single most salient modifier, by vendor precedence.
fn modifier_abbrev(vendor: Vendor, modifiers: &[String]) -> Option<&'static str> {
    let has = |needle: &str| modifiers.iter().any(|m| m == needle);
    match vendor {
        Vendor::OpenAI => {
            // Size/spark beat named variants, which beat the bare `codex` marker.
            if has("max") {
                Some("Max")
            } else if has("mini") {
                Some("Mini")
            } else if has("nano") {
                Some("Nano")
            } else if has("spark") {
                Some("Sprk")
            } else if has("sol") {
                Some("Sol")
            } else if has("luna") {
                Some("Luna")
            } else if has("codex") {
                Some("Cdx")
            } else {
                None
            }
        }
        Vendor::Google => {
            // `image` wins; `flash-lite` reads as `Lt`, not `Fl`.
            if has("image") {
                Some("Img")
            } else if has("lite") {
                Some("Lt")
            } else if has("flash") {
                Some("Fl")
            } else if has("pro") {
                Some("Pro")
            } else if has("ultra") {
                Some("Ult")
            } else {
                None
            }
        }
        Vendor::Moonshot => {
            // Speed variants beat the bare `coding` marker.
            if has("highspeed") {
                Some("HS")
            } else if has("turbo") {
                Some("Tb")
            } else if has("coding") {
                Some("Coding")
            } else {
                None
            }
        }
        Vendor::Zhipu => {
            // Combined `airx`/`flashx` beat their base tokens.
            if has("airx") {
                Some("AirX")
            } else if has("flashx") {
                Some("FlX")
            } else if has("air") {
                Some("Air")
            } else if has("flash") {
                Some("Fl")
            } else if has("x") {
                Some("X")
            } else {
                None
            }
        }
        _ => None,
    }
}

fn effort_abbrev(effort: &str) -> String {
    match effort {
        "low" => "L".to_string(),
        "medium" => "M".to_string(),
        "high" => "H".to_string(),
        "xhigh" => "XH".to_string(),
        "max" => "Max".to_string(),
        other => capitalize(&other[..1]),
    }
}

/// Render the compact display label for a model, e.g. `Opus 4.8`,
/// `GPT-5.5 Cdx`, `Gem 3.2 Pro`.
pub fn short_label(id: &ModelIdentity) -> String {
    let mut base = match id.vendor {
        Vendor::Anthropic => {
            let fam = capitalize(&id.family);
            if id.version.is_empty() {
                fam
            } else {
                format!("{} {}", fam, id.version)
            }
        }
        Vendor::OpenAI if id.family == "o" => id.normalized_id.clone(),
        Vendor::OpenAI => {
            let head = if id.family == "codex" {
                "Codex".to_string()
            } else if id.version.is_empty() {
                "GPT".to_string()
            } else {
                format!("GPT-{}", id.version)
            };
            match modifier_abbrev(id.vendor, &id.modifiers) {
                Some(abbr) => format!("{} {}", head, abbr),
                None => head,
            }
        }
        Vendor::Google => {
            let head = if id.version.is_empty() {
                "Gem".to_string()
            } else {
                format!("Gem {}", id.version)
            };
            match modifier_abbrev(id.vendor, &id.modifiers) {
                Some(abbr) => format!("{} {}", head, abbr),
                None => head,
            }
        }
        Vendor::Moonshot => {
            let head = if id.version.is_empty() {
                "Kimi".to_string()
            } else {
                format!("K{}", id.version)
            };
            match modifier_abbrev(id.vendor, &id.modifiers) {
                Some(abbr) => format!("{} {}", head, abbr),
                None => head,
            }
        }
        Vendor::Zhipu => {
            let head = if id.version.is_empty() {
                "GLM".to_string()
            } else {
                format!("GLM-{}", id.version)
            };
            match modifier_abbrev(id.vendor, &id.modifiers) {
                Some(abbr) => format!("{} {}", head, abbr),
                None => head,
            }
        }
        Vendor::Unknown => {
            if id.normalized_id == "<synthetic>" {
                "synthetic".to_string()
            } else {
                title_case(&id.normalized_id)
            }
        }
    };

    if let Some(effort) = &id.effort {
        base = format!("{}({})", base, effort_abbrev(effort));
    }
    base
}

fn family_rank(id: &ModelIdentity) -> u32 {
    match id.vendor {
        Vendor::Anthropic => match id.family.as_str() {
            "fable" | "mythos" => 0,
            "opus" => 1,
            "sonnet" => 2,
            "haiku" => 3,
            _ => 4,
        },
        Vendor::OpenAI => 10,
        Vendor::Google => 20,
        Vendor::Moonshot => 25,
        Vendor::Zhipu => 27,
        Vendor::Unknown => 30,
    }
}

/// Ordering key for charts: groups by family rank, then newest version first,
/// then by normalized id for stability.
pub fn sort_key(id: &ModelIdentity) -> (u32, std::cmp::Reverse<(u32, u32)>, String) {
    (
        family_rank(id),
        std::cmp::Reverse(id.version_key),
        id.normalized_id.clone(),
    )
}

/// Chart color family. Only Claude's sub-families get dedicated palette
/// entries today; everything else falls back to the indexed palette. Fable and
/// Mythos share one palette entry (same model behind two ids).
pub fn color_family(id: &ModelIdentity) -> Option<&'static str> {
    if id.vendor == Vendor::Anthropic {
        match id.family.as_str() {
            "fable" | "mythos" => Some("fable"),
            "opus" => Some("opus"),
            "sonnet" => Some("sonnet"),
            "haiku" => Some("haiku"),
            _ => None,
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(s: &str) -> String {
        short_label(&parse_model_identity(s))
    }

    #[test]
    fn claude_labels_derive_family_and_version() {
        assert_eq!(label("claude-fable-5"), "Fable 5");
        assert_eq!(label("claude-mythos-5"), "Mythos 5");
        assert_eq!(label("claude-mythos-preview"), "Mythos");
        assert_eq!(label("claude-opus-4-8"), "Opus 4.8");
        assert_eq!(label("claude-opus-4-7"), "Opus 4.7");
        assert_eq!(label("claude-opus-4-5-20251101"), "Opus 4.5");
        assert_eq!(label("claude-opus-4-1-20250805"), "Opus 4.1");
        assert_eq!(label("claude-sonnet-4-6"), "Sonnet 4.6");
        assert_eq!(label("claude-sonnet-4-5-20250929"), "Sonnet 4.5");
        assert_eq!(label("claude-sonnet-4-20250514"), "Sonnet 4");
        assert_eq!(label("claude-haiku-4-5-20251001"), "Haiku 4.5");
        // Older "claude-3-7-sonnet" ordering (family token after the numbers).
        assert_eq!(label("claude-3-7-sonnet-20250219"), "Sonnet 3.7");
        assert_eq!(label("claude-3-5-haiku-20241022"), "Haiku 3.5");
        // Provider prefixes (Bedrock dot form, OMP slash form).
        assert_eq!(label("anthropic.claude-opus-4-8"), "Opus 4.8");
        assert_eq!(label("anthropic/claude-opus-4-8"), "Opus 4.8");
    }

    #[test]
    fn codex_labels_handle_versions_modifiers_and_effort() {
        assert_eq!(label("gpt-5.5"), "GPT-5.5");
        assert_eq!(label("gpt-5"), "GPT-5");
        assert_eq!(label("gpt-4.1"), "GPT-4.1");
        assert_eq!(label("gpt-5.5-codex"), "GPT-5.5 Cdx");
        assert_eq!(label("gpt-5.4-mini"), "GPT-5.4 Mini");
        assert_eq!(label("gpt-5.4-nano"), "GPT-5.4 Nano");
        assert_eq!(label("gpt-5.1-codex-max"), "GPT-5.1 Max");
        assert_eq!(label("gpt-5.1-codex-mini"), "GPT-5.1 Mini");
        assert_eq!(label("gpt-5.3-codex-spark"), "GPT-5.3 Sprk");
        assert_eq!(label("gpt-5.6-sol"), "GPT-5.6 Sol");
        assert_eq!(label("gpt-5.6-luna"), "GPT-5.6 Luna");
        assert_eq!(label("gpt-4.1-mini"), "GPT-4.1 Mini");
        assert_eq!(label("codex-mini-latest"), "Codex Mini");
        assert_eq!(label("o3"), "o3");
        assert_eq!(label("o3-mini"), "o3-mini");
        assert_eq!(label("o4-mini"), "o4-mini");
        assert_eq!(label("gpt-5.5 (high)"), "GPT-5.5(H)");
        assert_eq!(label("gpt-5.4-codex (medium)"), "GPT-5.4 Cdx(M)");
        assert_eq!(label("gpt-5.5:xhigh"), "GPT-5.5(XH)");
        assert_eq!(label("gpt-5.5:high"), "GPT-5.5(H)");
        assert_eq!(label("gpt-5.5:max"), "GPT-5.5(Max)");
        assert_eq!(label("gpt-5.5 (rust-cat)"), "GPT-5.5");
    }

    #[test]
    fn gemini_labels_abbreviate_tiers() {
        assert_eq!(label("gemini-3.2-pro-preview"), "Gem 3.2 Pro");
        assert_eq!(label("gemini-3-pro-preview"), "Gem 3 Pro");
        assert_eq!(label("gemini-3-flash-preview"), "Gem 3 Fl");
        assert_eq!(label("gemini-3.1-flash-lite-preview"), "Gem 3.1 Lt");
        assert_eq!(label("gemini-3-pro-image-preview"), "Gem 3 Img");
        assert_eq!(label("gemini-2.5-pro"), "Gem 2.5 Pro");
        assert_eq!(label("gemini-2.5-flash"), "Gem 2.5 Fl");
        assert_eq!(label("gemini-2.5-flash-lite"), "Gem 2.5 Lt");
        assert_eq!(label("gemini-2.0-flash"), "Gem 2.0 Fl");
        assert_eq!(label("gemini-2.5-flash-preview-09-2025"), "Gem 2.5 Fl");
    }

    #[test]
    fn kimi_labels_derive_version_and_modifiers() {
        assert_eq!(label("k3"), "K3");
        assert_eq!(label("kimi-code/k3"), "K3");
        assert_eq!(label("kimi-k2.5"), "K2.5");
        assert_eq!(label("kimi-k2"), "K2");
        assert_eq!(label("kimi-for-coding"), "Kimi Coding");
        assert_eq!(label("kimi-for-coding-highspeed"), "Kimi HS");
        assert_eq!(label("kimi-latest"), "Kimi");
        assert_eq!(label("k3 (max)"), "K3(Max)");
        assert_eq!(label("k3:high"), "K3(H)");
    }

    #[test]
    fn kimi_parse_exposes_provider_family_version_modifiers() {
        let k = parse_model_identity("k3");
        assert_eq!(k.vendor, Vendor::Moonshot);
        assert_eq!(k.family, "kimi");
        assert_eq!(k.version_key, (3, 0));
        assert!(k.modifiers.is_empty());

        let k25 = parse_model_identity("kimi-k2.5");
        assert_eq!(k25.vendor, Vendor::Moonshot);
        assert_eq!(k25.family, "kimi");
        assert_eq!(k25.version_key, (2, 5));

        let coding = parse_model_identity("kimi-for-coding");
        assert_eq!(coding.vendor, Vendor::Moonshot);
        assert_eq!(coding.version_key, (0, 0));
        assert_eq!(coding.modifiers, vec!["coding".to_string()]);

        let highspeed = parse_model_identity("kimi-for-coding-highspeed");
        assert_eq!(
            highspeed.modifiers,
            vec!["coding".to_string(), "highspeed".to_string()]
        );
    }

    #[test]
    fn kimi_sorts_after_google_before_unknown_newest_first() {
        let k3 = sort_key(&parse_model_identity("k3"));
        let k25 = sort_key(&parse_model_identity("kimi-k2.5"));
        let gem = sort_key(&parse_model_identity("gemini-2.5-pro"));
        let unknown = sort_key(&parse_model_identity("mystery-model"));

        assert!(gem < k3);
        assert!(k3 < k25);
        assert!(k25 < unknown);
    }

    #[test]
    fn glm_labels_derive_version_and_modifiers() {
        assert_eq!(label("glm-5.2"), "GLM-5.2");
        assert_eq!(label("glm-4.6"), "GLM-4.6");
        assert_eq!(label("glm-4.5-air"), "GLM-4.5 Air");
        assert_eq!(label("glm-4.5-flash"), "GLM-4.5 Fl");
        assert_eq!(label("glm-4.5-x"), "GLM-4.5 X");
        assert_eq!(label("glm-4.5-airx"), "GLM-4.5 AirX");
        assert_eq!(label("zai-org/GLM-5.2"), "GLM-5.2");
        assert_eq!(label("glm-5.2:high"), "GLM-5.2(H)");
    }

    #[test]
    fn glm_parse_exposes_vendor_family_version() {
        let g = parse_model_identity("glm-5.2");
        assert_eq!(g.vendor, Vendor::Zhipu);
        assert_eq!(g.family, "glm");
        assert_eq!(g.version_key, (5, 2));
        assert!(g.modifiers.is_empty());

        let air = parse_model_identity("glm-4.5-air");
        assert_eq!(air.vendor, Vendor::Zhipu);
        assert_eq!(air.modifiers, vec!["air".to_string()]);
    }

    #[test]
    fn vendor_display_names_are_maker_names() {
        assert_eq!(Vendor::Anthropic.display_name(), "Anthropic");
        assert_eq!(Vendor::OpenAI.display_name(), "OpenAI");
        assert_eq!(Vendor::Google.display_name(), "Google");
        assert_eq!(Vendor::Moonshot.display_name(), "Moonshot");
        assert_eq!(Vendor::Zhipu.display_name(), "Zhipu");
        assert_eq!(Vendor::Unknown.display_name(), "Other");
    }

    #[test]
    fn vendor_is_inferred_from_model_id() {
        let vendor = |s: &str| parse_model_identity(s).vendor;
        assert_eq!(vendor("claude-opus-4-8"), Vendor::Anthropic);
        assert_eq!(vendor("anthropic/claude-opus-4-8"), Vendor::Anthropic);
        assert_eq!(vendor("gpt-5.5-codex"), Vendor::OpenAI);
        assert_eq!(vendor("o3-mini"), Vendor::OpenAI);
        assert_eq!(vendor("gemini-2.5-pro"), Vendor::Google);
        assert_eq!(vendor("kimi-k2.5"), Vendor::Moonshot);
        assert_eq!(vendor("k3"), Vendor::Moonshot);
        assert_eq!(vendor("glm-5.2"), Vendor::Zhipu);
        assert_eq!(vendor("mystery-model"), Vendor::Unknown);
    }

    #[test]
    fn glm_sorts_after_moonshot_before_unknown() {
        let kimi = sort_key(&parse_model_identity("kimi-k2.5"));
        let glm = sort_key(&parse_model_identity("glm-5.2"));
        let glm_old = sort_key(&parse_model_identity("glm-4.6"));
        let unknown = sort_key(&parse_model_identity("mystery-model"));

        assert!(kimi < glm);
        assert!(glm < glm_old);
        assert!(glm_old < unknown);
    }

    #[test]
    fn unknown_ids_title_case_instead_of_truncating() {
        assert_eq!(label("<synthetic>"), "synthetic");
        assert_eq!(label("foo-bar-9"), "Foo Bar 9");
        assert_eq!(label("some_weird_model"), "Some Weird Model");
    }

    #[test]
    fn color_family_is_claude_subfamily_only() {
        assert_eq!(
            color_family(&parse_model_identity("claude-fable-5")),
            Some("fable")
        );
        assert_eq!(
            color_family(&parse_model_identity("claude-mythos-5")),
            Some("fable")
        );
        assert_eq!(
            color_family(&parse_model_identity("claude-opus-4-8")),
            Some("opus")
        );
        assert_eq!(
            color_family(&parse_model_identity("claude-sonnet-4-6")),
            Some("sonnet")
        );
        assert_eq!(
            color_family(&parse_model_identity("claude-haiku-4-5-20251001")),
            Some("haiku")
        );
        assert_eq!(color_family(&parse_model_identity("gpt-5.5")), None);
        assert_eq!(color_family(&parse_model_identity("gemini-2.5-pro")), None);
        assert_eq!(color_family(&parse_model_identity("mystery-model")), None);
    }

    #[test]
    fn parse_exposes_provider_family_version_modifiers() {
        let c = parse_model_identity("claude-opus-4-8");
        assert_eq!(c.vendor, Vendor::Anthropic);
        assert_eq!(c.family, "opus");
        assert_eq!(c.version_key, (4, 8));
        assert!(c.modifiers.is_empty());

        let g = parse_model_identity("gpt-5.5-codex");
        assert_eq!(g.vendor, Vendor::OpenAI);
        assert_eq!(g.family, "gpt");
        assert_eq!(g.version_key, (5, 5));
        assert_eq!(g.modifiers, vec!["codex".to_string()]);

        let m = parse_model_identity("gpt-5.5-mini");
        assert_eq!(m.modifiers, vec!["mini".to_string()]);

        let fl = parse_model_identity("gemini-2.5-flash-lite");
        assert_eq!(fl.vendor, Vendor::Google);
        assert_eq!(fl.family, "gemini");
        assert_eq!(fl.version_key, (2, 5));
        assert!(fl.modifiers.contains(&"lite".to_string()));
    }

    #[test]
    fn sort_key_groups_family_then_newest_first() {
        let fable5 = sort_key(&parse_model_identity("claude-fable-5"));
        let opus8 = sort_key(&parse_model_identity("claude-opus-4-8"));
        let opus7 = sort_key(&parse_model_identity("claude-opus-4-7"));
        let sonnet6 = sort_key(&parse_model_identity("claude-sonnet-4-6"));
        let haiku5 = sort_key(&parse_model_identity("claude-haiku-4-5-20251001"));
        let gpt = sort_key(&parse_model_identity("gpt-5.5"));

        // Newer version sorts before older within a family.
        assert!(opus8 < opus7);
        // fable < opus < sonnet < haiku, all before OpenAI.
        assert!(fable5 < opus8);
        assert!(opus7 < sonnet6);
        assert!(sonnet6 < haiku5);
        assert!(haiku5 < gpt);
    }
}
