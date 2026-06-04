//! In-memory row + result types for the `memories` table and the unified
//! search-result shape returned by the handlers.

use serde::Serialize;

/// Max length of an individual tag.
const MAX_TAG_LENGTH: usize = 100;
/// Max tags per memory.
const MAX_TAGS_PER_MEMORY: usize = 100;
/// Max length of a tag JSON-array string before treating it as a literal tag.
const MAX_JSON_LENGTH: usize = 4096;

/// Normalize a list of tags: strip whitespace, replace internal commas with
/// hyphens, truncate to `MAX_TAG_LENGTH`, lowercase, drop empties, and dedup
/// case-insensitively (preserving first-seen order). Caps the total at
/// `MAX_TAGS_PER_MEMORY`. Lowercasing here is what keeps the stored tag CSV and
/// later tag filters comparable, since filtering matches on the stored bytes.
pub fn normalize_tags(tags: &[String]) -> Vec<String> {
    let mut seen_lower = std::collections::HashSet::new();
    let mut out = Vec::new();
    for tag in tags {
        let mut t = tag.trim().to_string();
        if t.is_empty() {
            continue;
        }
        // Tags are stored comma-joined and searched with LIKE, so an embedded
        // comma would split one tag into two; replace it with a hyphen.
        if t.contains(',') {
            t = t.replace(',', "-");
        }
        if t.len() > MAX_TAG_LENGTH {
            t.truncate(MAX_TAG_LENGTH);
        }
        let lower = t.to_lowercase();
        if seen_lower.contains(&lower) {
            continue;
        }
        seen_lower.insert(lower.clone());
        out.push(lower);
        if out.len() >= MAX_TAGS_PER_MEMORY {
            break;
        }
    }
    out
}

/// Split a single tag *string* into raw tokens, before any per-tag
/// normalization:
///   * a leading-`[` JSON array string -> the parsed array elements (or, if the
///     string is too long / not valid JSON / not a list, the literal string);
///   * a comma-containing string -> comma-split, trimmed, empties dropped;
///   * otherwise a single trimmed tag.
/// The per-tag lowercase/dedup/etc. is applied separately via [`normalize_tags`].
pub fn split_tag_string(s: &str) -> Vec<String> {
    let stripped = s.trim();
    if stripped.is_empty() {
        return Vec::new();
    }
    if stripped.starts_with('[') {
        if stripped.len() > MAX_JSON_LENGTH {
            return vec![stripped.to_string()];
        }
        match serde_json::from_str::<serde_json::Value>(stripped) {
            Ok(serde_json::Value::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect(),
            _ => vec![stripped.to_string()],
        }
    } else if stripped.contains(',') {
        stripped
            .split(',')
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
            .collect()
    } else {
        vec![stripped.to_string()]
    }
}

/// A `tags` argument that accepts EITHER a JSON array of strings OR a single
/// string (CSV / JSON-array-string / single tag). The MCP schema advertises this
/// as `oneOf: [array<string>, string]`.
///
/// Deserializing splits a string form into raw tokens (pre-normalization); the
/// per-tag lowercase/dedup/comma->hyphen normalization runs later via
/// [`normalize_tags`] at the resolve site. The custom `JsonSchema` impl emits the
/// matching `oneOf` shape so clients validating against the advertised schema
/// accept the CSV-string form, which agents commonly send.
#[derive(Debug, Clone, Default)]
pub struct Tags(pub Option<Vec<String>>);

impl Tags {
    /// The raw (pre-normalization) token list, empty if the field was absent/null.
    pub fn into_vec(self) -> Vec<String> {
        self.0.unwrap_or_default()
    }
    /// Borrow the inner option. (Test-only today.)
    #[cfg(test)]
    pub fn as_option(&self) -> Option<&Vec<String>> {
        self.0.as_ref()
    }
}

impl<'de> serde::Deserialize<'de> for Tags {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::Deserialize;
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum StringOrVec {
            Str(String),
            Vec(Vec<String>),
        }
        let v = Option::<StringOrVec>::deserialize(deserializer)?;
        Ok(Tags(match v {
            None => None,
            Some(StringOrVec::Str(s)) => Some(split_tag_string(&s)),
            Some(StringOrVec::Vec(v)) => Some(v),
        }))
    }
}

impl schemars::JsonSchema for Tags {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Tags".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        // Advertise `oneOf: [array<string>, string]` so MCP clients accept
        // either shape for the tags argument.
        schemars::json_schema!({
            "oneOf": [
                {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Tags as an array of strings"
                },
                {
                    "type": "string",
                    "description": "Tags as comma-separated string"
                }
            ],
            "description": "Tags. Accepts either an array of strings or a comma-separated string.",
            "examples": ["tag1,tag2,tag3", ["tag1", "tag2", "tag3"]]
        })
    }
}

/// A stored memory row (the columns the read paths select). `tags` is held as a
/// `Vec<String>` in Rust but is stored as a comma-joined TEXT column in SQLite
/// (default `"untagged"` when empty); conversion happens at the storage edge.
#[derive(Debug, Clone, Serialize)]
pub struct Memory {
    pub content_hash: String,
    pub content: String,
    /// Split from the comma-joined `tags` column on read; joined on write.
    pub tags: Vec<String>,
    pub memory_type: Option<String>,
    /// Parsed from the JSON-string `metadata` column.
    pub metadata: serde_json::Value,
    /// Unix epoch seconds (fractional).
    pub created_at: f64,
    pub updated_at: f64,
    /// Naive-UTC ISO 8601 with a literal trailing `Z` and microsecond precision
    /// (see [`epoch_to_iso`]).
    pub created_at_iso: String,
    pub updated_at_iso: String,
}

/// A search hit: a memory plus its vector distance + derived relevance score.
#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    #[serde(flatten)]
    pub memory: Memory,
    /// vec0 cosine distance (0 = identical .. 2 = opposite).
    pub distance: f64,
    /// `max(0, 1 - distance/2)` — normalizes cosine distance into a 0..1 score.
    pub relevance_score: f64,
}

impl SearchHit {
    /// Convert cosine distance into the 0..1 relevance score.
    pub fn relevance_from_distance(distance: f64) -> f64 {
        (1.0 - distance / 2.0).max(0.0)
    }
}

/// Search mode for `memory_search` (`mode` arg). v1 implements `Semantic`;
/// `Exact` (FTS5/BM25) and `Hybrid` (RRF fuse) are wired but may stub initially.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Semantic,
    Exact,
    Hybrid,
    Ranked,
}

impl SearchMode {
    pub fn parse(s: &str) -> Self {
        match s {
            "exact" => SearchMode::Exact,
            "hybrid" => SearchMode::Hybrid,
            "ranked" => SearchMode::Ranked,
            _ => SearchMode::Semantic,
        }
    }
}

/// How multiple tags combine in a filter (`tag_match` arg).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagMatch {
    Any,
    All,
}

impl TagMatch {
    pub fn parse(s: &str) -> Self {
        if s == "all" { TagMatch::All } else { TagMatch::Any }
    }
}

/// Convert an epoch-seconds timestamp to a naive-UTC ISO 8601 string with a
/// literal trailing `Z` and microsecond precision.
///
/// The fractional part is omitted entirely when the microsecond component is
/// exactly 0 (e.g. `2024-01-01T00:00:00Z`), otherwise exactly 6 fractional
/// digits are emitted (e.g. `...:00.123456Z`). This is the on-disk format for
/// the `*_iso` columns, so the two branches must be reproduced exactly for
/// stored strings to round-trip.
pub fn epoch_to_iso(ts: f64) -> String {
    use chrono::{DateTime, Timelike, Utc};
    // Split the timestamp into whole seconds (truncated toward the epoch) and
    // nanoseconds, then build a UTC datetime to format.
    let secs = ts.floor() as i64;
    let nanos = ((ts - ts.floor()) * 1_000_000_000.0).round() as u32;
    let dt: DateTime<Utc> =
        DateTime::from_timestamp(secs, nanos).unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap());
    let micros = dt.nanosecond() / 1_000;
    if micros == 0 {
        dt.format("%Y-%m-%dT%H:%M:%S").to_string() + "Z"
    } else {
        // Emit a dot plus exactly 6 fractional digits.
        format!("{}.{:06}Z", dt.format("%Y-%m-%dT%H:%M:%S"), micros)
    }
}

/// Join tags into the comma-separated TEXT column representation.
///
/// Empty tags join to `""` here; the `"untagged"` default is applied earlier at
/// the tool layer (where tags are normalized), so this function never sees the
/// empty-default case in practice.
pub fn tags_to_csv(tags: &[String]) -> String {
    tags.join(",")
}

/// Split the comma-separated `tags` column back into a `Vec<String>`,
/// trimming each token and dropping empties.
pub fn tags_from_csv(csv: &str) -> Vec<String> {
    if csv.is_empty() {
        return Vec::new();
    }
    csv.split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_matches_python() {
        assert_eq!(epoch_to_iso(0.0), "1970-01-01T00:00:00Z");
        assert_eq!(epoch_to_iso(1700000000.0), "2023-11-14T22:13:20Z");
        assert_eq!(epoch_to_iso(1700000000.5), "2023-11-14T22:13:20.500000Z");
        assert_eq!(epoch_to_iso(1717430400.0), "2024-06-03T16:00:00Z");
    }

    #[test]
    fn tags_roundtrip() {
        assert_eq!(tags_to_csv(&["a".into(), "b".into()]), "a,b");
        assert_eq!(tags_to_csv(&[]), "");
        assert_eq!(tags_from_csv("a, b ,c"), vec!["a", "b", "c"]);
        assert_eq!(tags_from_csv(""), Vec::<String>::new());
        assert_eq!(tags_from_csv("untagged"), vec!["untagged"]);
    }

    #[test]
    fn normalize_tags_lowercases_trims_dedups() {
        assert_eq!(
            normalize_tags(&["Python".into(), " python ".into(), "Reference".into()]),
            vec!["python", "reference"]
        );
        assert_eq!(normalize_tags(&["a,b".into()]), vec!["a-b"]);
        assert_eq!(normalize_tags(&["".into(), "  ".into(), "x".into()]), vec!["x"]);
    }

    #[test]
    fn split_tag_string_forms() {
        assert_eq!(split_tag_string("python,reference"), vec!["python", "reference"]);
        assert_eq!(split_tag_string("single"), vec!["single"]);
        assert_eq!(split_tag_string("  a , , b "), vec!["a", "b"]);
        assert_eq!(
            split_tag_string("[\"x\", \"y\"]"),
            vec!["x".to_string(), "y".to_string()]
        );
        assert_eq!(split_tag_string(""), Vec::<String>::new());
        assert_eq!(split_tag_string("[not json"), vec!["[not json".to_string()]);
    }

    #[test]
    fn tags_deserializes_string_and_array() {
        let t: Tags = serde_json::from_str("\"python,reference\"").unwrap();
        assert_eq!(t.into_vec(), vec!["python", "reference"]);
        let t: Tags = serde_json::from_str("[\"a\",\"b\"]").unwrap();
        assert_eq!(t.into_vec(), vec!["a", "b"]);
        let t: Tags = serde_json::from_str("null").unwrap();
        assert!(t.as_option().is_none());
    }

    #[test]
    fn relevance_from_distance_clamps() {
        assert_eq!(SearchHit::relevance_from_distance(0.0), 1.0);
        assert_eq!(SearchHit::relevance_from_distance(2.0), 0.0);
        assert!((SearchHit::relevance_from_distance(1.0) - 0.5).abs() < 1e-12);
        assert_eq!(SearchHit::relevance_from_distance(3.0), 0.0);
    }
}
