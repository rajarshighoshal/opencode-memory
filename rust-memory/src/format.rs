//! Per-tool output-string formatting. These templates are load-bearing:
//! callers parse the prose, so the exact wording and layout are part of the
//! tool contract and must not change casually.

use crate::models::SearchHit;

/// `memory_store` success: `"Memory stored successfully (hash: <hash>)"`.
pub fn store_ok(hash: &str) -> String {
    format!("Memory stored successfully (hash: {hash})")
}

/// `memory_store` exact-duplicate rejection. The store handler surfaces a failed
/// store as `"Error storing memory: <error>"`; here the error is
/// `"Duplicate content detected (exact match)"`.
pub fn store_duplicate(_hash: &str) -> String {
    "Error storing memory: Duplicate content detected (exact match)".to_string()
}

/// `memory_store` semantic-duplicate rejection. The store handler surfaces a failed
/// store as `"Error storing memory: <error>"`; here the error names the existing
/// near-match: `"Duplicate content detected (semantically similar to <hash>)"`.
pub fn store_semantic_duplicate(existing_hash: &str) -> String {
    format!(
        "Error storing memory: Duplicate content detected (semantically similar to {existing_hash})"
    )
}

/// `memory_search` result block.
///
/// Header: `"Found <total> memories"` + `" (mode: <mode>)"` + (if query non-empty)
/// `" for query: '<q>'"`. Then `header + "\n\n" + "\n\n".join(results)` where each
/// result is `"<i>. <content>\n   Hash: <hash>\n   Created: <created_at>[ [tags]]"`.
/// `created_at` uses the ISO timestamp, falling back to the raw float timestamp
/// rendered as a string when the ISO field is empty.
///
/// Empty: `"No memories found"` + (if query non-empty) `" for query: '<q>'"`.
pub fn search_results(query: &str, mode: &str, hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        let mut s = String::from("No memories found");
        if !query.is_empty() {
            s.push_str(&format!(" for query: '{query}'"));
        }
        return s;
    }

    let total = hits.len();
    let mut header = format!("Found {total} memories (mode: {mode})");
    if !query.is_empty() {
        header.push_str(&format!(" for query: '{query}'"));
    }

    let results: Vec<String> = hits
        .iter()
        .enumerate()
        .map(|(i, hit)| {
            let m = &hit.memory;
            let created_at = if !m.created_at_iso.is_empty() {
                m.created_at_iso.clone()
            } else {
                m.created_at.to_string()
            };
            let tags_display = if m.tags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", m.tags.join(", "))
            };
            format!(
                "{}. {}\n   Hash: {}\n   Created: {}{}",
                i + 1,
                m.content,
                m.content_hash,
                created_at,
                tags_display
            )
        })
        .collect();

    format!("{}\n\n{}", header, results.join("\n\n"))
}

/// `memory_delete` message.
///
/// Starts from the storage-layer message and appends:
///   * dry_run -> `"\n\nWould delete N memories"` and, if N>0,
///     `"\nHashes: <h1, h2, h3, h4, h5>"` (first 5) + (if N>5) `" ... and <N-5> more"`.
///   * else -> `"\n\nDeleted N memories"`.
/// `base_message` is the storage-layer message (e.g. `"Successfully deleted memory <h>"`).
pub fn delete_msg(n: usize, hashes: &[String], dry_run: bool, base_message: &str) -> String {
    let mut response = base_message.to_string();
    if dry_run {
        response.push_str(&format!("\n\nWould delete {n} memories"));
        if n > 0 {
            let shown: Vec<&str> = hashes.iter().take(5).map(|s| s.as_str()).collect();
            response.push_str(&format!("\nHashes: {}", shown.join(", ")));
            if n > 5 {
                response.push_str(&format!(" ... and {} more", n - 5));
            }
        }
    } else {
        response.push_str(&format!("\n\nDeleted {n} memories"));
    }
    response
}

/// `memory_health` payload: `"Database Health Check Results:\n"` + pretty JSON.
pub fn health(json: &serde_json::Value) -> String {
    format!(
        "Database Health Check Results:\n{}",
        serde_json::to_string_pretty(json).unwrap_or_else(|_| "{}".to_string())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Memory, SearchHit};

    fn hit(content: &str, hash: &str, iso: &str, tags: &[&str], distance: f64) -> SearchHit {
        SearchHit {
            memory: Memory {
                content_hash: hash.into(),
                content: content.into(),
                tags: tags.iter().map(|s| s.to_string()).collect(),
                memory_type: Some("note".into()),
                metadata: serde_json::json!({}),
                created_at: 0.0,
                updated_at: 0.0,
                created_at_iso: iso.into(),
                updated_at_iso: iso.into(),
            },
            distance,
            relevance_score: SearchHit::relevance_from_distance(distance),
        }
    }

    #[test]
    fn search_empty_with_query() {
        assert_eq!(
            search_results("sycophancy", "semantic", &[]),
            "No memories found for query: 'sycophancy'"
        );
    }

    #[test]
    fn search_empty_no_query() {
        assert_eq!(search_results("", "semantic", &[]), "No memories found");
    }

    #[test]
    fn search_one_result_with_tags() {
        let hits = vec![hit("hello", "abc123", "2024-01-01T00:00:00Z", &["x", "y"], 0.2)];
        let out = search_results("q", "semantic", &hits);
        assert_eq!(
            out,
            "Found 1 memories (mode: semantic) for query: 'q'\n\n\
             1. hello\n   Hash: abc123\n   Created: 2024-01-01T00:00:00Z [x, y]"
        );
    }

    #[test]
    fn search_result_no_tags() {
        let hits = vec![hit("hi", "h1", "2024-01-01T00:00:00Z", &[], 0.0)];
        let out = search_results("", "exact", &hits);
        // No query -> header omits the "for query" suffix.
        assert_eq!(
            out,
            "Found 1 memories (mode: exact)\n\n1. hi\n   Hash: h1\n   Created: 2024-01-01T00:00:00Z"
        );
    }

    #[test]
    fn delete_single_hash_message() {
        assert_eq!(
            delete_msg(1, &["abc".into()], false, "Successfully deleted memory abc"),
            "Successfully deleted memory abc\n\nDeleted 1 memories"
        );
    }

    #[test]
    fn delete_dry_run_truncates_hashes() {
        let hashes: Vec<String> = (0..7).map(|i| format!("h{i}")).collect();
        // For tag/filter deletes the dry-run base message is "Would delete N memories",
        // so it appears twice once delete_msg appends its own dry-run line.
        let out = delete_msg(7, &hashes, true, "Would delete 7 memories");
        assert_eq!(
            out,
            "Would delete 7 memories\n\nWould delete 7 memories\nHashes: h0, h1, h2, h3, h4 ... and 2 more"
        );
    }

    #[test]
    fn store_duplicate_is_error_line() {
        assert_eq!(
            store_duplicate("abc"),
            "Error storing memory: Duplicate content detected (exact match)"
        );
    }
}
