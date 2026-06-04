//! End-to-end integration test of Storage + EmbedClient against a COPY of the
//! live DB, using the live llama.cpp embedding server at :11434.
//!
//! HARD RULE: never touch the live DB. This test only runs when
//! `MCP_MEMORY_SQLITE_PATH` points at a non-live copy AND the embed server is up;
//! otherwise it is skipped (returns early) so `cargo test` stays green offline.
//!
//! The test runner is the crate binary's modules — but this integration test
//! lives outside the crate, so we re-exercise the SAME behaviors through a tiny
//! direct rusqlite read plus a real embedding round-trip, then a KNN self-query
//! to confirm distance ~0 for an identical stored vector.

#![allow(clippy::missing_transmute_annotations)]

use std::time::Duration;

const LIVE_GLOBAL: &str = "/.config/opencode/memory/global.db";

fn embed_server_up() -> bool {
    // Block on a tiny health check with a short timeout.
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap();
        client
            .get("http://127.0.0.1:11434/health")
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    })
}

#[test]
fn e2e_storage_embed_against_copy() {
    let db = match std::env::var("MCP_MEMORY_SQLITE_PATH") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SKIP: MCP_MEMORY_SQLITE_PATH not set");
            return;
        }
    };
    // Safety: refuse to run if the path resolves to the live DB.
    if db.ends_with(LIVE_GLOBAL) || db.contains("/.config/opencode/memory/") {
        panic!("REFUSING to run E2E against a live DB path: {db}");
    }
    if !embed_server_up() {
        eprintln!("SKIP: embed server not up at :11434");
        return;
    }

    // Register the static vec0 extension once (mirrors main.rs).
    unsafe {
        let rc = rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
        assert_eq!(rc, rusqlite::ffi::SQLITE_OK);
    }

    let conn = rusqlite::Connection::open(&db).expect("open copy");
    conn.pragma_update(None, "journal_mode", "WAL").unwrap();
    conn.pragma_update(None, "busy_timeout", 5000).unwrap();
    conn.pragma_update(None, "synchronous", "NORMAL").unwrap();

    // vec_version resolves (extension live).
    let vv: String = conn
        .query_row("SELECT vec_version()", [], |r| r.get(0))
        .expect("vec_version");
    eprintln!("vec_version = {vv}");
    assert!(vv.starts_with("v0.1"));

    // Count live memories + embeddings.
    let n_mem: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let n_emb: i64 = conn
        .query_row("SELECT COUNT(*) FROM memory_embeddings", [], |r| r.get(0))
        .unwrap();
    eprintln!("live memories={n_mem}, embeddings={n_emb}");
    assert!(n_mem > 0, "expected some live memories in the copy");

    // Grab one stored embedding blob; decode it; run a KNN self-query with that
    // exact blob and confirm the nearest distance is ~0 (identical vector).
    let (rowid, blob): (i64, Vec<u8>) = conn
        .query_row(
            "SELECT rowid, content_embedding FROM memory_embeddings LIMIT 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .expect("read one embedding");
    assert_eq!(blob.len(), 2560 * 4, "embedding blob must be 10240 bytes");

    // vec0 0.1.9 forbids `k = ?` and LIMIT in the SAME statement, so use the
    // production pattern: `k =` lives in an inner subquery, no outer LIMIT here.
    let nearest_dist: f64 = conn
        .query_row(
            "SELECT distance FROM memory_embeddings
             WHERE content_embedding MATCH ?1 AND k = 1",
            rusqlite::params![blob],
            |r| r.get(0),
        )
        .expect("knn self-query");
    eprintln!("self-query nearest distance = {nearest_dist} (rowid {rowid})");
    assert!(nearest_dist < 1e-4, "self-distance should be ~0");

    // Now exercise the REAL embed path: embed a query string via the live server
    // and run a KNN search over the copy. This proves embed dim + KNN wiring.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let query_vec: Vec<f32> = rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap();
        let resp: serde_json::Value = client
            .post("http://127.0.0.1:11434/v1/embeddings")
            .json(&serde_json::json!({"input": ["sycophancy"], "model": "qwen3-embedding-4b"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        resp["data"][0]["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    });
    assert_eq!(query_vec.len(), 2560, "embed dim must be 2560");

    let qblob: Vec<u8> = query_vec.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut stmt = conn
        .prepare(
            "SELECT m.content_hash, m.content, e.distance
             FROM memories m
             INNER JOIN (SELECT rowid, distance FROM memory_embeddings
                         WHERE content_embedding MATCH ?1 AND k = ?2) e
               ON m.id = e.rowid
             WHERE m.deleted_at IS NULL
             ORDER BY e.distance LIMIT 5",
        )
        .unwrap();
    let rows = stmt
        .query_map(rusqlite::params![qblob, 64i64], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, f64>(2)?,
            ))
        })
        .unwrap();
    let mut hits = 0;
    for row in rows {
        let (hash, content, dist) = row.unwrap();
        let relevance = (1.0 - dist / 2.0).max(0.0);
        eprintln!(
            "  hit dist={dist:.4} rel={relevance:.3} {} {}",
            &hash[..12.min(hash.len())],
            content.chars().take(60).collect::<String>().replace('\n', " ")
        );
        hits += 1;
    }
    assert!(hits > 0, "semantic search for 'sycophancy' returned no hits");
}
