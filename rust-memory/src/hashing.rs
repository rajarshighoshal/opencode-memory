//! content_hash contract — MUST match the Python impl exactly for dedup and
//! `memory_graph` source/target hash compatibility.
//!
//! Python (`utils/hashing.py`):
//! `content_hash = sha256(content.strip().lower().encode("utf-8")).hexdigest()`
//! — lowercase hex, 64 chars, identity is content-only (tags/type/metadata do
//! NOT affect the hash).
//!
//! Rust equivalent: `hex::encode(Sha256::digest(content.trim().to_lowercase().as_bytes()))`.
//! Note: Python `str.lower()` and Rust `to_lowercase()` are both Unicode-aware
//! and agree for ASCII/typical memory text; they can differ on exotic
//! locale-specific characters (e.g. Turkish dotless i) — acceptable for this
//! single-user English workload.

use sha2::{Digest, Sha256};

/// Compute the content hash for dedup. 64-char lowercase hex.
pub fn content_hash(content: &str) -> String {
    let normalized = content.trim().to_lowercase();
    hex::encode(Sha256::digest(normalized.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_lowercase_hex_64() {
        let h = content_hash("  Hello World  ");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        // trim + lowercase => same hash as the normalized form.
        assert_eq!(h, content_hash("hello world"));
    }

    /// Parity with the Python `generate_content_hash`:
    /// `sha256(content.strip().lower().encode("utf-8")).hexdigest()`.
    /// Reference values produced by running the live Python helper.
    #[test]
    fn hash_matches_python_reference() {
        assert_eq!(
            content_hash("  Hello World  "),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
        assert_eq!(
            content_hash("sycophancy"),
            "c9a4e9211db03b3d10c6750b9cbf7dbd292b48005d283f3cd8ea0ee29503ef97"
        );
        assert_eq!(
            content_hash("The Quick Brown Fox"),
            "9ecb36561341d18eb65484e833efea61edc74b84cf5e6ae1b81c63533e25fc8f"
        );
        assert_eq!(
            content_hash(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
