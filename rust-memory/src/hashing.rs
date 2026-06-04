//! content_hash contract — the stable identity used for dedup and for
//! `memory_graph` source/target hash matching. Stored hashes must stay
//! consistent across versions, so the normalization here is fixed:
//!
//! `content_hash = sha256(content.trim().to_lowercase())` as 64-char lowercase
//! hex. Identity is content-only — tags, type, and metadata do NOT affect the
//! hash.
//!
//! Note: `to_lowercase()` is Unicode-aware. It agrees with itself for ASCII and
//! typical memory text but can fold exotic locale-specific characters (e.g.
//! Turkish dotless i) differently than a naive ASCII lowercase — acceptable for
//! this single-user English workload.

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

    /// Locks in the exact hash values for known inputs. These are stored as
    /// memory identities, so any change to the normalization or digest would
    /// break dedup and graph hash matching against existing data.
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
