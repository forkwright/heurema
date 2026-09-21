//! `atmis` (ἀτμίς, vapor) is an in-memory [`PersistenceBackend`] adapter for
//! heurema indexes.
//!
//! Snapshots live in a `HashMap<String, Vec<u8>>` behind an [`RwLock`],
//! encoded through `serde_json` so save and load exercise the same
//! byte-level contract a durable backend depends on. A backend that instead
//! cloned the live `I` value directly would pass a save/load round trip
//! trivially without proving the encode/decode path. See `thesauros` for
//! the durable adapter this one is meant to stand in for during tests.
//!
//! Snapshots never reach disk and do not survive the process, the way vapor
//! leaves no residue once its condition (heat, or here, a live process)
//! ends. Nothing here reopens a store across a process boundary; that proof
//! belongs to `thesauros`'s test suite.

#![deny(missing_docs)]

use std::collections::HashMap;
use std::sync::{PoisonError, RwLock};

use heurema::{
    FtsIndex, HeuremaError, PersistenceBackend, PersistenceSource, SnapshotEnvelope,
    SnapshotFamily, VectorIndex, decode_snapshot_payload,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// WHY: vector-index and FTS-index snapshots live in separate maps because
/// [`PersistenceBackend::save_vector_index`] and
/// [`PersistenceBackend::save_fts_index`] are independent name spaces —
/// nothing in the trait contract says the two families share names.
#[derive(Debug, Default)]
pub struct AtmisBackend {
    vector_snapshots: RwLock<HashMap<String, Vec<u8>>>,
    fts_snapshots: RwLock<HashMap<String, Vec<u8>>>,
}

impl AtmisBackend {
    /// Construct an empty backend holding no saved snapshots.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn not_found(name: &str) -> HeuremaError {
        HeuremaError::IndexNotFound {
            name: name.to_owned(),
            location: std::panic::Location::caller(),
        }
    }

    fn codec_error(source: serde_json::Error) -> HeuremaError {
        HeuremaError::Persistence {
            source: PersistenceSource::new(source),
            location: std::panic::Location::caller(),
        }
    }

    // WHY: a poisoned writer panicked mid-`HashMap::insert`, which either
    // completed or did not; the map carries no partial-write state a reader
    // needs to distrust, so recovering the guard is safe for a backend with
    // no external durability contract to uphold.
    fn read(
        lock: &RwLock<HashMap<String, Vec<u8>>>,
    ) -> std::sync::RwLockReadGuard<'_, HashMap<String, Vec<u8>>> {
        lock.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(
        lock: &RwLock<HashMap<String, Vec<u8>>>,
    ) -> std::sync::RwLockWriteGuard<'_, HashMap<String, Vec<u8>>> {
        lock.write().unwrap_or_else(PoisonError::into_inner)
    }
}

impl PersistenceBackend for AtmisBackend {
    fn save_vector_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: VectorIndex + Serialize,
    {
        let bytes = serde_json::to_vec(&SnapshotEnvelope::new(SnapshotFamily::Vector, idx))
            .map_err(Self::codec_error)?;
        Self::write(&self.vector_snapshots).insert(name.to_owned(), bytes);
        Ok(())
    }

    fn load_vector_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: VectorIndex + DeserializeOwned,
    {
        let snapshots = Self::read(&self.vector_snapshots);
        let bytes = snapshots.get(name).ok_or_else(|| Self::not_found(name))?;
        decode_snapshot_payload(bytes, SnapshotFamily::Vector)
    }

    fn save_fts_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: FtsIndex + Serialize,
    {
        let bytes = serde_json::to_vec(&SnapshotEnvelope::new(SnapshotFamily::Fts, idx))
            .map_err(Self::codec_error)?;
        Self::write(&self.fts_snapshots).insert(name.to_owned(), bytes);
        Ok(())
    }

    fn load_fts_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: FtsIndex + DeserializeOwned,
    {
        let snapshots = Self::read(&self.fts_snapshots);
        let bytes = snapshots.get(name).ok_or_else(|| Self::not_found(name))?;
        decode_snapshot_payload(bytes, SnapshotFamily::Fts)
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests need concise private-store setup")]
mod tests {
    use super::*;
    use heurema::{HnswConfig, HnswIndex};

    fn inject_vector_snapshot(backend: &AtmisBackend, name: &str, bytes: &[u8]) {
        AtmisBackend::write(&backend.vector_snapshots).insert(name.to_owned(), bytes.to_vec());
    }

    #[test]
    fn format_header_refuses_alien_payload_before_vector_decode() {
        let backend = AtmisBackend::new();
        inject_vector_snapshot(
            &backend,
            "future",
            br#"{"format_version":2,"family":"Vector","payload":{"alien":true}}"#,
        );
        inject_vector_snapshot(
            &backend,
            "wrong-family",
            br#"{"format_version":1,"family":"Fts","payload":{"alien":true}}"#,
        );

        for name in ["future", "wrong-family"] {
            assert!(matches!(
                backend.load_vector_index::<HnswIndex<u64>>(name),
                Err(HeuremaError::SnapshotFormat { .. })
            ));
        }
    }

    #[test]
    fn current_header_with_an_invalid_payload_remains_a_decode_error() {
        let backend = AtmisBackend::new();
        inject_vector_snapshot(
            &backend,
            "invalid-current",
            br#"{"format_version":1,"family":"Vector","payload":{"alien":true}}"#,
        );
        assert!(matches!(
            backend.load_vector_index::<HnswIndex<u64>>("invalid-current"),
            Err(HeuremaError::Persistence { .. })
        ));

        let valid = HnswIndex::<u64>::new(HnswConfig::new(2));
        backend
            .save_vector_index("valid", &valid)
            .expect("valid current envelope saves");
        assert!(backend.load_vector_index::<HnswIndex<u64>>("valid").is_ok());
    }
}
