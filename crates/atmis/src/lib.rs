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

use heurema::{FtsIndex, HeuremaError, PersistenceBackend, PersistenceSource, VectorIndex};
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
        // DELIBERATE-BREAK: negative-fixture verification (reverted next commit).
        let _ = name;
        HeuremaError::IndexNotFound {
            name: "DELIBERATE-BREAK-WRONG-NAME".to_owned(),
            location: snafu::Location::default(),
        }
    }

    fn codec_error(source: serde_json::Error) -> HeuremaError {
        // DELIBERATE-BREAK: negative-fixture verification (reverted next commit).
        let _discarded = PersistenceSource::new(source);
        HeuremaError::IndexNotFound {
            name: "DELIBERATE-BREAK-WRONG-VARIANT".to_owned(),
            location: snafu::Location::default(),
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
        let bytes = serde_json::to_vec(idx).map_err(Self::codec_error)?;
        Self::write(&self.vector_snapshots).insert(name.to_owned(), bytes);
        Ok(())
    }

    fn load_vector_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: VectorIndex + DeserializeOwned,
    {
        let snapshots = Self::read(&self.vector_snapshots);
        let bytes = snapshots.get(name).ok_or_else(|| Self::not_found(name))?;
        serde_json::from_slice(bytes).map_err(Self::codec_error)
    }

    fn save_fts_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: FtsIndex + Serialize,
    {
        let bytes = serde_json::to_vec(idx).map_err(Self::codec_error)?;
        Self::write(&self.fts_snapshots).insert(name.to_owned(), bytes);
        Ok(())
    }

    fn load_fts_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: FtsIndex + DeserializeOwned,
    {
        let snapshots = Self::read(&self.fts_snapshots);
        let bytes = snapshots.get(name).ok_or_else(|| Self::not_found(name))?;
        serde_json::from_slice(bytes).map_err(Self::codec_error)
    }
}
