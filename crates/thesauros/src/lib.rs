//! `thesauros` (θησαυρός, storehouse, treasury) is an fjall-backed durable
//! [`PersistenceBackend`] adapter for heurema indexes.
//!
//! Every save encodes the index through `serde_json`, writes it to a fjall
//! keyspace, then fsyncs the database journal
//! (`fjall::PersistMode::SyncAll`) before returning `Ok`. A save that
//! returns `Ok` is durable on disk at that point, not merely buffered. That
//! is this crate's entire reason to exist over `atmis`, and the property
//! its test suite proves by closing and reopening the database rather than
//! trusting an in-process value.

#![deny(missing_docs)]

use std::path::Path;

use fjall::KeyspaceCreateOptions;
use heurema::{FtsIndex, HeuremaError, PersistenceBackend, PersistenceSource, VectorIndex};
use serde::Serialize;
use serde::de::DeserializeOwned;

const VECTOR_PARTITION: &str = "vector_indexes";
const FTS_PARTITION: &str = "fts_indexes";

/// fjall-backed [`PersistenceBackend`]. Vector-index and FTS-index
/// snapshots live in separate fjall keyspaces so the two name spaces the
/// trait itself keeps independent stay independent on disk.
pub struct ThesaurosBackend {
    db: fjall::Database,
    vector_indexes: fjall::Keyspace,
    fts_indexes: fjall::Keyspace,
}

impl ThesaurosBackend {
    /// Open (or create) a database at `path`.
    ///
    /// WARNING: fjall databases are single-process — opening the same
    /// `path` from two live backends at once is a fleet-wide known hazard
    /// (see kanon's `archeion` crate CLAUDE.md), not specific to this
    /// adapter. The caller owns process-level exclusivity over `path`.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::Persistence`] if the database or either
    /// keyspace fails to open.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, HeuremaError> {
        let db = fjall::Database::builder(path)
            .open()
            .map_err(Self::fjall_error)?;
        let vector_indexes = db
            .keyspace(VECTOR_PARTITION, KeyspaceCreateOptions::default)
            .map_err(Self::fjall_error)?;
        let fts_indexes = db
            .keyspace(FTS_PARTITION, KeyspaceCreateOptions::default)
            .map_err(Self::fjall_error)?;
        Ok(Self {
            db,
            vector_indexes,
            fts_indexes,
        })
    }

    fn not_found(name: &str) -> HeuremaError {
        HeuremaError::IndexNotFound {
            name: name.to_owned(),
            location: std::panic::Location::caller(),
        }
    }

    fn fjall_error(source: fjall::Error) -> HeuremaError {
        HeuremaError::Persistence {
            source: PersistenceSource::new(source),
            location: std::panic::Location::caller(),
        }
    }

    fn codec_error(source: serde_json::Error) -> HeuremaError {
        HeuremaError::Persistence {
            source: PersistenceSource::new(source),
            location: std::panic::Location::caller(),
        }
    }

    // WHY: fsyncing the journal on every write is what makes a returned
    // `Ok` mean "durable", not merely "buffered" — the property that
    // separates this adapter from atmis. A backend intended for
    // high-throughput bulk loading could relax this to a caller-chosen
    // `PersistMode` later; Phase 3 keeps it unconditional because no
    // caller has asked for the weaker mode yet.
    fn sync(&self) -> Result<(), HeuremaError> {
        self.db
            .persist(fjall::PersistMode::SyncAll)
            .map_err(Self::fjall_error)
    }
}

impl PersistenceBackend for ThesaurosBackend {
    fn save_vector_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: VectorIndex + Serialize,
    {
        let bytes = serde_json::to_vec(idx).map_err(Self::codec_error)?;
        self.vector_indexes
            .insert(name, bytes)
            .map_err(Self::fjall_error)?;
        self.sync()
    }

    fn load_vector_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: VectorIndex + DeserializeOwned,
    {
        let bytes = self
            .vector_indexes
            .get(name)
            .map_err(Self::fjall_error)?
            .ok_or_else(|| Self::not_found(name))?;
        serde_json::from_slice(&bytes).map_err(Self::codec_error)
    }

    fn save_fts_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: FtsIndex + Serialize,
    {
        let bytes = serde_json::to_vec(idx).map_err(Self::codec_error)?;
        self.fts_indexes
            .insert(name, bytes)
            .map_err(Self::fjall_error)?;
        self.sync()
    }

    fn load_fts_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: FtsIndex + DeserializeOwned,
    {
        let bytes = self
            .fts_indexes
            .get(name)
            .map_err(Self::fjall_error)?
            .ok_or_else(|| Self::not_found(name))?;
        serde_json::from_slice(&bytes).map_err(Self::codec_error)
    }
}
