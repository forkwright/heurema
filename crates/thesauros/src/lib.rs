//! `thesauros` (θησαυρός, storehouse, treasury) is an fjall-backed durable
//! [`PersistenceBackend`] adapter for heurema indexes.
//!
//! Every save encodes the index through `serde_json`, writes it to a fjall
//! partition, then fsyncs the keyspace journal
//! (`fjall::PersistMode::SyncAll`) before returning `Ok`. A save that
//! returns `Ok` is durable on disk at that point, not merely buffered. That
//! is this crate's entire reason to exist over `atmis`, and the property
//! its test suite proves by closing and reopening the keyspace rather than
//! trusting an in-process value.

#![deny(missing_docs)]

use std::path::Path;

use fjall::PartitionCreateOptions;
use heurema::{FtsIndex, HeuremaError, PersistenceBackend, PersistenceSource, VectorIndex};
use serde::Serialize;
use serde::de::DeserializeOwned;

const VECTOR_PARTITION: &str = "vector_indexes";
const FTS_PARTITION: &str = "fts_indexes";

/// fjall-backed [`PersistenceBackend`]. Vector-index and FTS-index
/// snapshots live in separate fjall partitions so the two name spaces the
/// trait itself keeps independent stay independent on disk.
pub struct ThesaurosBackend {
    keyspace: fjall::Keyspace,
    vector_indexes: fjall::PartitionHandle,
    fts_indexes: fjall::PartitionHandle,
}

impl ThesaurosBackend {
    /// Open (or create) a keyspace at `path`.
    ///
    /// WARNING: fjall keyspaces are single-process — opening the same
    /// `path` from two live backends at once is a fleet-wide known hazard
    /// (see kanon's `archeion` crate CLAUDE.md), not specific to this
    /// adapter. The caller owns process-level exclusivity over `path`.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::Persistence`] if the keyspace or either
    /// partition fails to open.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, HeuremaError> {
        let keyspace = fjall::Config::new(path).open().map_err(Self::fjall_error)?;
        let vector_indexes = keyspace
            .open_partition(VECTOR_PARTITION, PartitionCreateOptions::default())
            .map_err(Self::fjall_error)?;
        let fts_indexes = keyspace
            .open_partition(FTS_PARTITION, PartitionCreateOptions::default())
            .map_err(Self::fjall_error)?;
        Ok(Self {
            keyspace,
            vector_indexes,
            fts_indexes,
        })
    }

    fn not_found(name: &str) -> HeuremaError {
        HeuremaError::IndexNotFound {
            name: name.to_owned(),
            location: snafu::Location::default(),
        }
    }

    fn fjall_error(source: fjall::Error) -> HeuremaError {
        // DELIBERATE-BREAK: negative-fixture verification (reverted next commit).
        let _discarded = PersistenceSource::new(source);
        HeuremaError::IndexNotFound {
            name: "DELIBERATE-BREAK-WRONG-VARIANT".to_owned(),
            location: snafu::Location::default(),
        }
    }

    fn codec_error(source: serde_json::Error) -> HeuremaError {
        HeuremaError::Persistence {
            source: PersistenceSource::new(source),
            location: snafu::Location::default(),
        }
    }

    // WHY: fsyncing the journal on every write is what makes a returned
    // `Ok` mean "durable", not merely "buffered" — the property that
    // separates this adapter from atmis. A backend intended for
    // high-throughput bulk loading could relax this to a caller-chosen
    // `PersistMode` later; Phase 3 keeps it unconditional because no
    // caller has asked for the weaker mode yet.
    fn sync(&self) -> Result<(), HeuremaError> {
        self.keyspace
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
