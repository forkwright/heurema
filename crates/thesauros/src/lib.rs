//! `thesauros` (θησαυρός, storehouse, treasury) is an fjall-backed durable
//! [`PersistenceBackend`] and [`LifecycleBackend`] adapter for heurema
//! indexes.
//!
//! Every save encodes the index through `serde_json`, writes it to a fjall
//! keyspace, then fsyncs the database journal
//! (`fjall::PersistMode::SyncAll`) before returning `Ok`. A save that
//! returns `Ok` is durable on disk at that point, not merely buffered. That
//! is this crate's entire reason to exist over `atmis`, and the property
//! its test suite proves by closing and reopening the database rather than
//! trusting an in-process value.
//!
//! # Lifecycle storage
//!
//! Lifecycle state lives in five fixed keyspaces, created by
//! [`ThesaurosBackend::open`]: `lifecycle_heads`, `lifecycle_versions`,
//! `lifecycle_staging`, `lifecycle_operations`, and `lifecycle_quarantine`,
//! keyed by [`storage_key`] strings. No keyspace is created per index.
//!
//! Every lifecycle write is one fjall write batch committed with
//! `PersistMode::SyncAll`. fjall writes the batch to its journal as one
//! checksummed unit, fsyncs it, and only then applies it in memory; on
//! reopen it replays complete batches and discards an incomplete trailing
//! one. A write is therefore all or nothing across a crash, and durable
//! before `Ok`. Reads go through an fjall snapshot, which sees only fully
//! applied batches, so an in-process reader also sees each write entirely or
//! not at all.
//!
//! The backend holds two locks:
//!
//! - The commit lock covers one write's checks through its commit. fjall
//!   batches have no conditional write, so each lifecycle write takes it,
//!   checks the stored state (the head, any staging marker, any payload, any
//!   operation record), and commits its batch while still holding it. That
//!   makes the head comparison a compare-and-set within the process; fjall's
//!   single-process rule covers the rest.
//! - The shared [`WriterLock`] ([`LifecycleBackend::writer`]) covers a whole
//!   lifecycle operation, from its head read to its publish, across every
//!   lifecycle over this backend.
//!
//! WARNING: a failed commit poisons the fjall database (fjall's journal
//! writer marks it poisoned on any journal write or fsync error), and every
//! later write returns [`HeuremaError::Persistence`]. The caller must drop
//! this backend and [`open`](ThesaurosBackend::open) the path again; the
//! failed batch is then either fully replayed or absent.

#![deny(missing_docs)]

use std::fmt;
use std::panic::Location;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use fjall::{KeyspaceCreateOptions, Readable, Snapshot};
use heurema::lifecycle::storage_key::{self, QuarantinePart};
use heurema::{
    DestroyWrite, FtsIndex, HeuremaError, IndexIdentity, IndexVersion, LifecycleBackend,
    OperationKey, PersistenceBackend, PersistenceSource, PublishWrite, QuarantineWrite,
    QuarantinedEntry, SnapshotEnvelope, SnapshotFamily, StageWrite, StagedEntry, VectorIndex,
    WriterLock, decode_snapshot_payload,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

const VECTOR_PARTITION: &str = "vector_indexes";
const FTS_PARTITION: &str = "fts_indexes";
const LIFECYCLE_HEADS: &str = "lifecycle_heads";
const LIFECYCLE_VERSIONS: &str = "lifecycle_versions";
const LIFECYCLE_STAGING: &str = "lifecycle_staging";
const LIFECYCLE_OPERATIONS: &str = "lifecycle_operations";
const LIFECYCLE_QUARANTINE: &str = "lifecycle_quarantine";

/// fjall-backed [`PersistenceBackend`] and [`LifecycleBackend`].
/// Vector-index and FTS-index snapshots live in separate fjall keyspaces so
/// the two name spaces the trait itself keeps independent stay independent
/// on disk, and lifecycle state lives in its own five keyspaces beside them.
pub struct ThesaurosBackend {
    db: fjall::Database,
    vector_indexes: fjall::Keyspace,
    fts_indexes: fjall::Keyspace,
    lifecycle: LifecycleKeyspaces,
    /// WHY: fjall write batches have no conditional write, so a lifecycle
    /// write's checks and its batch could otherwise interleave with another
    /// writer's. Every lifecycle write holds this lock from its first check
    /// through its commit. It guards no data.
    commit_lock: Mutex<()>,
    /// The writer every lifecycle over this backend shares; see
    /// [`LifecycleBackend::writer`].
    writer: WriterLock,
}

/// The five lifecycle keyspaces.
///
/// WHY separate keyspaces: each kind of [`storage_key`] lives in its own, so
/// a digits-only operation key, which spells the same text as a version
/// key, can never overwrite a version payload.
struct LifecycleKeyspaces {
    heads: fjall::Keyspace,
    versions: fjall::Keyspace,
    staging: fjall::Keyspace,
    operations: fjall::Keyspace,
    quarantine: fjall::Keyspace,
}

/// A thread panicked while holding the lifecycle commit lock.
#[derive(Debug)]
struct CommitLockPoisoned;

/// The quarantine counter reached `u64::MAX`.
#[derive(Debug)]
struct QuarantineSequenceExhausted;

/// A quarantine keyspace entry that does not pair with its marker.
#[derive(Debug)]
struct UnpairedQuarantineEntry {
    sequence: u64,
    reason: &'static str,
}

/// Carry an fjall failure into [`HeuremaError::Persistence`] at the
/// caller's location.
trait OrPersistence<T> {
    fn or_persistence(self) -> Result<T, HeuremaError>;
}

impl ThesaurosBackend {
    /// Open (or create) a database at `path`, with its snapshot and
    /// lifecycle keyspaces.
    ///
    /// WARNING: fjall databases are single-process — opening the same
    /// `path` from two live backends at once is a fleet-wide known hazard
    /// (see kanon's `archeion` crate CLAUDE.md), not specific to this
    /// adapter. The caller owns process-level exclusivity over `path`. The
    /// shared lifecycle writer ([`LifecycleBackend::writer`]) covers only
    /// lifecycles over one backend instance.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::Persistence`] if the database or any keyspace
    /// fails to open.
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
        let lifecycle = LifecycleKeyspaces {
            heads: Self::lifecycle_keyspace(&db, LIFECYCLE_HEADS)?,
            versions: Self::lifecycle_keyspace(&db, LIFECYCLE_VERSIONS)?,
            staging: Self::lifecycle_keyspace(&db, LIFECYCLE_STAGING)?,
            operations: Self::lifecycle_keyspace(&db, LIFECYCLE_OPERATIONS)?,
            quarantine: Self::lifecycle_keyspace(&db, LIFECYCLE_QUARANTINE)?,
        };
        Ok(Self {
            db,
            vector_indexes,
            fts_indexes,
            lifecycle,
            commit_lock: Mutex::new(()),
            writer: WriterLock::new(),
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
    // `PersistMode` later; the adapter keeps it unconditional because no
    // caller has asked for the weaker mode yet.
    fn sync(&self) -> Result<(), HeuremaError> {
        self.db
            .persist(fjall::PersistMode::SyncAll)
            .map_err(Self::fjall_error)
    }

    #[track_caller]
    fn lifecycle_keyspace(
        db: &fjall::Database,
        name: &str,
    ) -> Result<fjall::Keyspace, HeuremaError> {
        db.keyspace(name, KeyspaceCreateOptions::default)
            .or_persistence()
    }

    /// The lifecycle commit lock, held from a write's first check through
    /// its commit.
    ///
    /// WHY a poisoned lock is refused: a panic between a write's checks and
    /// its commit leaves no half-written batch (fjall commits whole batches),
    /// but it does mean an unexpected failure inside the storage path. The
    /// backend refuses further lifecycle writes rather than guess; reopening
    /// it clears the lock. Reads take no lock and keep working.
    #[track_caller]
    fn commit_lock(&self) -> Result<MutexGuard<'_, ()>, HeuremaError> {
        match self.commit_lock.lock() {
            Ok(guard) => Ok(guard),
            Err(_poisoned) => Err(persistence(CommitLockPoisoned)),
        }
    }

    /// A write batch that is fsynced before `commit` returns.
    ///
    /// WHY explicit: `Database::batch` defaults to `PersistMode::Buffer`,
    /// which returns before the journal reaches the disk. Every lifecycle
    /// write goes through this one constructor, so none can commit with the
    /// weaker mode.
    fn lifecycle_batch(&self) -> fjall::OwnedWriteBatch {
        self.db
            .batch()
            .durability(Some(fjall::PersistMode::SyncAll))
    }
}

impl LifecycleKeyspaces {
    #[track_caller]
    fn get(
        snapshot: &Snapshot,
        keyspace: &fjall::Keyspace,
        key: &str,
    ) -> Result<Option<Vec<u8>>, HeuremaError> {
        Ok(snapshot
            .get(keyspace, key)
            .or_persistence()?
            .map(|value| value.to_vec()))
    }

    #[track_caller]
    fn contains(
        snapshot: &Snapshot,
        keyspace: &fjall::Keyspace,
        key: &str,
    ) -> Result<bool, HeuremaError> {
        snapshot.contains_key(keyspace, key).or_persistence()
    }

    /// The first staging marker among the keys that start with `prefix`.
    #[track_caller]
    fn first_staged(
        &self,
        snapshot: &Snapshot,
        prefix: &str,
    ) -> Result<Option<(IndexVersion, Vec<u8>)>, HeuremaError> {
        let Some(guard) = snapshot.prefix(&self.staging, prefix).next() else {
            return Ok(None);
        };
        let (key, marker) = guard.into_inner().or_persistence()?;
        let (_, version) = storage_key::parse_staging(&key)?;
        Ok(Some((version, marker.to_vec())))
    }

    #[track_caller]
    fn check_head(
        &self,
        snapshot: &Snapshot,
        index: &IndexIdentity,
        head_key: &str,
        expected: Option<&[u8]>,
    ) -> Result<(), HeuremaError> {
        let stored = snapshot.get(&self.heads, head_key).or_persistence()?;
        if stored.as_deref() == expected {
            Ok(())
        } else {
            Err(HeuremaError::HeadChanged {
                index: index.clone(),
                location: Location::caller(),
            })
        }
    }

    #[track_caller]
    fn refuse_staged(
        &self,
        snapshot: &Snapshot,
        index: &IndexIdentity,
        prefix: &str,
    ) -> Result<(), HeuremaError> {
        match self.first_staged(snapshot, prefix)? {
            None => Ok(()),
            Some((version, _)) => Err(staged_state_exists(index, version)),
        }
    }

    #[track_caller]
    fn refuse_stored_version(
        &self,
        snapshot: &Snapshot,
        index: &IndexIdentity,
        version: IndexVersion,
        version_key: &str,
    ) -> Result<(), HeuremaError> {
        if Self::contains(snapshot, &self.versions, version_key)? {
            Err(version_stored(index, version))
        } else {
            Ok(())
        }
    }

    #[track_caller]
    fn check_marker(
        &self,
        snapshot: &Snapshot,
        index: &IndexIdentity,
        version: IndexVersion,
        staging_key: &str,
        marker: &[u8],
    ) -> Result<(), HeuremaError> {
        let stored = snapshot.get(&self.staging, staging_key).or_persistence()?;
        if stored.as_deref() == Some(marker) {
            Ok(())
        } else {
            Err(staged_state_missing(index, version))
        }
    }

    #[track_caller]
    fn check_payload(
        &self,
        snapshot: &Snapshot,
        index: &IndexIdentity,
        version: IndexVersion,
        version_key: &str,
    ) -> Result<(), HeuremaError> {
        if Self::contains(snapshot, &self.versions, version_key)? {
            Ok(())
        } else {
            Err(staged_state_missing(index, version))
        }
    }

    #[track_caller]
    fn refuse_recorded(
        &self,
        snapshot: &Snapshot,
        index: &IndexIdentity,
        key: &OperationKey,
        operation_key: &str,
    ) -> Result<(), HeuremaError> {
        if Self::contains(snapshot, &self.operations, operation_key)? {
            Err(HeuremaError::OperationRecorded {
                index: index.clone(),
                key: key.clone(),
                location: Location::caller(),
            })
        } else {
            Ok(())
        }
    }

    #[track_caller]
    fn next_sequence(&self, snapshot: &Snapshot) -> Result<u64, HeuremaError> {
        let Some(guard) = snapshot.last_key_value(&self.quarantine) else {
            return Ok(QuarantinedEntry::FIRST_SEQUENCE);
        };
        let key = guard.key().or_persistence()?;
        let (last, ..) = storage_key::parse_quarantine(&key)?;
        match last.checked_add(1) {
            Some(next) => Ok(next),
            None => Err(persistence(QuarantineSequenceExhausted)),
        }
    }
}

impl PersistenceBackend for ThesaurosBackend {
    fn save_vector_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: VectorIndex + Serialize,
    {
        let bytes = serde_json::to_vec(&SnapshotEnvelope::new(SnapshotFamily::Vector, idx))
            .map_err(Self::codec_error)?;
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
        decode_snapshot_payload(&bytes, SnapshotFamily::Vector)
    }

    fn save_fts_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: FtsIndex + Serialize,
    {
        let bytes = serde_json::to_vec(&SnapshotEnvelope::new(SnapshotFamily::Fts, idx))
            .map_err(Self::codec_error)?;
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
        decode_snapshot_payload(&bytes, SnapshotFamily::Fts)
    }
}

impl LifecycleBackend for ThesaurosBackend {
    fn writer(&self) -> &WriterLock {
        &self.writer
    }

    fn read_head(&self, index: &IndexIdentity) -> Result<Option<Vec<u8>>, HeuremaError> {
        let key = storage_key::head(index);
        LifecycleKeyspaces::get(&self.db.snapshot(), &self.lifecycle.heads, &key)
    }

    fn read_version(
        &self,
        index: &IndexIdentity,
        version: IndexVersion,
    ) -> Result<Option<Vec<u8>>, HeuremaError> {
        let key = storage_key::version(index, version);
        LifecycleKeyspaces::get(&self.db.snapshot(), &self.lifecycle.versions, &key)
    }

    fn read_operation(
        &self,
        index: &IndexIdentity,
        key: &OperationKey,
    ) -> Result<Option<Vec<u8>>, HeuremaError> {
        let key = storage_key::operation(index, key);
        LifecycleKeyspaces::get(&self.db.snapshot(), &self.lifecycle.operations, &key)
    }

    fn read_staging(
        &self,
        index: &IndexIdentity,
    ) -> Result<Option<(IndexVersion, Vec<u8>)>, HeuremaError> {
        let prefix = storage_key::index_prefix(index);
        self.lifecycle.first_staged(&self.db.snapshot(), &prefix)
    }

    fn stage(&self, write: StageWrite<'_>) -> Result<(), HeuremaError> {
        let head_key = storage_key::head(write.index);
        let prefix = storage_key::index_prefix(write.index);
        let version_key = storage_key::version(write.index, write.version);
        let staging_key = storage_key::staging(write.index, write.version);
        let operation_key = storage_key::operation(write.index, write.key);

        let _commit = self.commit_lock()?;
        let snapshot = self.db.snapshot();
        let lifecycle = &self.lifecycle;
        lifecycle.check_head(&snapshot, write.index, &head_key, write.expected_head)?;
        lifecycle.refuse_staged(&snapshot, write.index, &prefix)?;
        lifecycle.refuse_stored_version(&snapshot, write.index, write.version, &version_key)?;
        lifecycle.refuse_recorded(&snapshot, write.index, write.key, &operation_key)?;
        let mut batch = self.lifecycle_batch();
        batch.insert(&lifecycle.versions, version_key, write.payload);
        batch.insert(&lifecycle.staging, staging_key, write.marker);
        batch.commit().or_persistence()
    }

    fn publish(&self, write: PublishWrite<'_>) -> Result<(), HeuremaError> {
        let head_key = storage_key::head(write.index);
        let version_key = storage_key::version(write.index, write.version);
        let staging_key = storage_key::staging(write.index, write.version);
        let operation_key = storage_key::operation(write.index, write.key);

        let _commit = self.commit_lock()?;
        let snapshot = self.db.snapshot();
        let lifecycle = &self.lifecycle;
        lifecycle.check_head(&snapshot, write.index, &head_key, write.expected_head)?;
        lifecycle.check_marker(
            &snapshot,
            write.index,
            write.version,
            &staging_key,
            write.marker,
        )?;
        lifecycle.check_payload(&snapshot, write.index, write.version, &version_key)?;
        lifecycle.refuse_recorded(&snapshot, write.index, write.key, &operation_key)?;
        let mut batch = self.lifecycle_batch();
        batch.insert(&lifecycle.heads, head_key, write.head);
        batch.insert(&lifecycle.operations, operation_key, write.operation);
        batch.remove(&lifecycle.staging, staging_key);
        batch.commit().or_persistence()
    }

    fn destroy(&self, write: DestroyWrite<'_>) -> Result<(), HeuremaError> {
        let head_key = storage_key::head(write.index);
        let prefix = storage_key::index_prefix(write.index);
        let operation_key = storage_key::operation(write.index, write.key);

        let _commit = self.commit_lock()?;
        let snapshot = self.db.snapshot();
        let lifecycle = &self.lifecycle;
        lifecycle.check_head(&snapshot, write.index, &head_key, Some(write.expected_head))?;
        lifecycle.refuse_staged(&snapshot, write.index, &prefix)?;
        lifecycle.refuse_recorded(&snapshot, write.index, write.key, &operation_key)?;
        let mut batch = self.lifecycle_batch();
        batch.insert(&lifecycle.heads, head_key, write.head);
        batch.insert(&lifecycle.operations, operation_key, write.operation);
        for version in write.versions {
            batch.remove(
                &lifecycle.versions,
                storage_key::version(write.index, *version),
            );
        }
        batch.commit().or_persistence()
    }

    fn list_heads(&self) -> Result<Vec<(IndexIdentity, Vec<u8>)>, HeuremaError> {
        let snapshot = self.db.snapshot();
        let mut heads = Vec::new();
        for guard in snapshot.iter(&self.lifecycle.heads) {
            let (key, head) = guard.into_inner().or_persistence()?;
            heads.push((storage_key::parse_head(&key)?, head.to_vec()));
        }
        Ok(heads)
    }

    fn list_staged(&self) -> Result<Vec<StagedEntry>, HeuremaError> {
        let snapshot = self.db.snapshot();
        let mut staged = Vec::new();
        for guard in snapshot.iter(&self.lifecycle.staging) {
            let (key, marker) = guard.into_inner().or_persistence()?;
            let (index, version) = storage_key::parse_staging(&key)?;
            staged.push(StagedEntry::new(index, version, marker.to_vec()));
        }
        Ok(staged)
    }

    fn list_operations(
        &self,
        index: &IndexIdentity,
    ) -> Result<Vec<(OperationKey, Vec<u8>)>, HeuremaError> {
        let prefix = storage_key::index_prefix(index);
        let snapshot = self.db.snapshot();
        let mut operations = Vec::new();
        for guard in snapshot.prefix(&self.lifecycle.operations, &prefix) {
            let (key, operation) = guard.into_inner().or_persistence()?;
            let (_, key) = storage_key::parse_operation(&key)?;
            operations.push((key, operation.to_vec()));
        }
        Ok(operations)
    }

    fn quarantine(&self, write: QuarantineWrite<'_>) -> Result<(), HeuremaError> {
        let version_key = storage_key::version(write.index, write.version);
        let staging_key = storage_key::staging(write.index, write.version);

        let _commit = self.commit_lock()?;
        let snapshot = self.db.snapshot();
        let lifecycle = &self.lifecycle;
        lifecycle.check_marker(
            &snapshot,
            write.index,
            write.version,
            &staging_key,
            write.marker,
        )?;
        let payload = snapshot
            .get(&lifecycle.versions, &version_key)
            .or_persistence()?;
        let sequence = lifecycle.next_sequence(&snapshot)?;
        let mut batch = self.lifecycle_batch();
        // INVARIANT: `check_marker` proved the stored marker equals
        // `write.marker`, so storing these bytes moves it byte for byte.
        batch.insert(
            &lifecycle.quarantine,
            storage_key::quarantine(sequence, write.index, write.version, QuarantinePart::Marker),
            write.marker,
        );
        if let Some(payload) = payload {
            batch.insert(
                &lifecycle.quarantine,
                storage_key::quarantine(
                    sequence,
                    write.index,
                    write.version,
                    QuarantinePart::Payload,
                ),
                payload,
            );
        }
        batch.remove(&lifecycle.staging, staging_key);
        batch.remove(&lifecycle.versions, version_key);
        batch.commit().or_persistence()
    }

    fn list_quarantined(&self) -> Result<Vec<QuarantinedEntry>, HeuremaError> {
        let snapshot = self.db.snapshot();
        let mut entries: Vec<QuarantinedEntry> = Vec::new();
        // INVARIANT: keys sort by sequence, and within one sequence
        // `marker` sorts before `payload`, so a payload always follows the
        // marker it belongs to.
        for guard in snapshot.iter(&self.lifecycle.quarantine) {
            let (key, value) = guard.into_inner().or_persistence()?;
            let (sequence, index, version, part) = storage_key::parse_quarantine(&key)?;
            match part {
                QuarantinePart::Marker => entries.push(QuarantinedEntry::new(
                    sequence,
                    index,
                    version,
                    value.to_vec(),
                    None,
                )),
                QuarantinePart::Payload => match entries.last_mut() {
                    Some(entry)
                        if entry.sequence == sequence
                            && entry.index == index
                            && entry.version == version
                            && entry.payload.is_none() =>
                    {
                        entry.payload = Some(value.to_vec());
                    }
                    _ => {
                        return Err(corrupt(UnpairedQuarantineEntry {
                            sequence,
                            reason: "holds a payload without its marker",
                        }));
                    }
                },
                _ => {
                    return Err(corrupt(UnpairedQuarantineEntry {
                        sequence,
                        reason: "holds a part this adapter does not store",
                    }));
                }
            }
        }
        Ok(entries)
    }
}

impl<T> OrPersistence<T> for Result<T, fjall::Error> {
    #[track_caller]
    fn or_persistence(self) -> Result<T, HeuremaError> {
        match self {
            Ok(value) => Ok(value),
            Err(source) => Err(persistence(source)),
        }
    }
}

#[track_caller]
fn staged_state_exists(index: &IndexIdentity, version: IndexVersion) -> HeuremaError {
    HeuremaError::StagedStateExists {
        index: index.clone(),
        version,
        location: Location::caller(),
    }
}

#[track_caller]
fn version_stored(index: &IndexIdentity, version: IndexVersion) -> HeuremaError {
    HeuremaError::VersionStored {
        index: index.clone(),
        version,
        location: Location::caller(),
    }
}

#[track_caller]
fn staged_state_missing(index: &IndexIdentity, version: IndexVersion) -> HeuremaError {
    HeuremaError::StagedStateMissing {
        index: index.clone(),
        version,
        location: Location::caller(),
    }
}

#[track_caller]
fn persistence<E>(source: E) -> HeuremaError
where
    E: std::error::Error + Send + Sync + 'static,
{
    HeuremaError::Persistence {
        source: PersistenceSource::new(source),
        location: Location::caller(),
    }
}

#[track_caller]
fn corrupt<E>(source: E) -> HeuremaError
where
    E: std::error::Error + Send + Sync + 'static,
{
    HeuremaError::CorruptSnapshot {
        source: PersistenceSource::new(source),
        location: Location::caller(),
    }
}

impl fmt::Display for CommitLockPoisoned {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(
            "thesauros lifecycle commit lock is poisoned: a writer panicked between its checks \
             and its commit; reopen the backend",
        )
    }
}

impl std::error::Error for CommitLockPoisoned {}

impl fmt::Display for QuarantineSequenceExhausted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("thesauros quarantine sequence is exhausted")
    }
}

impl std::error::Error for QuarantineSequenceExhausted {}

impl fmt::Display for UnpairedQuarantineEntry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "quarantine entry {} {}",
            self.sequence, self.reason
        )
    }
}

impl std::error::Error for UnpairedQuarantineEntry {}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests need concise fixture setup")]
mod tests {
    use super::*;
    use heurema::{IndexName, OwnerNamespace};

    #[test]
    fn poisoned_writer_lock_refuses_lifecycle_writes_but_not_reads() {
        let dir = tempfile::tempdir().expect("tempdir");
        let backend = ThesaurosBackend::open(dir.path()).expect("open");
        let index = IndexIdentity::new(
            OwnerNamespace::try_from("example").expect("namespace"),
            IndexName::try_from("notes").expect("name"),
        );

        let key = OperationKey::try_from("op-1").expect("key");

        let writer = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _commit = backend.commit_lock.lock();
                    panic!("writer panics while holding the lifecycle commit lock");
                })
                .join()
        });
        assert!(writer.is_err(), "the writer thread panicked");

        let error = backend
            .stage(StageWrite::new(
                &index,
                IndexVersion::FIRST,
                &key,
                None,
                b"marker",
                b"payload",
            ))
            .expect_err("a poisoned writer lock refuses writes");
        assert!(
            matches!(error, HeuremaError::Persistence { .. }),
            "{error:?}"
        );
        assert!(error.to_string().contains("poisoned"), "{error}");
        assert_eq!(
            backend.read_staging(&index).expect("reads take no lock"),
            None,
            "the refused stage wrote nothing"
        );
    }
}
