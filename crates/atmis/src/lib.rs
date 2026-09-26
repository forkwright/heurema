//! `atmis` (ἀτμίς, vapor) is an in-memory [`PersistenceBackend`] and
//! [`LifecycleBackend`] adapter for heurema indexes.
//!
//! Snapshots live in a `HashMap<String, Vec<u8>>` behind an [`RwLock`],
//! encoded through `serde_json` so save and load exercise the same
//! byte-level contract a durable backend depends on. A backend that instead
//! cloned the live `I` value directly would pass a save/load round trip
//! trivially without proving the encode/decode path. See `thesauros` for
//! the durable adapter this one is meant to stand in for during tests.
//!
//! Lifecycle state lives in five maps behind one [`Mutex`]: heads, version
//! payloads, staging markers, operation records, and quarantine. Every
//! lifecycle write runs its checks and its mutation under that single guard,
//! with its owned values prepared before the lock is taken, so a reader sees
//! each write entirely or not at all. That single lock is this adapter's
//! atomicity basis; `thesauros` gets the same property from fjall's write
//! batches.
//!
//! Snapshots and lifecycle state never reach disk and do not survive the
//! process, the way vapor leaves no residue once its condition (heat, or
//! here, a live process) ends. Nothing here reopens a store across a process
//! boundary; that proof belongs to `thesauros`'s test suite.
//!
//! # Examples
//!
//! ```
//! use atmis::AtmisBackend;
//! use heurema::{
//!     IndexIdentity, IndexName, IndexVersion, LifecycleBackend, OperationKey, OwnerNamespace,
//!     PublishWrite, StageWrite,
//! };
//!
//! let backend = AtmisBackend::new();
//! let index = IndexIdentity::new(
//!     OwnerNamespace::try_from("example")?,
//!     IndexName::try_from("notes")?,
//! );
//! let key = OperationKey::try_from("create-notes")?;
//! let version = IndexVersion::FIRST;
//!
//! // Values are opaque bytes here; heurēma's lifecycle encodes the real records.
//! let (head, operation) = (b"head", b"operation");
//! backend.stage(StageWrite::new(
//!     &index, version, &key, None, b"marker", b"payload", head.len(), operation.len(),
//! ))?;
//! assert_eq!(backend.read_head(&index)?, None, "staging moves no head");
//!
//! backend.publish(PublishWrite::new(
//!     &index, version, &key, None, b"marker", head, operation,
//! ))?;
//! assert_eq!(backend.read_head(&index)?, Some(b"head".to_vec()));
//! assert_eq!(backend.read_staging(&index)?, None, "publish removes the marker");
//! # Ok::<(), heurema::HeuremaError>(())
//! ```

#![deny(missing_docs)]

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::Bound;
use std::panic::Location;
use std::sync::{Mutex, MutexGuard, PoisonError, RwLock};

use heurema::lifecycle::storage_key;
use heurema::{
    DestroyWrite, FtsIndex, HeuremaError, IndexIdentity, IndexVersion, LifecycleBackend,
    OperationKey, PersistenceBackend, PersistenceSource, PublishWrite, QuarantineWrite,
    QuarantinedEntry, SnapshotEnvelope, SnapshotFamily, StageWrite, StagedEntry, VectorIndex,
    WriterLock, decode_snapshot_payload,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// WHY: vector-index and FTS-index snapshots live in separate maps because
/// [`PersistenceBackend::save_vector_index`] and
/// [`PersistenceBackend::save_fts_index`] are independent name spaces —
/// nothing in the trait contract says the two families share names.
/// Lifecycle state is a third, separate store, so a snapshot name never
/// meets a lifecycle key.
#[derive(Debug, Default)]
pub struct AtmisBackend {
    vector_snapshots: RwLock<HashMap<String, Vec<u8>>>,
    fts_snapshots: RwLock<HashMap<String, Vec<u8>>>,
    lifecycle: Mutex<LifecycleState>,
    /// The writer every lifecycle over this backend shares; see
    /// [`LifecycleBackend::writer`].
    writer: WriterLock,
}

/// Every lifecycle map, behind the one mutex in [`AtmisBackend`].
///
/// WHY one struct under one lock: publish changes the heads, operations,
/// and staging maps together, and a reader must never see one change
/// without the others. The four keyed maps use [`storage_key`] strings in
/// `BTreeMap`s, so listings come back in the same byte order fjall's
/// keyspaces give `thesauros`; quarantine is a `Vec` in sequence order.
/// Version and operation keys live in separate maps because a digits-only
/// operation key spells the same text as a version key.
#[derive(Debug, Default)]
struct LifecycleState {
    heads: BTreeMap<String, Vec<u8>>,
    versions: BTreeMap<String, Vec<u8>>,
    staging: BTreeMap<String, Vec<u8>>,
    operations: BTreeMap<String, Vec<u8>>,
    quarantine: Vec<QuarantinedEntry>,
}

/// A thread panicked while holding the lifecycle lock.
#[derive(Debug)]
struct LifecycleStatePoisoned;

/// The quarantine counter reached `u64::MAX`.
#[derive(Debug)]
struct QuarantineSequenceExhausted;

impl AtmisBackend {
    /// Construct an empty backend holding no saved snapshots and no
    /// lifecycle state.
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

    /// The lifecycle maps, locked.
    ///
    /// WHY a poisoned lock is refused, not recovered like the snapshot
    /// locks above: a panic while this guard was held may have left one
    /// write half applied across the maps (a head moved without its
    /// operation record, say). Recovering the guard would serve that half
    /// write as state.
    #[track_caller]
    fn lifecycle(&self) -> Result<MutexGuard<'_, LifecycleState>, HeuremaError> {
        match self.lifecycle.lock() {
            Ok(state) => Ok(state),
            Err(_poisoned) => Err(persistence(LifecycleStatePoisoned)),
        }
    }
}

impl LifecycleState {
    /// The first staging marker among the keys that start with `prefix`,
    /// as `(key, marker)`.
    fn first_staged(&self, prefix: &str) -> Option<(&String, &Vec<u8>)> {
        self.staging
            .range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
            .next()
            .filter(|(key, _)| key.starts_with(prefix))
    }

    #[track_caller]
    fn check_head(
        &self,
        index: &IndexIdentity,
        head_key: &str,
        expected: Option<&[u8]>,
    ) -> Result<(), HeuremaError> {
        if self.heads.get(head_key).map(Vec::as_slice) == expected {
            Ok(())
        } else {
            Err(HeuremaError::HeadChanged {
                index: index.clone(),
                location: Location::caller(),
            })
        }
    }

    #[track_caller]
    fn refuse_staged(&self, index: &IndexIdentity, prefix: &str) -> Result<(), HeuremaError> {
        let Some((key, _)) = self.first_staged(prefix) else {
            return Ok(());
        };
        let (_, version) = storage_key::parse_staging(key.as_bytes())?;
        Err(staged_state_exists(index, version))
    }

    #[track_caller]
    fn refuse_stored_version(
        &self,
        index: &IndexIdentity,
        version: IndexVersion,
        version_key: &str,
    ) -> Result<(), HeuremaError> {
        if self.versions.contains_key(version_key) {
            Err(version_stored(index, version))
        } else {
            Ok(())
        }
    }

    #[track_caller]
    fn check_marker(
        &self,
        index: &IndexIdentity,
        version: IndexVersion,
        staging_key: &str,
        marker: &[u8],
    ) -> Result<(), HeuremaError> {
        if self.staging.get(staging_key).map(Vec::as_slice) == Some(marker) {
            Ok(())
        } else {
            Err(staged_state_missing(index, version))
        }
    }

    #[track_caller]
    fn check_payload(
        &self,
        index: &IndexIdentity,
        version: IndexVersion,
        version_key: &str,
    ) -> Result<(), HeuremaError> {
        if self.versions.contains_key(version_key) {
            Ok(())
        } else {
            Err(staged_state_missing(index, version))
        }
    }

    #[track_caller]
    fn refuse_recorded(
        &self,
        index: &IndexIdentity,
        key: &OperationKey,
        operation_key: &str,
    ) -> Result<(), HeuremaError> {
        if self.operations.contains_key(operation_key) {
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
    fn next_sequence(&self) -> Result<u64, HeuremaError> {
        let Some(last) = self.quarantine.last() else {
            return Ok(QuarantinedEntry::FIRST_SEQUENCE);
        };
        match last.sequence.checked_add(1) {
            Some(next) => Ok(next),
            None => Err(persistence(QuarantineSequenceExhausted)),
        }
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

impl LifecycleBackend for AtmisBackend {
    fn writer(&self) -> &WriterLock {
        &self.writer
    }

    fn read_head(&self, index: &IndexIdentity) -> Result<Option<Vec<u8>>, HeuremaError> {
        let key = storage_key::head(index);
        Ok(self.lifecycle()?.heads.get(&key).cloned())
    }

    fn read_version(
        &self,
        index: &IndexIdentity,
        version: IndexVersion,
    ) -> Result<Option<Vec<u8>>, HeuremaError> {
        let key = storage_key::version(index, version);
        Ok(self.lifecycle()?.versions.get(&key).cloned())
    }

    fn read_operation(
        &self,
        index: &IndexIdentity,
        key: &OperationKey,
    ) -> Result<Option<Vec<u8>>, HeuremaError> {
        let key = storage_key::operation(index, key);
        Ok(self.lifecycle()?.operations.get(&key).cloned())
    }

    fn read_staging(
        &self,
        index: &IndexIdentity,
    ) -> Result<Option<(IndexVersion, Vec<u8>)>, HeuremaError> {
        let prefix = storage_key::index_prefix(index);
        let staged = self
            .lifecycle()?
            .first_staged(&prefix)
            .map(|(key, marker)| (key.clone(), marker.clone()));
        let Some((key, marker)) = staged else {
            return Ok(None);
        };
        let (_, version) = storage_key::parse_staging(key.as_bytes())?;
        Ok(Some((version, marker)))
    }

    fn stage(&self, write: StageWrite<'_>) -> Result<(), HeuremaError> {
        let head_key = storage_key::head(write.index);
        let prefix = storage_key::index_prefix(write.index);
        let version_key = storage_key::version(write.index, write.version);
        let staging_key = storage_key::staging(write.index, write.version);
        let operation_key = storage_key::operation(write.index, write.key);
        let payload = write.payload.to_vec();
        let marker = write.marker.to_vec();

        let mut state = self.lifecycle()?;
        state.check_head(write.index, &head_key, write.expected_head)?;
        state.refuse_staged(write.index, &prefix)?;
        state.refuse_stored_version(write.index, write.version, &version_key)?;
        state.refuse_recorded(write.index, write.key, &operation_key)?;
        state.versions.insert(version_key, payload);
        state.staging.insert(staging_key, marker);
        Ok(())
    }

    fn publish(&self, write: PublishWrite<'_>) -> Result<(), HeuremaError> {
        let head_key = storage_key::head(write.index);
        let version_key = storage_key::version(write.index, write.version);
        let staging_key = storage_key::staging(write.index, write.version);
        let operation_key = storage_key::operation(write.index, write.key);
        let head = write.head.to_vec();
        let operation = write.operation.to_vec();

        let mut state = self.lifecycle()?;
        state.check_head(write.index, &head_key, write.expected_head)?;
        state.check_marker(write.index, write.version, &staging_key, write.marker)?;
        state.check_payload(write.index, write.version, &version_key)?;
        state.refuse_recorded(write.index, write.key, &operation_key)?;
        state.heads.insert(head_key, head);
        state.operations.insert(operation_key, operation);
        state.staging.remove(&staging_key);
        Ok(())
    }

    fn destroy(&self, write: DestroyWrite<'_>) -> Result<(), HeuremaError> {
        let head_key = storage_key::head(write.index);
        let prefix = storage_key::index_prefix(write.index);
        let operation_key = storage_key::operation(write.index, write.key);
        let version_keys: Vec<String> = write
            .versions
            .iter()
            .map(|version| storage_key::version(write.index, *version))
            .collect();
        let head = write.head.to_vec();
        let operation = write.operation.to_vec();

        let mut state = self.lifecycle()?;
        state.check_head(write.index, &head_key, Some(write.expected_head))?;
        state.refuse_staged(write.index, &prefix)?;
        state.refuse_recorded(write.index, write.key, &operation_key)?;
        state.heads.insert(head_key, head);
        state.operations.insert(operation_key, operation);
        for key in &version_keys {
            state.versions.remove(key);
        }
        Ok(())
    }

    fn list_heads(&self) -> Result<Vec<(IndexIdentity, Vec<u8>)>, HeuremaError> {
        let heads: Vec<(String, Vec<u8>)> = self
            .lifecycle()?
            .heads
            .iter()
            .map(|(key, head)| (key.clone(), head.clone()))
            .collect();
        heads
            .into_iter()
            .map(|(key, head)| Ok((storage_key::parse_head(key.as_bytes())?, head)))
            .collect()
    }

    fn list_staged(&self) -> Result<Vec<StagedEntry>, HeuremaError> {
        let staged: Vec<(String, Vec<u8>)> = self
            .lifecycle()?
            .staging
            .iter()
            .map(|(key, marker)| (key.clone(), marker.clone()))
            .collect();
        staged
            .into_iter()
            .map(|(key, marker)| {
                let (index, version) = storage_key::parse_staging(key.as_bytes())?;
                Ok(StagedEntry::new(index, version, marker))
            })
            .collect()
    }

    fn list_operations(
        &self,
        index: &IndexIdentity,
    ) -> Result<Vec<(OperationKey, Vec<u8>)>, HeuremaError> {
        let prefix = storage_key::index_prefix(index);
        let operations: Vec<(String, Vec<u8>)> = self
            .lifecycle()?
            .operations
            .range::<str, _>((Bound::Included(prefix.as_str()), Bound::Unbounded))
            .take_while(|(key, _)| key.starts_with(&prefix))
            .map(|(key, operation)| (key.clone(), operation.clone()))
            .collect();
        operations
            .into_iter()
            .map(|(key, operation)| {
                let (_, key) = storage_key::parse_operation(key.as_bytes())?;
                Ok((key, operation))
            })
            .collect()
    }

    fn quarantine(&self, write: QuarantineWrite<'_>) -> Result<(), HeuremaError> {
        let version_key = storage_key::version(write.index, write.version);
        let staging_key = storage_key::staging(write.index, write.version);
        let index = write.index.clone();
        // INVARIANT: `check_marker` below proves the stored marker equals
        // these bytes, so storing them moves the marker byte for byte.
        let marker = write.marker.to_vec();

        let mut state = self.lifecycle()?;
        state.check_marker(write.index, write.version, &staging_key, write.marker)?;
        let sequence = state.next_sequence()?;
        let payload = state.versions.remove(&version_key);
        state.staging.remove(&staging_key);
        state.quarantine.push(QuarantinedEntry::new(
            sequence,
            index,
            write.version,
            marker,
            payload,
        ));
        Ok(())
    }

    fn list_quarantined(&self) -> Result<Vec<QuarantinedEntry>, HeuremaError> {
        Ok(self.lifecycle()?.quarantine.clone())
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

impl fmt::Display for LifecycleStatePoisoned {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(
            "atmis lifecycle state is poisoned: a writer panicked while holding the lock, \
             so the maps may hold part of one write; build a new backend",
        )
    }
}

impl std::error::Error for LifecycleStatePoisoned {}

impl fmt::Display for QuarantineSequenceExhausted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("atmis quarantine sequence is exhausted")
    }
}

impl std::error::Error for QuarantineSequenceExhausted {}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests need concise private-store setup")]
mod tests {
    use super::*;
    use heurema::{HnswConfig, HnswIndex, IndexName, OwnerNamespace};

    fn notes() -> IndexIdentity {
        IndexIdentity::new(
            OwnerNamespace::try_from("example").expect("namespace"),
            IndexName::try_from("notes").expect("name"),
        )
    }

    fn inject_vector_snapshot(backend: &AtmisBackend, name: &str, bytes: &[u8]) {
        AtmisBackend::write(&backend.vector_snapshots).insert(name.to_owned(), bytes.to_vec());
    }

    #[test]
    fn poisoned_lifecycle_state_is_refused_not_recovered() {
        let backend = AtmisBackend::new();
        let index = notes();
        let key = OperationKey::try_from("op-1").expect("key");
        backend
            .stage(StageWrite::new(
                &index,
                IndexVersion::FIRST,
                &key,
                None,
                b"marker",
                b"payload",
                0,
                0,
            ))
            .expect("stage before the panic");

        let writer = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _state = backend.lifecycle.lock();
                    panic!("writer panics while holding the lifecycle lock");
                })
                .join()
        });
        assert!(writer.is_err(), "the writer thread panicked");
        assert!(backend.lifecycle.is_poisoned());

        let refusals = [
            backend.read_head(&index).map(|_| ()),
            backend.read_staging(&index).map(|_| ()),
            backend.list_staged().map(|_| ()),
            backend.publish(PublishWrite::new(
                &index,
                IndexVersion::FIRST,
                &key,
                None,
                b"marker",
                b"head",
                b"operation",
            )),
        ];
        for refusal in refusals {
            let error = refusal.expect_err("a poisoned lifecycle lock is refused");
            assert!(
                matches!(error, HeuremaError::Persistence { .. }),
                "{error:?}"
            );
            assert!(error.to_string().contains("poisoned"), "{error}");
        }
        assert!(
            backend.lifecycle.is_poisoned(),
            "the refusal leaves the poison in place"
        );

        let snapshot = HnswIndex::<u64>::new(HnswConfig::new(2));
        backend
            .save_vector_index("unaffected", &snapshot)
            .expect("snapshot maps have their own locks");
    }

    #[test]
    fn a_marker_without_its_payload_cannot_publish_but_can_be_quarantined() {
        let backend = AtmisBackend::new();
        let index = notes();
        let version = IndexVersion::FIRST;
        let key = OperationKey::try_from("op-1").expect("key");
        backend
            .stage(StageWrite::new(
                &index, version, &key, None, b"marker", b"payload", 0, 0,
            ))
            .expect("stage");
        // WHY: stage writes both halves together, so only a damaged store
        // holds a marker without its payload; build one directly.
        backend
            .lifecycle()
            .expect("lock")
            .versions
            .remove(&storage_key::version(&index, version))
            .expect("the payload was staged");

        let refusal = backend
            .publish(PublishWrite::new(
                &index,
                version,
                &key,
                None,
                b"marker",
                b"head",
                b"operation",
            ))
            .expect_err("no payload to publish");
        assert!(
            matches!(refusal, HeuremaError::StagedStateMissing { .. }),
            "{refusal:?}"
        );
        assert_eq!(backend.read_head(&index).expect("read"), None);

        backend
            .quarantine(QuarantineWrite::new(&index, version, b"marker"))
            .expect("quarantine clears the marker");
        let quarantined = backend.list_quarantined().expect("list");
        assert_eq!(quarantined.len(), 1);
        assert_eq!(quarantined[0].marker, b"marker");
        assert_eq!(
            quarantined[0].payload, None,
            "the absent payload is recorded"
        );
        assert_eq!(backend.read_staging(&index).expect("read"), None);
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

        assert!(
            matches!(
                backend.load_vector_index::<HnswIndex<u64>>("future"),
                Err(HeuremaError::UnsupportedSnapshotVersion { .. })
            ),
            "a future format version is unsupported before the payload decodes"
        );
        assert!(
            matches!(
                backend.load_vector_index::<HnswIndex<u64>>("wrong-family"),
                Err(HeuremaError::SnapshotFormat { .. })
            ),
            "a wrong family is refused before the payload decodes"
        );
    }

    #[test]
    fn current_header_with_an_invalid_payload_remains_a_decode_error() {
        let backend = AtmisBackend::new();
        inject_vector_snapshot(
            &backend,
            "invalid-current",
            br#"{"format_version":1,"family":"Vector","payload":{"alien":true}}"#,
        );
        assert!(
            matches!(
                backend.load_vector_index::<HnswIndex<u64>>("invalid-current"),
                Err(HeuremaError::CorruptSnapshot { .. })
            ),
            "a current header over an undecodable payload is corrupt"
        );

        let valid = HnswIndex::<u64>::new(HnswConfig::new(2));
        backend
            .save_vector_index("valid", &valid)
            .expect("valid current envelope saves");
        assert!(backend.load_vector_index::<HnswIndex<u64>>("valid").is_ok());
    }
}
