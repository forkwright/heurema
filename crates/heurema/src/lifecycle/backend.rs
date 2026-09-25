//! The durable lifecycle's storage contract, [`LifecycleBackend`], with the
//! write descriptors its adapters receive and the [`storage_key`] grammar
//! they share. The contract itself is documented on the trait, because this
//! module is private and only its re-exports are public.

use std::sync::Arc;

use super::identity::{IndexIdentity, IndexVersion, OperationKey};
use super::writer::WriterLock;
use crate::HeuremaError;

/// The format version heurēma writes into every lifecycle record it stores
/// through a [`LifecycleBackend`]: heads, version payloads, staging markers,
/// and operation records. The [`storage_key`] grammar belongs to the same
/// format.
///
/// WHY: a stored record outlives the build that wrote it. A reader compares
/// this version before decoding, so a record from a newer build is reported
/// as unsupported and left alone rather than taken for corrupt bytes.
pub const LIFECYCLE_FORMAT_VERSION: u16 = 1;

/// The storage a durable index lifecycle runs on.
///
/// WHY a trait beside [`PersistenceBackend`](crate::PersistenceBackend):
/// that trait saves and loads one whole-index snapshot by name. It has no
/// delete, no existence check, no multi-key write, and no compare-and-set,
/// and its generic methods keep it from being a trait object. The lifecycle
/// needs every one of those, so it binds this separate, object-safe trait.
/// `PersistenceBackend` is unchanged.
///
/// # What an adapter stores
///
/// Every value is an opaque byte string heurēma encoded; an adapter stores
/// and returns it byte for byte and never decodes it. Every key is a
/// [`storage_key`] string built from validated identifiers. An adapter owns
/// no encoding and no lifecycle rule: it cannot tell what a head, payload,
/// marker, or operation record says. The only checks it makes are structural
/// and need no decoding: comparing a head's or marker's bytes
/// (compare-and-set), never overwriting a stored value, and refusing while
/// staged state exists.
///
/// A keyed store (the fjall keyspaces of `thesauros`) keeps five maps:
///
/// | map | key | value |
/// |---|---|---|
/// | heads | [`storage_key::head`] | the index's current head record |
/// | versions | [`storage_key::version`] | one version's immutable payload |
/// | staging | [`storage_key::staging`] | the marker of a staged, unpublished version |
/// | operations | [`storage_key::operation`] | the record of one published operation |
/// | quarantine | [`storage_key::quarantine`] | a staged marker and payload moved aside |
///
/// `thesauros` keys the quarantine map by [`storage_key::quarantine`];
/// `atmis` keeps quarantine entries in a `Vec` in sequence order.
///
/// # Contract
///
/// - Every lifecycle over one backend shares its writer (see
///   [`writer`](LifecycleBackend::writer)).
/// - Each write ([`stage`](LifecycleBackend::stage),
///   [`publish`](LifecycleBackend::publish),
///   [`destroy`](LifecycleBackend::destroy),
///   [`quarantine`](LifecycleBackend::quarantine)) is atomic and durable
///   before it returns `Ok`: a reader, in process or after a crash and
///   reopen, sees all of the write or none of it. (`atmis` keeps nothing past
///   the process; within it the same holds.)
/// - A read sees every write that returned before the read began, and each
///   write entirely or not at all.
/// - Each write runs its checks and its effect under one backend-owned lock,
///   so no other write lands between them. That makes an expected head or
///   marker a compare-and-set, and it holds even for writers that do not
///   hold the backend's writer.
/// - A write refused with [`HeuremaError::HeadChanged`],
///   [`HeuremaError::StagedStateExists`], [`HeuremaError::VersionStored`],
///   [`HeuremaError::StagedStateMissing`], or
///   [`HeuremaError::OperationRecorded`] wrote nothing.
/// - A write that fails with [`HeuremaError::Persistence`] may or may not
///   have taken effect (a lost acknowledgement): read the state back before
///   deciding what to do. After a failed `thesauros` commit the fjall
///   database refuses further writes until the backend is reopened, and
///   until then its reads may also miss a write that reached the journal.
///
/// # Atomicity basis
///
/// - `thesauros` commits each write as one fjall write batch with
///   `PersistMode::SyncAll`. fjall appends the batch to its journal as one
///   checksummed unit, fsyncs it, and only then applies it in memory; on
///   reopen it replays complete batches and discards an incomplete trailing
///   one, so a crash leaves all of the write or none of it.
/// - `atmis` applies each write as one mutation under a single mutex over
///   all five maps.
///
/// # Interrupted operations
///
/// A version that was staged but never published (the process stopped
/// between stage and publish) stays in storage as orphan staged state: its
/// payload and its marker. While it exists, [`stage`](LifecycleBackend::stage)
/// and [`destroy`](LifecycleBackend::destroy) of that index are refused with
/// [`HeuremaError::StagedStateExists`]; other indexes are unaffected. Only
/// recovery clears it, by moving it to quarantine through
/// [`quarantine`](LifecycleBackend::quarantine); nothing ever deletes it.
/// Recovery arrives with the lifecycle driver's reopen path in a later
/// Phase 02 change; until then an orphan blocks its index. With every writer
/// holding the backend's [`writer`](LifecycleBackend::writer), a marker
/// exists outside any operation only when that operation was interrupted.
///
/// The methods after [`destroy`](LifecycleBackend::destroy) serve recovery
/// and audit. They are declared together with the rest so a later recovery
/// pass adds no required method, which would break every implementor.
///
/// # Errors
///
/// Every method returns [`HeuremaError::Persistence`] when the backend
/// cannot read or write, and [`HeuremaError::CorruptSnapshot`] when a stored
/// key does not follow the [`storage_key`] grammar.
///
/// WHY object safe: a lifecycle is chosen at runtime (durable or in-memory,
/// or a test wrapper that injects faults), so `&dyn LifecycleBackend`,
/// `Box<dyn LifecycleBackend>`, and `Arc<dyn LifecycleBackend>` all
/// implement it.
pub trait LifecycleBackend {
    /// The writer every lifecycle over this backend shares.
    ///
    /// [`IndexLifecycle`](crate::IndexLifecycle) takes it after an
    /// operation's stateless checks, and [`Prepared`](crate::Prepared) and
    /// [`Staged`](crate::Staged) hold it until publish or drop. So on one
    /// backend at most one operation is between its head read and its
    /// publish at a time, whichever lifecycle or thread runs it, and any
    /// staging marker met under the writer is interrupted state.
    ///
    /// An adapter owns exactly one [`WriterLock`] and returns it on every
    /// call. A wrapper returns the lock of the backend it wraps; the pointer
    /// impls do.
    ///
    /// Direct callers of [`stage`](Self::stage), [`publish`](Self::publish),
    /// [`destroy`](Self::destroy), or [`quarantine`](Self::quarantine) should
    /// hold it. Without it the structural checks still hold, but a
    /// [`HeuremaError::StagedStateExists`] the caller causes or meets may
    /// name a live stage, and a `quarantine` it runs may move a live stage,
    /// whose publish then fails with [`HeuremaError::StagedStateMissing`],
    /// having published nothing.
    fn writer(&self) -> &WriterLock;

    /// The head record of `index`, or `None` when the index has no head.
    ///
    /// # Errors
    ///
    /// See the [trait-level errors](LifecycleBackend#errors).
    fn read_head(&self, index: &IndexIdentity) -> Result<Option<Vec<u8>>, HeuremaError>;

    /// The payload stored for `version` of `index`, or `None`.
    ///
    /// A staged, unpublished payload is returned too, because the backend
    /// cannot tell which versions are published. A reader resolves the head
    /// first and reads only the version it names.
    ///
    /// # Errors
    ///
    /// See the [trait-level errors](LifecycleBackend#errors).
    fn read_version(
        &self,
        index: &IndexIdentity,
        version: IndexVersion,
    ) -> Result<Option<Vec<u8>>, HeuremaError>;

    /// The record of the published operation `key` on `index`, or `None`.
    ///
    /// # Errors
    ///
    /// See the [trait-level errors](LifecycleBackend#errors).
    fn read_operation(
        &self,
        index: &IndexIdentity,
        key: &OperationKey,
    ) -> Result<Option<Vec<u8>>, HeuremaError>;

    /// The staged version of `index` and its marker, or `None` when nothing
    /// is staged for it.
    ///
    /// At most one version is staged per index, because
    /// [`stage`](LifecycleBackend::stage) refuses while one is. If a store
    /// holds several anyway, this returns the lowest, and
    /// [`list_staged`](LifecycleBackend::list_staged) lists them all.
    ///
    /// WHY: every mutation, destroy included, must see staged state before
    /// it acts, and destroy stages nothing that would reveal it.
    ///
    /// # Errors
    ///
    /// See the [trait-level errors](LifecycleBackend#errors).
    fn read_staging(
        &self,
        index: &IndexIdentity,
    ) -> Result<Option<(IndexVersion, Vec<u8>)>, HeuremaError>;

    /// Store a version's payload and its staging marker in one atomic write
    /// that is durable before `Ok`. The head does not change, so the version
    /// stays unreachable until [`publish`](LifecycleBackend::publish).
    ///
    /// Checks, in order, under the backend lock:
    ///
    /// 1. The head equals `write.expected_head` (`None`: no head exists).
    /// 2. No staging marker exists for the index, at any version.
    /// 3. No payload is stored under `write.version`: stage never
    ///    overwrites.
    /// 4. No operation record exists under `write.key`: stage never leaves
    ///    staged state for an operation that is already recorded.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::HeadChanged`] when check 1 fails,
    /// [`HeuremaError::StagedStateExists`] when check 2 fails, naming the
    /// version found, [`HeuremaError::VersionStored`] when check 3 fails,
    /// and [`HeuremaError::OperationRecorded`] when check 4 fails. Otherwise
    /// see the [trait-level errors](LifecycleBackend#errors).
    fn stage(&self, write: StageWrite<'_>) -> Result<(), HeuremaError>;

    /// Publish a staged version: in one atomic write that is durable before
    /// `Ok`, set the head to `write.head`, record `write.operation` under
    /// `write.key`, and remove the version's staging marker. This is the
    /// lifecycle's single publish point.
    ///
    /// Checks, in order, under the backend lock:
    ///
    /// 1. The head equals `write.expected_head`.
    /// 2. The staging marker of `write.version` equals `write.marker`, and a
    ///    payload is stored under that version.
    /// 3. No operation record exists under `write.key`.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::HeadChanged`] when check 1 fails,
    /// [`HeuremaError::StagedStateMissing`] when check 2 fails, and
    /// [`HeuremaError::OperationRecorded`] when check 3 fails. Otherwise see
    /// the [trait-level errors](LifecycleBackend#errors).
    fn publish(&self, write: PublishWrite<'_>) -> Result<(), HeuremaError>;

    /// Destroy an index: in one atomic write that is durable before `Ok`,
    /// set the head to `write.head` (heurēma's destroyed record), record
    /// `write.operation` under `write.key`, and remove the payload of every
    /// version in `write.versions`. A listed version with no payload is
    /// skipped. Operation records stay: a destroyed index keeps its audit
    /// chain.
    ///
    /// Checks, in order, under the backend lock:
    ///
    /// 1. The head equals `write.expected_head`.
    /// 2. No staging marker exists for the index, at any version.
    /// 3. No operation record exists under `write.key`.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::HeadChanged`] when check 1 fails,
    /// [`HeuremaError::StagedStateExists`] when check 2 fails, and
    /// [`HeuremaError::OperationRecorded`] when check 3 fails. Otherwise see
    /// the [trait-level errors](LifecycleBackend#errors).
    fn destroy(&self, write: DestroyWrite<'_>) -> Result<(), HeuremaError>;

    /// Every head, as `(index, head record)`, in ascending
    /// [`storage_key::head`] byte order.
    ///
    /// # Errors
    ///
    /// See the [trait-level errors](LifecycleBackend#errors).
    fn list_heads(&self) -> Result<Vec<(IndexIdentity, Vec<u8>)>, HeuremaError>;

    /// Every staging marker in the store, in ascending
    /// [`storage_key::staging`] byte order: by index, then by version.
    ///
    /// # Errors
    ///
    /// See the [trait-level errors](LifecycleBackend#errors).
    fn list_staged(&self) -> Result<Vec<StagedEntry>, HeuremaError>;

    /// Every operation record of `index`, as `(key, record)`, in ascending
    /// key byte order. The list is complete, never truncated.
    ///
    /// # Errors
    ///
    /// See the [trait-level errors](LifecycleBackend#errors).
    fn list_operations(
        &self,
        index: &IndexIdentity,
    ) -> Result<Vec<(OperationKey, Vec<u8>)>, HeuremaError>;

    /// Move a staged version to quarantine: in one atomic write that is
    /// durable before `Ok`, copy its marker and payload byte for byte into a
    /// new quarantine entry under the next sequence number, and remove both
    /// from the staging and versions maps. Nothing is deleted.
    ///
    /// Checks, under the backend lock: the staging marker of
    /// `write.version` equals `write.marker`. A payload missing beside the
    /// marker is recorded as absent rather than refused, so recovery can
    /// still clear the marker.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::StagedStateMissing`] when the check fails. Otherwise
    /// see the [trait-level errors](LifecycleBackend#errors).
    fn quarantine(&self, write: QuarantineWrite<'_>) -> Result<(), HeuremaError>;

    /// Every quarantine entry, in ascending sequence order.
    ///
    /// # Errors
    ///
    /// See the [trait-level errors](LifecycleBackend#errors).
    fn list_quarantined(&self) -> Result<Vec<QuarantinedEntry>, HeuremaError>;
}

/// Implements [`LifecycleBackend`] for a pointer type by forwarding every
/// method to the pointee.
///
/// WHY: pointer impls let lifecycles share one backend, and with it the
/// backend's writer; they exist now because adding a blanket impl later
/// would break a downstream impl for its own pointer type.
macro_rules! forward_lifecycle_backend {
    ($($pointer:ty),+ $(,)?) => {$(
        impl<B: LifecycleBackend + ?Sized> LifecycleBackend for $pointer {
            fn writer(&self) -> &WriterLock {
                (**self).writer()
            }

            fn read_head(&self, index: &IndexIdentity) -> Result<Option<Vec<u8>>, HeuremaError> {
                (**self).read_head(index)
            }

            fn read_version(
                &self,
                index: &IndexIdentity,
                version: IndexVersion,
            ) -> Result<Option<Vec<u8>>, HeuremaError> {
                (**self).read_version(index, version)
            }

            fn read_operation(
                &self,
                index: &IndexIdentity,
                key: &OperationKey,
            ) -> Result<Option<Vec<u8>>, HeuremaError> {
                (**self).read_operation(index, key)
            }

            fn read_staging(
                &self,
                index: &IndexIdentity,
            ) -> Result<Option<(IndexVersion, Vec<u8>)>, HeuremaError> {
                (**self).read_staging(index)
            }

            fn stage(&self, write: StageWrite<'_>) -> Result<(), HeuremaError> {
                (**self).stage(write)
            }

            fn publish(&self, write: PublishWrite<'_>) -> Result<(), HeuremaError> {
                (**self).publish(write)
            }

            fn destroy(&self, write: DestroyWrite<'_>) -> Result<(), HeuremaError> {
                (**self).destroy(write)
            }

            fn list_heads(&self) -> Result<Vec<(IndexIdentity, Vec<u8>)>, HeuremaError> {
                (**self).list_heads()
            }

            fn list_staged(&self) -> Result<Vec<StagedEntry>, HeuremaError> {
                (**self).list_staged()
            }

            fn list_operations(
                &self,
                index: &IndexIdentity,
            ) -> Result<Vec<(OperationKey, Vec<u8>)>, HeuremaError> {
                (**self).list_operations(index)
            }

            fn quarantine(&self, write: QuarantineWrite<'_>) -> Result<(), HeuremaError> {
                (**self).quarantine(write)
            }

            fn list_quarantined(&self) -> Result<Vec<QuarantinedEntry>, HeuremaError> {
                (**self).list_quarantined()
            }
        }
    )+};
}

forward_lifecycle_backend!(&B, Box<B>, Arc<B>);

/// What [`LifecycleBackend::stage`] writes, and the head it expects.
///
/// WHY `#[non_exhaustive]` with a constructor: a later field (a checksum,
/// say) must be addable without breaking adapters, which read these fields
/// but cannot build the struct by literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StageWrite<'a> {
    /// The index the version belongs to.
    pub index: &'a IndexIdentity,
    /// The version being staged.
    pub version: IndexVersion,
    /// The operation the stage belongs to.
    pub key: &'a OperationKey,
    /// The head the write was computed from; `None` when the index had no
    /// head.
    pub expected_head: Option<&'a [u8]>,
    /// The staging marker, stored under [`storage_key::staging`].
    pub marker: &'a [u8],
    /// The version payload, stored under [`storage_key::version`].
    pub payload: &'a [u8],
}

impl<'a> StageWrite<'a> {
    /// Describe a stage of `version` of `index` for the operation `key`,
    /// computed from `expected_head`.
    #[must_use]
    pub const fn new(
        index: &'a IndexIdentity,
        version: IndexVersion,
        key: &'a OperationKey,
        expected_head: Option<&'a [u8]>,
        marker: &'a [u8],
        payload: &'a [u8],
    ) -> Self {
        Self {
            index,
            version,
            key,
            expected_head,
            marker,
            payload,
        }
    }
}

/// What [`LifecycleBackend::publish`] writes, and the head and staged
/// marker it expects.
///
/// WHY the marker is carried: it fences the publish to the exact staged
/// state this writer staged. A version that was quarantined and staged again
/// by another writer has another marker, so publishing it is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PublishWrite<'a> {
    /// The index being published.
    pub index: &'a IndexIdentity,
    /// The staged version the new head names.
    pub version: IndexVersion,
    /// The key the operation record is stored under.
    pub key: &'a OperationKey,
    /// The head the write was computed from; `None` when the index had no
    /// head.
    pub expected_head: Option<&'a [u8]>,
    /// The staging marker [`LifecycleBackend::stage`] stored for `version`.
    pub marker: &'a [u8],
    /// The new head record.
    pub head: &'a [u8],
    /// The operation record.
    pub operation: &'a [u8],
}

impl<'a> PublishWrite<'a> {
    /// Describe a publish of `version` of `index` under `key`, computed
    /// from `expected_head` and fenced by the staged `marker`.
    #[must_use]
    pub const fn new(
        index: &'a IndexIdentity,
        version: IndexVersion,
        key: &'a OperationKey,
        expected_head: Option<&'a [u8]>,
        marker: &'a [u8],
        head: &'a [u8],
        operation: &'a [u8],
    ) -> Self {
        Self {
            index,
            version,
            key,
            expected_head,
            marker,
            head,
            operation,
        }
    }
}

/// What [`LifecycleBackend::destroy`] writes and removes, and the head it
/// expects.
///
/// WHY `expected_head` is not optional: only an index with a head can be
/// destroyed, so a destroy of an absent index cannot be described.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DestroyWrite<'a> {
    /// The index being destroyed.
    pub index: &'a IndexIdentity,
    /// The key the operation record is stored under.
    pub key: &'a OperationKey,
    /// The head the write was computed from.
    pub expected_head: &'a [u8],
    /// The new head record: heurēma's destroyed record.
    pub head: &'a [u8],
    /// The operation record.
    pub operation: &'a [u8],
    /// The versions whose payloads are removed.
    pub versions: &'a [IndexVersion],
}

impl<'a> DestroyWrite<'a> {
    /// Describe a destroy of `index` under `key`, computed from
    /// `expected_head`, removing the payloads of `versions`.
    #[must_use]
    pub const fn new(
        index: &'a IndexIdentity,
        key: &'a OperationKey,
        expected_head: &'a [u8],
        head: &'a [u8],
        operation: &'a [u8],
        versions: &'a [IndexVersion],
    ) -> Self {
        Self {
            index,
            key,
            expected_head,
            head,
            operation,
            versions,
        }
    }
}

/// Which staged version [`LifecycleBackend::quarantine`] moves, fenced by
/// its marker.
///
/// WHY the marker is carried: recovery decides from the marker it read. If
/// the staged state changed since (published, or quarantined and staged
/// again), moving whatever is there now would act on state recovery never
/// examined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct QuarantineWrite<'a> {
    /// The index the staged version belongs to.
    pub index: &'a IndexIdentity,
    /// The staged version.
    pub version: IndexVersion,
    /// The staging marker recovery read for `version`.
    pub marker: &'a [u8],
}

impl<'a> QuarantineWrite<'a> {
    /// Describe the quarantine of staged `version` of `index`, fenced by
    /// `marker`.
    #[must_use]
    pub const fn new(index: &'a IndexIdentity, version: IndexVersion, marker: &'a [u8]) -> Self {
        Self {
            index,
            version,
            marker,
        }
    }
}

/// One staging marker, as [`LifecycleBackend::list_staged`] returns it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StagedEntry {
    /// The index the staged version belongs to.
    pub index: IndexIdentity,
    /// The staged version.
    pub version: IndexVersion,
    /// The marker bytes, as staged.
    pub marker: Vec<u8>,
}

impl StagedEntry {
    /// WHY: adapters in other crates build entries they list, and the
    /// struct is `#[non_exhaustive]`.
    #[must_use]
    pub const fn new(index: IndexIdentity, version: IndexVersion, marker: Vec<u8>) -> Self {
        Self {
            index,
            version,
            marker,
        }
    }
}

/// One staged version moved to quarantine, as
/// [`LifecycleBackend::list_quarantined`] returns it.
///
/// WHY a sequence number: a version number can be staged again after its
/// staged state was quarantined, so the index and version alone would let a
/// second quarantine collide with, and overwrite, the first.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QuarantinedEntry {
    /// The entry's position in quarantine order, starting at
    /// [`QuarantinedEntry::FIRST_SEQUENCE`], one higher for each entry, never
    /// reused.
    pub sequence: u64,
    /// The index the staged version belonged to.
    pub index: IndexIdentity,
    /// The staged version.
    pub version: IndexVersion,
    /// The staging marker, byte for byte.
    pub marker: Vec<u8>,
    /// The staged payload, byte for byte; `None` when the marker had no
    /// payload beside it.
    pub payload: Option<Vec<u8>>,
}

impl QuarantinedEntry {
    /// The sequence number of the first entry a store quarantines.
    pub const FIRST_SEQUENCE: u64 = 1;

    /// WHY: adapters in other crates build entries they list, and the
    /// struct is `#[non_exhaustive]`.
    #[must_use]
    pub const fn new(
        sequence: u64,
        index: IndexIdentity,
        version: IndexVersion,
        marker: Vec<u8>,
        payload: Option<Vec<u8>>,
    ) -> Self {
        Self {
            sequence,
            index,
            version,
            marker,
            payload,
        }
    }
}

/// The one key encoding every [`LifecycleBackend`] adapter uses.
///
/// # Grammar
///
/// ```text
/// head       = namespace "/" name
/// prefix     = namespace "/" name "/"
/// version    = prefix decimal20                   ; in the versions map
/// staging    = prefix decimal20                   ; in the staging map
/// operation  = prefix operation-key               ; in the operations map
/// quarantine = decimal20 "/" namespace "/" name "/" decimal20 "/" part
/// part       = "marker" / "payload"
/// decimal20  = 20 ASCII digits: the zero-padded decimal value
/// ```
///
/// `namespace`, `name`, and `operation-key` follow the
/// [`OwnerNamespace`](crate::OwnerNamespace),
/// [`IndexName`](crate::IndexName), and [`OperationKey`] grammars, none of
/// which admits `/`. Every key therefore splits on `/` one way only, and a
/// `prefix` matches the keys of exactly one index: the trailing `/` keeps
/// `notes` from matching `notes.v2`. Padding to 20 digits, the width of
/// `u64::MAX`, makes byte order equal numeric order, so versions and
/// quarantine sequences list in ascending order.
///
/// Each kind of key lives in its own map. A digits-only operation key such
/// as `00000000000000000001` spells the same text as a version segment, so
/// version and operation keys must never share one.
///
/// Parsing a stored key refuses anything this grammar does not produce with
/// [`HeuremaError::CorruptSnapshot`]: the key's bytes are present but do not
/// decode.
///
/// # Examples
///
/// ```
/// use heurema::lifecycle::storage_key;
/// use heurema::{IndexIdentity, IndexName, IndexVersion, OperationKey, OwnerNamespace};
///
/// let index = IndexIdentity::new(
///     OwnerNamespace::try_from("example")?,
///     IndexName::try_from("notes")?,
/// );
/// let version = IndexVersion::try_from(12)?;
/// assert_eq!(storage_key::head(&index), "example/notes");
/// assert_eq!(
///     storage_key::version(&index, version),
///     "example/notes/00000000000000000012"
/// );
/// let key = OperationKey::try_from("import-7")?;
/// assert_eq!(storage_key::operation(&index, &key), "example/notes/import-7");
/// assert_eq!(
///     storage_key::parse_version(storage_key::version(&index, version).as_bytes())?,
///     (index, version)
/// );
/// # Ok::<(), heurema::HeuremaError>(())
/// ```
pub mod storage_key {
    use std::fmt;

    use snafu::IntoError;

    use crate::error::CorruptSnapshotSnafu;
    use crate::{
        HeuremaError, IndexIdentity, IndexName, IndexVersion, OperationKey, OwnerNamespace,
        PersistenceSource,
    };

    /// WHY: `u64::MAX` has 20 decimal digits, so padding every number to 20
    /// makes byte order equal numeric order.
    const DECIMAL_WIDTH: usize = 20;

    /// WHY: a refused key is echoed into the error, and a stored key of any
    /// length must not make the error arbitrarily large.
    const REPORTED_KEY_CHARS: usize = 256;

    const SEPARATOR: char = '/';

    /// Which half of a quarantine entry a [`quarantine`] key names.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum QuarantinePart {
        /// The staging marker.
        Marker,
        /// The version payload.
        Payload,
    }

    impl QuarantinePart {
        const fn as_str(self) -> &'static str {
            match self {
                Self::Marker => "marker",
                Self::Payload => "payload",
            }
        }
    }

    /// The key of `index`'s head: `namespace/name`.
    #[must_use]
    pub fn head(index: &IndexIdentity) -> String {
        format!("{}{SEPARATOR}{}", index.namespace, index.name)
    }

    /// The prefix every version, staging, and operation key of `index`
    /// starts with, and no key of another index does: `namespace/name/`.
    #[must_use]
    pub fn index_prefix(index: &IndexIdentity) -> String {
        format!("{}{SEPARATOR}", head(index))
    }

    /// The key of `version`'s payload: `namespace/name/{version:020}`.
    #[must_use]
    pub fn version(index: &IndexIdentity, version: IndexVersion) -> String {
        format!(
            "{}{:0width$}",
            index_prefix(index),
            version.get(),
            width = DECIMAL_WIDTH
        )
    }

    /// The key of `version`'s staging marker. It spells the same text as
    /// [`version`]; the two live in different maps.
    #[must_use]
    pub fn staging(index: &IndexIdentity, version: IndexVersion) -> String {
        self::version(index, version)
    }

    /// The key of the record of operation `key`: `namespace/name/key`.
    #[must_use]
    pub fn operation(index: &IndexIdentity, key: &OperationKey) -> String {
        format!("{}{key}", index_prefix(index))
    }

    /// The key of one half of quarantine entry `sequence`:
    /// `{sequence:020}/namespace/name/{version:020}/part`.
    #[must_use]
    pub fn quarantine(
        sequence: u64,
        index: &IndexIdentity,
        version: IndexVersion,
        part: QuarantinePart,
    ) -> String {
        format!(
            "{sequence:0width$}{SEPARATOR}{}{SEPARATOR}{}",
            self::version(index, version),
            part.as_str(),
            width = DECIMAL_WIDTH
        )
    }

    /// The index a [`head`] key names.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::CorruptSnapshot`] when `key` is not a head key.
    #[track_caller]
    pub fn parse_head(key: &[u8]) -> Result<IndexIdentity, HeuremaError> {
        let parts = split(key)?;
        match parts.as_slice() {
            [namespace, name] => identity(key, namespace, name),
            _ => Err(malformed(key, "expected namespace/name")),
        }
    }

    /// The index and version a [`version`] key names.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::CorruptSnapshot`] when `key` is not a version key.
    #[track_caller]
    pub fn parse_version(key: &[u8]) -> Result<(IndexIdentity, IndexVersion), HeuremaError> {
        let parts = split(key)?;
        match parts.as_slice() {
            [namespace, name, version] => Ok((
                identity(key, namespace, name)?,
                index_version(key, version)?,
            )),
            _ => Err(malformed(key, "expected namespace/name/version")),
        }
    }

    /// The index and version a [`staging`] key names; the grammar is
    /// [`parse_version`]'s.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::CorruptSnapshot`] when `key` is not a staging key.
    #[track_caller]
    pub fn parse_staging(key: &[u8]) -> Result<(IndexIdentity, IndexVersion), HeuremaError> {
        parse_version(key)
    }

    /// The index and operation key an [`operation`] key names.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::CorruptSnapshot`] when `key` is not an operation key.
    #[track_caller]
    pub fn parse_operation(key: &[u8]) -> Result<(IndexIdentity, OperationKey), HeuremaError> {
        let parts = split(key)?;
        match parts.as_slice() {
            [namespace, name, operation] => Ok((
                identity(key, namespace, name)?,
                checked(key, OperationKey::try_from(*operation))?,
            )),
            _ => Err(malformed(key, "expected namespace/name/operation-key")),
        }
    }

    /// The sequence, index, version, and part a [`quarantine`] key names.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::CorruptSnapshot`] when `key` is not a quarantine key.
    #[track_caller]
    pub fn parse_quarantine(
        key: &[u8],
    ) -> Result<(u64, IndexIdentity, IndexVersion, QuarantinePart), HeuremaError> {
        let parts = split(key)?;
        let [sequence, namespace, name, version, part] = parts.as_slice() else {
            return Err(malformed(
                key,
                "expected sequence/namespace/name/version/part",
            ));
        };
        let part = if *part == QuarantinePart::Marker.as_str() {
            QuarantinePart::Marker
        } else if *part == QuarantinePart::Payload.as_str() {
            QuarantinePart::Payload
        } else {
            return Err(malformed(key, "part is neither marker nor payload"));
        };
        Ok((
            decimal(key, sequence)?,
            identity(key, namespace, name)?,
            index_version(key, version)?,
            part,
        ))
    }

    /// A stored key that does not follow the lifecycle key grammar.
    #[derive(Debug)]
    struct MalformedKey {
        key: String,
        reason: String,
    }

    impl fmt::Display for MalformedKey {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "lifecycle storage key {:?} does not follow the key grammar: {}",
                self.key, self.reason
            )
        }
    }

    impl std::error::Error for MalformedKey {}

    #[track_caller]
    fn malformed(key: &[u8], reason: impl Into<String>) -> HeuremaError {
        CorruptSnapshotSnafu.into_error(PersistenceSource::new(MalformedKey {
            key: String::from_utf8_lossy(key)
                .chars()
                .take(REPORTED_KEY_CHARS)
                .collect(),
            reason: reason.into(),
        }))
    }

    #[track_caller]
    fn split(key: &[u8]) -> Result<Vec<&str>, HeuremaError> {
        match std::str::from_utf8(key) {
            Ok(text) => Ok(text.split(SEPARATOR).collect()),
            Err(error) => Err(malformed(key, format!("is not UTF-8: {error}"))),
        }
    }

    /// Carry an identifier refusal over as a malformed stored key.
    #[track_caller]
    fn checked<T>(key: &[u8], parsed: Result<T, HeuremaError>) -> Result<T, HeuremaError> {
        match parsed {
            Ok(value) => Ok(value),
            Err(refusal) => Err(malformed(key, refusal.to_string())),
        }
    }

    #[track_caller]
    fn identity(key: &[u8], namespace: &str, name: &str) -> Result<IndexIdentity, HeuremaError> {
        Ok(IndexIdentity::new(
            checked(key, OwnerNamespace::try_from(namespace))?,
            checked(key, IndexName::try_from(name))?,
        ))
    }

    #[track_caller]
    fn decimal(key: &[u8], digits: &str) -> Result<u64, HeuremaError> {
        if digits.len() != DECIMAL_WIDTH || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(malformed(
                key,
                format!("{digits:?} is not {DECIMAL_WIDTH} ASCII digits"),
            ));
        }
        match digits.parse::<u64>() {
            Ok(value) => Ok(value),
            Err(error) => Err(malformed(key, format!("{digits:?}: {error}"))),
        }
    }

    #[track_caller]
    fn index_version(key: &[u8], digits: &str) -> Result<IndexVersion, HeuremaError> {
        checked(key, IndexVersion::try_from(decimal(key, digits)?))
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests need concise identifier fixtures")]
mod tests {
    use super::storage_key::{self, QuarantinePart};
    use super::*;
    use crate::{ErrorCategory, IndexName, OwnerNamespace};

    fn index(namespace: &str, name: &str) -> IndexIdentity {
        IndexIdentity::new(
            OwnerNamespace::try_from(namespace).expect("namespace"),
            IndexName::try_from(name).expect("name"),
        )
    }

    fn version(value: u64) -> IndexVersion {
        IndexVersion::try_from(value).expect("non-zero version")
    }

    #[test]
    fn storage_keys_follow_the_documented_grammar() {
        let notes = index("example", "notes");
        let key = OperationKey::try_from("urn:op:1").expect("key");
        assert_eq!(storage_key::head(&notes), "example/notes");
        assert_eq!(storage_key::index_prefix(&notes), "example/notes/");
        assert_eq!(
            storage_key::version(&notes, version(1)),
            "example/notes/00000000000000000001"
        );
        assert_eq!(
            storage_key::staging(&notes, version(1)),
            storage_key::version(&notes, version(1))
        );
        assert_eq!(
            storage_key::version(&notes, version(u64::MAX)),
            "example/notes/18446744073709551615"
        );
        assert_eq!(
            storage_key::operation(&notes, &key),
            "example/notes/urn:op:1"
        );
        assert_eq!(
            storage_key::quarantine(7, &notes, version(2), QuarantinePart::Marker),
            "00000000000000000007/example/notes/00000000000000000002/marker"
        );
        assert_eq!(
            storage_key::quarantine(7, &notes, version(2), QuarantinePart::Payload),
            "00000000000000000007/example/notes/00000000000000000002/payload"
        );
    }

    #[test]
    fn every_key_kind_parses_back_to_what_built_it() {
        let notes = index("fleet.search", "notes-v2");
        let key = OperationKey::try_from("01J8ZQ6D4X5K2M9N7P3R8T1V0W").expect("key");
        let v = version(42);
        assert_eq!(
            storage_key::parse_head(storage_key::head(&notes).as_bytes()).expect("head"),
            notes
        );
        assert_eq!(
            storage_key::parse_version(storage_key::version(&notes, v).as_bytes())
                .expect("version"),
            (notes.clone(), v)
        );
        assert_eq!(
            storage_key::parse_staging(storage_key::staging(&notes, v).as_bytes())
                .expect("staging"),
            (notes.clone(), v)
        );
        assert_eq!(
            storage_key::parse_operation(storage_key::operation(&notes, &key).as_bytes())
                .expect("operation"),
            (notes.clone(), key)
        );
        for part in [QuarantinePart::Marker, QuarantinePart::Payload] {
            assert_eq!(
                storage_key::parse_quarantine(
                    storage_key::quarantine(u64::MAX, &notes, v, part).as_bytes()
                )
                .expect("quarantine"),
                (u64::MAX, notes.clone(), v, part)
            );
        }
    }

    #[test]
    fn padded_numbers_sort_in_numeric_order() {
        let notes = index("example", "notes");
        let mut keys: Vec<String> = [10_u64, 2, u64::MAX, 1, 9]
            .into_iter()
            .map(|value| storage_key::version(&notes, version(value)))
            .collect();
        keys.sort();
        let parsed: Vec<u64> = keys
            .iter()
            .map(|key| {
                storage_key::parse_version(key.as_bytes())
                    .expect("version key")
                    .1
                    .get()
            })
            .collect();
        assert_eq!(parsed, [1, 2, 9, 10, u64::MAX]);

        let ninth_marker = storage_key::quarantine(9, &notes, version(1), QuarantinePart::Marker);
        let ninth_payload = storage_key::quarantine(9, &notes, version(1), QuarantinePart::Payload);
        let tenth_marker = storage_key::quarantine(10, &notes, version(1), QuarantinePart::Marker);
        let mut sequences = [
            tenth_marker.clone(),
            ninth_payload.clone(),
            ninth_marker.clone(),
        ];
        sequences.sort();
        assert_eq!(
            sequences,
            [ninth_marker, ninth_payload, tenth_marker],
            "by sequence, then a marker before its payload"
        );
    }

    #[test]
    fn an_index_prefix_matches_only_its_own_index() {
        let notes = index("example", "notes");
        let prefix = storage_key::index_prefix(&notes);
        assert!(storage_key::version(&notes, version(1)).starts_with(&prefix));
        for other in [
            index("example", "notes.v2"),
            index("example", "notes-x"),
            index("example", "notes_x"),
            index("example.b", "notes"),
            index("exampl", "enotes"),
        ] {
            let key = storage_key::version(&other, version(1));
            assert!(!key.starts_with(&prefix), "{key} must not match {prefix}");
            let key = storage_key::operation(&other, &OperationKey::try_from("k").expect("key"));
            assert!(!key.starts_with(&prefix), "{key} must not match {prefix}");
        }
    }

    #[test]
    fn a_digits_only_operation_key_spells_a_version_key() {
        // WHY: this is why version and operation keys live in separate maps;
        // sharing one would let an operation record overwrite a payload.
        let notes = index("example", "notes");
        let digits = OperationKey::try_from("00000000000000000001").expect("digits-only key");
        assert_eq!(
            storage_key::operation(&notes, &digits),
            storage_key::version(&notes, version(1))
        );
    }

    #[test]
    fn malformed_stored_keys_are_corrupt_not_storage_failures() {
        let heads: [&[u8]; 6] = [
            b"",
            b"example",
            b"example/notes/extra",
            b"Example/notes",
            b"example/no tes",
            b"example/\xff",
        ];
        for key in heads {
            assert_corrupt(storage_key::parse_head(key), key);
        }
        let versions: [&[u8]; 8] = [
            b"example/notes",
            b"example/notes/1",
            b"example/notes/0000000000000000001",
            b"example/notes/000000000000000000001",
            b"example/notes/00000000000000000000",
            b"example/notes/99999999999999999999",
            b"example/notes/0000000000000000000x",
            b"example/notes/+0000000000000000001",
        ];
        for key in versions {
            assert_corrupt(storage_key::parse_version(key), key);
            assert_corrupt(storage_key::parse_staging(key), key);
        }
        let operations: [&[u8]; 3] = [b"example/notes", b"example/notes/", b"example/notes/a b"];
        for key in operations {
            assert_corrupt(storage_key::parse_operation(key), key);
        }
        let quarantine: [&[u8]; 4] = [
            b"00000000000000000001/example/notes/00000000000000000001/manifest",
            b"00000000000000000001/example/notes/00000000000000000001",
            b"1/example/notes/00000000000000000001/marker",
            b"00000000000000000001/example/notes/00000000000000000000/marker",
        ];
        for key in quarantine {
            assert_corrupt(storage_key::parse_quarantine(key), key);
        }
    }

    fn assert_corrupt<T: std::fmt::Debug>(result: Result<T, HeuremaError>, key: &[u8]) {
        let error = result.expect_err("a malformed key is refused");
        assert!(
            matches!(error, HeuremaError::CorruptSnapshot { .. }),
            "{:?}: {error:?}",
            String::from_utf8_lossy(key)
        );
        assert_eq!(error.category(), ErrorCategory::Corrupt, "{error}");
        assert!(
            error.to_string().contains("lifecycle storage key"),
            "{error}"
        );
    }

    #[test]
    fn lifecycle_backend_is_object_safe() {
        fn borrowed(_backend: Option<&dyn LifecycleBackend>) {}
        fn shared(_backend: Option<Arc<dyn LifecycleBackend>>) {}
        borrowed(None);
        shared(None);
    }
}
