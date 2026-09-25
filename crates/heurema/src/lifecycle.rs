//! The durable retrieval lifecycle: its vocabulary (which index an operation
//! names, what it asks to change, and which state the index is in), the
//! pre-publish validation that refuses an invalid operation before any write,
//! and the driver, [`IndexLifecycle`], that stages each operation's version
//! and publishes it at one atomic point.
//!
//! WHY: the lifecycle contract has to be readable from types before any
//! durable write depends on it. A consumer can tell from these types alone
//! which index an operation targets ([`IndexIdentity`]), which transition it
//! requests ([`LifecycleOperation::transition`]), and, through
//! [`HeuremaError::category`](crate::HeuremaError::category), which class of
//! failure it met.
//!
//! # States
//!
//! A named index is in one of three states ([`IndexStateKind`]):
//!
//! - **Absent**: no record exists for the identity.
//! - **Active { version }**: a published [`IndexVersion`] is current.
//!   Create publishes version 1, and each later mutation publishes the
//!   successor of the active version.
//! - **Destroyed { last_version, retention, operation }**: the index was
//!   destroyed under a retention reference; the record keeps the last
//!   version, that reference, and the destroying operation's key.
//!
//! A published version is immutable. A later operation publishes a new
//! version; it never rewrites one that was already published.
//!
//! # Permitted transitions
//!
//! - **Create** is permitted only from Absent, and publishes version 1. A
//!   destroyed identity is never created again, because reusing it would
//!   splice two audit histories into one.
//! - **Insert** (insert or replace), **Remove** (removing an absent member
//!   is a no-op), and **Rebuild** (a new configuration of the same family)
//!   are permitted only from Active, and each publishes the successor
//!   version.
//! - **Destroy** is permitted only from Active, moves the index to
//!   Destroyed, and requires a retention reference.
//! - **Publish**, the step that makes a staged operation visible, is
//!   permitted from Absent or Active, never from Destroyed.
//! - **Recover**, the step that repairs interrupted state after reopen, is
//!   permitted from every state.
//!
//! [`LifecycleTransition::is_permitted_from`] encodes this table, and
//! [`CheckedOperation::permit`] enforces it.
//!
//! # Validation
//!
//! An operation is validated in two pure stages, neither of which reads or
//! writes a backend:
//!
//! 1. [`CheckedOperation::check`] runs every check that needs no index state
//!    (empty or duplicate batches, member identity shape, non-finite
//!    vectors, configuration validity, family and dimension agreement within
//!    the operation, and whether every consumer value reads back from its
//!    JSON encoding) and computes the operation's [`OperationIdentity`]: its
//!    key plus the [`OperationDigest`] of its canonical encoding.
//! 2. [`CheckedOperation::permit`] takes the index's current record (`None`
//!    when absent) and refuses a transition the table forbids, an Insert the
//!    current configuration cannot hold, or a Rebuild that changes the
//!    index's family. It yields a [`ValidatedOperation`], the only form of
//!    an operation later lifecycle steps accept.
//!
//! # Consumer types
//!
//! Member identity, provenance, and retention are the consumer's own types,
//! bound by the [`MemberIdentity`], [`ProvenanceReference`], and
//! [`RetentionReference`] marker traits. heurēma defines no provenance or
//! retention shape and never interprets either value.
//!
//! # Storage
//!
//! [`LifecycleBackend`] is the storage contract the lifecycle runs on: five
//! maps of opaque heurēma-encoded bytes under one
//! [`storage_key`](crate::lifecycle::storage_key) grammar, and four writes
//! (stage, publish, destroy, quarantine), each atomic and durable before it
//! returns. The backend enforces only structural rules (compare-and-set on
//! the head, never overwriting, refusing while staged state exists); it
//! decodes nothing. heurēma owns every encoding: a head names the index's
//! [`IndexRecord`], a version payload holds one version's engine and its
//! member table ([`MemberEntry`] per member: provenance, introducing
//! version, superseded version), a staging marker names the operation that
//! staged a version, and an operation record keeps a published operation's
//! outcome and every member it changed ([`MemberChange`]).
//!
//! `atmis` and `thesauros` implement it. `atmis` applies each write under
//! one mutex over all five maps; its quarantine is a `Vec` in sequence
//! order, and the other four are keyed by `storage_key`. `thesauros`
//! commits each write as one fjall write batch with `PersistMode::SyncAll`,
//! which fjall journals as one checksummed unit, fsyncs, and replays only
//! whole on reopen, so a crash leaves all of a write or none of it. That
//! journal property is relied on, not simulated: the tests stop an
//! operation between writes, never inside one.
//!
//! `PersistenceBackend` and its whole-index snapshots are unchanged and
//! separate: a snapshot is a single save, not a lifecycle version, and the
//! two never share keys.
//!
//! # Publishing and reading
//!
//! [`IndexLifecycle`] runs every operation in one order: stateless checks,
//! the backend's writer, head read, replay lookup by key and digest, the
//! active version's payload, permission, staged-state check, an in-memory
//! build of the successor version, stage, publish. The checks write
//! nothing, so a refused operation leaves storage as it was, and a
//! stateless refusal makes no backend call at all.
//!
//! Every lifecycle over one backend shares the backend's [`WriterLock`]
//! ([`LifecycleBackend::writer`]), held from before the head read until the
//! operation publishes or is dropped. Another thread's operation waits for
//! it; a thread that already holds it is refused with
//! [`HeuremaError::WriterHeld`](crate::HeuremaError::WriterHeld) instead of
//! waiting for itself. Among writers that hold it for their whole operation,
//! as every lifecycle does, a staged version that one meets is therefore
//! interrupted state, never another operation's live stage. Code that
//! writes through the backend directly and bypasses the writer, or holds it
//! only around each call, can meet or leave a live stage that is reported
//! the same way.
//!
//! The publish point is one atomic backend write. Before it, readers see
//! the old version; after it, they see the new head, the operation record,
//! and the staging marker's removal together. The version payload was
//! already durable from the stage but unreachable, because a reader resolves
//! the head first and then reads only the immutable payload the head names.
//! A reader therefore sees all of one version or all of the next, and never
//! a staged one. No version is published without its member identities and
//! provenance: the payload decoder refuses an engine whose members differ
//! from its member table.
//!
//! The same key with the same digest replays the recorded outcome and writes
//! nothing; the same key with another digest is refused with
//! [`HeuremaError::OperationConflict`](crate::HeuremaError::OperationConflict).
//! Only published operations are recorded, so the key of a refused or
//! interrupted operation stays free. One refused before it staged anything
//! can be retried at once. One interrupted after its stage can be retried
//! once recovery, which a later Phase 02 change adds, has moved its staged
//! state to quarantine.
//!
//! A version staged but never published is orphan staged state. It refuses
//! every mutation of its index, Create and Destroy included, with
//! [`HeuremaError::StagedStateExists`](crate::HeuremaError::StagedStateExists)
//! until recovery moves it to quarantine; other indexes are unaffected.
//! Recovery on open lands in a later Phase 02 change; nothing clears orphan
//! staged state before then, and nothing ever deletes it.
//!
//! # Limits
//!
//! - Every published version payload is retained until the index is
//!   destroyed, so storage grows with every operation. Pruning superseded
//!   payloads needs a retention decision and is not implemented.
//! - Each operation decodes the active version's payload and applies its
//!   change to that copy, so its cost grows with the index's size.
//! - Destroy removes every version payload in its one atomic write, and
//!   keeps the head and every operation record, so the destroyed index stays
//!   explainable.
//! - `thesauros` stores a lifecycle value of at most 4 GiB. A version
//!   payload holds one index's whole engine and member table as JSON, so a
//!   large enough index is refused with
//!   [`HeuremaError::Persistence`](crate::HeuremaError::Persistence) before
//!   anything is written.
//!
//! # Example
//!
//! ```
//! use heurema::{
//!     HnswConfig, IndexChange, IndexConfig, IndexIdentity, IndexMember, IndexName,
//!     LifecycleOperation, LifecycleTransition, MemberContent, MemberIdentity, OperationKey,
//!     OwnerNamespace, ProvenanceReference, RetentionReference,
//! };
//! use serde::{Deserialize, Serialize};
//!
//! /// test-local placeholder; heurēma defines no provenance shape
//! #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
//! struct TestMember(u64);
//! impl MemberIdentity for TestMember {}
//!
//! /// test-local placeholder; heurēma defines no provenance shape
//! #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
//! struct PlaceholderProvenance(u32);
//! impl ProvenanceReference for PlaceholderProvenance {}
//!
//! /// test-local placeholder; heurēma defines no provenance shape
//! #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
//! struct PlaceholderRetention(u32);
//! impl RetentionReference for PlaceholderRetention {}
//!
//! let index = IndexIdentity::new(
//!     OwnerNamespace::try_from("example")?,
//!     IndexName::try_from("notes")?,
//! );
//! assert_eq!(index.to_string(), "example/notes");
//!
//! let create = LifecycleOperation::<TestMember, PlaceholderProvenance, PlaceholderRetention>::new(
//!     index.clone(),
//!     OperationKey::try_from("01J8ZQ6D4X5K2M9N7P3R8T1V0W")?,
//!     IndexChange::Create {
//!         config: IndexConfig::Vector(HnswConfig::new(2)),
//!     },
//! );
//! assert_eq!(create.transition(), LifecycleTransition::Create);
//!
//! let insert = LifecycleOperation::<_, _, PlaceholderRetention>::new(
//!     index,
//!     OperationKey::try_from("urn:example:insert-1")?,
//!     IndexChange::Insert {
//!         members: vec![IndexMember::new(
//!             TestMember(1),
//!             PlaceholderProvenance(7),
//!             MemberContent::Vector(vec![0.0, 1.0]),
//!         )],
//!     },
//! );
//! assert_eq!(insert.transition(), LifecycleTransition::Insert);
//! # Ok::<(), heurema::HeuremaError>(())
//! ```

mod apply;
mod backend;
mod digest;
mod driver;
mod encoding;
mod identity;
mod member;
mod operation;
mod published;
mod record;
mod validate;
mod writer;

pub use backend::{
    DestroyWrite, LIFECYCLE_FORMAT_VERSION, LifecycleBackend, PublishWrite, QuarantineWrite,
    QuarantinedEntry, StageWrite, StagedEntry, storage_key,
};

pub use identity::{
    IdentifierKind, IndexIdentity, IndexName, IndexVersion, OperationDigest, OperationIdentity,
    OperationKey, OwnerNamespace,
};
pub use member::{
    IndexMember, MemberContent, MemberIdentity, ProvenanceReference, RetentionReference,
};
pub use operation::{IndexChange, IndexConfig, LifecycleOperation, LifecycleTransition};
pub use record::{IndexRecord, IndexState, IndexStateKind};
pub use validate::{CheckedOperation, ValidatedOperation};
pub use writer::{WriterGuard, WriterLock};

pub use apply::MemberChange;
pub use driver::{IndexLifecycle, Preparation, Prepared, PublishReceipt, Staged};
pub use published::{IndexHit, MemberEntry, PublishedIndex};

/// Opaque stand-ins for consumer types, shared by this module's unit tests.
#[cfg(test)]
mod test_placeholders {
    use serde::{Deserialize, Serialize};

    use super::{MemberIdentity, ProvenanceReference, RetentionReference};

    /// test-local placeholder; heurēma defines no provenance shape
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    pub(crate) struct TestMember(pub(crate) u64);

    impl MemberIdentity for TestMember {}

    /// test-local placeholder; heurēma defines no provenance shape
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub(crate) struct PlaceholderProvenance(pub(crate) u32);

    impl ProvenanceReference for PlaceholderProvenance {}

    /// test-local placeholder; heurēma defines no provenance shape
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub(crate) struct PlaceholderRetention(pub(crate) u32);

    impl RetentionReference for PlaceholderRetention {}
}
