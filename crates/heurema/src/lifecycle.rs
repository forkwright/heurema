//! The durable retrieval lifecycle's vocabulary: which index an operation
//! names, what it asks to change, and which state the index is in.
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
//! This module states the table; it does not enforce it. Enforcement
//! arrives with pre-publish validation, which refuses a disallowed
//! transition before any write.
//!
//! # Consumer types
//!
//! Member identity, provenance, and retention are the consumer's own types,
//! bound by the [`MemberIdentity`], [`ProvenanceReference`], and
//! [`RetentionReference`] marker traits. heurēma defines no provenance or
//! retention shape and never interprets either value.
//!
//! # Limits
//!
//! Snapshots are not transactions; this module defines the durable
//! lifecycle's vocabulary, not its storage. Nothing here reads or writes a
//! backend, checks a transition against a state, or computes an
//! [`OperationDigest`].
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

mod identity;
mod member;
mod operation;
mod record;

pub use identity::{
    IdentifierKind, IndexIdentity, IndexName, IndexVersion, OperationDigest, OperationIdentity,
    OperationKey, OwnerNamespace,
};
pub use member::{
    IndexMember, MemberContent, MemberIdentity, ProvenanceReference, RetentionReference,
};
pub use operation::{IndexChange, IndexConfig, LifecycleOperation, LifecycleTransition};
pub use record::{IndexRecord, IndexState, IndexStateKind};

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
