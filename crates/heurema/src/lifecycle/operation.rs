//! Lifecycle operations: what a caller asks to change about a named index.

use serde::{Deserialize, Serialize};

use super::identity::{IndexIdentity, OperationKey};
use super::member::{IndexMember, MemberIdentity, ProvenanceReference, RetentionReference};
use super::record::IndexStateKind;
use crate::{FtsConfig, HnswConfig, SnapshotFamily};

/// The configuration a named index is created or rebuilt with.
///
/// WHY: the variant fixes the index's family, and the family decides which
/// member content the index accepts; reusing the engines' own config types
/// keeps one definition of each engine's parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IndexConfig {
    /// An HNSW vector index.
    Vector(HnswConfig),
    /// A BM25 full-text index.
    Fts(FtsConfig),
}

impl IndexConfig {
    /// The index family this configuration builds.
    #[must_use]
    pub const fn family(&self) -> SnapshotFamily {
        match self {
            Self::Vector(_) => SnapshotFamily::Vector,
            Self::Fts(_) => SnapshotFamily::Fts,
        }
    }
}

/// Every step that moves a named index between lifecycle states.
///
/// `Create`, `Insert`, `Remove`, `Rebuild`, and `Destroy` are requested by a
/// caller through an [`IndexChange`]. `Publish` and `Recover` are steps the
/// lifecycle itself takes: `Publish` makes a staged operation visible, and
/// `Recover` repairs interrupted state on reopen. No `IndexChange` names
/// either of them.
///
/// WHY: the permission table and error reports name transitions, including
/// the two no caller can request, so they share one vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LifecycleTransition {
    /// Create an absent index at version 1.
    Create,
    /// Insert members, replacing any member with the same identity.
    Insert,
    /// Remove members; removing an absent member is a no-op.
    Remove,
    /// Replace the configuration and the whole member set, keeping the family.
    Rebuild,
    /// Make a staged operation visible at one point.
    Publish,
    /// Repair interrupted state after reopen.
    Recover,
    /// Destroy an active index under a retention reference.
    Destroy,
}

impl LifecycleTransition {
    /// Whether this transition is permitted on an index in `state`.
    ///
    /// | transition | Absent | Active | Destroyed |
    /// |---|---|---|---|
    /// | Create | yes | no | no |
    /// | Insert | no | yes | no |
    /// | Remove | no | yes | no |
    /// | Rebuild | no | yes | no |
    /// | Destroy | no | yes | no |
    /// | Publish | yes | yes | no |
    /// | Recover | yes | yes | yes |
    ///
    /// A destroyed identity is never created again, because reusing it would
    /// splice two audit histories into one.
    ///
    /// WHY exhaustive with no wildcard: a new transition or state does not
    /// compile until every cell of its row or column is decided.
    #[must_use]
    pub const fn is_permitted_from(self, state: IndexStateKind) -> bool {
        match (self, state) {
            (Self::Create, IndexStateKind::Absent)
            | (
                Self::Insert | Self::Remove | Self::Rebuild | Self::Destroy,
                IndexStateKind::Active,
            )
            | (Self::Publish, IndexStateKind::Absent | IndexStateKind::Active)
            | (
                Self::Recover,
                IndexStateKind::Absent | IndexStateKind::Active | IndexStateKind::Destroyed,
            ) => true,
            (Self::Create, IndexStateKind::Active | IndexStateKind::Destroyed)
            | (
                Self::Insert | Self::Remove | Self::Rebuild | Self::Destroy,
                IndexStateKind::Absent | IndexStateKind::Destroyed,
            )
            | (Self::Publish, IndexStateKind::Destroyed) => false,
        }
    }
}

/// The change one operation requests.
///
/// WHY: each variant carries exactly what its transition needs, so a request
/// that lacks a required part does not type-check. Destroy carries a
/// retention reference because physical deletion is authorized only by a
/// retention decision.
///
/// # Examples
///
/// ```
/// # use heurema::{
/// #     IndexChange, IndexIdentity, IndexName, LifecycleOperation, LifecycleTransition,
/// #     MemberIdentity, OperationKey, OwnerNamespace, ProvenanceReference, RetentionReference,
/// # };
/// # use serde::{Deserialize, Serialize};
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// # struct TestMember(u64);
/// # impl MemberIdentity for TestMember {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderProvenance(u32);
/// # impl ProvenanceReference for PlaceholderProvenance {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderRetention(u32);
/// # impl RetentionReference for PlaceholderRetention {}
/// # let index = IndexIdentity::new(OwnerNamespace::try_from("example")?, IndexName::try_from("notes")?);
/// # let key = OperationKey::try_from("destroy-notes")?;
/// let destroy = LifecycleOperation::<TestMember, PlaceholderProvenance, PlaceholderRetention>::new(
///     index,
///     key,
///     IndexChange::Destroy { retention: PlaceholderRetention(3) },
/// );
/// assert_eq!(destroy.transition(), LifecycleTransition::Destroy);
/// # Ok::<(), heurema::HeuremaError>(())
/// ```
///
/// A destroy without a retention reference does not compile. This is the
/// example above with the retention field removed:
///
/// ```compile_fail
/// # use heurema::{
/// #     IndexChange, IndexIdentity, IndexName, LifecycleOperation, LifecycleTransition,
/// #     MemberIdentity, OperationKey, OwnerNamespace, ProvenanceReference, RetentionReference,
/// # };
/// # use serde::{Deserialize, Serialize};
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// # struct TestMember(u64);
/// # impl MemberIdentity for TestMember {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderProvenance(u32);
/// # impl ProvenanceReference for PlaceholderProvenance {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderRetention(u32);
/// # impl RetentionReference for PlaceholderRetention {}
/// # let index = IndexIdentity::new(OwnerNamespace::try_from("example")?, IndexName::try_from("notes")?);
/// # let key = OperationKey::try_from("destroy-notes")?;
/// let destroy = LifecycleOperation::<TestMember, PlaceholderProvenance, PlaceholderRetention>::new(
///     index,
///     key,
///     IndexChange::Destroy {},
/// );
/// assert_eq!(destroy.transition(), LifecycleTransition::Destroy);
/// # Ok::<(), heurema::HeuremaError>(())
/// ```
///
/// Nor does `None` for the retention reference: it is required, not optional.
/// This is the first example with the reference replaced by `None`; it starts
/// compiling if the field ever becomes an `Option`:
///
/// ```compile_fail
/// # use heurema::{
/// #     IndexChange, IndexIdentity, IndexName, LifecycleOperation, LifecycleTransition,
/// #     MemberIdentity, OperationKey, OwnerNamespace, ProvenanceReference, RetentionReference,
/// # };
/// # use serde::{Deserialize, Serialize};
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// # struct TestMember(u64);
/// # impl MemberIdentity for TestMember {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderProvenance(u32);
/// # impl ProvenanceReference for PlaceholderProvenance {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderRetention(u32);
/// # impl RetentionReference for PlaceholderRetention {}
/// # let index = IndexIdentity::new(OwnerNamespace::try_from("example")?, IndexName::try_from("notes")?);
/// # let key = OperationKey::try_from("destroy-notes")?;
/// let destroy = LifecycleOperation::<TestMember, PlaceholderProvenance, PlaceholderRetention>::new(
///     index,
///     key,
///     IndexChange::Destroy { retention: None },
/// );
/// assert_eq!(destroy.transition(), LifecycleTransition::Destroy);
/// # Ok::<(), heurema::HeuremaError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        serialize = "M: Serialize, P: Serialize, R: Serialize",
        deserialize = "M: MemberIdentity, P: ProvenanceReference, R: RetentionReference"
    )
)]
#[non_exhaustive]
pub enum IndexChange<M, P, R> {
    /// Create an absent index.
    Create {
        /// The new index's configuration; its variant fixes the family.
        config: IndexConfig,
    },
    /// Insert members, replacing any member with the same identity.
    Insert {
        /// The members to insert or replace.
        members: Vec<IndexMember<M, P>>,
    },
    /// Remove members by identity; an absent identity is a no-op.
    Remove {
        /// The identities to remove.
        members: Vec<M>,
    },
    /// Rebuild the index under a new configuration of the same family.
    Rebuild {
        /// The replacement configuration.
        config: IndexConfig,
        /// The complete member set of the rebuilt index.
        members: Vec<IndexMember<M, P>>,
    },
    /// Destroy the index.
    Destroy {
        /// The retention decision that authorizes the deletion.
        retention: R,
    },
}

impl<M, P, R> IndexChange<M, P, R> {
    /// The transition this change requests.
    #[must_use]
    pub const fn transition(&self) -> LifecycleTransition {
        match self {
            Self::Create { .. } => LifecycleTransition::Create,
            Self::Insert { .. } => LifecycleTransition::Insert,
            Self::Remove { .. } => LifecycleTransition::Remove,
            Self::Rebuild { .. } => LifecycleTransition::Rebuild,
            Self::Destroy { .. } => LifecycleTransition::Destroy,
        }
    }
}

/// One requested lifecycle operation: which index, under which idempotency
/// key, and what change.
///
/// WHY: the index, the key, and the change are the whole of what a caller
/// asks for, so they form one value that later validation, digesting, and
/// staging take as a unit. Construction outside this crate goes through
/// [`LifecycleOperation::new`] or deserialization, and both require the
/// marker traits, so `()` or a bare `String` cannot stand in for a member
/// identity, provenance, or retention reference. The lifecycle module's
/// documentation builds one end to end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        serialize = "M: Serialize, P: Serialize, R: Serialize",
        deserialize = "M: MemberIdentity, P: ProvenanceReference, R: RetentionReference"
    )
)]
#[non_exhaustive]
pub struct LifecycleOperation<M, P, R> {
    /// The index the operation targets.
    pub index: IndexIdentity,
    /// The caller's idempotency key.
    pub key: OperationKey,
    /// The requested change.
    pub change: IndexChange<M, P, R>,
}

impl<M: MemberIdentity, P: ProvenanceReference, R: RetentionReference> LifecycleOperation<M, P, R> {
    /// WHY: the marker bounds live on the constructor, so an operation cannot
    /// be built over types that were never declared identities, provenance,
    /// or retention references.
    #[must_use]
    pub const fn new(
        index: IndexIdentity,
        key: OperationKey,
        change: IndexChange<M, P, R>,
    ) -> Self {
        Self { index, key, change }
    }

    /// The transition this operation requests.
    #[must_use]
    pub const fn transition(&self) -> LifecycleTransition {
        self.change.transition()
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "tests need concise operation assertions"
)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::lifecycle::MemberContent;
    use crate::lifecycle::test_placeholders::{
        PlaceholderProvenance, PlaceholderRetention, TestMember,
    };
    use crate::{IndexName, OwnerNamespace};

    type Change = IndexChange<TestMember, PlaceholderProvenance, PlaceholderRetention>;

    fn member(id: u64) -> IndexMember<TestMember, PlaceholderProvenance> {
        IndexMember::new(
            TestMember(id),
            PlaceholderProvenance(7),
            MemberContent::Vector(vec![0.0, 1.0]),
        )
    }

    #[test]
    fn every_index_change_names_exactly_one_transition() {
        let index = IndexIdentity::new(
            OwnerNamespace::try_from("example").expect("namespace"),
            IndexName::try_from("notes").expect("name"),
        );
        let key = OperationKey::try_from("op-1").expect("key");
        let config = IndexConfig::Vector(HnswConfig::new(2));
        let cases: [(Change, LifecycleTransition); 5] = [
            (
                IndexChange::Create {
                    config: config.clone(),
                },
                LifecycleTransition::Create,
            ),
            (
                IndexChange::Insert {
                    members: vec![member(1)],
                },
                LifecycleTransition::Insert,
            ),
            (
                IndexChange::Remove {
                    members: vec![TestMember(1)],
                },
                LifecycleTransition::Remove,
            ),
            (
                IndexChange::Rebuild {
                    config,
                    members: vec![member(2)],
                },
                LifecycleTransition::Rebuild,
            ),
            (
                IndexChange::Destroy {
                    retention: PlaceholderRetention(3),
                },
                LifecycleTransition::Destroy,
            ),
        ];

        let mut named = HashSet::new();
        for (change, expected) in cases {
            assert_eq!(change.transition(), expected, "{change:?}");
            assert!(named.insert(expected), "two changes name {expected:?}");
            let operation = LifecycleOperation::new(index.clone(), key.clone(), change);
            assert_eq!(operation.transition(), expected);
        }
        assert!(
            !named.contains(&LifecycleTransition::Publish)
                && !named.contains(&LifecycleTransition::Recover),
            "publish and recover are lifecycle steps, never a caller's change"
        );
    }

    #[test]
    fn member_content_and_index_config_name_their_snapshot_family() {
        assert_eq!(
            MemberContent::Vector(vec![1.0]).family(),
            SnapshotFamily::Vector
        );
        assert_eq!(
            MemberContent::Document("text".to_owned()).family(),
            SnapshotFamily::Fts
        );
        assert_eq!(
            IndexConfig::Vector(HnswConfig::new(3)).family(),
            SnapshotFamily::Vector
        );
        assert_eq!(
            IndexConfig::Fts(FtsConfig::simple()).family(),
            SnapshotFamily::Fts
        );
    }

    #[test]
    fn lifecycle_operation_round_trips_through_serde() {
        let operation = LifecycleOperation::new(
            IndexIdentity::new(
                OwnerNamespace::try_from("example").expect("namespace"),
                IndexName::try_from("notes").expect("name"),
            ),
            OperationKey::try_from("urn:example:op-2").expect("key"),
            Change::Insert {
                members: vec![member(4)],
            },
        );
        let encoded = serde_json::to_string(&operation).expect("operation encodes");
        let decoded: LifecycleOperation<TestMember, PlaceholderProvenance, PlaceholderRetention> =
            serde_json::from_str(&encoded).expect("operation decodes");
        assert_eq!(decoded, operation);

        let with_unknown_field = encoded.replacen('{', r#"{"extra":1,"#, 1);
        assert!(
            serde_json::from_str::<
                LifecycleOperation<TestMember, PlaceholderProvenance, PlaceholderRetention>,
            >(&with_unknown_field)
            .is_err(),
            "unknown fields are refused: {with_unknown_field}"
        );
    }
}
