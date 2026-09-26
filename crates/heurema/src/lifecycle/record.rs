//! The named index record: which index, how it is configured, and which state
//! it is in.

use serde::{Deserialize, Serialize};

use super::identity::{IndexIdentity, IndexVersion, OperationKey};
use super::member::RetentionReference;
use super::operation::IndexConfig;

/// The three lifecycle states of a named index, without their data.
///
/// WHY: the permission table and refusals are stated per state, and
/// `Absent` has no record to carry data, so the discriminant stands alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IndexStateKind {
    /// No record exists for the identity.
    Absent,
    /// A published version is current.
    Active,
    /// The index was destroyed; only its audit record remains.
    Destroyed,
}

/// The state a recorded index is in.
///
/// There is no `Absent` variant: an absent index has no record at all.
///
/// WHY: a destroyed index keeps the last version it published, the retention
/// reference that authorized the deletion, and the key of the operation that
/// performed it, so the deletion stays explainable after its data is gone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(serialize = "R: Serialize", deserialize = "R: RetentionReference")
)]
#[non_exhaustive]
pub enum IndexState<R> {
    /// A published version is current.
    Active {
        /// The current published version.
        version: IndexVersion,
    },
    /// The index was destroyed.
    Destroyed {
        /// The last version published before the destroy.
        last_version: IndexVersion,
        /// The retention decision that authorized the deletion.
        retention: R,
        /// The key of the operation that destroyed the index.
        operation: OperationKey,
    },
}

impl<R> IndexState<R> {
    /// The state's discriminant.
    #[must_use]
    pub const fn kind(&self) -> IndexStateKind {
        match self {
            Self::Active { .. } => IndexStateKind::Active,
            Self::Destroyed { .. } => IndexStateKind::Destroyed,
        }
    }
}

/// The named index record: the head that names an index's identity, its
/// configuration (and so its family), and its current state.
///
/// Per-member identity and provenance are not in the head. Each published
/// version's immutable payload carries its own member table, mapping every
/// member's identity to its provenance, and the head names the version whose
/// payload is current. The lifecycle contract's "named index record" (index
/// identity, kind, member identities, provenance, active version or deletion
/// state) is therefore this head together with the payload of the version it
/// names. For a destroyed index, whose version payloads are removed under a
/// retention reference, the last member identities and provenance are kept in
/// the destroy operation's record instead.
/// [`IndexLifecycle::index`](crate::IndexLifecycle::index) reads the head and
/// then that payload, as a [`PublishedIndex`](crate::PublishedIndex).
///
/// WHY: members and their provenance belong to a version, not to the index.
/// A published version never changes, so its member table is fixed with it,
/// and the head only names which version is current. Publishing a version
/// then changes one small record instead of a member table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(serialize = "R: Serialize", deserialize = "R: RetentionReference")
)]
#[non_exhaustive]
pub struct IndexRecord<R> {
    /// The index this record describes.
    pub identity: IndexIdentity,
    /// The configuration of the current (or, once destroyed, last) version.
    pub config: IndexConfig,
    /// The index's lifecycle state.
    pub state: IndexState<R>,
}

impl<R: RetentionReference> IndexRecord<R> {
    /// WHY: the retention bound lives on the constructor, so a record cannot
    /// carry a type that was never declared a retention reference.
    #[must_use]
    pub const fn new(identity: IndexIdentity, config: IndexConfig, state: IndexState<R>) -> Self {
        Self {
            identity,
            config,
            state,
        }
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests need concise record assertions")]
mod tests {
    use super::*;
    use crate::lifecycle::test_placeholders::PlaceholderRetention;
    use crate::{FtsConfig, IndexName, OwnerNamespace};

    #[test]
    fn index_state_kind_names_the_variant() {
        let active = IndexState::<PlaceholderRetention>::Active {
            version: IndexVersion::FIRST,
        };
        assert_eq!(active.kind(), IndexStateKind::Active);

        let destroyed = IndexState::Destroyed {
            last_version: IndexVersion::try_from(3).expect("version"),
            retention: PlaceholderRetention(9),
            operation: OperationKey::try_from("destroy-1").expect("key"),
        };
        assert_eq!(destroyed.kind(), IndexStateKind::Destroyed);
    }

    #[test]
    fn destroyed_record_round_trips_with_its_retention_reference() {
        let record = IndexRecord::new(
            IndexIdentity::new(
                OwnerNamespace::try_from("example").expect("namespace"),
                IndexName::try_from("notes").expect("name"),
            ),
            IndexConfig::Fts(FtsConfig::simple()),
            IndexState::Destroyed {
                last_version: IndexVersion::try_from(4).expect("version"),
                retention: PlaceholderRetention(9),
                operation: OperationKey::try_from("destroy-1").expect("key"),
            },
        );
        let encoded = serde_json::to_string(&record).expect("record encodes");
        let decoded: IndexRecord<PlaceholderRetention> =
            serde_json::from_str(&encoded).expect("record decodes");
        assert_eq!(decoded, record);
        assert_eq!(decoded.state.kind(), IndexStateKind::Destroyed);

        let zero_version = encoded.replace(r#""last_version":4"#, r#""last_version":0"#);
        assert_ne!(zero_version, encoded, "the replacement must hit the field");
        assert!(
            serde_json::from_str::<IndexRecord<PlaceholderRetention>>(&zero_version).is_err(),
            "a zero version is refused on decode"
        );
    }
}
