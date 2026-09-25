//! Building what an operation publishes, in memory, before anything is
//! staged: the successor version's engine and member table, the new head,
//! and the operation's audit record.
//!
//! WHY in memory first: every engine refusal happens here, so an operation
//! the engine cannot apply is refused with nothing written.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::encoding::{OperationRecord, VersionEngine, VersionPayload};
use super::identity::{IndexVersion, OperationIdentity};
use super::member::{IndexMember, MemberIdentity, ProvenanceReference, RetentionReference};
use super::operation::{IndexChange, IndexConfig, LifecycleOperation, LifecycleTransition};
use super::published::MemberEntry;
use super::record::{IndexRecord, IndexState, IndexStateKind};
use super::validate::{ValidatedOperation, sorted_members};
use crate::HeuremaError;
use crate::error::TransitionNotPermittedSnafu;

/// How one operation changed one member, with the entry it displaced.
///
/// An operation's audit record lists one change per member it touched, in
/// ascending identity order. Every variant that displaces an entry carries
/// that entry whole (provenance, introducing version, and the version it
/// superseded), so the transition stays explainable after the version that
/// held the entry is gone. The read API that returns these records lands in
/// a later Phase 02 change; the record format carries them now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        serialize = "M: Serialize, P: Serialize",
        deserialize = "M: MemberIdentity, P: ProvenanceReference"
    )
)]
#[non_exhaustive]
pub enum MemberChange<M, P> {
    /// An Insert or Rebuild added a member that was not in the version it
    /// started from.
    Inserted {
        /// The member's identity.
        id: M,
    },
    /// An Insert or Rebuild replaced an existing member; the new entry's
    /// `supersedes` names `previous.introduced`.
    Replaced {
        /// The member's identity.
        id: M,
        /// The entry that was replaced.
        previous: MemberEntry<P>,
    },
    /// A Remove removed a member.
    Removed {
        /// The member's identity.
        id: M,
        /// The entry that was removed.
        previous: MemberEntry<P>,
    },
    /// A Remove named a member that was not in the index; nothing changed
    /// for it.
    AbsentOnRemove {
        /// The identity the Remove named.
        id: M,
    },
    /// A Rebuild left out a member of the version it started from.
    Dropped {
        /// The member's identity.
        id: M,
        /// The entry the rebuild left out.
        previous: MemberEntry<P>,
    },
    /// A Destroy retired a member of the index's last version.
    Retired {
        /// The member's identity.
        id: M,
        /// The member's last entry.
        previous: MemberEntry<P>,
    },
}

impl<M, P> MemberChange<M, P> {
    /// The identity of the member this change concerns.
    #[must_use]
    pub const fn id(&self) -> &M {
        match self {
            Self::Inserted { id }
            | Self::Replaced { id, .. }
            | Self::Removed { id, .. }
            | Self::AbsentOnRemove { id }
            | Self::Dropped { id, .. }
            | Self::Retired { id, .. } => id,
        }
    }
}

/// What publishing one validated operation writes.
pub(super) struct Successor<M, P, R> {
    /// The index's new head.
    pub(super) head: IndexRecord<R>,
    /// The operation's audit record.
    pub(super) record: OperationRecord<M, P, R>,
    /// The version payload to stage; `None` for Destroy, which publishes no
    /// version.
    pub(super) payload: Option<VersionPayload<M, P>>,
}

/// The state the index is in, as the operation was permitted from it: its
/// head record and the payload of its active version, or `None` when the
/// index is absent.
pub(super) type Current<'a, M, P, R> = Option<(&'a IndexRecord<R>, VersionPayload<M, P>)>;

/// Builds what `validated` publishes on top of `current`.
///
/// Members are applied in ascending identity order, so the engine a batch
/// builds does not depend on the order the caller listed them in. Insert,
/// Remove, and Rebuild publish the successor of the active version, and
/// every entry they write is introduced at it.
///
/// PERF: the active payload is decoded and then mutated, so each operation
/// costs O(members) before its own changes; a later change can keep the
/// active engine in memory between operations.
///
/// # Errors
///
/// The engine's own refusals, and [`HeuremaError::InvalidIdentifier`] when
/// the version counter is exhausted. [`HeuremaError::TransitionNotPermitted`]
/// if `current` contradicts the state the operation was permitted from,
/// which [`CheckedOperation::permit`](super::CheckedOperation::permit) rules
/// out.
pub(super) fn successor<M, P, R>(
    validated: &ValidatedOperation<M, P, R>,
    current: Current<'_, M, P, R>,
) -> Result<Successor<M, P, R>, HeuremaError>
where
    M: MemberIdentity,
    P: ProvenanceReference,
    R: RetentionReference,
{
    let operation = validated.operation();
    let identity = validated.identity();
    let built = match (&operation.change, current) {
        (IndexChange::Create { config }, None) => Built {
            from: None,
            target: IndexVersion::FIRST,
            config: config.clone(),
            engine: VersionEngine::empty(config),
            members: BTreeMap::new(),
            changes: Vec::new(),
            previous_config: None,
        },
        (IndexChange::Insert { members }, Some((_, current))) => insert(members, current)?,
        (IndexChange::Remove { members }, Some((_, current))) => remove(members, current)?,
        (IndexChange::Rebuild { config, members }, Some((_, current))) => {
            rebuild(config, members, current)?
        }
        (IndexChange::Destroy { retention }, Some((record, current))) => {
            return Ok(destroy(operation, identity, retention, record, current));
        }
        (change, current) => {
            return TransitionNotPermittedSnafu {
                index: operation.index.clone(),
                transition: change.transition(),
                state: if current.is_some() {
                    IndexStateKind::Active
                } else {
                    IndexStateKind::Absent
                },
            }
            .fail();
        }
    };
    Ok(published(operation, identity, built))
}

/// A new version an operation built, before it is encoded.
struct Built<M, P> {
    /// The version the operation was applied to; `None` for Create.
    from: Option<IndexVersion>,
    /// The version it publishes.
    target: IndexVersion,
    config: IndexConfig,
    engine: VersionEngine<M>,
    members: BTreeMap<M, MemberEntry<P>>,
    changes: Vec<MemberChange<M, P>>,
    previous_config: Option<IndexConfig>,
}

/// Inserts or replaces `members` in the active version.
fn insert<M, P>(
    members: &[IndexMember<M, P>],
    current: VersionPayload<M, P>,
) -> Result<Built<M, P>, HeuremaError>
where
    M: MemberIdentity,
    P: ProvenanceReference,
{
    let target = current.version.successor()?;
    let mut engine = current.engine;
    let mut table = current.members;
    let mut changes = Vec::with_capacity(members.len());
    for member in sorted_members(members) {
        engine.insert(member.id.clone(), &member.content)?;
        let previous = table.remove(&member.id);
        let (entry, change) = written(member, target, previous);
        table.insert(member.id.clone(), entry);
        changes.push(change);
    }
    Ok(Built {
        from: Some(current.version),
        target,
        config: current.config,
        engine,
        members: table,
        changes,
        previous_config: None,
    })
}

/// Removes `members` from the active version; an absent identity is
/// recorded and otherwise ignored.
fn remove<M, P>(members: &[M], current: VersionPayload<M, P>) -> Result<Built<M, P>, HeuremaError>
where
    M: MemberIdentity,
{
    let target = current.version.successor()?;
    let mut engine = current.engine;
    let mut table = current.members;
    let mut ids: Vec<&M> = members.iter().collect();
    ids.sort_unstable();
    let mut changes = Vec::with_capacity(ids.len());
    for id in ids {
        let change = match table.remove(id) {
            Some(previous) => {
                engine.remove(id)?;
                MemberChange::Removed {
                    id: id.clone(),
                    previous,
                }
            }
            None => MemberChange::AbsentOnRemove { id: id.clone() },
        };
        changes.push(change);
    }
    Ok(Built {
        from: Some(current.version),
        target,
        config: current.config,
        engine,
        members: table,
        changes,
        previous_config: None,
    })
}

/// Builds a fresh engine under `config` holding exactly `members`; every
/// member of the active version that `members` leaves out is dropped.
fn rebuild<M, P>(
    config: &IndexConfig,
    members: &[IndexMember<M, P>],
    current: VersionPayload<M, P>,
) -> Result<Built<M, P>, HeuremaError>
where
    M: MemberIdentity,
    P: ProvenanceReference,
{
    let target = current.version.successor()?;
    let mut engine = VersionEngine::empty(config);
    let mut previous_table = current.members;
    let mut table = BTreeMap::new();
    let mut changes = BTreeMap::new();
    for member in sorted_members(members) {
        engine.insert(member.id.clone(), &member.content)?;
        let previous = previous_table.remove(&member.id);
        let (entry, change) = written(member, target, previous);
        table.insert(member.id.clone(), entry);
        changes.insert(member.id.clone(), change);
    }
    for (id, previous) in previous_table {
        changes.insert(id.clone(), MemberChange::Dropped { id, previous });
    }
    Ok(Built {
        from: Some(current.version),
        target,
        config: config.clone(),
        engine,
        members: table,
        changes: changes.into_values().collect(),
        previous_config: Some(current.config),
    })
}

/// Destroys the index: a destroyed head that keeps the last configuration,
/// and a record retiring every member of the last version. Nothing is
/// staged.
fn destroy<M, P, R>(
    operation: &LifecycleOperation<M, P, R>,
    identity: &OperationIdentity,
    retention: &R,
    head: &IndexRecord<R>,
    current: VersionPayload<M, P>,
) -> Successor<M, P, R>
where
    R: RetentionReference,
{
    let state = IndexState::Destroyed {
        last_version: current.version,
        retention: retention.clone(),
        operation: identity.key.clone(),
    };
    let changes = current
        .members
        .into_iter()
        .map(|(id, previous)| MemberChange::Retired { id, previous })
        .collect();
    Successor {
        head: IndexRecord::new(operation.index.clone(), head.config.clone(), state.clone()),
        record: OperationRecord::new(
            operation.index.clone(),
            identity.clone(),
            LifecycleTransition::Destroy,
            Some(current.version),
            state,
            None,
            changes,
        ),
        payload: None,
    }
}

/// The entry `member` gets at `target`, linked to the entry it replaces,
/// and the change that records it.
fn written<M, P>(
    member: &IndexMember<M, P>,
    target: IndexVersion,
    previous: Option<MemberEntry<P>>,
) -> (MemberEntry<P>, MemberChange<M, P>)
where
    M: Clone,
    P: Clone,
{
    let entry = MemberEntry {
        provenance: member.provenance.clone(),
        introduced: target,
        supersedes: previous.as_ref().map(|entry| entry.introduced),
    };
    let id = member.id.clone();
    let change = match previous {
        Some(previous) => MemberChange::Replaced { id, previous },
        None => MemberChange::Inserted { id },
    };
    (entry, change)
}

/// What publishing `built` writes: the active head, the operation record,
/// and the version payload.
fn published<M, P, R: RetentionReference>(
    operation: &LifecycleOperation<M, P, R>,
    identity: &OperationIdentity,
    built: Built<M, P>,
) -> Successor<M, P, R> {
    let state = IndexState::Active {
        version: built.target,
    };
    Successor {
        head: IndexRecord::new(operation.index.clone(), built.config.clone(), state.clone()),
        record: OperationRecord::new(
            operation.index.clone(),
            identity.clone(),
            operation.change.transition(),
            built.from,
            state,
            built.previous_config,
            built.changes,
        ),
        payload: Some(VersionPayload::new(
            operation.index.clone(),
            built.target,
            identity.clone(),
            built.config,
            built.engine,
            built.members,
        )),
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests need concise lifecycle fixtures")]
mod tests {
    use super::*;
    use crate::lifecycle::test_placeholders::{
        PlaceholderProvenance, PlaceholderRetention, TestMember,
    };
    use crate::{
        CheckedOperation, HnswConfig, IndexIdentity, IndexMember, IndexName, LifecycleOperation,
        MemberContent, OperationKey, OwnerNamespace,
    };

    type Operation = LifecycleOperation<TestMember, PlaceholderProvenance, PlaceholderRetention>;
    type Built = Successor<TestMember, PlaceholderProvenance, PlaceholderRetention>;

    fn notes() -> IndexIdentity {
        IndexIdentity::new(
            OwnerNamespace::try_from("example").expect("namespace"),
            IndexName::try_from("notes").expect("name"),
        )
    }

    fn version(value: u64) -> IndexVersion {
        IndexVersion::try_from(value).expect("version")
    }

    fn operation(
        key: &str,
        change: IndexChange<TestMember, PlaceholderProvenance, PlaceholderRetention>,
    ) -> Operation {
        LifecycleOperation::new(notes(), OperationKey::try_from(key).expect("key"), change)
    }

    fn member(
        id: u64,
        provenance: u32,
        vector: [f32; 2],
    ) -> IndexMember<TestMember, PlaceholderProvenance> {
        IndexMember::new(
            TestMember(id),
            PlaceholderProvenance(provenance),
            MemberContent::Vector(vector.to_vec()),
        )
    }

    /// Validates `operation` against `previous` and builds its successor.
    fn step(operation: Operation, previous: Option<&Built>) -> Built {
        let validated = CheckedOperation::check(operation)
            .expect("stateless checks pass")
            .permit(previous.map(|built| &built.head))
            .expect("permitted");
        let current = previous.map(|built| {
            (
                &built.head,
                built
                    .payload
                    .clone()
                    .expect("an active version has a payload"),
            )
        });
        successor(&validated, current).expect("successor builds")
    }

    fn payload(built: &Built) -> &VersionPayload<TestMember, PlaceholderProvenance> {
        built.payload.as_ref().expect("payload")
    }

    /// Version 1 (empty), version 2 (members 1 and 2), and version 3
    /// (member 1 replaced under new provenance).
    fn replaced_once() -> (Built, Built, Built) {
        let created = step(
            operation(
                "create",
                IndexChange::Create {
                    config: IndexConfig::Vector(HnswConfig::new(2)),
                },
            ),
            None,
        );
        let first = step(
            operation(
                "insert-1",
                IndexChange::Insert {
                    members: vec![member(1, 10, [1.0, 0.0]), member(2, 20, [0.0, 1.0])],
                },
            ),
            Some(&created),
        );
        let second = step(
            operation(
                "insert-2",
                IndexChange::Insert {
                    members: vec![member(1, 11, [0.5, 0.5])],
                },
            ),
            Some(&first),
        );
        (created, first, second)
    }

    #[test]
    fn apply_links_replaced_members_to_the_entry_they_supersede() {
        let (_, first, second) = replaced_once();
        let replaced = payload(&second)
            .members
            .get(&TestMember(1))
            .expect("member 1");
        assert_eq!(replaced.provenance, PlaceholderProvenance(11));
        assert_eq!(replaced.introduced, version(3));
        assert_eq!(
            replaced.supersedes,
            Some(version(2)),
            "links to the entry it replaced"
        );
        let untouched = payload(&second)
            .members
            .get(&TestMember(2))
            .expect("member 2");
        assert_eq!(
            (untouched.introduced, untouched.supersedes),
            (version(2), None),
            "a member the operation did not name keeps its entry"
        );
        assert_eq!(
            second.record.changes,
            vec![MemberChange::Replaced {
                id: TestMember(1),
                previous: MemberEntry {
                    provenance: PlaceholderProvenance(10),
                    introduced: version(2),
                    supersedes: None,
                },
            }],
            "the audit record keeps the superseded entry whole"
        );
        assert_eq!(second.record.from, Some(version(2)));
        assert_eq!(
            payload(&first)
                .members
                .get(&TestMember(1))
                .map(|entry| &entry.provenance),
            Some(&PlaceholderProvenance(10)),
            "the superseded entry stays in the version that introduced it"
        );
    }

    #[test]
    fn rebuild_links_kept_members_and_records_the_ones_it_drops() {
        let (_, _, second) = replaced_once();
        let rebuilt = step(
            operation(
                "rebuild",
                IndexChange::Rebuild {
                    config: IndexConfig::Vector(HnswConfig::new(3)),
                    members: vec![IndexMember::new(
                        TestMember(1),
                        PlaceholderProvenance(12),
                        MemberContent::Vector(vec![1.0, 0.0, 0.0]),
                    )],
                },
            ),
            Some(&second),
        );
        let rebuilt_entry = payload(&rebuilt)
            .members
            .get(&TestMember(1))
            .expect("member 1");
        assert_eq!(rebuilt_entry.supersedes, Some(version(3)));
        assert_eq!(
            rebuilt.record.previous_config,
            Some(IndexConfig::Vector(HnswConfig::new(2)))
        );
        assert!(
            matches!(
                rebuilt.record.changes.as_slice(),
                [
                    MemberChange::Replaced { id: TestMember(1), previous: first_previous },
                    MemberChange::Dropped { id: TestMember(2), previous: dropped },
                ] if first_previous.introduced == version(3) && dropped.introduced == version(2)
            ),
            "{:?}",
            rebuilt.record.changes
        );
    }

    /// Forty two-dimensional members: enough that HNSW level draws and
    /// neighbour pruning depend on the order the members are inserted in.
    fn order_sensitive_members() -> Vec<IndexMember<TestMember, PlaceholderProvenance>> {
        (0..40_u64)
            .map(|i| {
                let a = i as f32 * 0.7;
                member(i, i as u32, [a.cos(), a.sin() + (i % 3) as f32 * 0.1])
            })
            .collect()
    }

    fn reversed(
        members: &[IndexMember<TestMember, PlaceholderProvenance>],
    ) -> Vec<IndexMember<TestMember, PlaceholderProvenance>> {
        members.iter().rev().cloned().collect()
    }

    fn payload_bytes(built: &Built) -> Vec<u8> {
        super::super::encoding::encode(payload(built)).expect("payload encodes")
    }

    #[test]
    fn member_order_in_a_batch_does_not_change_the_built_version() {
        let config = IndexConfig::Vector(HnswConfig::new(2));
        let members = order_sensitive_members();

        // NOTE: the precondition that makes this test able to fail. Inserted
        // in the caller's order, this fixture builds different engines.
        let engine_bytes = |members: &[IndexMember<TestMember, PlaceholderProvenance>]| {
            let mut engine = VersionEngine::empty(&config);
            for member in members {
                engine
                    .insert(member.id.clone(), &member.content)
                    .expect("insert");
            }
            serde_json::to_vec(&engine).expect("engine encodes")
        };
        assert_ne!(
            engine_bytes(&members),
            engine_bytes(&reversed(&members)),
            "precondition: the engine depends on insertion order"
        );

        let created = step(operation("create", IndexChange::Create { config }), None);
        let inserted = |members| {
            step(
                operation("insert", IndexChange::Insert { members }),
                Some(&created),
            )
        };
        assert_eq!(
            payload_bytes(&inserted(members.clone())),
            payload_bytes(&inserted(reversed(&members))),
            "members apply in identity order, so the payload bytes agree"
        );
    }

    #[test]
    fn member_order_in_a_rebuild_does_not_change_the_built_version() {
        let (_, second, _) = replaced_once();
        let members = order_sensitive_members();
        let rebuilt = |members| {
            step(
                operation(
                    "rebuild",
                    IndexChange::Rebuild {
                        config: IndexConfig::Vector(HnswConfig::new(2)),
                        members,
                    },
                ),
                Some(&second),
            )
        };
        assert_eq!(
            payload_bytes(&rebuilt(members.clone())),
            payload_bytes(&rebuilt(reversed(&members))),
            "a rebuild applies members in identity order, so the payload bytes agree"
        );
    }

    #[test]
    fn destroy_retires_every_member_of_the_last_version() {
        let created = step(
            operation(
                "create",
                IndexChange::Create {
                    config: IndexConfig::Vector(HnswConfig::new(2)),
                },
            ),
            None,
        );
        let inserted = step(
            operation(
                "insert",
                IndexChange::Insert {
                    members: vec![member(1, 10, [1.0, 0.0]), member(2, 20, [0.0, 1.0])],
                },
            ),
            Some(&created),
        );
        let destroyed = step(
            operation(
                "destroy",
                IndexChange::Destroy {
                    retention: PlaceholderRetention(4),
                },
            ),
            Some(&inserted),
        );
        assert!(destroyed.payload.is_none(), "destroy publishes no version");
        assert_eq!(
            destroyed.head.state,
            IndexState::Destroyed {
                last_version: version(2),
                retention: PlaceholderRetention(4),
                operation: OperationKey::try_from("destroy").expect("key"),
            }
        );
        let retired: Vec<_> = destroyed
            .record
            .changes
            .iter()
            .map(MemberChange::id)
            .collect();
        assert_eq!(retired, [&TestMember(1), &TestMember(2)]);
        assert!(
            destroyed
                .record
                .changes
                .iter()
                .all(|change| matches!(change, MemberChange::Retired { .. }))
        );
    }
}
