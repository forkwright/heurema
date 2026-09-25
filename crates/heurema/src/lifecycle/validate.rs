//! Pre-publish validation: an operation is parsed into a [`CheckedOperation`]
//! by its stateless checks, then into a [`ValidatedOperation`] by the
//! permission table and the checks that need the index's current record.

use std::collections::BTreeMap;

use serde::Serialize;
use serde::de::DeserializeOwned;
use snafu::ensure;

use super::digest;
use super::identity::{self, IdentifierKind, OperationIdentity};
use super::member::{
    IndexMember, MemberContent, MemberIdentity, ProvenanceReference, RetentionReference,
};
use super::operation::{IndexChange, IndexConfig, LifecycleOperation, LifecycleTransition};
use super::record::{IndexRecord, IndexStateKind};
use crate::HeuremaError;
use crate::error::{
    DuplicateMemberSnafu, EmptyBatchSnafu, FamilyMismatchSnafu, RecordMismatchSnafu,
    TransitionNotPermittedSnafu, UnencodableOperationSnafu,
};
use crate::fts::require_simple_pipeline;
use crate::hnsw::{check_config, check_finite, check_vector};

/// A lifecycle operation that passed every check that needs no index state,
/// together with its [`OperationIdentity`].
///
/// [`check`](Self::check) is a pure function: it reads no backend and
/// writes nothing. It refuses, in this order:
///
/// 1. an Insert or Remove that names no members
///    ([`HeuremaError::EmptyBatch`]);
/// 2. a batch that names one member identity twice
///    ([`HeuremaError::DuplicateMember`]);
/// 3. a member identity that does not encode as a string or an integer, or
///    that does not read back as itself from a JSON object key or from a
///    JSON value ([`HeuremaError::InvalidIdentifier`] of kind
///    [`IdentifierKind::MemberIdentity`]), because engine snapshots and the
///    stored records key members by identity and also store identities as
///    values;
/// 4. a vector with a NaN or infinite component
///    ([`HeuremaError::InvalidVector`]);
/// 5. a Create or Rebuild configuration the engine refuses: an HNSW
///    configuration with a zero dimension, `m_neighbours`, or
///    `ef_construction` ([`HeuremaError::InvalidHnswConfig`]), or a BM25
///    pipeline other than `Simple` ([`HeuremaError::NotYetImplemented`]);
/// 6. a Rebuild member that does not fit the new configuration: first any
///    member of the other family ([`HeuremaError::FamilyMismatch`]), checked
///    for every member before any member's dimension, then a vector of the
///    wrong dimension ([`HeuremaError::DimensionMismatch`]);
/// 7. an Insert that mixes vector and document members
///    ([`HeuremaError::FamilyMismatch`]);
/// 8. an operation whose provenance or retention value has no canonical
///    encoding, such as one holding a float or a map keyed by something
///    other than strings or integers
///    ([`HeuremaError::UnencodableOperation`]);
/// 9. a provenance or retention value the operation stores that does not
///    read back as itself from its `serde_json` encoding: each Insert and
///    Rebuild member's provenance, in ascending identity order, and a
///    Destroy's retention ([`HeuremaError::UnencodableOperation`]).
///
/// Each step covers the whole batch before the next begins, and per-member
/// checks visit members in ascending identity order, so the refusal an
/// operation meets does not depend on the order its members were listed
/// in. Step 8 computes the [`OperationDigest`] as documented there.
/// Every refusal here precedes any refusal [`permit`](Self::permit) can
/// give, because `permit` takes a `CheckedOperation`.
///
/// A missing provenance or retention reference is not a refusal here: it
/// does not compile, because every member carries a required
/// [`ProvenanceReference`] and Destroy a required [`RetentionReference`].
///
/// WHY two stages: the stateless checks and the digest need no storage, so a
/// driver can run them, and refuse, before any adapter I/O. Only
/// [`permit`](Self::permit), given the current record, yields the
/// [`ValidatedOperation`] that later steps accept.
///
/// # Examples
///
/// ```
/// use heurema::{
///     CheckedOperation, HeuremaError, HnswConfig, IndexChange, IndexConfig, IndexIdentity,
///     IndexMember, IndexName, IndexStateKind, LifecycleOperation, MemberContent,
///     MemberIdentity, OperationKey, OwnerNamespace, ProvenanceReference, RetentionReference,
/// };
/// use serde::{Deserialize, Serialize};
///
/// /// test-local placeholder; heurēma defines no provenance shape
/// #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// struct TestMember(u64);
/// impl MemberIdentity for TestMember {}
///
/// /// test-local placeholder; heurēma defines no provenance shape
/// #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// struct PlaceholderProvenance(u32);
/// impl ProvenanceReference for PlaceholderProvenance {}
///
/// /// test-local placeholder; heurēma defines no provenance shape
/// #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// struct PlaceholderRetention(u32);
/// impl RetentionReference for PlaceholderRetention {}
///
/// let index = IndexIdentity::new(
///     OwnerNamespace::try_from("example")?,
///     IndexName::try_from("notes")?,
/// );
///
/// let create = LifecycleOperation::<TestMember, PlaceholderProvenance, PlaceholderRetention>::new(
///     index.clone(),
///     OperationKey::try_from("create-notes")?,
///     IndexChange::Create {
///         config: IndexConfig::Vector(HnswConfig::new(2)),
///     },
/// );
/// let checked = CheckedOperation::check(create)?;
/// assert_eq!(checked.identity().key.as_str(), "create-notes");
///
/// // Create is permitted on an absent index: no record exists yet.
/// let validated = checked.permit(None)?;
/// assert_eq!(validated.from_state(), IndexStateKind::Absent);
///
/// // Insert is not: the index has to exist first.
/// let insert = LifecycleOperation::<_, _, PlaceholderRetention>::new(
///     index,
///     OperationKey::try_from("insert-1")?,
///     IndexChange::Insert {
///         members: vec![IndexMember::new(
///             TestMember(1),
///             PlaceholderProvenance(7),
///             MemberContent::Vector(vec![0.0, 1.0]),
///         )],
///     },
/// );
/// let refused = CheckedOperation::check(insert)?.permit(None);
/// assert!(matches!(refused, Err(HeuremaError::TransitionNotPermitted { .. })));
/// # Ok::<(), HeuremaError>(())
/// ```
///
/// [`OperationDigest`]: super::OperationDigest
#[derive(Debug, Clone)]
pub struct CheckedOperation<M, P, R> {
    operation: LifecycleOperation<M, P, R>,
    identity: OperationIdentity,
}

/// A lifecycle operation that passed its stateless checks and was permitted
/// against the index's current record.
///
/// WHY: parse, don't validate. There is no way to build one except
/// [`CheckedOperation::permit`], so a step that takes a
/// `ValidatedOperation` cannot receive an operation that skipped a check.
#[derive(Debug, Clone)]
pub struct ValidatedOperation<M, P, R> {
    checked: CheckedOperation<M, P, R>,
    from: IndexStateKind,
}

impl<M: MemberIdentity, P: ProvenanceReference, R: RetentionReference> CheckedOperation<M, P, R> {
    /// Runs every check that needs no index state, then computes the
    /// operation's identity. See [`CheckedOperation`] for the checks and
    /// their order.
    ///
    /// # Errors
    ///
    /// Returns the first refusal in the order listed on
    /// [`CheckedOperation`]. Every refusal except `NotYetImplemented`
    /// (category `Unsupported`) has the category `Refused`.
    pub fn check(operation: LifecycleOperation<M, P, R>) -> Result<Self, HeuremaError> {
        check_change(&operation.change)?;
        let digest = digest::operation_digest(&operation)?;
        check_stored_values(&operation.change)?;
        let identity = OperationIdentity {
            key: operation.key.clone(),
            digest,
        };
        Ok(Self {
            operation,
            identity,
        })
    }

    /// The operation's key and digest.
    #[must_use]
    pub const fn identity(&self) -> &OperationIdentity {
        &self.identity
    }

    /// The checked operation.
    #[must_use]
    pub const fn operation(&self) -> &LifecycleOperation<M, P, R> {
        &self.operation
    }

    /// Checks the operation against the index's current record: `None` for
    /// an absent index, otherwise the record read for
    /// [`operation().index`](LifecycleOperation::index). This is a pure
    /// function; the caller reads the record.
    ///
    /// It refuses, in this order:
    ///
    /// 1. a record whose identity is not the operation's index
    ///    ([`HeuremaError::RecordMismatch`]);
    /// 2. a transition the permission table forbids from the current state
    ///    ([`HeuremaError::TransitionNotPermitted`]; see
    ///    [`LifecycleTransition::is_permitted_from`]);
    /// 3. an Insert member the current configuration cannot hold: first any
    ///    member of the other family ([`HeuremaError::FamilyMismatch`]),
    ///    checked for every member before any member's dimension, then a
    ///    vector of the wrong dimension ([`HeuremaError::DimensionMismatch`]);
    /// 4. a Rebuild whose configuration changes the index's family
    ///    ([`HeuremaError::FamilyMismatch`]), since a named index keeps one
    ///    family for life.
    ///
    /// # Errors
    ///
    /// Returns the first refusal above, all of category `Refused`.
    pub fn permit(
        self,
        current: Option<&IndexRecord<R>>,
    ) -> Result<ValidatedOperation<M, P, R>, HeuremaError> {
        if let Some(record) = current {
            ensure!(
                record.identity == self.operation.index,
                RecordMismatchSnafu {
                    expected: self.operation.index.clone(),
                    actual: record.identity.clone(),
                }
            );
        }
        let from = current.map_or(IndexStateKind::Absent, |record| record.state.kind());
        let transition = self.operation.transition();
        ensure!(
            transition.is_permitted_from(from),
            TransitionNotPermittedSnafu {
                index: self.operation.index.clone(),
                transition,
                state: from,
            }
        );
        if let Some(record) = current {
            check_against_record(&self.operation.change, &record.config)?;
        }
        Ok(ValidatedOperation {
            checked: self,
            from,
        })
    }
}

impl<M, P, R> ValidatedOperation<M, P, R> {
    /// The operation's key and digest.
    #[must_use]
    pub const fn identity(&self) -> &OperationIdentity {
        &self.checked.identity
    }

    /// The validated operation.
    #[must_use]
    pub const fn operation(&self) -> &LifecycleOperation<M, P, R> {
        &self.checked.operation
    }

    /// The state the index was in when the operation was permitted.
    #[must_use]
    pub const fn from_state(&self) -> IndexStateKind {
        self.from
    }
}

/// The stateless checks, steps 1 to 7 of the order on [`CheckedOperation`].
fn check_change<M, P, R>(change: &IndexChange<M, P, R>) -> Result<(), HeuremaError>
where
    M: MemberIdentity,
{
    match change {
        IndexChange::Create { config } => check_index_config(config),
        IndexChange::Insert { members } => {
            ensure!(
                !members.is_empty(),
                EmptyBatchSnafu {
                    transition: LifecycleTransition::Insert,
                }
            );
            let members = sorted_members(members);
            check_member_identities(&member_ids(&members))?;
            check_finite_vectors(&members)?;
            check_single_family(&members)
        }
        IndexChange::Remove { members } => {
            ensure!(
                !members.is_empty(),
                EmptyBatchSnafu {
                    transition: LifecycleTransition::Remove,
                }
            );
            let mut ids: Vec<&M> = members.iter().collect();
            ids.sort_unstable();
            check_member_identities(&ids)
        }
        IndexChange::Rebuild { config, members } => {
            let members = sorted_members(members);
            check_member_identities(&member_ids(&members))?;
            check_finite_vectors(&members)?;
            check_index_config(config)?;
            check_members_fit(config, &members)
        }
        IndexChange::Destroy { .. } => Ok(()),
    }
}

/// The checks that need the index's current configuration, steps 3 and 4 of
/// [`CheckedOperation::permit`].
fn check_against_record<M, P, R>(
    change: &IndexChange<M, P, R>,
    current: &IndexConfig,
) -> Result<(), HeuremaError>
where
    M: Ord,
{
    match change {
        IndexChange::Insert { members } => check_members_fit(current, &sorted_members(members)),
        IndexChange::Rebuild { config, .. } => {
            ensure!(
                config.family() == current.family(),
                FamilyMismatchSnafu {
                    expected: current.family(),
                    actual: config.family(),
                }
            );
            Ok(())
        }
        IndexChange::Create { .. } | IndexChange::Remove { .. } | IndexChange::Destroy { .. } => {
            Ok(())
        }
    }
}

/// `members` in ascending identity order.
///
/// WHY: every per-member check walks this order, so which member a refusal
/// names does not depend on the order the caller listed them in; building a
/// version applies members in the same order, so the engine a batch builds
/// does not depend on it either.
pub(super) fn sorted_members<M: Ord, P>(members: &[IndexMember<M, P>]) -> Vec<&IndexMember<M, P>> {
    let mut sorted: Vec<&IndexMember<M, P>> = members.iter().collect();
    sorted.sort_unstable_by(|left, right| left.id.cmp(&right.id));
    sorted
}

fn member_ids<'a, M, P>(members: &[&'a IndexMember<M, P>]) -> Vec<&'a M> {
    members.iter().map(|member| &member.id).collect()
}

/// Steps 2 and 3: no identity twice, and each one a string or an integer
/// that reads back as itself from a JSON object key and from a JSON value.
///
/// INVARIANT: `sorted_ids` ascends, so equal identities are adjacent.
fn check_member_identities<M: MemberIdentity>(sorted_ids: &[&M]) -> Result<(), HeuremaError> {
    let successors = sorted_ids.iter().skip(1);
    if let Some((duplicate, _)) = sorted_ids
        .iter()
        .zip(successors)
        .find(|(id, next)| id == next)
    {
        return DuplicateMemberSnafu {
            member: identity::reported(&format!("{duplicate:?}")),
        }
        .fail();
    }
    for id in sorted_ids {
        if let Some(reason) = digest::member_identity_refusal(id)
            .or_else(|| json_key_refusal(*id))
            .or_else(|| json_value_refusal(*id))
        {
            return Err(identity::refusal(
                IdentifierKind::MemberIdentity,
                &format!("{id:?}"),
                reason,
            ));
        }
    }
    Ok(())
}

/// Why `id` does not survive a round trip through a JSON object key, or
/// `None` when it does.
///
/// WHY: both adapters encode through `serde_json`, and engine snapshots hold
/// their members in maps keyed by identity, which JSON writes as objects. A
/// JSON object key is always a string: serde_json writes an integer key as
/// its decimal digits and reads it back through the identity's own
/// `Deserialize`. An identity that reads those digits back as another value,
/// such as an untagged enum with an integer and a string variant, would load
/// as a different member than was stored.
fn json_key_refusal<M: MemberIdentity>(id: &M) -> Option<String> {
    let stored = match serde_json::to_string(&BTreeMap::from([(id, 0_u8)])) {
        Ok(stored) => stored,
        Err(error) => return Some(format!("cannot be written as a JSON object key: {error}")),
    };
    // WHY `reported`: the refusal already carries the identity truncated to
    // 64 characters, and the JSON text would repeat it in full.
    match serde_json::from_str::<BTreeMap<M, u8>>(&stored) {
        Ok(read) if read.len() == 1 && read.contains_key(id) => None,
        Ok(_) => Some(format!(
            "reads back from the JSON object {} as a different identity; adapters store \
             members in JSON objects keyed by identity",
            identity::reported(&stored)
        )),
        Err(error) => Some(format!(
            "cannot be read back from the JSON object {}: {error}",
            identity::reported(&stored)
        )),
    }
}

/// Why `id` does not survive a round trip through a JSON value, or `None`
/// when it does.
///
/// WHY: besides keying maps, heurēma stores member identities as JSON
/// values: an HNSW engine's entry point and neighbour lists, and each
/// member change in an operation record. An identity that reads back only
/// from a key would publish a version that never decodes again.
fn json_value_refusal<M: MemberIdentity>(id: &M) -> Option<String> {
    json_round_trip(id)
        .err()
        .map(|reason| format!("{reason}; heurēma also stores member identities as JSON values"))
}

/// Step 9: every provenance or retention value the operation stores reads
/// back as itself from its JSON encoding.
///
/// WHY after the digest: a value the canonical encoder refuses, such as a
/// float, is reported as having no canonical encoding, the more specific
/// refusal, before its round trip is tried.
fn check_stored_values<M, P, R>(change: &IndexChange<M, P, R>) -> Result<(), HeuremaError>
where
    M: MemberIdentity,
    P: ProvenanceReference,
    R: RetentionReference,
{
    let refusal = match change {
        IndexChange::Insert { members } | IndexChange::Rebuild { members, .. } => {
            sorted_members(members).into_iter().find_map(|member| {
                json_round_trip(&member.provenance).err().map(|reason| {
                    format!(
                        "provenance of member {} {reason}",
                        identity::reported(&format!("{:?}", member.id))
                    )
                })
            })
        }
        IndexChange::Destroy { retention } => json_round_trip(retention)
            .err()
            .map(|reason| format!("retention {reason}")),
        IndexChange::Create { .. } | IndexChange::Remove { .. } => None,
    };
    match refusal {
        Some(reason) => UnencodableOperationSnafu { reason }.fail(),
        None => Ok(()),
    }
}

/// Why `value` does not read back as itself from the JSON bytes
/// `serde_json` writes for it, or `Ok` when it does.
///
/// WHY the byte path, never `serde_json::Value`: storage writes and reads
/// bytes, and a `Value` round trip differs from that for a `u128` or for
/// borrowed data.
fn json_round_trip<T: Serialize + DeserializeOwned + Eq>(value: &T) -> Result<(), String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("cannot be written as a JSON value: {error}"))?;
    // WHY `reported`: the value can be arbitrarily long, and a refusal must
    // not be.
    let json = identity::reported(&String::from_utf8_lossy(&bytes));
    match serde_json::from_slice::<T>(&bytes) {
        Ok(read) if read == *value => Ok(()),
        Ok(_) => Err(format!(
            "reads back from its JSON value {json} as a different value"
        )),
        Err(error) => Err(format!(
            "cannot be read back from its JSON value {json}: {error}"
        )),
    }
}

/// Step 4: every vector component is finite.
fn check_finite_vectors<M, P>(members: &[&IndexMember<M, P>]) -> Result<(), HeuremaError> {
    members.iter().try_for_each(|member| match &member.content {
        MemberContent::Vector(vector) => check_finite(vector),
        MemberContent::Document(_) => Ok(()),
    })
}

/// Step 5: the configuration is one the engine accepts.
fn check_index_config(config: &IndexConfig) -> Result<(), HeuremaError> {
    match config {
        IndexConfig::Vector(hnsw) => check_config(hnsw),
        IndexConfig::Fts(fts) => require_simple_pipeline(fts),
    }
}

/// Step 7: every member of an Insert belongs to one family, that of the
/// member with the lowest identity.
fn check_single_family<M, P>(members: &[&IndexMember<M, P>]) -> Result<(), HeuremaError> {
    let Some(first) = members.first() else {
        return Ok(());
    };
    let expected = first.content.family();
    members.iter().try_for_each(|member| {
        let actual = member.content.family();
        ensure!(actual == expected, FamilyMismatchSnafu { expected, actual });
        Ok(())
    })
}

/// Every member fits an index built under `config`: first every member's
/// family, then, member by member, the check the engine applies when the
/// content enters it (for a vector, its dimension).
///
/// WHY two passes: a batch holding both a member of the other family and a
/// vector of the wrong dimension is refused with `FamilyMismatch` whatever
/// order its members sort in, because no member's dimension is checked
/// before every member's family is.
fn check_members_fit<M, P>(
    config: &IndexConfig,
    sorted_members: &[&IndexMember<M, P>],
) -> Result<(), HeuremaError> {
    let expected = config.family();
    for member in sorted_members {
        let actual = member.content.family();
        ensure!(actual == expected, FamilyMismatchSnafu { expected, actual });
    }
    sorted_members
        .iter()
        .try_for_each(|member| check_content(config, &member.content))
}

/// The check an engine built under `config` applies when `content` enters
/// it.
///
/// WHY: calling the engines' own checks keeps validation and application in
/// agreement; an operation validated here is not refused when applied. The
/// other-family arms repeat [`check_members_fit`]'s first pass, so the match
/// stays exhaustive without a panic.
fn check_content(config: &IndexConfig, content: &MemberContent) -> Result<(), HeuremaError> {
    match (config, content) {
        (IndexConfig::Vector(hnsw), MemberContent::Vector(vector)) => check_vector(hnsw, vector),
        (IndexConfig::Fts(fts), MemberContent::Document(_)) => require_simple_pipeline(fts),
        (IndexConfig::Vector(_), MemberContent::Document(_))
        | (IndexConfig::Fts(_), MemberContent::Vector(_)) => FamilyMismatchSnafu {
            expected: config.family(),
            actual: content.family(),
        }
        .fail(),
    }
}
