//! The lifecycle driver: [`IndexLifecycle`] validates an operation, stages
//! the version it builds, and publishes it at one atomic point.

use std::fmt;
use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::apply::{self, Successor};
use super::backend::{DestroyWrite, LifecycleBackend, PublishWrite, StageWrite};
use super::encoding::{
    StagingMarker, VersionPayload, corrupt, decode_head, decode_operation, decode_payload, encode,
    encode_head,
};
use super::identity::{IndexIdentity, IndexVersion, OperationIdentity};
use super::member::{MemberIdentity, ProvenanceReference, RetentionReference};
use super::operation::{LifecycleOperation, LifecycleTransition};
use super::published::PublishedIndex;
use super::record::{IndexRecord, IndexState, IndexStateKind};
use super::validate::CheckedOperation;
use crate::error::{
    HeadChangedSnafu, IndexNotFoundSnafu, OperationConflictSnafu, StagedStateExistsSnafu,
    TransitionNotPermittedSnafu,
};
use crate::{HeuremaError, OperationConflictDetail};

/// The durable lifecycle of named indexes over one [`LifecycleBackend`].
///
/// Every mutation runs the same steps, in this order:
///
/// 1. [`CheckedOperation::check`]: the stateless checks and the operation's
///    digest. No backend call is made, so a stateless refusal touches no
///    storage.
/// 2. Replay lookup: the operation record stored under the operation's key
///    for its index. The same digest returns the recorded outcome as a
///    [`PublishReceipt`] with `replayed` set and writes nothing; a different
///    digest is [`HeuremaError::OperationConflict`]. The lookup precedes the
///    permission check, so a replayed Create still answers after the index
///    has moved on.
/// 3. Head read, then [`CheckedOperation::permit`]: the permission table
///    and the checks against the current configuration.
/// 4. Staged-state check: an index holding a staged, unpublished version
///    refuses every mutation, Create and Destroy included, with
///    [`HeuremaError::StagedStateExists`] until recovery clears it.
/// 5. Build: the successor version is built in memory from the active
///    version's payload (a Rebuild starts from an empty engine), applying
///    members in ascending identity order. Every engine refusal happens here.
/// 6. Stage: the version payload and its staging marker, in one durable
///    write that changes no head (Destroy stages nothing).
/// 7. Publish: the new head, the operation record, and the removal of the
///    staging marker, in one atomic durable write. This is the only point at
///    which the operation becomes visible.
///
/// Steps 1 to 5 write nothing; a refusal at any of them leaves storage as it
/// was. Steps 6 and 7 compare-and-set the head read in step 3, so a head
/// that moved in between (another lifecycle over the same backend) refuses
/// the write with [`HeuremaError::HeadChanged`] and nothing is written.
///
/// [`apply`](Self::apply) runs every step. [`prepare`](Self::prepare) runs
/// steps 1 to 5 and returns the remaining two as separate calls
/// ([`Prepared::stage`], [`Staged::publish`]), so a caller can stop between
/// them.
///
/// # Reads
///
/// [`record`](Self::record) reads an index's head, and
/// [`index`](Self::index) reads the head and then the immutable payload of
/// the version it names. A staged version is never readable: no read takes
/// a version the head does not name, and the head names a version only once
/// it is published.
///
/// # Concurrency
///
/// A lifecycle runs one mutation at a time: [`prepare`](Self::prepare) takes
/// a writer lock that the returned [`Prepared`] and [`Staged`] hold until
/// they publish or are dropped. Reads take no lock.
///
/// WARNING: a thread that still holds a [`Prepared`] or [`Staged`] must drop
/// it before it prepares or applies another operation on the same
/// lifecycle; locking the writer again from that thread does not return
/// normally (the standard library's mutex deadlocks or panics).
#[must_use = "a lifecycle does nothing until an operation is applied"]
pub struct IndexLifecycle<B, M, P, R> {
    backend: B,
    writer: Mutex<()>,
    types: ConsumerTypes<M, P, R>,
}

/// The consumer's member identity, provenance, and retention types, which a
/// lifecycle decodes and returns but never stores in memory.
///
/// WHY `fn() -> _`: the lifecycle holds no value of these types, so it is
/// `Send` and `Sync` whenever its backend is, whatever they are.
type ConsumerTypes<M, P, R> = PhantomData<fn() -> (M, P, R)>;

/// The outcome of a published or replayed operation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PublishReceipt<R> {
    /// The index the operation changed.
    pub index: IndexIdentity,
    /// The operation's key and digest.
    pub operation: OperationIdentity,
    /// The transition it performed.
    pub transition: LifecycleTransition,
    /// The state publishing the operation left the index in. On a replay
    /// this is the state recorded when the operation was first published,
    /// not the index's current state.
    pub state: IndexState<R>,
    /// Whether the operation had already been published under this key and
    /// digest, so this call wrote nothing.
    pub replayed: bool,
}

impl<R> PublishReceipt<R> {
    /// The version the operation published, or, for a Destroy, the last
    /// version the index had.
    #[must_use]
    pub const fn version(&self) -> IndexVersion {
        match &self.state {
            IndexState::Active { version } => *version,
            IndexState::Destroyed { last_version, .. } => *last_version,
        }
    }
}

/// What [`IndexLifecycle::prepare`] found.
#[non_exhaustive]
pub enum Preparation<'a, B, M, P, R> {
    /// The operation was already published under its key with the same
    /// digest; nothing is left to do and nothing was written.
    Replayed(PublishReceipt<R>),
    /// The operation passed every check and its writes are encoded, ready
    /// to stage.
    Ready(Prepared<'a, B, M, P, R>),
}

/// A validated operation whose writes are encoded and not yet staged.
///
/// Dropping it abandons the operation with nothing written.
#[must_use = "a prepared operation does nothing until it is staged and published"]
pub struct Prepared<'a, B, M, P, R> {
    lifecycle: &'a IndexLifecycle<B, M, P, R>,
    writer: MutexGuard<'a, ()>,
    plan: Plan<R>,
    write: PlannedWrite,
}

/// A staged operation, not yet published.
///
/// Dropping it after a version was staged leaves that version as orphan
/// staged state: durable, never readable, and refusing every later mutation
/// of its index with [`HeuremaError::StagedStateExists`] until recovery
/// clears it.
#[must_use = "a staged operation is invisible until it is published"]
pub struct Staged<'a, B, M, P, R> {
    lifecycle: &'a IndexLifecycle<B, M, P, R>,
    writer: MutexGuard<'a, ()>,
    plan: Plan<R>,
    write: StagedWrite,
}

/// The encoded writes and outcome shared by every step of one operation.
struct Plan<R> {
    index: IndexIdentity,
    operation: OperationIdentity,
    transition: LifecycleTransition,
    resulting: IndexState<R>,
    head: Vec<u8>,
    record: Vec<u8>,
}

/// What [`Prepared::stage`] writes.
enum PlannedWrite {
    /// Stage a version, then publish it.
    Version {
        expected_head: Option<Vec<u8>>,
        version: IndexVersion,
        marker: Vec<u8>,
        payload: Vec<u8>,
    },
    /// Destroy the index; nothing is staged.
    Destroy {
        expected_head: Vec<u8>,
        versions: Vec<IndexVersion>,
    },
}

/// What [`Staged::publish`] writes.
enum StagedWrite {
    /// Publish the staged version, fenced by its marker.
    Version {
        expected_head: Option<Vec<u8>>,
        version: IndexVersion,
        marker: Vec<u8>,
    },
    /// Destroy the index, removing the payload of every listed version.
    Destroy {
        expected_head: Vec<u8>,
        versions: Vec<IndexVersion>,
    },
}

impl<B, M, P, R> IndexLifecycle<B, M, P, R>
where
    B: LifecycleBackend,
    M: MemberIdentity,
    P: ProvenanceReference,
    R: RetentionReference,
{
    /// Opens the lifecycle over `backend`.
    ///
    /// Recovery of interrupted operations on open arrives in a later Phase 02
    /// change. Until then, an index whose operation was interrupted between
    /// stage and publish keeps its orphan staged state, and every mutation
    /// of that index is refused with [`HeuremaError::StagedStateExists`];
    /// other indexes are unaffected.
    ///
    /// # Errors
    ///
    /// None yet: opening reads nothing. The `Result` is where recovery on
    /// open will report the state it refuses to repair.
    pub fn open(backend: B) -> Result<Self, HeuremaError> {
        Ok(Self {
            backend,
            writer: Mutex::new(()),
            types: PhantomData,
        })
    }

    /// Validates, stages, and publishes `operation`, or returns its recorded
    /// outcome when it was already published under the same key and digest.
    ///
    /// # Errors
    ///
    /// Every refusal of [`prepare`](Self::prepare),
    /// [`Prepared::stage`], and [`Staged::publish`].
    pub fn apply(
        &self,
        operation: LifecycleOperation<M, P, R>,
    ) -> Result<PublishReceipt<R>, HeuremaError> {
        match self.prepare(operation)? {
            Preparation::Replayed(receipt) => Ok(receipt),
            Preparation::Ready(prepared) => prepared.stage()?.publish(),
        }
    }

    /// Runs steps 1 to 5 of the lifecycle (see [`IndexLifecycle`]): checks,
    /// replay lookup, permission, staged-state check, and the in-memory
    /// build, writing nothing.
    ///
    /// # Errors
    ///
    /// - The stateless refusals of [`CheckedOperation::check`], before any
    ///   backend call.
    /// - [`HeuremaError::OperationConflict`] when the key is recorded for
    ///   the index with a different digest.
    /// - The refusals of [`CheckedOperation::permit`].
    /// - [`HeuremaError::StagedStateExists`] when the index holds a staged,
    ///   unpublished version.
    /// - The engine's refusals while building the successor version.
    /// - [`HeuremaError::HeadChanged`] when another lifecycle over the same
    ///   backend destroyed the index between the head read and the payload
    ///   read.
    /// - [`HeuremaError::CorruptSnapshot`] or
    ///   [`HeuremaError::UnsupportedSnapshotVersion`] when a stored record
    ///   cannot be read, and [`HeuremaError::Persistence`] when the backend
    ///   fails.
    pub fn prepare(
        &self,
        operation: LifecycleOperation<M, P, R>,
    ) -> Result<Preparation<'_, B, M, P, R>, HeuremaError> {
        let checked = CheckedOperation::check(operation)?;
        // INVARIANT: the guard protects no in-memory state (storage is the
        // state, and every backend write is atomic), so a writer that
        // panicked leaves nothing half-applied and the poison is ignored.
        let writer = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(receipt) = self.replay(&checked)? {
            return Ok(Preparation::Replayed(receipt));
        }
        let index = checked.operation().index.clone();
        let expected_head = self.backend.read_head(&index)?;
        let head = expected_head
            .as_deref()
            .map(|bytes| decode_head::<R>(bytes, &index))
            .transpose()?;
        let validated = checked.permit(head.as_ref())?;
        if let Some((version, _)) = self.backend.read_staging(&index)? {
            return StagedStateExistsSnafu { index, version }.fail();
        }
        let current = match &head {
            Some(record) => match record.state {
                IndexState::Active { version } => {
                    let Some(payload) = self.active_payload(&index, version, record)? else {
                        return HeadChangedSnafu { index }.fail();
                    };
                    Some((record, payload))
                }
                IndexState::Destroyed { .. } => None,
            },
            None => None,
        };
        let successor = apply::successor(&validated, current)?;
        let (plan, write) = encode_plan(successor, expected_head, validated.operation())?;
        Ok(Preparation::Ready(Prepared {
            lifecycle: self,
            writer,
            plan,
            write,
        }))
    }

    /// The head record of `index`, or `None` when the index has never been
    /// created. A destroyed index's record stays readable.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::CorruptSnapshot`] or
    /// [`HeuremaError::UnsupportedSnapshotVersion`] when the stored head
    /// cannot be read, and [`HeuremaError::Persistence`] when the backend
    /// fails.
    pub fn record(&self, index: &IndexIdentity) -> Result<Option<IndexRecord<R>>, HeuremaError> {
        self.backend
            .read_head(index)?
            .map(|bytes| decode_head(&bytes, index))
            .transpose()
    }

    /// The active version of `index`: the head, then the immutable payload
    /// it names.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::IndexNotFound`] when the index is absent or
    /// destroyed; [`HeuremaError::CorruptSnapshot`] or
    /// [`HeuremaError::UnsupportedSnapshotVersion`] when a stored record
    /// cannot be read, or when the head names a version whose payload is
    /// missing; and [`HeuremaError::Persistence`] when the backend fails.
    pub fn index(&self, index: &IndexIdentity) -> Result<PublishedIndex<M, P>, HeuremaError> {
        let Some(head_bytes) = self.backend.read_head(index)? else {
            return not_found(index);
        };
        let head: IndexRecord<R> = decode_head(&head_bytes, index)?;
        let IndexState::Active { version } = head.state else {
            return not_found(index);
        };
        match self.active_payload(index, version, &head)? {
            Some(payload) => Ok(PublishedIndex::from_payload(payload)),
            None => not_found(index),
        }
    }

    /// The backend the lifecycle runs on.
    #[must_use]
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// Closes the lifecycle and returns its backend.
    #[must_use]
    pub fn into_backend(self) -> B {
        self.backend
    }

    /// Step 2: the recorded outcome when the operation's key is already
    /// recorded for its index with the same digest.
    fn replay(
        &self,
        checked: &CheckedOperation<M, P, R>,
    ) -> Result<Option<PublishReceipt<R>>, HeuremaError> {
        let operation = checked.operation();
        let Some(bytes) = self
            .backend
            .read_operation(&operation.index, &operation.key)?
        else {
            return Ok(None);
        };
        let record = decode_operation::<M, P, R>(&bytes, &operation.index, &operation.key)?;
        let requested = checked.identity();
        if record.operation.digest != requested.digest {
            return OperationConflictSnafu {
                conflict: Box::new(OperationConflictDetail::new(
                    operation.index.clone(),
                    operation.key.clone(),
                    record.operation.digest,
                    requested.digest,
                )),
            }
            .fail();
        }
        Ok(Some(PublishReceipt {
            index: record.index,
            operation: record.operation,
            transition: record.transition,
            state: record.resulting,
            replayed: true,
        }))
    }

    /// The payload of `version`, which `head` names as active, or `None`
    /// when the payload is gone because the index was destroyed since
    /// `head` was read.
    ///
    /// WHY the second head read: payloads are removed only by destroy, in
    /// the same write that marks the head destroyed. A payload missing under
    /// a head that is still active is damage; one missing because the head
    /// changed since it was read is a concurrent destroy.
    fn active_payload(
        &self,
        index: &IndexIdentity,
        version: IndexVersion,
        head: &IndexRecord<R>,
    ) -> Result<Option<VersionPayload<M, P>>, HeuremaError> {
        let Some(bytes) = self.backend.read_version(index, version)? else {
            return match self.record(index)?.map(|record| record.state) {
                Some(IndexState::Active { .. }) => Err(corrupt(format!(
                    "head of index {index} names version {version}, whose payload is missing"
                ))),
                Some(IndexState::Destroyed { .. }) | None => Ok(None),
            };
        };
        let payload = decode_payload(&bytes, index, version)?;
        if payload.config != head.config {
            return Err(corrupt(format!(
                "head of index {index} and the payload of version {version} name different configurations"
            )));
        }
        Ok(Some(payload))
    }
}

impl<'a, B: LifecycleBackend, M, P, R> Prepared<'a, B, M, P, R> {
    /// Step 6: stages the version payload and its marker in one durable
    /// write that changes no head. A Destroy stages nothing; its only write
    /// is [`Staged::publish`].
    ///
    /// # Errors
    ///
    /// [`HeuremaError::HeadChanged`] when the head moved since it was read,
    /// [`HeuremaError::StagedStateExists`] when the index gained staged
    /// state, and [`HeuremaError::Persistence`] when the backend fails; see
    /// [`LifecycleBackend::stage`].
    pub fn stage(self) -> Result<Staged<'a, B, M, P, R>, HeuremaError> {
        let Self {
            lifecycle,
            writer,
            plan,
            write,
        } = self;
        let write = match write {
            PlannedWrite::Version {
                expected_head,
                version,
                marker,
                payload,
            } => {
                lifecycle.backend.stage(StageWrite::new(
                    &plan.index,
                    version,
                    expected_head.as_deref(),
                    &marker,
                    &payload,
                ))?;
                StagedWrite::Version {
                    expected_head,
                    version,
                    marker,
                }
            }
            PlannedWrite::Destroy {
                expected_head,
                versions,
            } => StagedWrite::Destroy {
                expected_head,
                versions,
            },
        };
        Ok(Staged {
            lifecycle,
            writer,
            plan,
            write,
        })
    }

    /// The index the operation changes.
    #[must_use]
    pub const fn index(&self) -> &IndexIdentity {
        &self.plan.index
    }

    /// The operation's key and digest.
    #[must_use]
    pub const fn operation(&self) -> &OperationIdentity {
        &self.plan.operation
    }

    /// The state publishing the operation will leave the index in.
    #[must_use]
    pub const fn resulting(&self) -> &IndexState<R> {
        &self.plan.resulting
    }
}

impl<B: LifecycleBackend, M, P, R> Staged<'_, B, M, P, R> {
    /// Step 7, the publish point: in one atomic durable write, the new head,
    /// the operation record, and the removal of the staging marker (for a
    /// Destroy, the destroyed head, the operation record, and the removal of
    /// every version payload). Before it the old version is visible; after
    /// it the new one is, whole.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::HeadChanged`], [`HeuremaError::StagedStateMissing`],
    /// [`HeuremaError::StagedStateExists`], or
    /// [`HeuremaError::OperationRecorded`] when another writer changed the
    /// index since it was read, with nothing written; and
    /// [`HeuremaError::Persistence`] when the backend fails, in which case
    /// the write may or may not have taken effect. Applying the same
    /// operation again tells which: a publish that took effect replays at
    /// the same version.
    pub fn publish(self) -> Result<PublishReceipt<R>, HeuremaError> {
        let Self {
            lifecycle,
            writer,
            plan,
            write,
        } = self;
        match &write {
            StagedWrite::Version {
                expected_head,
                version,
                marker,
            } => lifecycle.backend.publish(PublishWrite::new(
                &plan.index,
                *version,
                &plan.operation.key,
                expected_head.as_deref(),
                marker,
                &plan.head,
                &plan.record,
            ))?,
            StagedWrite::Destroy {
                expected_head,
                versions,
            } => lifecycle.backend.destroy(DestroyWrite::new(
                &plan.index,
                &plan.operation.key,
                expected_head,
                &plan.head,
                &plan.record,
                versions,
            ))?,
        }
        drop(writer);
        Ok(PublishReceipt {
            index: plan.index,
            operation: plan.operation,
            transition: plan.transition,
            state: plan.resulting,
            replayed: false,
        })
    }

    /// The index the operation changes.
    #[must_use]
    pub const fn index(&self) -> &IndexIdentity {
        &self.plan.index
    }

    /// The operation's key and digest.
    #[must_use]
    pub const fn operation(&self) -> &OperationIdentity {
        &self.plan.operation
    }

    /// The state publishing the operation will leave the index in.
    #[must_use]
    pub const fn resulting(&self) -> &IndexState<R> {
        &self.plan.resulting
    }
}

/// Encodes every write `successor` needs.
fn encode_plan<M, P, R>(
    successor: Successor<M, P, R>,
    expected_head: Option<Vec<u8>>,
    operation: &LifecycleOperation<M, P, R>,
) -> Result<(Plan<R>, PlannedWrite), HeuremaError>
where
    M: MemberIdentity,
    P: ProvenanceReference,
    R: RetentionReference,
{
    let Successor {
        head,
        record,
        payload,
    } = successor;
    let write = match (payload, expected_head) {
        (Some(payload), expected_head) => PlannedWrite::Version {
            expected_head,
            version: payload.version,
            marker: encode(&StagingMarker::new(
                payload.index.clone(),
                payload.version,
                record.operation.clone(),
                record.transition,
            ))?,
            payload: encode(&payload)?,
        },
        (None, Some(expected_head)) => {
            let IndexState::Destroyed { last_version, .. } = &head.state else {
                return not_permitted(operation, IndexStateKind::Active);
            };
            let versions = (1..=last_version.get())
                .map(IndexVersion::try_from)
                .collect::<Result<_, _>>()?;
            PlannedWrite::Destroy {
                expected_head,
                versions,
            }
        }
        // INVARIANT: only Destroy publishes no payload, and it is permitted
        // only from Active, which has a head.
        (None, None) => return not_permitted(operation, IndexStateKind::Absent),
    };
    let plan = Plan {
        index: record.index.clone(),
        operation: record.operation.clone(),
        transition: record.transition,
        resulting: record.resulting.clone(),
        head: encode_head(&head)?,
        record: encode(&record)?,
    };
    Ok((plan, write))
}

fn not_permitted<T, M, P, R>(
    operation: &LifecycleOperation<M, P, R>,
    state: IndexStateKind,
) -> Result<T, HeuremaError> {
    TransitionNotPermittedSnafu {
        index: operation.index.clone(),
        transition: operation.change.transition(),
        state,
    }
    .fail()
}

#[track_caller]
fn not_found<T>(index: &IndexIdentity) -> Result<T, HeuremaError> {
    IndexNotFoundSnafu {
        name: index.to_string(),
    }
    .fail()
}

impl<B: fmt::Debug, M, P, R> fmt::Debug for IndexLifecycle<B, M, P, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IndexLifecycle")
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl<R: fmt::Debug> fmt::Debug for Plan<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Plan")
            .field("index", &self.index)
            .field("operation", &self.operation)
            .field("transition", &self.transition)
            .field("resulting", &self.resulting)
            .finish_non_exhaustive()
    }
}

impl<B, M, P, R: fmt::Debug> fmt::Debug for Prepared<'_, B, M, P, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Prepared")
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}

impl<B, M, P, R: fmt::Debug> fmt::Debug for Staged<'_, B, M, P, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Staged")
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}

impl<B, M, P, R: fmt::Debug> fmt::Debug for Preparation<'_, B, M, P, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Replayed(receipt) => formatter.debug_tuple("Replayed").field(receipt).finish(),
            Self::Ready(prepared) => formatter.debug_tuple("Ready").field(prepared).finish(),
        }
    }
}
