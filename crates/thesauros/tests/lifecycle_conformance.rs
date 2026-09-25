//! Conformance of heurēma's lifecycle driver, [`heurema::IndexLifecycle`]:
//! every case runs against `atmis::AtmisBackend` and against a tempdir
//! `thesauros::ThesaurosBackend`, and the durability cases close and reopen
//! the fjall database between steps.
//!
//! WHY one suite for both adapters: the driver's guarantees (one publish
//! point, refusals that write nothing, replay by key and digest, reads that
//! never see a staged version) must hold on either adapter, so each is pinned
//! once and checked against both.
//!
//! How interruption is injected, and what that proves:
//!
//! - The step API ([`IndexLifecycle::prepare`], [`Prepared::stage`],
//!   [`Staged::publish`]) stops an operation between two durable steps.
//!   Each step is durable before it returns, so dropping the lifecycle
//!   afterwards and reopening the fjall database reads exactly what a crash
//!   after that step would leave.
//! - [`Faulting`] makes one write fail with an injected
//!   [`HeuremaError::Persistence`], either before the wrapped backend applies
//!   it or after (a lost acknowledgement).
//! - [`Counting`] counts backend reads and writes, which proves a refusal
//!   happened before any adapter I/O (stateless) or before any write
//!   (stateful).
//!
//! What this cannot prove: a crash in the middle of one fjall batch. That
//! rests on fjall's journal (one checksummed batch, replayed whole or not at
//! all), which is relied on and not simulated here; and a regression from
//! `PersistMode::SyncAll` to fjall's buffered default would pass these tests,
//! because a clean drop flushes the journal. Both are review items.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt;

use atmis::AtmisBackend;
use heurema::{
    DestroyWrite, ErrorCategory, FtsConfig, HeuremaError, HnswConfig, HnswIndex, IdentifierKind,
    IndexChange, IndexConfig, IndexIdentity, IndexLifecycle, IndexMember, IndexName, IndexState,
    IndexStateKind, IndexVersion, LifecycleBackend, LifecycleOperation, MemberContent,
    MemberIdentity, OperationKey, OwnerNamespace, PersistenceBackend, PersistenceSource,
    Preparation, Prepared, ProvenanceReference, PublishReceipt, PublishWrite, QuarantineWrite,
    QuarantinedEntry, RetentionReference, StageWrite, Staged, StagedEntry, TokenizerConfig,
    VectorIndex,
};
use serde::{Deserialize, Serialize};
use thesauros::ThesaurosBackend;

type TestResult = Result<(), HeuremaError>;

/// test-local placeholder; heurēma defines no provenance shape
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct TestMember(u64);

impl MemberIdentity for TestMember {}

/// test-local placeholder; heurēma defines no provenance shape
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PlaceholderProvenance(u32);

impl ProvenanceReference for PlaceholderProvenance {}

/// test-local placeholder; heurēma defines no provenance shape
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PlaceholderRetention(u32);

impl RetentionReference for PlaceholderRetention {}

type Lifecycle<B> = IndexLifecycle<B, TestMember, PlaceholderProvenance, PlaceholderRetention>;
type Operation = LifecycleOperation<TestMember, PlaceholderProvenance, PlaceholderRetention>;
type Member = IndexMember<TestMember, PlaceholderProvenance>;

/// Runs each named case against both adapters, as `case::atmis` and
/// `case::thesauros`, so a failure names the adapter that broke.
macro_rules! conformance {
    ($($case:ident),+ $(,)?) => {$(
        mod $case {
            #[test]
            fn atmis() -> super::TestResult {
                super::$case(&super::Memory)
            }

            #[test]
            fn thesauros() -> super::TestResult {
                super::$case(&super::Disk::new()?)
            }
        }
    )+};
}

conformance!(
    create_insert_remove_rebuild_publish_successive_versions,
    interruption_before_publish_reopens_to_the_old_version,
    interruption_after_publish_reopens_to_the_new_version,
    publish_failure_before_effect_leaves_the_old_version,
    publish_failure_after_effect_retry_replays_at_the_same_version,
    stage_failure_leaves_nothing_or_orphan_staged_state,
    every_visible_hit_after_reopen_carries_identity_and_provenance,
    stateless_refusals_make_no_backend_call,
    stateful_refusals_read_but_never_write,
    replayed_key_with_same_content_is_a_no_op_across_reopen,
    replayed_key_with_different_content_is_refused_across_reopen,
    orphan_staged_state_refuses_the_next_mutation_of_that_index,
    other_indexes_proceed_while_one_holds_staged_state,
    destroy_is_atomic_across_reopen,
    staged_version_is_not_readable,
    two_lifecycles_over_one_backend_cannot_both_publish,
    lifecycle_and_snapshot_keyspaces_are_independent,
);

/// Where a case's lifecycle state lives, and how it is closed and reopened.
trait Store {
    type Backend: LifecycleBackend + PersistenceBackend;

    /// A backend over the store.
    fn open(&self) -> Result<Self::Backend, HeuremaError>;

    /// Closes `backend` and opens the store again.
    fn reopen(&self, backend: Self::Backend) -> Result<Self::Backend, HeuremaError>;
}

/// The in-memory adapter. Its state lives exactly as long as the backend
/// value, so a reopen hands the same backend to a new lifecycle.
struct Memory;

impl Store for Memory {
    type Backend = AtmisBackend;

    fn open(&self) -> Result<AtmisBackend, HeuremaError> {
        Ok(AtmisBackend::new())
    }

    fn reopen(&self, backend: AtmisBackend) -> Result<AtmisBackend, HeuremaError> {
        Ok(backend)
    }
}

/// The durable adapter over a temporary directory. A reopen drops the only
/// handle on the fjall database and opens the path again, so the new
/// backend reads what reached the journal.
struct Disk {
    dir: tempfile::TempDir,
}

impl Disk {
    fn new() -> Result<Self, HeuremaError> {
        Ok(Self {
            dir: tempfile::tempdir().map_err(storage_error)?,
        })
    }
}

impl Store for Disk {
    type Backend = ThesaurosBackend;

    fn open(&self) -> Result<ThesaurosBackend, HeuremaError> {
        ThesaurosBackend::open(self.dir.path())
    }

    fn reopen(&self, backend: ThesaurosBackend) -> Result<ThesaurosBackend, HeuremaError> {
        drop(backend);
        self.open()
    }
}

/// A new lifecycle over `store`'s backend.
fn open<S: Store>(store: &S) -> Result<Lifecycle<S::Backend>, HeuremaError> {
    Lifecycle::open(store.open()?)
}

/// Closes `lifecycle` and its backend, and opens both again.
fn reopen<S: Store>(
    store: &S,
    lifecycle: Lifecycle<S::Backend>,
) -> Result<Lifecycle<S::Backend>, HeuremaError> {
    Lifecycle::open(store.reopen(lifecycle.into_backend())?)
}

/// Counts every backend read and write that passes through it.
struct Counting<B> {
    inner: B,
    reads: Cell<usize>,
    writes: Cell<usize>,
}

impl<B> Counting<B> {
    const fn new(inner: B) -> Self {
        Self {
            inner,
            reads: Cell::new(0),
            writes: Cell::new(0),
        }
    }

    fn reset(&self) {
        self.reads.set(0);
        self.writes.set(0);
    }

    /// `(reads, writes)` since the last reset.
    fn counts(&self) -> (usize, usize) {
        (self.reads.get(), self.writes.get())
    }

    fn read(&self) -> &B {
        self.reads.set(self.reads.get() + 1);
        &self.inner
    }

    fn write(&self) -> &B {
        self.writes.set(self.writes.get() + 1);
        &self.inner
    }
}

impl<B: LifecycleBackend> LifecycleBackend for Counting<B> {
    fn read_head(&self, index: &IndexIdentity) -> Result<Option<Vec<u8>>, HeuremaError> {
        self.read().read_head(index)
    }

    fn read_version(
        &self,
        index: &IndexIdentity,
        version: IndexVersion,
    ) -> Result<Option<Vec<u8>>, HeuremaError> {
        self.read().read_version(index, version)
    }

    fn read_operation(
        &self,
        index: &IndexIdentity,
        key: &OperationKey,
    ) -> Result<Option<Vec<u8>>, HeuremaError> {
        self.read().read_operation(index, key)
    }

    fn read_staging(
        &self,
        index: &IndexIdentity,
    ) -> Result<Option<(IndexVersion, Vec<u8>)>, HeuremaError> {
        self.read().read_staging(index)
    }

    fn stage(&self, write: StageWrite<'_>) -> Result<(), HeuremaError> {
        self.write().stage(write)
    }

    fn publish(&self, write: PublishWrite<'_>) -> Result<(), HeuremaError> {
        self.write().publish(write)
    }

    fn destroy(&self, write: DestroyWrite<'_>) -> Result<(), HeuremaError> {
        self.write().destroy(write)
    }

    fn list_heads(&self) -> Result<Vec<(IndexIdentity, Vec<u8>)>, HeuremaError> {
        self.read().list_heads()
    }

    fn list_staged(&self) -> Result<Vec<StagedEntry>, HeuremaError> {
        self.read().list_staged()
    }

    fn list_operations(
        &self,
        index: &IndexIdentity,
    ) -> Result<Vec<(OperationKey, Vec<u8>)>, HeuremaError> {
        self.read().list_operations(index)
    }

    fn quarantine(&self, write: QuarantineWrite<'_>) -> Result<(), HeuremaError> {
        self.write().quarantine(write)
    }

    fn list_quarantined(&self) -> Result<Vec<QuarantinedEntry>, HeuremaError> {
        self.read().list_quarantined()
    }
}

/// Which backend write a fault targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Stage,
    Publish,
    Destroy,
}

/// Whether an injected fault fires before the wrapped write takes effect,
/// or after it (the write is durable, but its acknowledgement is lost).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum When {
    BeforeEffect,
    AfterEffect,
}

/// Delegates to `inner`, and fails the next write of an armed [`Step`] with
/// an injected [`HeuremaError::Persistence`]. A fault fires once.
struct Faulting<B> {
    inner: B,
    fault: Cell<Option<(Step, When)>>,
}

impl<B> Faulting<B> {
    const fn new(inner: B) -> Self {
        Self {
            inner,
            fault: Cell::new(None),
        }
    }

    fn arm(&self, step: Step, when: When) {
        self.fault.set(Some((step, when)));
    }

    fn into_inner(self) -> B {
        self.inner
    }

    fn write(&self, step: Step, effect: impl FnOnce(&B) -> TestResult) -> TestResult {
        match self.fault.get() {
            Some((armed, When::BeforeEffect)) if armed == step => {
                self.fault.set(None);
                Err(injected())
            }
            Some((armed, When::AfterEffect)) if armed == step => {
                self.fault.set(None);
                effect(&self.inner)?;
                Err(injected())
            }
            _ => effect(&self.inner),
        }
    }
}

impl<B: LifecycleBackend> LifecycleBackend for Faulting<B> {
    fn read_head(&self, index: &IndexIdentity) -> Result<Option<Vec<u8>>, HeuremaError> {
        self.inner.read_head(index)
    }

    fn read_version(
        &self,
        index: &IndexIdentity,
        version: IndexVersion,
    ) -> Result<Option<Vec<u8>>, HeuremaError> {
        self.inner.read_version(index, version)
    }

    fn read_operation(
        &self,
        index: &IndexIdentity,
        key: &OperationKey,
    ) -> Result<Option<Vec<u8>>, HeuremaError> {
        self.inner.read_operation(index, key)
    }

    fn read_staging(
        &self,
        index: &IndexIdentity,
    ) -> Result<Option<(IndexVersion, Vec<u8>)>, HeuremaError> {
        self.inner.read_staging(index)
    }

    fn stage(&self, write: StageWrite<'_>) -> Result<(), HeuremaError> {
        self.write(Step::Stage, |inner| inner.stage(write))
    }

    fn publish(&self, write: PublishWrite<'_>) -> Result<(), HeuremaError> {
        self.write(Step::Publish, |inner| inner.publish(write))
    }

    fn destroy(&self, write: DestroyWrite<'_>) -> Result<(), HeuremaError> {
        self.write(Step::Destroy, |inner| inner.destroy(write))
    }

    fn list_heads(&self) -> Result<Vec<(IndexIdentity, Vec<u8>)>, HeuremaError> {
        self.inner.list_heads()
    }

    fn list_staged(&self) -> Result<Vec<StagedEntry>, HeuremaError> {
        self.inner.list_staged()
    }

    fn list_operations(
        &self,
        index: &IndexIdentity,
    ) -> Result<Vec<(OperationKey, Vec<u8>)>, HeuremaError> {
        self.inner.list_operations(index)
    }

    fn quarantine(&self, write: QuarantineWrite<'_>) -> Result<(), HeuremaError> {
        self.inner.quarantine(write)
    }

    fn list_quarantined(&self) -> Result<Vec<QuarantinedEntry>, HeuremaError> {
        self.inner.list_quarantined()
    }
}

/// A lifecycle over a [`Faulting`] wrapper of `store`'s backend.
fn open_faulting<S: Store>(store: &S) -> Result<Lifecycle<Faulting<S::Backend>>, HeuremaError> {
    Lifecycle::open(Faulting::new(store.open()?))
}

/// Closes a faulting lifecycle and its backend, and opens both again.
fn reopen_faulting<S: Store>(
    store: &S,
    lifecycle: Lifecycle<Faulting<S::Backend>>,
) -> Result<Lifecycle<Faulting<S::Backend>>, HeuremaError> {
    let backend = store.reopen(lifecycle.into_backend().into_inner())?;
    Lifecycle::open(Faulting::new(backend))
}

/// Why an injected write failed.
#[derive(Debug)]
struct InjectedFault;

impl fmt::Display for InjectedFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("injected fault")
    }
}

impl std::error::Error for InjectedFault {}

#[track_caller]
fn injected() -> HeuremaError {
    HeuremaError::Persistence {
        source: PersistenceSource::new(InjectedFault),
        location: std::panic::Location::caller(),
    }
}

// WHY: tempdir creation fails with `io::Error`; routing it through
// `PersistenceSource` lets every case return `Result<(), HeuremaError>` and
// use `?`, since the workspace lints deny `.expect()`.
fn storage_error(source: std::io::Error) -> HeuremaError {
    HeuremaError::Persistence {
        source: PersistenceSource::new(source),
        location: std::panic::Location::caller(),
    }
}

fn index(name: &str) -> Result<IndexIdentity, HeuremaError> {
    Ok(IndexIdentity::new(
        OwnerNamespace::try_from("example")?,
        IndexName::try_from(name)?,
    ))
}

fn version(value: u64) -> Result<IndexVersion, HeuremaError> {
    IndexVersion::try_from(value)
}

fn operation(
    index: &IndexIdentity,
    key: &str,
    change: IndexChange<TestMember, PlaceholderProvenance, PlaceholderRetention>,
) -> Result<Operation, HeuremaError> {
    Ok(LifecycleOperation::new(
        index.clone(),
        OperationKey::try_from(key)?,
        change,
    ))
}

fn create_vector(index: &IndexIdentity, key: &str, dims: usize) -> Result<Operation, HeuremaError> {
    operation(
        index,
        key,
        IndexChange::Create {
            config: IndexConfig::Vector(HnswConfig::new(dims)),
        },
    )
}

fn create_text(index: &IndexIdentity, key: &str) -> Result<Operation, HeuremaError> {
    operation(
        index,
        key,
        IndexChange::Create {
            config: IndexConfig::Fts(FtsConfig::simple()),
        },
    )
}

fn insert(
    index: &IndexIdentity,
    key: &str,
    members: Vec<Member>,
) -> Result<Operation, HeuremaError> {
    operation(index, key, IndexChange::Insert { members })
}

fn remove(index: &IndexIdentity, key: &str, ids: &[u64]) -> Result<Operation, HeuremaError> {
    operation(
        index,
        key,
        IndexChange::Remove {
            members: ids.iter().copied().map(TestMember).collect(),
        },
    )
}

fn rebuild(
    index: &IndexIdentity,
    key: &str,
    dims: usize,
    members: Vec<Member>,
) -> Result<Operation, HeuremaError> {
    operation(
        index,
        key,
        IndexChange::Rebuild {
            config: IndexConfig::Vector(HnswConfig::new(dims)),
            members,
        },
    )
}

fn destroy(index: &IndexIdentity, key: &str, retention: u32) -> Result<Operation, HeuremaError> {
    operation(
        index,
        key,
        IndexChange::Destroy {
            retention: PlaceholderRetention(retention),
        },
    )
}

fn vector(id: u64, provenance: u32, components: &[f32]) -> Member {
    IndexMember::new(
        TestMember(id),
        PlaceholderProvenance(provenance),
        MemberContent::Vector(components.to_vec()),
    )
}

fn document(id: u64, provenance: u32, text: &str) -> Member {
    IndexMember::new(
        TestMember(id),
        PlaceholderProvenance(provenance),
        MemberContent::Document(text.to_owned()),
    )
}

/// An active two-dimensional vector index at version 2, holding members 1
/// and 2 (provenance 10 and 20).
fn seeded<B: LifecycleBackend>(
    lifecycle: &Lifecycle<B>,
    index: &IndexIdentity,
) -> Result<(), HeuremaError> {
    lifecycle.apply(create_vector(index, "create", 2)?)?;
    lifecycle.apply(insert(
        index,
        "insert",
        vec![vector(1, 10, &[1.0, 0.0]), vector(2, 20, &[0.0, 1.0])],
    )?)?;
    Ok(())
}

/// One member as `(identity, provenance, introduced version)`.
type Observed = (u64, u32, u64);

/// The active version of `index` and its members, as a reader sees them.
fn observe<B: LifecycleBackend>(
    lifecycle: &Lifecycle<B>,
    index: &IndexIdentity,
) -> Result<(u64, Vec<Observed>), HeuremaError> {
    let published = lifecycle.index(index)?;
    let members = published
        .members()
        .map(|(id, entry)| (id.0, entry.provenance.0, entry.introduced.get()))
        .collect();
    Ok((published.version().get(), members))
}

/// The state `index`'s head records, as `(kind, version)`.
fn head_state<B: LifecycleBackend>(
    lifecycle: &Lifecycle<B>,
    index: &IndexIdentity,
) -> Result<Option<(IndexStateKind, u64)>, HeuremaError> {
    Ok(lifecycle.record(index)?.map(|record| {
        let kind = record.state.kind();
        let version = match record.state {
            IndexState::Active { version } => version.get(),
            IndexState::Destroyed { last_version, .. } => last_version.get(),
            other => panic!("unexpected {other:?}"),
        };
        (kind, version)
    }))
}

/// The staged version `index` holds, if any.
fn staged_version<B: LifecycleBackend>(
    lifecycle: &Lifecycle<B>,
    index: &IndexIdentity,
) -> Result<Option<u64>, HeuremaError> {
    Ok(lifecycle
        .backend()
        .read_staging(index)?
        .map(|(staged, _)| staged.get()))
}

/// Unwraps a refusal, failing the case if the call succeeded.
#[track_caller]
fn refused<T: fmt::Debug>(result: Result<T, HeuremaError>) -> HeuremaError {
    match result {
        Ok(value) => panic!("expected a refusal, got {value:?}"),
        Err(error) => error,
    }
}

/// Requires `error` to be a staged-state refusal naming `expected`.
#[track_caller]
fn assert_staged_state(error: &HeuremaError, expected: u64) {
    match error {
        HeuremaError::StagedStateExists { version, .. } if version.get() == expected => {
            assert_eq!(error.category(), ErrorCategory::RecoveryRequired);
        }
        other => panic!("unexpected {other:?}"),
    }
}

/// Requires `error` to refuse `transition` from `state`.
#[track_caller]
fn assert_not_permitted(error: &HeuremaError, state: IndexStateKind) {
    match error {
        HeuremaError::TransitionNotPermitted { state: found, .. } if *found == state => {}
        other => panic!("unexpected {other:?}"),
    }
}

/// Requires `error` to be an injected storage failure.
#[track_caller]
fn assert_injected(error: &HeuremaError) {
    match error {
        HeuremaError::Persistence { source, .. } if source.to_string() == "injected fault" => {}
        other => panic!("unexpected {other:?}"),
    }
}

/// Requires `receipt` to be a fresh publish (or a replay) of `version`.
#[track_caller]
fn assert_receipt(receipt: &PublishReceipt<PlaceholderRetention>, expected: u64, replayed: bool) {
    assert_eq!(receipt.version().get(), expected, "{receipt:?}");
    assert_eq!(receipt.replayed, replayed, "{receipt:?}");
}

/// Stages `operation` and returns the staged step without publishing it.
fn stage<B: LifecycleBackend>(
    lifecycle: &Lifecycle<B>,
    operation: Operation,
) -> Result<Staged<'_, B, TestMember, PlaceholderProvenance, PlaceholderRetention>, HeuremaError> {
    ready(lifecycle.prepare(operation)?).stage()
}

/// Unwraps a preparation that must be ready to stage.
#[track_caller]
fn ready<B>(
    preparation: Preparation<'_, B, TestMember, PlaceholderProvenance, PlaceholderRetention>,
) -> Prepared<'_, B, TestMember, PlaceholderProvenance, PlaceholderRetention> {
    match preparation {
        Preparation::Ready(prepared) => prepared,
        other => panic!("unexpected {other:?}"),
    }
}

fn create_insert_remove_rebuild_publish_successive_versions<S: Store>(store: &S) -> TestResult {
    let lifecycle = open(store)?;
    let notes = index("notes")?;

    let created = lifecycle.apply(create_vector(&notes, "create", 2)?)?;
    assert_receipt(&created, 1, false);
    assert_eq!(observe(&lifecycle, &notes)?, (1, vec![]));

    let inserted = lifecycle.apply(insert(
        &notes,
        "insert",
        vec![
            vector(3, 30, &[1.0, 1.0]),
            vector(1, 10, &[1.0, 0.0]),
            vector(2, 20, &[0.0, 1.0]),
        ],
    )?)?;
    assert_receipt(&inserted, 2, false);
    assert_eq!(
        observe(&lifecycle, &notes)?,
        (2, vec![(1, 10, 2), (2, 20, 2), (3, 30, 2)])
    );

    assert_receipt(&lifecycle.apply(remove(&notes, "remove", &[2])?)?, 3, false);
    assert_eq!(
        observe(&lifecycle, &notes)?,
        (3, vec![(1, 10, 2), (3, 30, 2)])
    );

    let rebuilt = lifecycle.apply(rebuild(
        &notes,
        "rebuild",
        3,
        vec![
            vector(1, 11, &[1.0, 0.0, 0.0]),
            vector(4, 40, &[0.0, 1.0, 0.0]),
        ],
    )?)?;
    assert_receipt(&rebuilt, 4, false);
    let published = lifecycle.index(&notes)?;
    assert_eq!(published.config(), &IndexConfig::Vector(HnswConfig::new(3)));
    assert_eq!(
        published
            .member(&TestMember(1))
            .map(|entry| entry.supersedes),
        Some(Some(version(2)?)),
        "a rebuilt member links to the entry it replaced"
    );
    assert_eq!(
        observe(&lifecycle, &notes)?,
        (4, vec![(1, 11, 4), (4, 40, 4)])
    );

    // NOTE: a Remove naming only absent members still publishes a version.
    let absent = lifecycle.apply(remove(&notes, "remove-absent", &[9])?)?;
    assert_receipt(&absent, 5, false);
    assert_eq!(
        observe(&lifecycle, &notes)?,
        (5, vec![(1, 11, 4), (4, 40, 4)])
    );

    let destroyed = lifecycle.apply(destroy(&notes, "destroy", 7)?)?;
    assert_receipt(&destroyed, 5, false);
    assert_eq!(
        destroyed.state,
        IndexState::Destroyed {
            last_version: version(5)?,
            retention: PlaceholderRetention(7),
            operation: OperationKey::try_from("destroy")?,
        }
    );
    assert_eq!(
        head_state(&lifecycle, &notes)?,
        Some((IndexStateKind::Destroyed, 5))
    );
    match refused(lifecycle.index(&notes)) {
        HeuremaError::IndexNotFound { .. } => Ok(()),
        other => panic!("unexpected {other:?}"),
    }
}

fn interruption_before_publish_reopens_to_the_old_version<S: Store>(store: &S) -> TestResult {
    let lifecycle = open(store)?;
    let notes = index("notes")?;
    seeded(&lifecycle, &notes)?;
    let before = observe(&lifecycle, &notes)?;

    let interrupted = || insert(&notes, "interrupted", vec![vector(3, 30, &[1.0, 1.0])]);
    let staged = stage(&lifecycle, interrupted()?)?;
    assert_eq!(
        staged.resulting(),
        &IndexState::Active {
            version: version(3)?
        }
    );
    drop(staged);
    let lifecycle = reopen(store, lifecycle)?;

    assert_eq!(
        observe(&lifecycle, &notes)?,
        before,
        "the old version, whole"
    );
    assert_eq!(
        head_state(&lifecycle, &notes)?,
        Some((IndexStateKind::Active, 2))
    );
    assert_eq!(
        staged_version(&lifecycle, &notes)?,
        Some(3),
        "the staged version survives, unpublished"
    );
    assert!(
        lifecycle
            .backend()
            .read_version(&notes, version(3)?)?
            .is_some(),
        "its payload is durable but unreachable"
    );

    // WHY: until recovery lands, orphan staged state refuses every mutation
    // of its index, the interrupted operation's own retry included.
    for blocked in [
        interrupted()?,
        insert(&notes, "next", vec![vector(4, 40, &[0.0, 0.0])])?,
        destroy(&notes, "destroy", 1)?,
    ] {
        assert_staged_state(&refused(lifecycle.apply(blocked)), 3);
    }
    assert_eq!(observe(&lifecycle, &notes)?, before);
    Ok(())
}

fn interruption_after_publish_reopens_to_the_new_version<S: Store>(store: &S) -> TestResult {
    let lifecycle = open(store)?;
    let notes = index("notes")?;
    seeded(&lifecycle, &notes)?;

    let published = stage(
        &lifecycle,
        insert(&notes, "published", vec![vector(3, 30, &[1.0, 1.0])])?,
    )?
    .publish()?;
    assert_receipt(&published, 3, false);
    let lifecycle = reopen(store, lifecycle)?;

    assert_eq!(
        observe(&lifecycle, &notes)?,
        (3, vec![(1, 10, 2), (2, 20, 2), (3, 30, 3)])
    );
    assert_eq!(
        staged_version(&lifecycle, &notes)?,
        None,
        "the marker is gone"
    );
    let replayed = lifecycle.apply(insert(
        &notes,
        "published",
        vec![vector(3, 30, &[1.0, 1.0])],
    )?)?;
    assert_receipt(&replayed, 3, true);
    Ok(())
}

fn publish_failure_before_effect_leaves_the_old_version<S: Store>(store: &S) -> TestResult {
    let lifecycle = open_faulting(store)?;
    let notes = index("notes")?;
    seeded(&lifecycle, &notes)?;
    let before = observe(&lifecycle, &notes)?;

    let failing = || insert(&notes, "failing", vec![vector(3, 30, &[1.0, 1.0])]);
    lifecycle.backend().arm(Step::Publish, When::BeforeEffect);
    assert_injected(&refused(lifecycle.apply(failing()?)));
    assert_eq!(
        observe(&lifecycle, &notes)?,
        before,
        "the old version, whole"
    );
    assert_eq!(
        staged_version(&lifecycle, &notes)?,
        Some(3),
        "the stage before the failed publish stays"
    );

    let lifecycle = reopen_faulting(store, lifecycle)?;
    assert_eq!(observe(&lifecycle, &notes)?, before);
    assert_eq!(
        lifecycle
            .backend()
            .read_operation(&notes, &OperationKey::try_from("failing")?)?,
        None,
        "an operation that never published is not recorded"
    );
    assert_staged_state(&refused(lifecycle.apply(failing()?)), 3);
    Ok(())
}

fn publish_failure_after_effect_retry_replays_at_the_same_version<S: Store>(
    store: &S,
) -> TestResult {
    let lifecycle = open_faulting(store)?;
    let notes = index("notes")?;
    seeded(&lifecycle, &notes)?;

    let lost = || insert(&notes, "lost-ack", vec![vector(3, 30, &[1.0, 1.0])]);
    lifecycle.backend().arm(Step::Publish, When::AfterEffect);
    assert_injected(&refused(lifecycle.apply(lost()?)));

    // WHY: the failure hid a publish that took effect. The retry must find
    // the operation record and replay at that version, not publish again.
    let retried = lifecycle.apply(lost()?)?;
    assert_receipt(&retried, 3, true);
    assert_eq!(
        head_state(&lifecycle, &notes)?,
        Some((IndexStateKind::Active, 3))
    );

    let lifecycle = reopen_faulting(store, lifecycle)?;
    assert_receipt(&lifecycle.apply(lost()?)?, 3, true);
    let changed = insert(&notes, "lost-ack", vec![vector(3, 31, &[1.0, 1.0])])?;
    match refused(lifecycle.apply(changed)) {
        HeuremaError::OperationConflict { conflict, .. } => {
            assert_eq!(conflict.recorded(), &retried.operation.digest);
            assert_ne!(conflict.requested(), conflict.recorded());
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(
        observe(&lifecycle, &notes)?,
        (3, vec![(1, 10, 2), (2, 20, 2), (3, 30, 3)])
    );

    // NOTE: The same holds for a destroy whose acknowledgement is lost.
    lifecycle.backend().arm(Step::Destroy, When::AfterEffect);
    assert_injected(&refused(lifecycle.apply(destroy(&notes, "destroy", 1)?)));
    let destroyed = lifecycle.apply(destroy(&notes, "destroy", 1)?)?;
    assert_receipt(&destroyed, 3, true);
    assert_eq!(destroyed.state.kind(), IndexStateKind::Destroyed);
    Ok(())
}

fn stage_failure_leaves_nothing_or_orphan_staged_state<S: Store>(store: &S) -> TestResult {
    let lifecycle = open_faulting(store)?;
    let notes = index("notes")?;
    seeded(&lifecycle, &notes)?;

    lifecycle.backend().arm(Step::Stage, When::BeforeEffect);
    let first = || insert(&notes, "first", vec![vector(3, 30, &[1.0, 1.0])]);
    assert_injected(&refused(lifecycle.apply(first()?)));
    assert_eq!(
        staged_version(&lifecycle, &notes)?,
        None,
        "nothing was staged"
    );
    assert_receipt(&lifecycle.apply(first()?)?, 3, false);

    lifecycle.backend().arm(Step::Stage, When::AfterEffect);
    let second = || insert(&notes, "second", vec![vector(4, 40, &[0.5, 0.5])]);
    assert_injected(&refused(lifecycle.apply(second()?)));
    assert_eq!(
        staged_version(&lifecycle, &notes)?,
        Some(4),
        "staged, never published"
    );
    assert_eq!(
        head_state(&lifecycle, &notes)?,
        Some((IndexStateKind::Active, 3))
    );
    let lifecycle = reopen_faulting(store, lifecycle)?;
    assert_staged_state(&refused(lifecycle.apply(second()?)), 4);
    Ok(())
}

fn every_visible_hit_after_reopen_carries_identity_and_provenance<S: Store>(
    store: &S,
) -> TestResult {
    let lifecycle = open(store)?;
    let vectors = index("vectors")?;
    let texts = index("texts")?;
    lifecycle.apply(create_vector(&vectors, "create", 2)?)?;
    let points = (1..=6_u32).map(|id| {
        let x = id as f32;
        vector(u64::from(id), id * 10, &[x, 6.0 - x])
    });
    lifecycle.apply(insert(&vectors, "insert", points.collect())?)?;
    lifecycle.apply(insert(
        &vectors,
        "replace",
        vec![vector(2, 21, &[2.5, 3.5])],
    )?)?;
    lifecycle.apply(remove(&vectors, "remove", &[5])?)?;
    lifecycle.apply(create_text(&texts, "create")?)?;
    lifecycle.apply(insert(
        &texts,
        "insert",
        vec![
            document(1, 100, "alpha beta"),
            document(2, 200, "alpha gamma gamma"),
            document(3, 300, "alpha delta"),
        ],
    )?)?;
    lifecycle.apply(insert(
        &texts,
        "replace",
        vec![document(3, 301, "alpha epsilon")],
    )?)?;
    let lifecycle = reopen(store, lifecycle)?;

    let expected_vectors = BTreeMap::from([(1, 10), (2, 21), (3, 30), (4, 40), (6, 60)]);
    let published = lifecycle.index(&vectors)?;
    let hits = published.query_vector(&[0.0, 0.0], published.len())?;
    let seen: BTreeMap<u64, u32> = hits
        .iter()
        .map(|hit| (hit.id.0, hit.provenance.0))
        .collect();
    assert_eq!(
        seen, expected_vectors,
        "every member is a hit, with its provenance"
    );
    assert!(hits.iter().all(|hit| hit.version == published.version()));

    let expected_texts = BTreeMap::from([(1, 100), (2, 200), (3, 301)]);
    let published = lifecycle.index(&texts)?;
    let hits = published.query_text("alpha", published.len())?;
    let seen: BTreeMap<u64, u32> = hits
        .iter()
        .map(|hit| (hit.id.0, hit.provenance.0))
        .collect();
    assert_eq!(seen, expected_texts);
    assert!(hits.iter().all(|hit| hit.version == published.version()));

    match refused(published.query_vector(&[0.0, 0.0], 1)) {
        HeuremaError::FamilyMismatch { .. } => Ok(()),
        other => panic!("unexpected {other:?}"),
    }
}

/// test-local placeholder; heurēma defines no provenance shape. It encodes
/// as a map, which no engine can key its members by.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct CompositeMember {
    shard: u8,
    id: u64,
}

impl MemberIdentity for CompositeMember {}

/// test-local placeholder; heurēma defines no provenance shape. It holds a
/// float, which the canonical operation encoding refuses.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FloatProvenance(f64);

impl PartialEq for FloatProvenance {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for FloatProvenance {}

impl ProvenanceReference for FloatProvenance {}

/// Whether a refusal is the one a case expects.
type Expectation = fn(&HeuremaError) -> bool;

fn stateless_refusals_make_no_backend_call<S: Store>(store: &S) -> TestResult {
    let counting = Counting::new(store.open()?);
    let lifecycle = Lifecycle::open(&counting)?;
    let notes = index("notes")?;
    let fresh = index("fresh")?;
    seeded(&lifecycle, &notes)?;
    let mut ngram = FtsConfig::simple();
    ngram.tokenizer = TokenizerConfig::new("NGram", Vec::new());

    let cases: Vec<(&str, Operation, Expectation)> = vec![
        ("empty insert", insert(&notes, "k", vec![])?, |error| {
            matches!(error, HeuremaError::EmptyBatch { .. })
        }),
        ("empty remove", remove(&notes, "k", &[])?, |error| {
            matches!(error, HeuremaError::EmptyBatch { .. })
        }),
        (
            "member named twice",
            insert(
                &notes,
                "k",
                vec![vector(1, 1, &[0.0, 0.0]), vector(1, 2, &[1.0, 1.0])],
            )?,
            |error| matches!(error, HeuremaError::DuplicateMember { .. }),
        ),
        (
            "non-finite vector",
            insert(&notes, "k", vec![vector(3, 1, &[f32::NAN, 0.0])])?,
            |error| matches!(error, HeuremaError::InvalidVector { .. }),
        ),
        (
            "zero-dimension configuration",
            create_vector(&fresh, "k", 0)?,
            |error| matches!(error, HeuremaError::InvalidHnswConfig { .. }),
        ),
        (
            "unimplemented analyzer",
            operation(
                &fresh,
                "k",
                IndexChange::Create {
                    config: IndexConfig::Fts(ngram),
                },
            )?,
            |error| matches!(error, HeuremaError::NotYetImplemented { .. }),
        ),
        (
            "mixed-family insert",
            insert(
                &notes,
                "k",
                vec![vector(3, 1, &[0.0, 0.0]), document(4, 1, "text")],
            )?,
            |error| matches!(error, HeuremaError::FamilyMismatch { .. }),
        ),
        (
            "rebuild member of the wrong dimension",
            rebuild(&notes, "k", 3, vec![vector(1, 1, &[0.0, 0.0])])?,
            |error| matches!(error, HeuremaError::DimensionMismatch { .. }),
        ),
    ];
    for (label, refused_operation, expected) in cases {
        counting.reset();
        let error = refused(lifecycle.apply(refused_operation));
        assert!(expected(&error), "{label}: {error:?}");
        assert_eq!(counting.counts(), (0, 0), "{label}: {error}");
    }

    consumer_type_refusals_make_no_backend_call(&counting, &notes)
}

/// The two stateless refusals that depend on the consumer's own types: a
/// member identity that encodes as neither a string nor an integer, and a
/// provenance holding a float.
fn consumer_type_refusals_make_no_backend_call<B: LifecycleBackend>(
    counting: &Counting<B>,
    notes: &IndexIdentity,
) -> TestResult {
    counting.reset();
    let composite =
        IndexLifecycle::<_, CompositeMember, PlaceholderProvenance, PlaceholderRetention>::open(
            counting,
        )?;
    let error = refused(composite.apply(LifecycleOperation::new(
        notes.clone(),
        OperationKey::try_from("k")?,
        IndexChange::Insert {
            members: vec![IndexMember::new(
                CompositeMember { shard: 1, id: 1 },
                PlaceholderProvenance(1),
                MemberContent::Vector(vec![0.0, 0.0]),
            )],
        },
    )));
    assert!(
        matches!(
            error,
            HeuremaError::InvalidIdentifier {
                kind: IdentifierKind::MemberIdentity,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(counting.counts(), (0, 0), "{error}");

    let floating =
        IndexLifecycle::<_, TestMember, FloatProvenance, PlaceholderRetention>::open(counting)?;
    let error = refused(floating.apply(LifecycleOperation::new(
        notes.clone(),
        OperationKey::try_from("k")?,
        IndexChange::Insert {
            members: vec![IndexMember::new(
                TestMember(3),
                FloatProvenance(0.5),
                MemberContent::Vector(vec![0.0, 0.0]),
            )],
        },
    )));
    assert!(
        matches!(error, HeuremaError::UnencodableOperation { .. }),
        "{error:?}"
    );
    assert_eq!(counting.counts(), (0, 0), "{error}");
    Ok(())
}

fn refused_from_absent(error: &HeuremaError) -> bool {
    matches!(
        error,
        HeuremaError::TransitionNotPermitted {
            state: IndexStateKind::Absent,
            ..
        }
    )
}

fn refused_from_active(error: &HeuremaError) -> bool {
    matches!(
        error,
        HeuremaError::TransitionNotPermitted {
            state: IndexStateKind::Active,
            ..
        }
    )
}

fn refused_from_destroyed(error: &HeuremaError) -> bool {
    matches!(
        error,
        HeuremaError::TransitionNotPermitted {
            state: IndexStateKind::Destroyed,
            ..
        }
    )
}

/// One index in each state a stateful refusal can meet: `notes` (active
/// vector index, [`seeded`]), `texts` (active text index), `gone`
/// (destroyed), `blocked` (active, holding orphan staged state), and
/// `absent`.
fn stateful_fixture<B: LifecycleBackend>(
    lifecycle: &Lifecycle<B>,
) -> Result<[IndexIdentity; 5], HeuremaError> {
    let [notes, texts, gone, blocked, absent] =
        ["notes", "texts", "gone", "blocked", "absent"].map(index);
    let indexes = [notes?, texts?, gone?, blocked?, absent?];
    let [notes, texts, gone, blocked, _] = &indexes;
    seeded(lifecycle, notes)?;
    lifecycle.apply(create_text(texts, "create")?)?;
    lifecycle.apply(create_vector(gone, "create", 2)?)?;
    lifecycle.apply(destroy(gone, "destroy", 1)?)?;
    lifecycle.apply(create_vector(blocked, "create", 2)?)?;
    drop(stage(
        lifecycle,
        insert(blocked, "interrupted", vec![vector(1, 1, &[0.0, 0.0])])?,
    )?);
    Ok(indexes)
}

fn stateful_refusals_read_but_never_write<S: Store>(store: &S) -> TestResult {
    let counting = Counting::new(store.open()?);
    let lifecycle = Lifecycle::open(&counting)?;
    let [notes, texts, gone, blocked, absent] = stateful_fixture(&lifecycle)?;
    let point = || vec![vector(9, 9, &[0.0, 0.0])];

    let cases: Vec<(&str, Operation, Expectation)> = vec![
        (
            "insert on an absent index",
            insert(&absent, "k", point())?,
            refused_from_absent,
        ),
        (
            "create on an active index",
            create_vector(&notes, "k", 2)?,
            refused_from_active,
        ),
        (
            "create on a destroyed index",
            create_vector(&gone, "k", 2)?,
            refused_from_destroyed,
        ),
        (
            "insert on a destroyed index",
            insert(&gone, "k", point())?,
            refused_from_destroyed,
        ),
        (
            "document into a vector index",
            insert(&notes, "k", vec![document(3, 1, "text")])?,
            |error| matches!(error, HeuremaError::FamilyMismatch { .. }),
        ),
        (
            "vector into a text index",
            insert(&texts, "k", point())?,
            |error| matches!(error, HeuremaError::FamilyMismatch { .. }),
        ),
        (
            "vector of the wrong dimension",
            insert(&notes, "k", vec![vector(3, 1, &[0.0, 0.0, 0.0])])?,
            |error| matches!(error, HeuremaError::DimensionMismatch { .. }),
        ),
        (
            "rebuild into the other family",
            operation(
                &notes,
                "k",
                IndexChange::Rebuild {
                    config: IndexConfig::Fts(FtsConfig::simple()),
                    members: vec![],
                },
            )?,
            |error| matches!(error, HeuremaError::FamilyMismatch { .. }),
        ),
        (
            "key reused with other content",
            insert(&notes, "insert", point())?,
            |error| matches!(error, HeuremaError::OperationConflict { .. }),
        ),
        (
            "insert while staged state exists",
            insert(&blocked, "k", point())?,
            |error| matches!(error, HeuremaError::StagedStateExists { .. }),
        ),
        (
            "destroy while staged state exists",
            destroy(&blocked, "k", 1)?,
            |error| matches!(error, HeuremaError::StagedStateExists { .. }),
        ),
    ];
    for (label, refused_operation, expected) in cases {
        counting.reset();
        let error = refused(lifecycle.apply(refused_operation));
        assert!(expected(&error), "{label}: {error:?}");
        let (reads, writes) = counting.counts();
        assert!(
            reads > 0,
            "{label}: a stateful refusal reads the state it refuses"
        );
        assert_eq!(writes, 0, "{label}: {error}");
    }
    Ok(())
}

fn replayed_key_with_same_content_is_a_no_op_across_reopen<S: Store>(store: &S) -> TestResult {
    let lifecycle = open(store)?;
    let notes = index("notes")?;
    let create = || create_vector(&notes, "create", 2);
    let batch = |members| insert(&notes, "batch", members);
    let created = lifecycle.apply(create()?)?;
    let first = lifecycle.apply(batch(vec![
        vector(2, 20, &[0.0, 1.0]),
        vector(1, 10, &[1.0, 0.0]),
    ])?)?;
    lifecycle.apply(remove(&notes, "remove", &[2])?)?;

    let counting = Counting::new(store.reopen(lifecycle.into_backend())?);
    let lifecycle = Lifecycle::open(&counting)?;
    counting.reset();
    // NOTE: the same members listed in another order are the same content.
    let replayed = lifecycle.apply(batch(vec![
        vector(1, 10, &[1.0, 0.0]),
        vector(2, 20, &[0.0, 1.0]),
    ])?)?;
    assert_receipt(&replayed, 2, true);
    assert_eq!(replayed.operation, first.operation);
    assert_eq!(replayed.state, first.state);

    // WHY: the replay lookup precedes the permission check, so a Create
    // replays after the index exists instead of being refused.
    let recreated = lifecycle.apply(create()?)?;
    assert_receipt(&recreated, 1, true);
    assert_eq!(recreated.operation, created.operation);
    assert_eq!(counting.counts().1, 0, "a replay writes nothing");
    assert_eq!(observe(&lifecycle, &notes)?, (3, vec![(1, 10, 2)]));
    Ok(())
}

fn replayed_key_with_different_content_is_refused_across_reopen<S: Store>(store: &S) -> TestResult {
    let lifecycle = open(store)?;
    let notes = index("notes")?;
    lifecycle.apply(create_vector(&notes, "create", 2)?)?;
    let first = lifecycle.apply(insert(&notes, "insert", vec![vector(1, 10, &[1.0, 0.0])])?)?;

    let counting = Counting::new(store.reopen(lifecycle.into_backend())?);
    let lifecycle = Lifecycle::open(&counting)?;
    counting.reset();
    for changed in [
        insert(&notes, "insert", vec![vector(1, 11, &[1.0, 0.0])])?,
        insert(&notes, "insert", vec![vector(1, 10, &[0.0, 1.0])])?,
        remove(&notes, "insert", &[1])?,
        destroy(&notes, "insert", 1)?,
    ] {
        match refused(lifecycle.apply(changed)) {
            HeuremaError::OperationConflict { conflict, .. } => {
                assert_eq!(conflict.index(), &notes);
                assert_eq!(conflict.key().as_str(), "insert");
                assert_eq!(conflict.recorded(), &first.operation.digest);
                assert_ne!(conflict.requested(), conflict.recorded());
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(counting.counts().1, 0, "a conflict writes nothing");
    assert_eq!(observe(&lifecycle, &notes)?, (2, vec![(1, 10, 2)]));

    // NOTE: a key is unique per index; on another index it names another
    // operation.
    let other = index("other")?;
    assert_receipt(
        &lifecycle.apply(create_vector(&other, "insert", 2)?)?,
        1,
        false,
    );
    Ok(())
}

fn orphan_staged_state_refuses_the_next_mutation_of_that_index<S: Store>(store: &S) -> TestResult {
    let lifecycle = open(store)?;
    let fresh = index("fresh")?;
    let notes = index("notes")?;
    drop(stage(&lifecycle, create_vector(&fresh, "create", 2)?)?);
    seeded(&lifecycle, &notes)?;
    let interrupted = || insert(&notes, "interrupted", vec![vector(3, 30, &[1.0, 1.0])]);
    drop(stage(&lifecycle, interrupted()?)?);
    let lifecycle = reopen(store, lifecycle)?;
    let point = || vec![vector(9, 9, &[0.0, 0.0])];

    // NOTE: An interrupted Create: absent, with version 1 staged. Create is
    // permitted from Absent and then meets the staged state; a transition
    // the table forbids is refused as such first.
    assert_staged_state(
        &refused(lifecycle.prepare(create_vector(&fresh, "create", 2)?)),
        1,
    );
    assert_staged_state(
        &refused(lifecycle.prepare(create_vector(&fresh, "again", 2)?)),
        1,
    );
    assert_not_permitted(
        &refused(lifecycle.prepare(insert(&fresh, "insert", point())?)),
        IndexStateKind::Absent,
    );
    assert_not_permitted(
        &refused(lifecycle.prepare(destroy(&fresh, "destroy", 1)?)),
        IndexStateKind::Absent,
    );
    assert_eq!(
        lifecycle.record(&fresh)?,
        None,
        "the staged Create stays unpublished"
    );

    // NOTE: An interrupted Insert on an active index blocks every mutation of it,
    // Destroy and the interrupted operation's own retry included. The
    // refusal comes from `prepare`, before any write is attempted.
    for blocked in [
        interrupted()?,
        insert(&notes, "insert-2", point())?,
        remove(&notes, "remove", &[1])?,
        rebuild(&notes, "rebuild", 2, point())?,
        destroy(&notes, "destroy", 1)?,
    ] {
        assert_staged_state(&refused(lifecycle.prepare(blocked)), 3);
    }
    assert_not_permitted(
        &refused(lifecycle.prepare(create_vector(&notes, "create-again", 2)?)),
        IndexStateKind::Active,
    );
    assert_eq!(
        head_state(&lifecycle, &notes)?,
        Some((IndexStateKind::Active, 2))
    );
    Ok(())
}

fn other_indexes_proceed_while_one_holds_staged_state<S: Store>(store: &S) -> TestResult {
    let lifecycle = open(store)?;
    let notes = index("notes")?;
    seeded(&lifecycle, &notes)?;
    drop(stage(
        &lifecycle,
        insert(&notes, "interrupted", vec![vector(3, 30, &[1.0, 1.0])])?,
    )?);
    let before = observe(&lifecycle, &notes)?;

    // NOTE: `notes.v2` shares `notes`' key prefix up to the dot.
    for other in [index("notes.v2")?, index("other")?] {
        assert_receipt(
            &lifecycle.apply(create_vector(&other, "create", 2)?)?,
            1,
            false,
        );
        let inserted = insert(&other, "insert", vec![vector(1, 1, &[0.0, 0.0])])?;
        assert_receipt(&lifecycle.apply(inserted)?, 2, false);
        assert_receipt(&lifecycle.apply(destroy(&other, "destroy", 1)?)?, 2, false);
    }
    let lifecycle = reopen(store, lifecycle)?;
    for other in [index("notes.v2")?, index("other")?] {
        assert_eq!(
            head_state(&lifecycle, &other)?,
            Some((IndexStateKind::Destroyed, 2))
        );
        assert_eq!(staged_version(&lifecycle, &other)?, None);
    }
    assert_eq!(
        observe(&lifecycle, &notes)?,
        before,
        "the blocked index still reads"
    );
    assert_staged_state(
        &refused(lifecycle.apply(insert(&notes, "next", vec![vector(4, 4, &[0.0, 0.0])])?)),
        3,
    );
    Ok(())
}

fn destroy_is_atomic_across_reopen<S: Store>(store: &S) -> TestResult {
    let lifecycle = open_faulting(store)?;
    let notes = index("notes")?;
    seeded(&lifecycle, &notes)?;
    lifecycle.apply(insert(&notes, "more", vec![vector(3, 30, &[1.0, 1.0])])?)?;
    let before = observe(&lifecycle, &notes)?;

    lifecycle.backend().arm(Step::Destroy, When::BeforeEffect);
    assert_injected(&refused(lifecycle.apply(destroy(&notes, "destroy", 9)?)));
    let lifecycle = reopen_faulting(store, lifecycle)?;
    assert_eq!(
        observe(&lifecycle, &notes)?,
        before,
        "a failed destroy changed nothing"
    );

    assert_receipt(&lifecycle.apply(destroy(&notes, "destroy", 9)?)?, 3, false);
    let lifecycle = reopen_faulting(store, lifecycle)?;
    let record = lifecycle.record(&notes)?;
    assert_eq!(
        record.map(|record| (record.config, record.state)),
        Some((
            IndexConfig::Vector(HnswConfig::new(2)),
            IndexState::Destroyed {
                last_version: version(3)?,
                retention: PlaceholderRetention(9),
                operation: OperationKey::try_from("destroy")?,
            }
        ))
    );
    match refused(lifecycle.index(&notes)) {
        HeuremaError::IndexNotFound { .. } => {}
        other => panic!("unexpected {other:?}"),
    }
    for published in 1..=3 {
        assert_eq!(
            lifecycle
                .backend()
                .read_version(&notes, version(published)?)?,
            None,
            "every payload goes in the destroy write"
        );
    }
    for key in ["create", "insert", "more", "destroy"] {
        assert!(
            lifecycle
                .backend()
                .read_operation(&notes, &OperationKey::try_from(key)?)?
                .is_some(),
            "the audit record of {key} stays"
        );
    }
    assert_not_permitted(
        &refused(lifecycle.apply(create_vector(&notes, "recreate", 2)?)),
        IndexStateKind::Destroyed,
    );
    assert_not_permitted(
        &refused(lifecycle.apply(insert(&notes, "late", vec![vector(4, 4, &[0.0, 0.0])])?)),
        IndexStateKind::Destroyed,
    );
    assert_receipt(&lifecycle.apply(destroy(&notes, "destroy", 9)?)?, 3, true);
    assert_receipt(
        &lifecycle.apply(create_vector(&notes, "create", 2)?)?,
        1,
        true,
    );
    Ok(())
}

/// Requires `index` to read as version 2 of [`seeded`], with no trace of
/// member 3 or of member 1's staged replacement.
fn assert_reads_version_two<B: LifecycleBackend>(
    lifecycle: &Lifecycle<B>,
    index: &IndexIdentity,
) -> TestResult {
    assert_eq!(
        head_state(lifecycle, index)?,
        Some((IndexStateKind::Active, 2))
    );
    assert_eq!(
        observe(lifecycle, index)?,
        (2, vec![(1, 10, 2), (2, 20, 2)])
    );
    let published = lifecycle.index(index)?;
    assert_eq!(published.member(&TestMember(3)), None);
    let two = version(2)?;
    let hits = published.query_vector(&[0.9, 0.9], 10)?;
    assert!(
        hits.iter()
            .all(|hit| hit.id != TestMember(3) && hit.version == two),
        "{hits:?}"
    );
    assert!(
        hits.iter()
            .any(|hit| hit.id == TestMember(1) && hit.provenance == PlaceholderProvenance(10)),
        "{hits:?}"
    );
    Ok(())
}

fn staged_version_is_not_readable<S: Store>(store: &S) -> TestResult {
    let lifecycle = open(store)?;
    let notes = index("notes")?;
    seeded(&lifecycle, &notes)?;

    // WHY read while the staged step is held: version 3's payload is durable
    // from here on, and no read may expose it before the publish point.
    let staged = stage(
        &lifecycle,
        insert(
            &notes,
            "staged",
            vec![vector(3, 30, &[0.9, 0.9]), vector(1, 11, &[1.0, 0.0])],
        )?,
    )?;
    assert!(
        lifecycle
            .backend()
            .read_version(&notes, version(3)?)?
            .is_some()
    );
    assert_reads_version_two(&lifecycle, &notes)?;
    assert_receipt(&staged.publish()?, 3, false);
    assert_eq!(
        observe(&lifecycle, &notes)?,
        (3, vec![(1, 11, 3), (2, 20, 2), (3, 30, 3)]),
        "the publish point makes the whole version visible at once"
    );

    let other = index("other")?;
    seeded(&lifecycle, &other)?;
    drop(stage(
        &lifecycle,
        insert(
            &other,
            "orphan",
            vec![vector(3, 30, &[0.9, 0.9]), vector(1, 11, &[1.0, 0.0])],
        )?,
    )?);
    let lifecycle = reopen(store, lifecycle)?;
    assert_reads_version_two(&lifecycle, &other)
}

fn two_lifecycles_over_one_backend_cannot_both_publish<S: Store>(store: &S) -> TestResult {
    let backend = store.open()?;
    let first = Lifecycle::open(&backend)?;
    let second = Lifecycle::open(&backend)?;
    let notes = index("notes")?;
    seeded(&first, &notes)?;
    let member = |id: u64| vec![vector(id, 1, &[0.25 * id as f32, 0.0])];

    // NOTE: Both prepare from version 2; the first to stage takes version 3.
    let a = ready(first.prepare(insert(&notes, "a", member(3))?)?);
    let b = ready(second.prepare(insert(&notes, "b", member(4))?)?);
    let a = a.stage()?;
    assert_staged_state(&refused(b.stage()), 3);
    assert_receipt(&a.publish()?, 3, false);

    // NOTE: Both prepare from version 3; the first publishes, and the second's
    // stage finds the head it was computed from gone.
    let c = ready(first.prepare(insert(&notes, "c", member(5))?)?);
    let d = ready(second.prepare(insert(&notes, "d", member(6))?)?);
    assert_receipt(&c.stage()?.publish()?, 4, false);
    match refused(d.stage()) {
        HeuremaError::HeadChanged { .. } => {}
        other => panic!("unexpected {other:?}"),
    }

    // NOTE: A destroy computed from version 4 is refused at its publish point
    // once the other lifecycle has published version 5.
    let destroying = ready(first.prepare(destroy(&notes, "destroy", 1)?)?).stage()?;
    assert_receipt(&second.apply(insert(&notes, "e", member(7))?)?, 5, false);
    match refused(destroying.publish()) {
        HeuremaError::HeadChanged { .. } => {}
        other => panic!("unexpected {other:?}"),
    }

    let (active, members) = observe(&second, &notes)?;
    assert_eq!(active, 5);
    let ids: Vec<u64> = members.iter().map(|(id, _, _)| *id).collect();
    assert_eq!(
        ids,
        [1, 2, 3, 5, 7],
        "exactly one of each racing pair published"
    );
    assert_eq!(
        staged_version(&first, &notes)?,
        None,
        "no loser left staged state"
    );
    // NOTE: a refused operation was never recorded, so its key can be
    // retried.
    assert_receipt(&second.apply(insert(&notes, "b", member(4))?)?, 6, false);
    Ok(())
}

fn lifecycle_and_snapshot_keyspaces_are_independent<S: Store>(store: &S) -> TestResult {
    let lifecycle = open(store)?;
    let notes = index("notes")?;
    seeded(&lifecycle, &notes)?;
    let mut snapshot = HnswIndex::new(HnswConfig::new(2));
    snapshot.insert(7_u64, &[0.5, 0.5])?;
    // WHY these names: they spell the lifecycle's own storage keys for the
    // head and version 2, so a shared map would collide.
    for name in ["example/notes", "example/notes/00000000000000000002"] {
        lifecycle.backend().save_vector_index(name, &snapshot)?;
    }

    let lifecycle = reopen(store, lifecycle)?;
    for name in ["example/notes", "example/notes/00000000000000000002"] {
        let loaded: HnswIndex<u64> = lifecycle.backend().load_vector_index(name)?;
        assert_eq!(loaded.len(), 1, "{name}");
    }
    assert_eq!(
        observe(&lifecycle, &notes)?,
        (2, vec![(1, 10, 2), (2, 20, 2)])
    );
    Ok(())
}
