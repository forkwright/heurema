//! Conformance of the two [`heurema::LifecycleBackend`] adapters: every
//! backend-agnostic case runs against `atmis::AtmisBackend` and against a
//! tempdir `thesauros::ThesaurosBackend`, and the durability cases close and
//! reopen the fjall database.
//!
//! WHY one suite for both: the lifecycle driver must behave the same on
//! either adapter, so each refusal, each listing order, the all-or-nothing
//! visibility of each stage, publish, destroy, and quarantine to a
//! concurrent reader, and each such write's exclusion of racing writers are
//! pinned once and checked against both. Values here are opaque byte
//! strings on purpose: the adapter contract is independent of heurēma's
//! record encodings (`crates/heurema/src/lifecycle/encoding.rs`), which
//! `lifecycle_conformance.rs` exercises through the driver, and the backend
//! must not care what the bytes say.
//!
//! WHY the concurrent cases prove more on `thesauros` than on `atmis`: a
//! reader or a racing writer can land inside a write that was split in two
//! only while the write is suspended between its halves. Each `thesauros`
//! batch is an fsync wide, which on a local disk leaves the reader at most
//! tens of polls inside each write (see [`READER_ROUNDS`]); every split
//! tried there so far was caught on every run. That margin is the fsync's
//! latency: with the tempdir on tmpfs, or on a device whose fsync is very
//! fast, it shrinks toward the `atmis` case. An
//! `atmis` write takes nanoseconds, so the same split is almost never
//! caught; its atomicity rests on one `MutexGuard` held across each write's
//! checks and effects, which is a review item. On `atmis` these cases check
//! only that a correct write never shows a torn state.
//!
//! WHY the durability cases drop the backend before reading: dropping it
//! closes the only handle on the fjall database, so the reopened backend
//! reads what reached the journal, as it would after a crash and restart. A
//! clean drop also persists fjall's journal (`Journal::drop` runs
//! `SyncAll`), so drop-and-reopen matches a crash only because every
//! lifecycle write is already `SyncAll`-durable; see the admission in
//! `lifecycle_conformance.rs` (its "What this cannot prove" paragraph).

use std::cell::Cell;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use atmis::AtmisBackend;
use heurema::lifecycle::storage_key;
use heurema::{
    DestroyWrite, ErrorCategory, HeuremaError, HnswConfig, HnswIndex, IndexIdentity, IndexName,
    IndexVersion, LifecycleBackend, OperationKey, OwnerNamespace, PersistenceBackend,
    PersistenceSource, PublishWrite, QuarantineWrite, QuarantinedEntry, StageWrite, StagedEntry,
    WriterLock,
};
use thesauros::ThesaurosBackend;

type TestResult = Result<(), HeuremaError>;

/// Highest version [`observe`] reads per index; no case publishes more.
const OBSERVED_VERSIONS: u64 = 4;

/// Runs each named case against both adapters, as `case::atmis` and
/// `case::thesauros`, so a failure names the adapter that broke.
macro_rules! conformance {
    ($($case:ident),+ $(,)?) => {$(
        mod $case {
            #[test]
            fn atmis() -> super::TestResult {
                super::$case(&atmis::AtmisBackend::new())
            }

            #[test]
            fn thesauros() -> super::TestResult {
                let dir = tempfile::tempdir().map_err(super::io_error)?;
                super::$case(&thesauros::ThesaurosBackend::open(dir.path())?)
            }
        }
    )+};
}

conformance!(
    reads_of_absent_keys_return_none,
    stage_then_publish_makes_head_and_operation_visible_and_removes_the_marker,
    stage_never_overwrites_an_existing_version_or_marker,
    stage_refuses_an_operation_key_already_recorded,
    stage_with_a_stale_expected_head_is_refused_and_changes_nothing,
    publish_with_a_stale_expected_head_is_refused_and_changes_nothing,
    publish_refuses_without_the_matching_staged_marker,
    operation_records_are_never_overwritten,
    destroy_with_a_staging_marker_present_is_refused,
    destroy_with_a_stale_expected_head_is_refused,
    destroy_removes_every_listed_version_and_keeps_the_audit_chain,
    quarantine_moves_staged_state_and_never_deletes_it,
    quarantine_refuses_a_changed_or_missing_marker,
    list_staged_enumerates_exactly_the_markers,
    indexes_whose_keys_share_a_prefix_stay_separate,
    listings_come_back_in_storage_key_order,
    a_concurrent_reader_never_sees_part_of_a_write,
    racing_writers_on_one_index_admit_exactly_one,
);

// WHY: tempdir creation fails with `io::Error`, distinct from every backend
// call's `HeuremaError`; routing it through `PersistenceSource` lets each
// test return `Result<(), HeuremaError>` and use `?` instead of `.expect()`,
// which the workspace lints deny.
fn io_error(source: std::io::Error) -> HeuremaError {
    HeuremaError::Persistence {
        source: PersistenceSource::new(source),
        location: std::panic::Location::caller(),
    }
}

fn fjall_error(source: fjall::Error) -> HeuremaError {
    HeuremaError::Persistence {
        source: PersistenceSource::new(source),
        location: std::panic::Location::caller(),
    }
}

fn index(namespace: &str, name: &str) -> Result<IndexIdentity, HeuremaError> {
    Ok(IndexIdentity::new(
        OwnerNamespace::try_from(namespace)?,
        IndexName::try_from(name)?,
    ))
}

fn notes() -> Result<IndexIdentity, HeuremaError> {
    index("example", "notes")
}

fn version(value: u64) -> Result<IndexVersion, HeuremaError> {
    IndexVersion::try_from(value)
}

fn key(text: &str) -> Result<OperationKey, HeuremaError> {
    OperationKey::try_from(text)
}

/// The operation key [`publish_version`] records version `value` under.
fn version_key(value: u64) -> Result<OperationKey, HeuremaError> {
    key(&format!("op-{value}"))
}

/// Distinct opaque bytes of one kind for one version of one index.
fn bytes(kind: &str, index: &IndexIdentity, value: u64) -> Vec<u8> {
    format!("{kind} {index} v{value}").into_bytes()
}

/// Stage version `value` of `index` under [`version_key`], with [`bytes`]
/// for marker and payload, and the sizes of the head and operation record
/// [`publish_staged`] writes.
fn stage_version(
    backend: &dyn LifecycleBackend,
    index: &IndexIdentity,
    value: u64,
    expected_head: Option<&[u8]>,
) -> TestResult {
    backend.stage(StageWrite::new(
        index,
        version(value)?,
        &version_key(value)?,
        expected_head,
        &bytes("marker", index, value),
        &bytes("payload", index, value),
        bytes("head", index, value).len(),
        bytes("operation", index, value).len(),
    ))
}

/// Publish the version [`stage_version`] staged, under [`version_key`].
fn publish_staged(
    backend: &dyn LifecycleBackend,
    index: &IndexIdentity,
    value: u64,
    expected_head: Option<&[u8]>,
) -> Result<Vec<u8>, HeuremaError> {
    let head = bytes("head", index, value);
    backend.publish(PublishWrite::new(
        index,
        version(value)?,
        &version_key(value)?,
        expected_head,
        &bytes("marker", index, value),
        &head,
        &bytes("operation", index, value),
    ))?;
    Ok(head)
}

/// Stage and publish version `value`; returns the new head.
fn publish_version(
    backend: &dyn LifecycleBackend,
    index: &IndexIdentity,
    value: u64,
    expected_head: Option<&[u8]>,
) -> Result<Vec<u8>, HeuremaError> {
    stage_version(backend, index, value, expected_head)?;
    publish_staged(backend, index, value, expected_head)
}

/// Unwrap a refused write, failing the test if it succeeded.
#[track_caller]
fn refusal(result: TestResult) -> HeuremaError {
    match result {
        Ok(()) => panic!("the write must be refused"),
        Err(error) => error,
    }
}

/// The version a staged-state or stored-version refusal names.
fn refused_version(error: &HeuremaError) -> Option<u64> {
    match error {
        HeuremaError::StagedStateExists { version, .. }
        | HeuremaError::StagedStateMissing { version, .. }
        | HeuremaError::VersionStored { version, .. } => Some(version.get()),
        _ => None,
    }
}

/// One index's operation records, as `list_operations` returns them.
type Operations = Vec<(OperationKey, Vec<u8>)>;

/// One index's staged version, as `read_staging` returns it.
type Staging = Option<(IndexVersion, Vec<u8>)>;

/// Everything a backend holds for `indexes`, for "changed nothing" and
/// "survived reopen" comparisons.
#[derive(Debug, PartialEq, Eq)]
struct Observed {
    heads: Vec<(IndexIdentity, Vec<u8>)>,
    staged: Vec<StagedEntry>,
    quarantined: Vec<QuarantinedEntry>,
    operations: Vec<(IndexIdentity, Operations)>,
    versions: Vec<(IndexIdentity, u64, Option<Vec<u8>>)>,
    staging: Vec<(IndexIdentity, Staging)>,
}

fn observe(
    backend: &dyn LifecycleBackend,
    indexes: &[&IndexIdentity],
) -> Result<Observed, HeuremaError> {
    let mut operations = Vec::new();
    let mut versions = Vec::new();
    let mut staging = Vec::new();
    for index in indexes {
        operations.push(((*index).clone(), backend.list_operations(index)?));
        staging.push(((*index).clone(), backend.read_staging(index)?));
        for value in 1..=OBSERVED_VERSIONS {
            versions.push((
                (*index).clone(),
                value,
                backend.read_version(index, version(value)?)?,
            ));
        }
    }
    Ok(Observed {
        heads: backend.list_heads()?,
        staged: backend.list_staged()?,
        quarantined: backend.list_quarantined()?,
        operations,
        versions,
        staging,
    })
}

fn reads_of_absent_keys_return_none(backend: &dyn LifecycleBackend) -> TestResult {
    let notes = notes()?;
    assert_eq!(backend.read_head(&notes)?, None);
    assert_eq!(backend.read_version(&notes, IndexVersion::FIRST)?, None);
    assert_eq!(backend.read_operation(&notes, &key("op-1")?)?, None);
    assert_eq!(backend.read_staging(&notes)?, None);
    assert!(backend.list_heads()?.is_empty());
    assert!(backend.list_staged()?.is_empty());
    assert!(backend.list_operations(&notes)?.is_empty());
    assert!(backend.list_quarantined()?.is_empty());
    Ok(())
}

fn stage_then_publish_makes_head_and_operation_visible_and_removes_the_marker(
    backend: &dyn LifecycleBackend,
) -> TestResult {
    let notes = notes()?;
    let v1 = version(1)?;
    stage_version(backend, &notes, 1, None)?;
    assert_eq!(backend.read_head(&notes)?, None, "stage moves no head");
    assert_eq!(
        backend.read_staging(&notes)?,
        Some((v1, bytes("marker", &notes, 1)))
    );
    assert_eq!(
        backend.read_version(&notes, v1)?,
        Some(bytes("payload", &notes, 1))
    );
    assert_eq!(backend.read_operation(&notes, &version_key(1)?)?, None);
    assert_eq!(
        backend.list_staged()?,
        [StagedEntry::new(
            notes.clone(),
            v1,
            bytes("marker", &notes, 1)
        )]
    );

    let head = publish_staged(backend, &notes, 1, None)?;
    assert_eq!(backend.read_head(&notes)?, Some(head.clone()));
    assert_eq!(
        backend.read_operation(&notes, &version_key(1)?)?,
        Some(bytes("operation", &notes, 1))
    );
    assert_eq!(
        backend.read_staging(&notes)?,
        None,
        "publish removes the marker"
    );
    assert!(backend.list_staged()?.is_empty());
    assert_eq!(
        backend.read_version(&notes, v1)?,
        Some(bytes("payload", &notes, 1)),
        "the published payload stays"
    );

    let head_2 = publish_version(backend, &notes, 2, Some(&head))?;
    assert_eq!(backend.read_head(&notes)?, Some(head_2.clone()));
    assert_eq!(backend.list_heads()?, [(notes.clone(), head_2)]);
    assert_eq!(
        backend.read_version(&notes, v1)?,
        Some(bytes("payload", &notes, 1)),
        "an earlier version's payload is retained"
    );
    assert_eq!(
        backend.list_operations(&notes)?,
        [
            (version_key(1)?, bytes("operation", &notes, 1)),
            (version_key(2)?, bytes("operation", &notes, 2)),
        ]
    );
    Ok(())
}

fn stage_never_overwrites_an_existing_version_or_marker(
    backend: &dyn LifecycleBackend,
) -> TestResult {
    let notes = notes()?;
    stage_version(backend, &notes, 1, None)?;
    let before = observe(backend, &[&notes])?;

    let another = key("another")?;
    for value in [1, 2] {
        let error = refusal(backend.stage(StageWrite::new(
            &notes,
            version(value)?,
            &another,
            None,
            b"another marker",
            b"another payload",
            0,
            0,
        )));
        assert!(
            matches!(error, HeuremaError::StagedStateExists { .. })
                && refused_version(&error) == Some(1),
            "a marker at any version blocks staging: {error:?}"
        );
        assert_eq!(error.category(), ErrorCategory::RecoveryRequired);
    }
    assert_eq!(
        observe(backend, &[&notes])?,
        before,
        "refusals wrote nothing"
    );

    // NOTE: a published version has no marker, only its payload, and staging
    // that version number again must still be refused. The store is
    // consistent: the head names that payload. The adapter never decodes a
    // head, so it cannot tell a published payload from one no head or marker
    // names, which is how a lifecycle caller meets this refusal (it stages
    // the version after the head it compare-and-sets, so only damage puts a
    // payload there). `VersionStored` is categorised for that caller, as
    // Corrupt; a direct caller that stages a published version gets the same
    // category, as the variant's documentation says.
    let head = publish_staged(backend, &notes, 1, None)?;
    let before = observe(backend, &[&notes])?;
    let error = refusal(backend.stage(StageWrite::new(
        &notes,
        version(1)?,
        &another,
        Some(&head),
        b"another marker",
        b"another payload",
        0,
        0,
    )));
    assert!(
        matches!(error, HeuremaError::VersionStored { .. }) && refused_version(&error) == Some(1),
        "a stored payload is never overwritten: {error:?}"
    );
    assert_eq!(error.category(), ErrorCategory::Corrupt);
    assert_eq!(
        observe(backend, &[&notes])?,
        before,
        "refusal wrote nothing"
    );
    Ok(())
}

fn stage_refuses_an_operation_key_already_recorded(backend: &dyn LifecycleBackend) -> TestResult {
    let notes = notes()?;
    let head = publish_version(backend, &notes, 1, None)?;
    let before = observe(backend, &[&notes])?;

    let error = refusal(backend.stage(StageWrite::new(
        &notes,
        version(2)?,
        &version_key(1)?,
        Some(&head),
        &bytes("marker", &notes, 2),
        &bytes("payload", &notes, 2),
        0,
        0,
    )));
    assert!(
        matches!(error, HeuremaError::OperationRecorded { ref key, .. } if key.as_str() == "op-1"),
        "a stage never leaves staged state for a recorded operation: {error:?}"
    );
    assert_eq!(error.category(), ErrorCategory::Conflict);
    assert_eq!(
        observe(backend, &[&notes])?,
        before,
        "refusal wrote nothing"
    );
    Ok(())
}

fn stage_with_a_stale_expected_head_is_refused_and_changes_nothing(
    backend: &dyn LifecycleBackend,
) -> TestResult {
    let notes = notes()?;
    let other = index("example", "other")?;
    let head = publish_version(backend, &notes, 1, None)?;
    let before = observe(backend, &[&notes, &other])?;

    let stale: [(&IndexIdentity, Option<&[u8]>); 3] = [
        (&notes, None),
        (&notes, Some(b"stale head")),
        (&other, Some(&head)),
    ];
    for (target, expected_head) in stale {
        let error = refusal(stage_version(backend, target, 2, expected_head));
        assert!(
            matches!(error, HeuremaError::HeadChanged { ref index, .. } if index == target),
            "{error:?}"
        );
        assert_eq!(error.category(), ErrorCategory::Conflict);
    }
    assert_eq!(
        observe(backend, &[&notes, &other])?,
        before,
        "refusals wrote nothing"
    );
    Ok(())
}

fn publish_with_a_stale_expected_head_is_refused_and_changes_nothing(
    backend: &dyn LifecycleBackend,
) -> TestResult {
    let notes = notes()?;
    let head = publish_version(backend, &notes, 1, None)?;
    stage_version(backend, &notes, 2, Some(&head))?;
    let before = observe(backend, &[&notes])?;

    for expected_head in [None, Some(&b"stale head"[..])] {
        let error = refusal(publish_staged(backend, &notes, 2, expected_head).map(|_| ()));
        assert!(
            matches!(error, HeuremaError::HeadChanged { .. }),
            "{error:?}"
        );
        assert_eq!(error.category(), ErrorCategory::Conflict);
    }
    assert_eq!(
        observe(backend, &[&notes])?,
        before,
        "refusals wrote nothing"
    );

    publish_staged(backend, &notes, 2, Some(&head))?;
    assert_eq!(backend.read_head(&notes)?, Some(bytes("head", &notes, 2)));
    Ok(())
}

fn publish_refuses_without_the_matching_staged_marker(
    backend: &dyn LifecycleBackend,
) -> TestResult {
    let notes = notes()?;
    let error = refusal(publish_staged(backend, &notes, 1, None).map(|_| ()));
    assert!(
        matches!(error, HeuremaError::StagedStateMissing { .. }),
        "nothing staged: {error:?}"
    );
    assert_eq!(error.category(), ErrorCategory::Conflict);

    stage_version(backend, &notes, 1, None)?;
    let before = observe(backend, &[&notes])?;
    let changed_marker = refusal(backend.publish(PublishWrite::new(
        &notes,
        version(1)?,
        &version_key(1)?,
        None,
        b"a marker this writer never staged",
        b"head",
        b"operation",
    )));
    assert!(
        matches!(changed_marker, HeuremaError::StagedStateMissing { .. }),
        "{changed_marker:?}"
    );
    let other_version = refusal(backend.publish(PublishWrite::new(
        &notes,
        version(2)?,
        &version_key(2)?,
        None,
        &bytes("marker", &notes, 1),
        b"head",
        b"operation",
    )));
    assert!(
        matches!(other_version, HeuremaError::StagedStateMissing { .. })
            && refused_version(&other_version) == Some(2),
        "{other_version:?}"
    );
    assert_eq!(
        observe(backend, &[&notes])?,
        before,
        "refusals wrote nothing"
    );

    publish_staged(backend, &notes, 1, None)?;
    assert_eq!(backend.read_staging(&notes)?, None);
    Ok(())
}

fn operation_records_are_never_overwritten(backend: &dyn LifecycleBackend) -> TestResult {
    let notes = notes()?;
    let v2 = version(2)?;
    let head = publish_version(backend, &notes, 1, None)?;
    stage_version(backend, &notes, 2, Some(&head))?;
    let before = observe(backend, &[&notes])?;

    let error = refusal(backend.publish(PublishWrite::new(
        &notes,
        v2,
        &version_key(1)?,
        Some(&head),
        &bytes("marker", &notes, 2),
        b"head reusing op-1",
        b"operation reusing op-1",
    )));
    assert!(
        matches!(error, HeuremaError::OperationRecorded { ref key, .. } if key.as_str() == "op-1"),
        "publish: {error:?}"
    );
    assert_eq!(error.category(), ErrorCategory::Conflict);
    assert_eq!(
        observe(backend, &[&notes])?,
        before,
        "refusal wrote nothing"
    );

    backend.quarantine(QuarantineWrite::new(
        &notes,
        v2,
        &bytes("marker", &notes, 2),
    ))?;
    let before = observe(backend, &[&notes])?;
    let error = refusal(backend.destroy(DestroyWrite::new(
        &notes,
        &version_key(1)?,
        &head,
        b"destroyed",
        b"destroy reusing op-1",
        &[version(1)?],
    )));
    assert!(
        matches!(error, HeuremaError::OperationRecorded { .. }),
        "destroy: {error:?}"
    );
    assert_eq!(
        observe(backend, &[&notes])?,
        before,
        "refusal wrote nothing"
    );
    assert_eq!(
        backend.read_operation(&notes, &version_key(1)?)?,
        Some(bytes("operation", &notes, 1))
    );
    Ok(())
}

fn destroy_with_a_staging_marker_present_is_refused(backend: &dyn LifecycleBackend) -> TestResult {
    let notes = notes()?;
    let head = publish_version(backend, &notes, 1, None)?;
    stage_version(backend, &notes, 2, Some(&head))?;
    let before = observe(backend, &[&notes])?;

    let error = refusal(backend.destroy(DestroyWrite::new(
        &notes,
        &key("destroy")?,
        &head,
        b"destroyed",
        b"destroy",
        &[version(1)?, version(2)?],
    )));
    assert!(
        matches!(error, HeuremaError::StagedStateExists { .. })
            && refused_version(&error) == Some(2),
        "{error:?}"
    );
    assert_eq!(error.category(), ErrorCategory::RecoveryRequired);
    assert_eq!(
        observe(backend, &[&notes])?,
        before,
        "the staged payload, its marker, and every version stay"
    );
    Ok(())
}

fn destroy_with_a_stale_expected_head_is_refused(backend: &dyn LifecycleBackend) -> TestResult {
    let notes = notes()?;
    let absent = index("example", "absent")?;
    let head = publish_version(backend, &notes, 1, None)?;
    let before = observe(backend, &[&notes, &absent])?;

    for (target, expected_head) in [(&notes, &b"stale head"[..]), (&absent, &head[..])] {
        let error = refusal(backend.destroy(DestroyWrite::new(
            target,
            &key("destroy")?,
            expected_head,
            b"destroyed",
            b"destroy",
            &[version(1)?],
        )));
        assert!(
            matches!(error, HeuremaError::HeadChanged { ref index, .. } if index == target),
            "{error:?}"
        );
    }
    assert_eq!(
        observe(backend, &[&notes, &absent])?,
        before,
        "refusals wrote nothing"
    );
    Ok(())
}

fn destroy_removes_every_listed_version_and_keeps_the_audit_chain(
    backend: &dyn LifecycleBackend,
) -> TestResult {
    let notes = notes()?;
    let other = index("example", "other")?;
    let head_1 = publish_version(backend, &notes, 1, None)?;
    let head_2 = publish_version(backend, &notes, 2, Some(&head_1))?;
    let other_head = publish_version(backend, &other, 1, None)?;

    let destroy_key = key("destroy-notes")?;
    backend.destroy(DestroyWrite::new(
        &notes,
        &destroy_key,
        &head_2,
        b"destroyed",
        b"destroy record",
        &[version(1)?, version(2)?, version(3)?],
    ))?;

    assert_eq!(backend.read_head(&notes)?, Some(b"destroyed".to_vec()));
    for value in 1..=3 {
        assert_eq!(backend.read_version(&notes, version(value)?)?, None);
    }
    assert_eq!(
        backend.list_operations(&notes)?,
        [
            (destroy_key, b"destroy record".to_vec()),
            (version_key(1)?, bytes("operation", &notes, 1)),
            (version_key(2)?, bytes("operation", &notes, 2)),
        ],
        "the destroyed index keeps its audit chain"
    );
    assert_eq!(backend.read_head(&other)?, Some(other_head));
    assert_eq!(
        backend.read_version(&other, version(1)?)?,
        Some(bytes("payload", &other, 1)),
        "another index's versions are untouched"
    );
    Ok(())
}

fn quarantine_moves_staged_state_and_never_deletes_it(
    backend: &dyn LifecycleBackend,
) -> TestResult {
    let notes = notes()?;
    let v1 = version(1)?;
    stage_version(backend, &notes, 1, None)?;
    backend.quarantine(QuarantineWrite::new(
        &notes,
        v1,
        &bytes("marker", &notes, 1),
    ))?;

    assert_eq!(backend.read_staging(&notes)?, None);
    assert_eq!(backend.read_version(&notes, v1)?, None);
    assert!(backend.list_staged()?.is_empty());
    let first = QuarantinedEntry::new(
        QuarantinedEntry::FIRST_SEQUENCE,
        notes.clone(),
        v1,
        bytes("marker", &notes, 1),
        Some(bytes("payload", &notes, 1)),
    );
    assert_eq!(backend.list_quarantined()?, std::slice::from_ref(&first));

    // NOTE: the index is unblocked, and quarantining the same version number
    // again keeps the first entry.
    backend.stage(StageWrite::new(
        &notes,
        v1,
        &key("retry")?,
        None,
        b"retry marker",
        b"retry payload",
        0,
        0,
    ))?;
    backend.quarantine(QuarantineWrite::new(&notes, v1, b"retry marker"))?;
    let second = QuarantinedEntry::new(
        QuarantinedEntry::FIRST_SEQUENCE + 1,
        notes.clone(),
        v1,
        b"retry marker".to_vec(),
        Some(b"retry payload".to_vec()),
    );
    assert_eq!(backend.list_quarantined()?, [first, second]);

    publish_version(backend, &notes, 1, None)?;
    assert_eq!(backend.read_head(&notes)?, Some(bytes("head", &notes, 1)));
    assert_eq!(
        backend.list_quarantined()?.len(),
        2,
        "publishing deletes nothing"
    );
    Ok(())
}

fn quarantine_refuses_a_changed_or_missing_marker(backend: &dyn LifecycleBackend) -> TestResult {
    let notes = notes()?;
    let error = refusal(backend.quarantine(QuarantineWrite::new(
        &notes,
        version(1)?,
        &bytes("marker", &notes, 1),
    )));
    assert!(
        matches!(error, HeuremaError::StagedStateMissing { .. }),
        "nothing staged: {error:?}"
    );

    stage_version(backend, &notes, 1, None)?;
    let before = observe(backend, &[&notes])?;
    for (value, marker) in [
        (1, b"a marker recovery never read".to_vec()),
        (2, bytes("marker", &notes, 1)),
    ] {
        let error =
            refusal(backend.quarantine(QuarantineWrite::new(&notes, version(value)?, &marker)));
        assert!(
            matches!(error, HeuremaError::StagedStateMissing { .. }),
            "{error:?}"
        );
        assert_eq!(error.category(), ErrorCategory::Conflict);
    }
    assert_eq!(
        observe(backend, &[&notes])?,
        before,
        "refusals moved nothing"
    );
    Ok(())
}

fn list_staged_enumerates_exactly_the_markers(backend: &dyn LifecycleBackend) -> TestResult {
    let orphan = index("example", "a-orphan")?;
    let advancing = index("example", "b-advancing")?;
    let published = index("example", "c-published")?;
    let quarantined = index("example", "d-quarantined")?;

    stage_version(backend, &orphan, 1, None)?;
    let head = publish_version(backend, &advancing, 1, None)?;
    stage_version(backend, &advancing, 2, Some(&head))?;
    publish_version(backend, &published, 1, None)?;
    stage_version(backend, &quarantined, 1, None)?;
    backend.quarantine(QuarantineWrite::new(
        &quarantined,
        version(1)?,
        &bytes("marker", &quarantined, 1),
    ))?;

    assert_eq!(
        backend.list_staged()?,
        [
            StagedEntry::new(orphan.clone(), version(1)?, bytes("marker", &orphan, 1)),
            StagedEntry::new(
                advancing.clone(),
                version(2)?,
                bytes("marker", &advancing, 2)
            ),
        ]
    );
    Ok(())
}

fn indexes_whose_keys_share_a_prefix_stay_separate(backend: &dyn LifecycleBackend) -> TestResult {
    let notes = notes()?;
    let dotted = index("example", "notes.v2")?;
    let dashed = index("example", "notes-x")?;
    let namespaced = index("example.b", "notes")?;

    stage_version(backend, &dotted, 1, None)?;
    publish_version(backend, &dashed, 1, None)?;
    stage_version(backend, &namespaced, 1, None)?;

    assert_eq!(backend.read_staging(&notes)?, None);
    let head = publish_version(backend, &notes, 1, None)?;
    backend.destroy(DestroyWrite::new(
        &notes,
        &key("destroy")?,
        &head,
        b"destroyed",
        b"destroy",
        &[version(1)?],
    ))?;
    assert_eq!(
        backend.list_operations(&notes)?,
        [
            (key("destroy")?, b"destroy".to_vec()),
            (version_key(1)?, bytes("operation", &notes, 1)),
        ]
    );
    assert!(backend.read_staging(&dotted)?.is_some());
    assert!(backend.read_staging(&namespaced)?.is_some());
    assert_eq!(
        backend.read_version(&dashed, version(1)?)?,
        Some(bytes("payload", &dashed, 1))
    );
    Ok(())
}

fn listings_come_back_in_storage_key_order(backend: &dyn LifecycleBackend) -> TestResult {
    // WHY these namespaces: `-` (0x2D) and `.` (0x2E) sort before the `/`
    // (0x2F) that ends a namespace, so key byte order differs from the
    // order of (namespace, name) pairs. Both adapters must agree on it.
    let ordered = [
        index("a-b", "x")?,
        index("a.b", "x")?,
        index("a", "x")?,
        index("b", "x")?,
    ];
    for target in ordered.iter().rev() {
        publish_version(backend, target, 1, None)?;
    }
    let heads: Vec<IndexIdentity> = backend
        .list_heads()?
        .into_iter()
        .map(|(identity, _)| identity)
        .collect();
    assert_eq!(heads, ordered);
    let mut by_key = heads.clone();
    by_key.sort_by_key(storage_key::head);
    assert_eq!(heads, by_key, "the order is storage key byte order");

    let target = &ordered[3];
    let mut head = bytes("head", target, 1);
    for (value, text) in [(2, "b"), (3, "a"), (4, "A1")] {
        stage_version(backend, target, value, Some(&head))?;
        let next = bytes("head", target, value);
        backend.publish(PublishWrite::new(
            target,
            version(value)?,
            &key(text)?,
            Some(&head),
            &bytes("marker", target, value),
            &next,
            text.as_bytes(),
        ))?;
        head = next;
    }
    let keys: Vec<String> = backend
        .list_operations(target)?
        .into_iter()
        .map(|(key, _)| key.as_str().to_owned())
        .collect();
    assert_eq!(keys, ["A1", "a", "b", "op-1"]);
    Ok(())
}

/// Rounds of [`a_concurrent_reader_never_sees_part_of_a_write`].
///
/// WHY this few: on `thesauros` each round is eleven `SyncAll` writes, each
/// an fsync. On a local disk the reader polls at most tens of times per
/// write (about 20,000 polls over a run's 275 writes, and a few per write
/// under a parallel build), which has been enough for every split write
/// tried to fail this case on every run. Detection depends on fsync
/// latency: a tempdir on tmpfs, or a very fast fsync, weakens the
/// `thesauros` case. The failure message reports the poll count.
const READER_ROUNDS: usize = 25;

/// Versions each reader round stages and publishes before it destroys.
const READER_VERSIONS: u64 = 4;

/// The version each reader round stages after [`READER_VERSIONS`] and
/// quarantines instead of publishing.
const QUARANTINED_VERSION: u64 = READER_VERSIONS + 1;

/// The index of reader round `round`.
fn race_index(round: usize) -> Result<IndexIdentity, HeuremaError> {
    index("example", &format!("race-{round}"))
}

/// The version `head` names, when it is one [`bytes`] wrote for `index`.
fn active_version(index: &IndexIdentity, head: Option<&[u8]>) -> Option<u64> {
    let head = head?;
    (1..=READER_VERSIONS).find(|&value| head == bytes("head", index, value).as_slice())
}

const DESTROYED: &[u8] = b"destroyed";

/// The quarantine entry of `index`'s [`QUARANTINED_VERSION`], if any.
fn quarantine_entry(
    backend: &dyn LifecycleBackend,
    index: &IndexIdentity,
) -> Result<Option<QuarantinedEntry>, HeuremaError> {
    let quarantined = version(QUARANTINED_VERSION)?;
    Ok(backend
        .list_quarantined()?
        .into_iter()
        .find(|entry| entry.index == *index && entry.version == quarantined))
}

/// Whether `index`'s staged version is [`QUARANTINED_VERSION`].
fn quarantined_version_marked(
    backend: &dyn LifecycleBackend,
    index: &IndexIdentity,
) -> Result<bool, HeuremaError> {
    Ok(backend
        .read_staging(index)?
        .is_some_and(|(staged, _)| staged.get() == QUARANTINED_VERSION))
}

/// One reader poll of `round`'s index: the invariants every atomic write
/// keeps, each read in an order where the fact that implies the other is
/// read first, so a correct adapter never fails one (a read sees every
/// write that returned before it began). `quarantine_staged` says the
/// round's stage of [`QUARANTINED_VERSION`] returned before the poll began.
fn poll(
    backend: &dyn LifecycleBackend,
    round: usize,
    quarantine_staged: bool,
) -> Result<Vec<String>, HeuremaError> {
    let index = race_index(round)?;
    let mut violations = Vec::new();

    let head = backend.read_head(&index)?;
    let active = active_version(&index, head.as_deref());
    if let Some(value) = active {
        // I1: publish writes the head and its operation record together.
        if backend
            .read_operation(&index, &version_key(value)?)?
            .is_none()
        {
            violations.push(format!(
                "{index}: head v{value} without its operation record"
            ));
        }
        // I2: publish removes the marker of the version it publishes.
        if let Some((staged, _)) = backend.read_staging(&index)?
            && staged.get() <= value
        {
            violations.push(format!(
                "{index}: head v{value} beside a marker for v{staged}"
            ));
        }
        // I6: a payload the head names leaves only in the destroy write that
        // marks the head destroyed, and the head is read again after the
        // payload.
        if backend.read_version(&index, version(value)?)?.is_none()
            && backend.read_head(&index)?.as_deref() != Some(DESTROYED)
        {
            violations.push(format!("{index}: head v{value} without its payload"));
        }
    }
    if head.as_deref() == Some(DESTROYED) {
        // I3: destroy writes the head and its record and removes every
        // listed payload together.
        if backend.read_operation(&index, &key("destroy")?)?.is_none() {
            violations.push(format!(
                "{index}: destroyed head without its operation record"
            ));
        }
        for value in 1..=READER_VERSIONS {
            if backend.read_version(&index, version(value)?)?.is_some() {
                violations.push(format!(
                    "{index}: destroyed head beside the payload of v{value}"
                ));
            }
        }
    }
    // I4: stage writes a marker and its payload together. Only quarantine,
    // which removes the marker in the same write, or destroy, which runs
    // only once no marker is left, removes the payload; so the marker is
    // read again after the payload.
    if let Some((staged, marker)) = backend.read_staging(&index)? {
        if marker != bytes("marker", &index, staged.get()) {
            violations.push(format!("{index}: marker of v{staged} holds other bytes"));
        }
        if backend.read_version(&index, staged)?.is_none()
            && backend.read_staging(&index)? == Some((staged, marker))
        {
            violations.push(format!("{index}: marker of v{staged} without its payload"));
        }
    }
    // I5: a payload past the head is staged (its marker is there) or was
    // published or destroyed since the head was read.
    if head.as_deref() != Some(DESTROYED) {
        let next = active.unwrap_or(0) + 1;
        if next <= READER_VERSIONS && backend.read_version(&index, version(next)?)?.is_some() {
            let marked = backend
                .read_staging(&index)?
                .is_some_and(|(staged, _)| staged.get() == next);
            let fresh = backend.read_head(&index)?;
            let moved = fresh.as_deref() == Some(DESTROYED)
                || active_version(&index, fresh.as_deref()).is_some_and(|value| value >= next);
            if !marked && !moved {
                violations.push(format!(
                    "{index}: payload of v{next} that no marker or head names"
                ));
            }
        }
    }
    // I7: quarantine creates an entry and removes the marker and payload it
    // moved in one write, and nothing stages that version again.
    if let Some(entry) = quarantine_entry(backend, &index)? {
        if entry.marker != bytes("marker", &index, QUARANTINED_VERSION)
            || entry.payload != Some(bytes("payload", &index, QUARANTINED_VERSION))
        {
            violations.push(format!(
                "{index}: quarantine entry of v{QUARANTINED_VERSION} holds other bytes"
            ));
        }
        if quarantined_version_marked(backend, &index)? {
            violations.push(format!(
                "{index}: quarantine entry of v{QUARANTINED_VERSION} beside its marker"
            ));
        }
        if backend
            .read_version(&index, version(QUARANTINED_VERSION)?)?
            .is_some()
        {
            violations.push(format!(
                "{index}: quarantine entry of v{QUARANTINED_VERSION} beside its payload"
            ));
        }
    }
    // I8: once staged, the quarantined version's marker stays until the
    // write that creates its quarantine entry.
    if quarantine_staged
        && !quarantined_version_marked(backend, &index)?
        && quarantine_entry(backend, &index)?.is_none()
    {
        violations.push(format!(
            "{index}: v{QUARANTINED_VERSION} left staging without reaching quarantine"
        ));
    }
    Ok(violations)
}

/// Stages and publishes [`READER_VERSIONS`] versions of `round`'s index,
/// stages [`QUARANTINED_VERSION`] and quarantines it, then destroys the
/// index. Once that stage returns, `quarantine_staged` holds `round + 1`.
fn write_round(
    backend: &dyn LifecycleBackend,
    round: usize,
    quarantine_staged: &AtomicUsize,
) -> TestResult {
    let index = race_index(round)?;
    let mut head: Option<Vec<u8>> = None;
    for value in 1..=READER_VERSIONS {
        head = Some(publish_version(backend, &index, value, head.as_deref())?);
    }
    let head = head.unwrap_or_default();
    stage_version(backend, &index, QUARANTINED_VERSION, Some(&head))?;
    quarantine_staged.store(round + 1, Ordering::SeqCst);
    backend.quarantine(QuarantineWrite::new(
        &index,
        version(QUARANTINED_VERSION)?,
        &bytes("marker", &index, QUARANTINED_VERSION),
    ))?;
    let versions = (1..=READER_VERSIONS)
        .map(version)
        .collect::<Result<Vec<_>, _>>()?;
    backend.destroy(DestroyWrite::new(
        &index,
        &key("destroy")?,
        &head,
        DESTROYED,
        b"destroy",
        &versions,
    ))
}

/// What the writer tells the reader of [`a_concurrent_reader_never_sees_part_of_a_write`].
#[derive(Default)]
struct ReaderProgress {
    /// The round being written.
    current: AtomicUsize,
    /// The last round whose [`QUARANTINED_VERSION`] stage returned, plus
    /// one; zero before any.
    quarantine_staged: AtomicUsize,
    /// Polls the reader has finished.
    polls: AtomicUsize,
    /// Set once the writer is done.
    done: AtomicBool,
    /// Set once the reader has stopped.
    reader_exited: AtomicBool,
}

/// Writes every reader round, publishing its number before each, and waits
/// after each until the reader has polled since the round began.
fn write_rounds(backend: &dyn LifecycleBackend, progress: &ReaderProgress) -> TestResult {
    for round in 0..READER_ROUNDS {
        let polled = progress.polls.load(Ordering::SeqCst);
        progress.current.store(round, Ordering::SeqCst);
        write_round(backend, round, &progress.quarantine_staged)?;
        // WHY: every round is polled at least once, without a sleep.
        while progress.polls.load(Ordering::SeqCst) == polled
            && !progress.reader_exited.load(Ordering::SeqCst)
        {
            thread::yield_now();
        }
    }
    Ok(())
}

/// Polls until the writer is done, and returns how many violations and
/// backend errors it met, with the first ten.
///
/// WHY no panic here: a violation or a backend error is collected, so the
/// writer is never left waiting for a dead reader.
fn read_rounds(backend: &dyn LifecycleBackend, progress: &ReaderProgress) -> (usize, Vec<String>) {
    let mut violations = Vec::new();
    let mut seen = 0_usize;
    while !progress.done.load(Ordering::SeqCst) {
        let round = progress.current.load(Ordering::SeqCst);
        let staged = progress.quarantine_staged.load(Ordering::SeqCst) == round + 1;
        match poll(backend, round, staged) {
            Ok(found) => {
                seen += found.len();
                violations.extend(found.into_iter().take(10 - violations.len().min(10)));
            }
            Err(error) => {
                seen += 1;
                if violations.len() < 10 {
                    violations.push(format!("backend error: {error}"));
                }
            }
        }
        progress.polls.fetch_add(1, Ordering::SeqCst);
    }
    progress.reader_exited.store(true, Ordering::SeqCst);
    (seen, violations)
}

/// A reader polls while a writer stages, publishes, quarantines, and
/// destroys, and never sees part of a write.
///
/// A split write is caught only while it is suspended between its halves
/// long enough for a poll to land, which each `thesauros` batch is (an
/// fsync) and an `atmis` write is not; see the module documentation.
fn a_concurrent_reader_never_sees_part_of_a_write<B: LifecycleBackend + Sync>(
    backend: &B,
) -> TestResult {
    let progress = ReaderProgress::default();
    let (written, (seen, first)) = thread::scope(|scope| {
        let reader = scope.spawn(|| read_rounds(backend, &progress));
        let written = write_rounds(backend, &progress);
        progress.done.store(true, Ordering::SeqCst);
        let violations = match reader.join() {
            Ok(violations) => violations,
            Err(panic) => std::panic::resume_unwind(panic),
        };
        (written, violations)
    });
    written?;
    assert_eq!(
        seen,
        0,
        "{seen} violations over {} polls; first: {first:#?}",
        progress.polls.load(Ordering::SeqCst)
    );
    Ok(())
}

/// Rounds of [`racing_writers_on_one_index_admit_exactly_one`], and the
/// threads that race each write.
const CONTENTION_ROUNDS: usize = 20;
const CONTENTION_THREADS: usize = 4;

/// Runs `write` on [`CONTENTION_THREADS`] threads released together, and
/// returns each thread's result in thread order.
fn race<F>(write: F) -> Vec<TestResult>
where
    F: Fn(usize) -> TestResult + Sync,
{
    let barrier = Barrier::new(CONTENTION_THREADS);
    thread::scope(|scope| {
        let racers: Vec<_> = (0..CONTENTION_THREADS)
            .map(|thread| {
                let (barrier, write) = (&barrier, &write);
                scope.spawn(move || {
                    barrier.wait();
                    write(thread)
                })
            })
            .collect();
        racers
            .into_iter()
            .map(|racer| match racer.join() {
                Ok(result) => result,
                Err(panic) => std::panic::resume_unwind(panic),
            })
            .collect()
    })
}

/// The one thread whose write succeeded, requiring every other to have
/// been refused as `lost` expects.
#[track_caller]
fn sole_winner(results: Vec<TestResult>, lost: fn(&HeuremaError) -> bool, what: &str) -> usize {
    let mut winners = Vec::new();
    for (thread, result) in results.into_iter().enumerate() {
        match result {
            Ok(()) => winners.push(thread),
            Err(error) => assert!(lost(&error), "{what}: thread {thread}: {error:?}"),
        }
    }
    match winners.as_slice() {
        [winner] => *winner,
        _ => panic!("{what}: exactly one write must succeed, not threads {winners:?}"),
    }
}

/// Racing stages, publishes, destroys, and quarantines of one index each
/// admit exactly one writer; see the module documentation for what this
/// proves on each adapter.
fn racing_writers_on_one_index_admit_exactly_one<B: LifecycleBackend + Sync>(
    backend: &B,
) -> TestResult {
    let v1 = version(1)?;
    for round in 0..CONTENTION_ROUNDS {
        let index = race_index(round)?;
        let own = |kind: &str, thread: usize| bytes(&format!("{kind}-{thread}"), &index, 1);

        let staged = race(|thread| {
            backend.stage(StageWrite::new(
                &index,
                v1,
                &key(&format!("op-{thread}"))?,
                None,
                &own("marker", thread),
                &own("payload", thread),
                own("head", thread).len(),
                own("operation", thread).len(),
            ))
        });
        let stager = sole_winner(
            staged,
            |error| matches!(error, HeuremaError::StagedStateExists { version, .. } if version.get() == 1),
            &format!("round {round} stage"),
        );
        assert_eq!(
            backend.read_staging(&index)?,
            Some((v1, own("marker", stager)))
        );
        assert_eq!(
            backend.read_version(&index, v1)?,
            Some(own("payload", stager))
        );

        let marker = own("marker", stager);
        let published = race(|thread| {
            backend.publish(PublishWrite::new(
                &index,
                v1,
                &key(&format!("k-{thread}"))?,
                None,
                &marker,
                &own("head", thread),
                &own("operation", thread),
            ))
        });
        let publisher = sole_winner(
            published,
            |error| matches!(error, HeuremaError::HeadChanged { .. }),
            &format!("round {round} publish"),
        );
        let head = own("head", publisher);
        assert_eq!(backend.read_head(&index)?, Some(head.clone()));
        assert_eq!(
            backend.list_operations(&index)?,
            [(key(&format!("k-{publisher}"))?, own("operation", publisher))]
        );

        let destroyed = race(|thread| {
            backend.destroy(DestroyWrite::new(
                &index,
                &key(&format!("d-{thread}"))?,
                &head,
                &own("destroyed", thread),
                &own("destroy", thread),
                &[v1],
            ))
        });
        let destroyer = sole_winner(
            destroyed,
            |error| matches!(error, HeuremaError::HeadChanged { .. }),
            &format!("round {round} destroy"),
        );
        let destroyed = own("destroyed", destroyer);
        assert_eq!(backend.read_head(&index)?, Some(destroyed.clone()));
        race_quarantines(backend, round, &index, &destroyed)?;
    }
    Ok(())
}

/// The last phase of [`racing_writers_on_one_index_admit_exactly_one`]:
/// racing quarantines of one staged marker admit exactly one.
fn race_quarantines<B: LifecycleBackend + Sync>(
    backend: &B,
    round: usize,
    index: &IndexIdentity,
    destroyed: &[u8],
) -> TestResult {
    // NOTE: the backend reads a destroyed head as opaque bytes like any
    // other, so a version staged against it gives the quarantine race a
    // marker to move.
    let v2 = version(2)?;
    let (marker, payload) = (bytes("marker", index, 2), bytes("payload", index, 2));
    backend.stage(StageWrite::new(
        index,
        v2,
        &key("q")?,
        Some(destroyed),
        &marker,
        &payload,
        0,
        0,
    ))?;
    let quarantined = race(|_| backend.quarantine(QuarantineWrite::new(index, v2, &marker)));
    sole_winner(
        quarantined,
        |error| matches!(error, HeuremaError::StagedStateMissing { version, .. } if version.get() == 2),
        &format!("round {round} quarantine"),
    );
    let entries: Vec<QuarantinedEntry> = backend
        .list_quarantined()?
        .into_iter()
        .filter(|entry| entry.index == *index)
        .collect();
    let [entry] = entries.as_slice() else {
        panic!("round {round}: exactly one quarantine entry, not {entries:?}");
    };
    assert_eq!(
        (entry.version, &entry.marker, &entry.payload),
        (v2, &marker, &Some(payload)),
        "round {round}"
    );
    assert_eq!(backend.read_staging(index)?, None);
    assert_eq!(backend.read_version(index, v2)?, None);
    Ok(())
}

/// Every call through the reference, box, and shared-pointer impls reaches
/// the backend behind it.
fn forwards_through<B: LifecycleBackend>(backend: B, name: &str) -> TestResult {
    let target = index("example", name)?;
    let head = publish_version(&backend, &target, 1, None)?;
    assert_eq!(backend.read_head(&target)?, Some(head));
    assert_eq!(
        backend.read_operation(&target, &version_key(1)?)?,
        Some(bytes("operation", &target, 1))
    );
    Ok(())
}

#[test]
fn pointer_impls_forward_every_call() -> TestResult {
    let atmis = AtmisBackend::new();
    forwards_through(&atmis, "borrowed")?;
    let boxed: Box<dyn LifecycleBackend> = Box::new(AtmisBackend::new());
    forwards_through(boxed, "boxed")?;
    let shared: Arc<dyn LifecycleBackend> = Arc::new(AtmisBackend::new());
    forwards_through(Arc::clone(&shared), "shared")?;
    assert!(shared.read_head(&index("example", "shared")?)?.is_some());

    let dir = tempfile::tempdir().map_err(io_error)?;
    let durable: Arc<dyn LifecycleBackend> = Arc::new(ThesaurosBackend::open(dir.path())?);
    forwards_through(Arc::clone(&durable), "shared")?;
    forwards_through(&*durable, "borrowed")?;
    assert_eq!(durable.list_heads()?.len(), 2);
    Ok(())
}

/// A snapshot saved under a name that spells a lifecycle key does not
/// reach the lifecycle, and the reverse.
fn stores_are_independent<B: LifecycleBackend + PersistenceBackend>(backend: &B) -> TestResult {
    let notes = notes()?;
    let name = storage_key::head(&notes);
    backend.save_vector_index(&name, &HnswIndex::<u64>::new(HnswConfig::new(2)))?;
    assert!(backend.list_heads()?.is_empty());
    assert_eq!(backend.read_head(&notes)?, None);

    publish_version(backend, &notes, 1, None)?;
    let loaded: HnswIndex<u64> = backend.load_vector_index(&name)?;
    assert_eq!(loaded.config(), &HnswConfig::new(2));
    Ok(())
}

#[test]
fn lifecycle_and_snapshot_stores_are_independent() -> TestResult {
    stores_are_independent(&AtmisBackend::new())?;
    let dir = tempfile::tempdir().map_err(io_error)?;
    stores_are_independent(&ThesaurosBackend::open(dir.path())?)
}

#[test]
fn interrupted_before_publish_reopens_with_the_staged_state_and_the_old_head() -> TestResult {
    let dir = tempfile::tempdir().map_err(io_error)?;
    let notes = notes()?;
    let head = {
        let backend = ThesaurosBackend::open(dir.path())?;
        let head = publish_version(&backend, &notes, 1, None)?;
        stage_version(&backend, &notes, 2, Some(&head))?;
        head
    }; // WHY: the process "stops" here, after stage and before publish.

    let reopened = ThesaurosBackend::open(dir.path())?;
    assert_eq!(reopened.read_head(&notes)?, Some(head.clone()), "old head");
    assert_eq!(
        reopened.read_staging(&notes)?,
        Some((version(2)?, bytes("marker", &notes, 2)))
    );
    assert_eq!(
        reopened.read_version(&notes, version(2)?)?,
        Some(bytes("payload", &notes, 2))
    );
    assert_eq!(reopened.read_operation(&notes, &version_key(2)?)?, None);

    let error = refusal(stage_version(&reopened, &notes, 3, Some(&head)));
    assert!(
        matches!(error, HeuremaError::StagedStateExists { .. }),
        "the orphan still blocks its index after reopen: {error:?}"
    );
    Ok(())
}

#[test]
fn interrupted_after_publish_reopens_with_the_new_head_and_no_marker() -> TestResult {
    let dir = tempfile::tempdir().map_err(io_error)?;
    let notes = notes()?;
    let head = {
        let backend = ThesaurosBackend::open(dir.path())?;
        publish_version(&backend, &notes, 1, None)?
    };

    let reopened = ThesaurosBackend::open(dir.path())?;
    assert_eq!(reopened.read_head(&notes)?, Some(head));
    assert_eq!(reopened.read_staging(&notes)?, None);
    assert_eq!(
        reopened.read_operation(&notes, &version_key(1)?)?,
        Some(bytes("operation", &notes, 1))
    );
    assert_eq!(
        reopened.read_version(&notes, version(1)?)?,
        Some(bytes("payload", &notes, 1))
    );
    Ok(())
}

#[test]
fn destroy_and_quarantine_survive_reopen() -> TestResult {
    let dir = tempfile::tempdir().map_err(io_error)?;
    let notes = notes()?;
    let orphan = index("example", "orphan")?;
    {
        let backend = ThesaurosBackend::open(dir.path())?;
        let head = publish_version(&backend, &notes, 1, None)?;
        backend.destroy(DestroyWrite::new(
            &notes,
            &key("destroy")?,
            &head,
            b"destroyed",
            b"destroy",
            &[version(1)?],
        ))?;
        stage_version(&backend, &orphan, 1, None)?;
        backend.quarantine(QuarantineWrite::new(
            &orphan,
            version(1)?,
            &bytes("marker", &orphan, 1),
        ))?;
    }

    let reopened = ThesaurosBackend::open(dir.path())?;
    assert_eq!(reopened.read_head(&notes)?, Some(b"destroyed".to_vec()));
    assert_eq!(reopened.read_version(&notes, version(1)?)?, None);
    assert_eq!(reopened.list_quarantined()?.len(), 1);
    assert_eq!(reopened.read_staging(&orphan)?, None);

    // NOTE: the sequence continues from what reached the disk.
    stage_version(&reopened, &orphan, 1, None)?;
    reopened.quarantine(QuarantineWrite::new(
        &orphan,
        version(1)?,
        &bytes("marker", &orphan, 1),
    ))?;
    let sequences: Vec<u64> = reopened
        .list_quarantined()?
        .iter()
        .map(|entry| entry.sequence)
        .collect();
    assert_eq!(
        sequences,
        [
            QuarantinedEntry::FIRST_SEQUENCE,
            QuarantinedEntry::FIRST_SEQUENCE + 1
        ]
    );
    Ok(())
}

/// Open the fjall database at `path` directly and edit one keyspace, the
/// way only a damaged or foreign-written store would change it.
fn edit_raw_keyspace(
    path: &Path,
    keyspace: &str,
    edit: impl FnOnce(&fjall::Keyspace) -> Result<(), fjall::Error>,
) -> TestResult {
    let db = fjall::Database::builder(path).open().map_err(fjall_error)?;
    let keyspace = db
        .keyspace(keyspace, fjall::KeyspaceCreateOptions::default)
        .map_err(fjall_error)?;
    edit(&keyspace).map_err(fjall_error)?;
    db.persist(fjall::PersistMode::SyncAll).map_err(fjall_error)
}

#[test]
fn a_marker_without_its_payload_cannot_publish_but_can_be_quarantined() -> TestResult {
    let dir = tempfile::tempdir().map_err(io_error)?;
    let notes = notes()?;
    let v1 = version(1)?;
    stage_version(&ThesaurosBackend::open(dir.path())?, &notes, 1, None)?;
    edit_raw_keyspace(dir.path(), "lifecycle_versions", |versions| {
        versions.remove(storage_key::version(&notes, v1))
    })?;

    let backend = ThesaurosBackend::open(dir.path())?;
    let error = refusal(publish_staged(&backend, &notes, 1, None).map(|_| ()));
    assert!(
        matches!(error, HeuremaError::StagedStateMissing { .. }),
        "{error:?}"
    );
    assert_eq!(backend.read_head(&notes)?, None);

    backend.quarantine(QuarantineWrite::new(
        &notes,
        v1,
        &bytes("marker", &notes, 1),
    ))?;
    assert_eq!(
        backend.list_quarantined()?,
        [QuarantinedEntry::new(
            QuarantinedEntry::FIRST_SEQUENCE,
            notes.clone(),
            v1,
            bytes("marker", &notes, 1),
            None,
        )]
    );
    assert_eq!(backend.read_staging(&notes)?, None);
    Ok(())
}

#[test]
fn malformed_stored_keys_are_reported_as_corrupt() -> TestResult {
    let dir = tempfile::tempdir().map_err(io_error)?;
    let notes = notes()?;
    let payload_only = storage_key::quarantine(
        QuarantinedEntry::FIRST_SEQUENCE,
        &notes,
        IndexVersion::FIRST,
        storage_key::QuarantinePart::Payload,
    );
    edit_raw_keyspace(dir.path(), "lifecycle_heads", |heads| {
        heads.insert("not a head key", "head")
    })?;
    edit_raw_keyspace(dir.path(), "lifecycle_quarantine", |quarantine| {
        quarantine.insert(payload_only.as_str(), "payload")
    })?;

    let backend = ThesaurosBackend::open(dir.path())?;
    for (listing, result) in [
        ("heads", backend.list_heads().map(|_| ())),
        ("quarantine", backend.list_quarantined().map(|_| ())),
    ] {
        let error = refusal(result);
        assert!(
            matches!(error, HeuremaError::CorruptSnapshot { .. }),
            "{listing}: {error:?}"
        );
        assert_eq!(error.category(), ErrorCategory::Corrupt, "{listing}");
    }
    Ok(())
}

/// A write step [`FaultingBackend`] can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Stage,
    Publish,
    Destroy,
    Quarantine,
}

/// Whether an injected failure comes before the delegated write or after
/// it (a lost acknowledgement).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum When {
    BeforeEffect,
    AfterEffect,
}

#[derive(Debug)]
struct InjectedFault {
    step: Step,
    when: When,
}

impl fmt::Display for InjectedFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "injected {:?} fault {:?}", self.step, self.when)
    }
}

impl std::error::Error for InjectedFault {}

/// Delegates every call to `inner`; the first write of the chosen step
/// fails with an injected [`HeuremaError::Persistence`], either before it
/// reaches `inner` or after `inner` applied it.
struct FaultingBackend<B> {
    inner: B,
    fault: Cell<Option<(Step, When)>>,
}

impl<B: LifecycleBackend> FaultingBackend<B> {
    fn new(inner: B, step: Step, when: When) -> Self {
        Self {
            inner,
            fault: Cell::new(Some((step, when))),
        }
    }

    fn write(&self, step: Step, effect: impl FnOnce(&B) -> TestResult) -> TestResult {
        match self.fault.get() {
            Some((faulted, when)) if faulted == step => {
                self.fault.set(None);
                if when == When::AfterEffect {
                    effect(&self.inner)?;
                }
                Err(HeuremaError::Persistence {
                    source: PersistenceSource::new(InjectedFault { step, when }),
                    location: std::panic::Location::caller(),
                })
            }
            _ => effect(&self.inner),
        }
    }
}

impl<B: LifecycleBackend> LifecycleBackend for FaultingBackend<B> {
    fn writer(&self) -> &WriterLock {
        self.inner.writer()
    }

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

    fn stage(&self, write: StageWrite<'_>) -> TestResult {
        self.write(Step::Stage, |inner| inner.stage(write))
    }

    fn publish(&self, write: PublishWrite<'_>) -> TestResult {
        self.write(Step::Publish, |inner| inner.publish(write))
    }

    fn destroy(&self, write: DestroyWrite<'_>) -> TestResult {
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

    fn quarantine(&self, write: QuarantineWrite<'_>) -> TestResult {
        self.write(Step::Quarantine, |inner| inner.quarantine(write))
    }

    fn list_quarantined(&self) -> Result<Vec<QuarantinedEntry>, HeuremaError> {
        self.inner.list_quarantined()
    }
}

/// Bring a backend to the state just before `step`'s write.
fn prepare(backend: &dyn LifecycleBackend, step: Step) -> TestResult {
    let notes = notes()?;
    let head = publish_version(backend, &notes, 1, None)?;
    match step {
        Step::Stage | Step::Destroy => Ok(()),
        Step::Publish | Step::Quarantine => stage_version(backend, &notes, 2, Some(&head)),
    }
}

/// `step`'s write, valid against the state [`prepare`] left.
fn apply(backend: &dyn LifecycleBackend, step: Step) -> TestResult {
    let notes = notes()?;
    let head = bytes("head", &notes, 1);
    match step {
        Step::Stage => stage_version(backend, &notes, 2, Some(&head)),
        Step::Publish => publish_staged(backend, &notes, 2, Some(&head)).map(|_| ()),
        Step::Destroy => backend.destroy(DestroyWrite::new(
            &notes,
            &key("destroy")?,
            &head,
            b"destroyed",
            b"destroy",
            &[version(1)?],
        )),
        Step::Quarantine => backend.quarantine(QuarantineWrite::new(
            &notes,
            version(2)?,
            &bytes("marker", &notes, 2),
        )),
    }
}

const STEPS: [Step; 4] = [Step::Stage, Step::Publish, Step::Destroy, Step::Quarantine];

#[track_caller]
fn assert_injected(result: TestResult, step: Step, when: When) {
    let error = refusal(result);
    assert!(
        matches!(error, HeuremaError::Persistence { .. }) && error.to_string().contains("injected"),
        "{step:?} {when:?}: {error:?}"
    );
}

/// Requires the retry of `step`'s write after a fault at `when` to take
/// effect only if the faulted write did not: before the effect the retry
/// applies, and after it the retry meets the state the write left and is
/// refused.
#[track_caller]
fn assert_retry(retry: TestResult, step: Step, when: When) {
    let error = match (retry, when) {
        (Ok(()), When::BeforeEffect) => return,
        (Ok(()), When::AfterEffect) => panic!("{step:?} {when:?}: the retry applied twice"),
        (Err(error), When::BeforeEffect) => panic!("{step:?} {when:?}: {error:?}"),
        (Err(error), When::AfterEffect) => error,
    };
    let expected = match step {
        Step::Stage => {
            matches!(error, HeuremaError::StagedStateExists { .. })
                && refused_version(&error) == Some(2)
        }
        Step::Publish | Step::Destroy => matches!(error, HeuremaError::HeadChanged { .. }),
        Step::Quarantine => {
            matches!(error, HeuremaError::StagedStateMissing { .. })
                && refused_version(&error) == Some(2)
        }
    };
    assert!(expected, "{step:?} {when:?}: {error:?}");
}

#[test]
fn durable_write_retried_after_a_fault_takes_effect_exactly_once() -> TestResult {
    let notes = notes()?;
    for step in STEPS {
        // NOTE: what the write leaves when it succeeds, after reopen.
        let reference = tempfile::tempdir().map_err(io_error)?;
        let (before, after) = {
            let backend = ThesaurosBackend::open(reference.path())?;
            prepare(&backend, step)?;
            let before = observe(&backend, &[&notes])?;
            apply(&backend, step)?;
            (before, observe(&backend, &[&notes])?)
        };
        assert_ne!(before, after, "{step:?} changes the store");
        assert_eq!(
            observe(&ThesaurosBackend::open(reference.path())?, &[&notes])?,
            after,
            "{step:?} survives reopen"
        );

        for (when, expected) in [(When::BeforeEffect, &before), (When::AfterEffect, &after)] {
            let dir = tempfile::tempdir().map_err(io_error)?;
            {
                let backend = ThesaurosBackend::open(dir.path())?;
                prepare(&backend, step)?;
                let faulting = FaultingBackend::new(backend, step, when);
                assert_injected(apply(&faulting, step), step, when);
            } // WHY: the process stops right after the failed call.
            let reopened = ThesaurosBackend::open(dir.path())?;
            assert_eq!(
                &observe(&reopened, &[&notes])?,
                expected,
                "{step:?} {when:?}"
            );
            assert_retry(apply(&reopened, step), step, when);
            assert_eq!(
                observe(&reopened, &[&notes])?,
                after,
                "{step:?} {when:?}: the write took effect exactly once"
            );
        }
    }
    Ok(())
}

#[test]
fn in_memory_write_retried_after_a_fault_takes_effect_exactly_once() -> TestResult {
    let notes = notes()?;
    for step in STEPS {
        let reference = AtmisBackend::new();
        prepare(&reference, step)?;
        let before = observe(&reference, &[&notes])?;
        apply(&reference, step)?;
        let after = observe(&reference, &[&notes])?;

        for (when, expected) in [(When::BeforeEffect, &before), (When::AfterEffect, &after)] {
            let backend = AtmisBackend::new();
            prepare(&backend, step)?;
            assert_injected(
                apply(&FaultingBackend::new(&backend, step, when), step),
                step,
                when,
            );
            assert_eq!(
                &observe(&backend, &[&notes])?,
                expected,
                "{step:?} {when:?}"
            );
            assert_retry(apply(&backend, step), step, when);
            assert_eq!(
                observe(&backend, &[&notes])?,
                after,
                "{step:?} {when:?}: the write took effect exactly once"
            );
        }
    }
    Ok(())
}

#[test]
fn a_lost_publish_acknowledgement_is_found_by_reading_back_and_never_applies_twice() -> TestResult {
    let dir = tempfile::tempdir().map_err(io_error)?;
    let notes = notes()?;
    {
        let backend = ThesaurosBackend::open(dir.path())?;
        prepare(&backend, Step::Publish)?;
        let faulting = FaultingBackend::new(backend, Step::Publish, When::AfterEffect);
        assert_injected(
            apply(&faulting, Step::Publish),
            Step::Publish,
            When::AfterEffect,
        );
    }

    let reopened = ThesaurosBackend::open(dir.path())?;
    assert_eq!(
        reopened.read_operation(&notes, &version_key(2)?)?,
        Some(bytes("operation", &notes, 2)),
        "the caller finds its operation's record by key"
    );
    let retry = refusal(apply(&reopened, Step::Publish));
    assert!(
        matches!(retry, HeuremaError::HeadChanged { .. }),
        "a blind retry is refused, not applied twice: {retry:?}"
    );
    assert_eq!(reopened.read_head(&notes)?, Some(bytes("head", &notes, 2)));
    Ok(())
}
