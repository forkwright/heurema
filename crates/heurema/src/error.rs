//! Error types for Heurēma.

use std::fmt;
use std::sync::Arc;

use crate::SnapshotFamily;
use crate::lifecycle::{
    IdentifierKind, IndexIdentity, IndexStateKind, IndexVersion, LifecycleTransition,
    OperationDigest, OperationKey,
};

/// WHY: Backend errors are type-erased only at the persistence boundary while
/// SNAFU still receives a concrete source type for error-chain reporting.
#[derive(Debug, Clone)]
pub struct PersistenceSource {
    source: Arc<dyn std::error::Error + Send + Sync + 'static>,
}

impl PersistenceSource {
    /// WHY: Backend adapters need a single conversion point from their concrete
    /// error type into Heurēma's persistence error source.
    #[must_use]
    pub fn new<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            source: Arc::new(source),
        }
    }
}

impl fmt::Display for PersistenceSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.source)
    }
}

impl std::error::Error for PersistenceSource {
    /// WHY: `Display` already forwards the wrapped error's message, so the
    /// chain continues at the wrapped error's own source — returning the
    /// wrapped error here would repeat the same message as a dead hop.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.source()
    }
}

/// WHY: Consumers need one concrete error surface for vector, FTS, fusion, and
/// persistence operations.
#[derive(Debug, snafu::Snafu)]
#[snafu(visibility(pub))]
#[non_exhaustive]
pub enum HeuremaError {
    /// WHY: The HNSW engine reproduces krites's strict vector dimension
    /// checks instead of silently accepting malformed vectors.
    #[snafu(display("vector dimension mismatch: expected {expected}, got {actual}"))]
    DimensionMismatch {
        /// Expected vector dimension.
        expected: usize,
        /// Actual vector dimension.
        actual: usize,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: graph distances require finite coordinates; accepting a NaN would
    /// make ordering and persisted graph topology non-deterministic.
    #[snafu(display("invalid vector: {reason}"))]
    InvalidVector {
        /// Validation failure.
        reason: String,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: The public vector contract exposes `f32` distances. Ranking is
    /// performed in `f64` so finite coordinates cannot poison graph ordering,
    /// but a mathematically finite result that cannot be represented by that
    /// public score type must be reported rather than clamped or mislabeled.
    #[snafu(display("vector distance is not representable as a finite f32: {reason}"))]
    DistanceNotRepresentable {
        /// Why the finite internal distance cannot cross the public boundary.
        reason: String,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: `HnswConfig::new` remains a data constructor for compatibility,
    /// so an invalid deserialized or manually assembled configuration is
    /// refused at the engine boundary rather than creating a graph with no
    /// navigable-link capacity.
    #[snafu(display("invalid HNSW configuration: {reason}"))]
    InvalidHnswConfig {
        /// Validation failure.
        reason: String,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: persisted index bytes must name the index family the caller
    /// loads before an adapter decodes engine state; bytes of another family
    /// under the requested name are refused, never reinterpreted.
    #[snafu(display("index snapshot does not match the requested family: {reason}"))]
    SnapshotFormat {
        /// Validation failure.
        reason: String,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: stored index data (a snapshot or a lifecycle record) whose format
    /// version this build does not read may be valid for a newer build. It is
    /// its own variant, in the `Unsupported` category, so a consumer never
    /// treats it as corrupt and overwrites it.
    #[snafu(display(
        "stored index data format version {found} is not supported; this build reads version {supported}"
    ))]
    UnsupportedSnapshotVersion {
        /// Format version the stored bytes declare.
        found: u16,
        /// Format version this build reads and writes.
        supported: u16,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: stored bytes that cannot be decoded, or that decode into a state
    /// violating an engine invariant or contradicting the key they were
    /// stored under, are corrupt rather than a backend I/O failure, and
    /// retrying the read cannot help. A snapshot is saved again from a
    /// rebuilt index; a lifecycle record has no repair path yet (see
    /// [`ErrorCategory::Corrupt`]).
    #[snafu(display("corrupt stored index data: {source}"))]
    CorruptSnapshot {
        /// Decoder error describing why the bytes were refused.
        source: PersistenceSource,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: Rank fusion must reject a malformed dampening constant with a
    /// typed error instead of panicking inside library code.
    #[snafu(display("invalid RRF k_constant: {k_constant} (must be finite and positive)"))]
    InvalidKConstant {
        /// Rejected rank dampening constant.
        k_constant: f32,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: Persistence backends address indexes by engine-owned names, so a
    /// missing name must be distinguishable from storage failure.
    #[snafu(display("index not found: {name}"))]
    IndexNotFound {
        /// Missing index name.
        name: String,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: Storage failures are external to the index algorithms but still
    /// need to remain in the same error chain for callers. Stored bytes that
    /// fail to decode are [`HeuremaError::CorruptSnapshot`], not this variant.
    #[snafu(display("persistence backend error: {source}"))]
    Persistence {
        /// Backend-specific source error.
        source: PersistenceSource,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: A configured capability an engine does not implement must fail
    /// as a typed error before any state changes. Today only `Bm25Index`
    /// returns it, for tokenizer or filter pipelines beyond `Simple`.
    #[snafu(display("not yet implemented: {feature}"))]
    NotYetImplemented {
        /// The unimplemented capability the caller configured.
        feature: String,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: lifecycle identifiers become storage keys and audit references,
    /// so a malformed one is refused when it is constructed, before any
    /// record or key could carry it.
    #[snafu(display("invalid {kind} {value:?}: {reason}"))]
    InvalidIdentifier {
        /// Which identifier was refused.
        kind: IdentifierKind,
        /// The refused value, truncated to its first 64 characters.
        value: String,
        /// Why the value was refused.
        reason: String,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: an index's family fixes which member content it accepts, and a
    /// named index keeps one family for life, so content or a rebuild of the
    /// other family is refused before anything is staged.
    #[snafu(display("index family mismatch: expected {expected:?}, got {actual:?}"))]
    FamilyMismatch {
        /// The family the index, configuration, or batch requires.
        expected: SnapshotFamily,
        /// The family the operation supplied.
        actual: SnapshotFamily,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: one operation that names a member twice has no single meaning
    /// (which content or provenance wins?), and its digest would depend on
    /// the order the duplicates arrived in, so it is refused outright.
    #[snafu(display("member {member} appears more than once in one operation"))]
    DuplicateMember {
        /// The repeated member identity's `Debug` form, truncated to its
        /// first 64 characters.
        member: String,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: an insert or remove that names no members would publish a new
    /// version identical to the old one; it is almost always a caller bug,
    /// so it is refused rather than recorded.
    #[snafu(display("{transition:?} operation names no members"))]
    EmptyBatch {
        /// The transition whose batch was empty.
        transition: LifecycleTransition,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: the lifecycle's permission table decides which transitions each
    /// state allows; a transition outside it is refused before any write, and
    /// the error names the index, the transition, and the state it met.
    #[snafu(display("{transition:?} is not permitted on index {index} in state {state:?}"))]
    TransitionNotPermitted {
        /// The index the operation targets.
        index: IndexIdentity,
        /// The refused transition.
        transition: LifecycleTransition,
        /// The index's state when the transition was requested.
        state: IndexStateKind,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: `permit` checks an operation against the record its caller read.
    /// A record for another index would check the transition and the
    /// members against the wrong state and configuration, so it is refused
    /// rather than trusted.
    #[snafu(display(
        "operation on index {expected} was checked against the record of index {actual}"
    ))]
    RecordMismatch {
        /// The index the operation targets.
        expected: IndexIdentity,
        /// The index the supplied record describes.
        actual: IndexIdentity,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: an operation's digest is computed over a canonical encoding that
    /// admits no floating-point numbers and only string or integer map keys,
    /// and every consumer value the lifecycle stores must read back from its
    /// `serde_json` encoding as the value written. A consumer provenance or
    /// retention value outside that grammar, or one `serde_json` writes but
    /// cannot read back as itself, is the caller's input to fix, not a
    /// storage failure.
    #[snafu(display("operation cannot be encoded: {reason}"))]
    UnencodableOperation {
        /// Why the encoder refused the operation.
        reason: String,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    // NOTE: lifecycle storage refusals, raised by a `LifecycleBackend` write or
    // by the replay check that reads the operation records it keeps.
    /// WHY: a version staged but never published is the only record of an
    /// interrupted operation. Staging over it would destroy that record, and
    /// destroying the index beside it would leave a marker no head can
    /// explain, so every later stage or destroy of the index is refused
    /// until recovery moves the staged state to quarantine. Every lifecycle
    /// over a backend holds its writer
    /// ([`LifecycleBackend::writer`](crate::LifecycleBackend::writer)) from
    /// its head read through its publish, so among writers that hold it for
    /// their whole operation a marker met here belongs to no running
    /// operation. A writer that bypasses the writer, or holds it only around
    /// each backend call, can meet or cause another writer's live stage
    /// here, and is still refused with this variant.
    #[snafu(display(
        "index {index} holds interrupted staged state for version {version}; nothing was written"
    ))]
    StagedStateExists {
        /// The index holding the staged state.
        index: IndexIdentity,
        /// The staged version found.
        version: IndexVersion,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: a lifecycle write is computed from the head it read. If another
    /// writer moved the head in between, applying the write would build on a
    /// state that is no longer current, so the backend compares the head
    /// under its own lock and refuses instead.
    #[snafu(display("index {index} changed since its head was read; nothing was written"))]
    HeadChanged {
        /// The index whose head changed.
        index: IndexIdentity,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: an operation key names one operation forever. The same key with
    /// a different digest is a different operation reusing the key; applying
    /// it would make the key's recorded outcome describe content it never
    /// saw, so it is refused and both digests are reported. The detail is
    /// boxed to keep `HeuremaError` within clippy's `result_large_err` bound.
    #[snafu(display(
        "operation key {} on index {} is recorded with digest {}, not the requested {}",
        conflict.key(),
        conflict.index(),
        conflict.recorded(),
        conflict.requested()
    ))]
    OperationConflict {
        /// The index, key, and both digests.
        conflict: Box<OperationConflictDetail>,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: publish and quarantine act on one exact staged state, named by
    /// its marker bytes. If the marker is gone, holds other bytes, or lost
    /// its payload, another writer published, quarantined, or restaged that
    /// version, and acting anyway would publish or move state this caller
    /// never staged.
    #[snafu(display(
        "index {index} holds no staged state for version {version} matching this write; nothing was written"
    ))]
    StagedStateMissing {
        /// The index the write named.
        index: IndexIdentity,
        /// The staged version the write named.
        version: IndexVersion,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: an operation record is the audit entry of a published operation
    /// and is never overwritten. A second publish under a recorded key means
    /// the caller missed that record; it must read the record back and
    /// replay or refuse, never write over it.
    #[snafu(display("index {index} already records operation {key}; nothing was written"))]
    OperationRecorded {
        /// The index the write named.
        index: IndexIdentity,
        /// The operation key already recorded.
        key: OperationKey,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: every lifecycle over one backend shares the backend's writer. A
    /// thread asking for it again while it holds it would wait for itself
    /// forever, so the request is refused instead. Dropping what holds it (a
    /// [`WriterGuard`](crate::WriterGuard) from
    /// [`WriterLock::acquire`](crate::WriterLock::acquire), or a
    /// [`Prepared`](crate::Prepared) or [`Staged`](crate::Staged)
    /// operation, which publishing consumes) releases the writer.
    #[snafu(display(
        "this thread already holds the backend's lifecycle writer (a WriterGuard, or a Prepared or Staged operation, not yet dropped); nothing was written"
    ))]
    WriterHeld {
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: stage never overwrites a stored payload. A lifecycle stages the
    /// version after the head it compare-and-sets, so a payload already
    /// under that version with no marker is published by no head and staged
    /// by no marker. The stored state contradicts itself, and recovery
    /// (which moves only marked state) cannot clear it, so the category is
    /// Corrupt. The backend never decodes a head, so it cannot tell that
    /// payload from a published one: a direct backend caller that stages a
    /// version number already published gets this variant too, category
    /// Corrupt included, over an intact store. That caller reads the head
    /// again and stages the version after the one it names.
    #[snafu(display(
        "index {index} already stores a payload for version {version} that no staging marker names; nothing was written"
    ))]
    VersionStored {
        /// The index the write named.
        index: IndexIdentity,
        /// The version whose payload is already stored.
        version: IndexVersion,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}

/// The index, key, and digests of an [`HeuremaError::OperationConflict`].
///
/// WHY: the conflict carries two identifiers and two digests, which would
/// push `HeuremaError` past clippy's 128-byte `result_large_err` threshold;
/// the variant boxes this detail instead.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OperationConflictDetail {
    index: IndexIdentity,
    key: OperationKey,
    recorded: OperationDigest,
    requested: OperationDigest,
}

impl OperationConflictDetail {
    /// WHY: every part is an already validated identifier, so pairing them
    /// cannot fail.
    #[must_use]
    pub const fn new(
        index: IndexIdentity,
        key: OperationKey,
        recorded: OperationDigest,
        requested: OperationDigest,
    ) -> Self {
        Self {
            index,
            key,
            recorded,
            requested,
        }
    }

    /// The index the key belongs to.
    #[must_use]
    pub const fn index(&self) -> &IndexIdentity {
        &self.index
    }

    /// The operation key both operations use.
    #[must_use]
    pub const fn key(&self) -> &OperationKey {
        &self.key
    }

    /// The digest recorded when the key's operation was published.
    #[must_use]
    pub const fn recorded(&self) -> &OperationDigest {
        &self.recorded
    }

    /// The digest of the operation that reused the key.
    #[must_use]
    pub const fn requested(&self) -> &OperationDigest {
        &self.requested
    }
}

/// The class of failure a [`HeuremaError`] belongs to.
///
/// WHY: a consumer decides whether to retry, report, or repair by the class
/// of a failure rather than by its variant. Every variant maps to exactly one
/// category through [`HeuremaError::category`], so a new variant cannot leave
/// a consumer's handling undefined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCategory {
    /// The input was refused, on its own or against the current state,
    /// before anything changed. The same input against the same state is
    /// refused again.
    Refused,
    /// The named index or snapshot does not exist.
    NotFound,
    /// The input or the stored bytes ask for a capability or format this
    /// build does not implement, such as an analyzer pipeline beyond `Simple`
    /// or a snapshot format version newer than this build reads. Stored bytes
    /// in this category may be valid for a newer build: do not overwrite them.
    Unsupported,
    /// Stored bytes are present but cannot be decoded, violate an engine
    /// invariant, or belong to another index family, or stored lifecycle
    /// state contradicts itself. Retrying the read cannot succeed. A
    /// snapshot is saved again from a rebuilt index; a lifecycle record
    /// (head, payload, or operation record) has no repair path yet, and
    /// recovery and quarantine arrive in a later Phase 02 change. One
    /// exception: [`HeuremaError::VersionStored`] also reaches a direct
    /// [`LifecycleBackend`](crate::LifecycleBackend) caller that stages a
    /// version already published, whose store is intact (see that variant).
    Corrupt,
    /// The persistence backend failed to encode, write, or read bytes; the
    /// error's source chain carries the backend's cause. Whether a retry can
    /// succeed depends on that cause.
    Storage,
    /// The write collides with state another writer recorded: the head
    /// moved after it was read, the staged state it names changed, or its
    /// operation key is already recorded. Nothing was written. Reading the
    /// state again decides what follows: a replay, a fresh attempt, or a
    /// refusal.
    Conflict,
    /// The index holds interrupted staged state from an operation that never
    /// published. Staging and destroying that index are refused until
    /// recovery moves the state to quarantine; other indexes are unaffected.
    /// For writers that hold the backend's writer from their head read
    /// through their publish (every [`IndexLifecycle`](crate::IndexLifecycle)
    /// does), this never names a stage another operation is still running.
    /// A live stage of a writer that bypasses the writer, or holds it only
    /// around each backend call, is reported here too.
    RecoveryRequired,
}

impl HeuremaError {
    /// The class of failure this error belongs to.
    ///
    /// WHY: the match is exhaustive with no wildcard arm, so adding a
    /// variant does not compile until it is given a category.
    #[must_use]
    pub const fn category(&self) -> ErrorCategory {
        match self {
            Self::DimensionMismatch { .. }
            | Self::InvalidVector { .. }
            | Self::DistanceNotRepresentable { .. }
            | Self::InvalidHnswConfig { .. }
            | Self::InvalidKConstant { .. }
            | Self::InvalidIdentifier { .. }
            | Self::FamilyMismatch { .. }
            | Self::DuplicateMember { .. }
            | Self::EmptyBatch { .. }
            | Self::TransitionNotPermitted { .. }
            | Self::RecordMismatch { .. }
            | Self::UnencodableOperation { .. }
            | Self::WriterHeld { .. } => ErrorCategory::Refused,
            Self::IndexNotFound { .. } => ErrorCategory::NotFound,
            Self::NotYetImplemented { .. } | Self::UnsupportedSnapshotVersion { .. } => {
                ErrorCategory::Unsupported
            }
            Self::SnapshotFormat { .. }
            | Self::CorruptSnapshot { .. }
            | Self::VersionStored { .. } => ErrorCategory::Corrupt,
            Self::Persistence { .. } => ErrorCategory::Storage,
            // NOTE: lifecycle storage refusals.
            Self::HeadChanged { .. }
            | Self::OperationConflict { .. }
            | Self::StagedStateMissing { .. }
            | Self::OperationRecorded { .. } => ErrorCategory::Conflict,
            Self::StagedStateExists { .. } => ErrorCategory::RecoveryRequired,
        }
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests need concise identifier fixtures")]
mod tests {
    use std::collections::BTreeSet;

    use snafu::IntoError;

    use super::*;

    /// One sample of every variant.
    fn every_variant() -> Vec<HeuremaError> {
        vec![
            DimensionMismatchSnafu {
                expected: 3_usize,
                actual: 2_usize,
            }
            .build(),
            InvalidVectorSnafu { reason: "NaN" }.build(),
            DistanceNotRepresentableSnafu { reason: "overflow" }.build(),
            InvalidHnswConfigSnafu { reason: "zero" }.build(),
            SnapshotFormatSnafu {
                reason: "wrong family",
            }
            .build(),
            UnsupportedSnapshotVersionSnafu {
                found: 2_u16,
                supported: 1_u16,
            }
            .build(),
            CorruptSnapshotSnafu.into_error(PersistenceSource::new(std::io::Error::other("torn"))),
            InvalidKConstantSnafu {
                k_constant: -1.0_f32,
            }
            .build(),
            IndexNotFoundSnafu { name: "missing" }.build(),
            PersistenceSnafu.into_error(PersistenceSource::new(std::io::Error::other("disk"))),
            NotYetImplementedSnafu { feature: "NGram" }.build(),
            InvalidIdentifierSnafu {
                kind: IdentifierKind::IndexName,
                value: "a/b",
                reason: "slash",
            }
            .build(),
            FamilyMismatchSnafu {
                expected: SnapshotFamily::Vector,
                actual: SnapshotFamily::Fts,
            }
            .build(),
            DuplicateMemberSnafu { member: "7" }.build(),
            EmptyBatchSnafu {
                transition: LifecycleTransition::Insert,
            }
            .build(),
            TransitionNotPermittedSnafu {
                index: sample_index(),
                transition: LifecycleTransition::Create,
                state: IndexStateKind::Destroyed,
            }
            .build(),
            RecordMismatchSnafu {
                expected: sample_index(),
                actual: other_index(),
            }
            .build(),
            UnencodableOperationSnafu { reason: "f64" }.build(),
            // NOTE: lifecycle storage refusals.
            StagedStateExistsSnafu {
                index: sample_index(),
                version: IndexVersion::FIRST,
            }
            .build(),
            HeadChangedSnafu {
                index: sample_index(),
            }
            .build(),
            OperationConflictSnafu {
                conflict: Box::new(sample_conflict()),
            }
            .build(),
            StagedStateMissingSnafu {
                index: sample_index(),
                version: IndexVersion::FIRST,
            }
            .build(),
            OperationRecordedSnafu {
                index: sample_index(),
                key: sample_key(),
            }
            .build(),
            WriterHeldSnafu.build(),
            VersionStoredSnafu {
                index: sample_index(),
                version: IndexVersion::FIRST,
            }
            .build(),
        ]
    }

    fn sample_index() -> IndexIdentity {
        let (Ok(namespace), Ok(name)) = (
            crate::OwnerNamespace::try_from("example"),
            crate::IndexName::try_from("notes"),
        ) else {
            panic!("sample identifiers are grammar-conformant");
        };
        IndexIdentity::new(namespace, name)
    }

    fn other_index() -> IndexIdentity {
        let (Ok(namespace), Ok(name)) = (
            crate::OwnerNamespace::try_from("example"),
            crate::IndexName::try_from("drafts"),
        ) else {
            panic!("sample identifiers are grammar-conformant");
        };
        IndexIdentity::new(namespace, name)
    }

    fn sample_key() -> OperationKey {
        OperationKey::try_from("op-1").expect("key")
    }

    fn sample_conflict() -> OperationConflictDetail {
        OperationConflictDetail::new(
            sample_index(),
            sample_key(),
            OperationDigest::try_from("0a".repeat(32)).expect("recorded digest"),
            OperationDigest::try_from("0b".repeat(32)).expect("requested digest"),
        )
    }

    /// The expected category of each variant, one arm per variant and no
    /// wildcard, written independently of `category()`'s grouping. A new
    /// variant fails to compile here until it has an arm, and the arm's name
    /// must then appear among `every_variant()`'s samples.
    fn expected(error: &HeuremaError) -> (&'static str, ErrorCategory) {
        match error {
            HeuremaError::DimensionMismatch { .. } => ("DimensionMismatch", ErrorCategory::Refused),
            HeuremaError::InvalidVector { .. } => ("InvalidVector", ErrorCategory::Refused),
            HeuremaError::DistanceNotRepresentable { .. } => {
                ("DistanceNotRepresentable", ErrorCategory::Refused)
            }
            HeuremaError::InvalidHnswConfig { .. } => ("InvalidHnswConfig", ErrorCategory::Refused),
            HeuremaError::SnapshotFormat { .. } => ("SnapshotFormat", ErrorCategory::Corrupt),
            HeuremaError::UnsupportedSnapshotVersion { .. } => {
                ("UnsupportedSnapshotVersion", ErrorCategory::Unsupported)
            }
            HeuremaError::CorruptSnapshot { .. } => ("CorruptSnapshot", ErrorCategory::Corrupt),
            HeuremaError::InvalidKConstant { .. } => ("InvalidKConstant", ErrorCategory::Refused),
            HeuremaError::IndexNotFound { .. } => ("IndexNotFound", ErrorCategory::NotFound),
            HeuremaError::Persistence { .. } => ("Persistence", ErrorCategory::Storage),
            HeuremaError::NotYetImplemented { .. } => {
                ("NotYetImplemented", ErrorCategory::Unsupported)
            }
            HeuremaError::InvalidIdentifier { .. } => ("InvalidIdentifier", ErrorCategory::Refused),
            HeuremaError::FamilyMismatch { .. } => ("FamilyMismatch", ErrorCategory::Refused),
            HeuremaError::DuplicateMember { .. } => ("DuplicateMember", ErrorCategory::Refused),
            HeuremaError::EmptyBatch { .. } => ("EmptyBatch", ErrorCategory::Refused),
            HeuremaError::TransitionNotPermitted { .. } => {
                ("TransitionNotPermitted", ErrorCategory::Refused)
            }
            HeuremaError::RecordMismatch { .. } => ("RecordMismatch", ErrorCategory::Refused),
            HeuremaError::UnencodableOperation { .. } => {
                ("UnencodableOperation", ErrorCategory::Refused)
            }
            // NOTE: lifecycle storage refusals.
            HeuremaError::StagedStateExists { .. } => {
                ("StagedStateExists", ErrorCategory::RecoveryRequired)
            }
            HeuremaError::HeadChanged { .. } => ("HeadChanged", ErrorCategory::Conflict),
            HeuremaError::OperationConflict { .. } => {
                ("OperationConflict", ErrorCategory::Conflict)
            }
            HeuremaError::StagedStateMissing { .. } => {
                ("StagedStateMissing", ErrorCategory::Conflict)
            }
            HeuremaError::OperationRecorded { .. } => {
                ("OperationRecorded", ErrorCategory::Conflict)
            }
            HeuremaError::WriterHeld { .. } => ("WriterHeld", ErrorCategory::Refused),
            HeuremaError::VersionStored { .. } => ("VersionStored", ErrorCategory::Corrupt),
        }
    }

    #[test]
    fn error_category_classifies_every_variant() {
        let samples = every_variant();
        let mut named = BTreeSet::new();
        for error in &samples {
            let (name, category) = expected(error);
            assert_eq!(error.category(), category, "{name}: {error}");
            assert!(named.insert(name), "{name} is sampled twice");
        }
        assert_eq!(
            named,
            BTreeSet::from([
                "CorruptSnapshot",
                "DimensionMismatch",
                "DistanceNotRepresentable",
                "DuplicateMember",
                "EmptyBatch",
                "FamilyMismatch",
                "IndexNotFound",
                "InvalidHnswConfig",
                "InvalidIdentifier",
                "InvalidKConstant",
                "InvalidVector",
                "NotYetImplemented",
                "Persistence",
                "RecordMismatch",
                "SnapshotFormat",
                "TransitionNotPermitted",
                "UnencodableOperation",
                "UnsupportedSnapshotVersion",
                // NOTE: lifecycle storage refusals.
                "StagedStateExists",
                "HeadChanged",
                "OperationConflict",
                "StagedStateMissing",
                "OperationRecorded",
                "WriterHeld",
                "VersionStored",
            ]),
            "every variant is sampled"
        );
    }

    #[test]
    fn error_displays_carry_no_em_dash() {
        for error in every_variant() {
            let message = error.to_string();
            assert!(!message.contains('\u{2014}'), "{message}");
        }
    }

    #[test]
    fn lifecycle_refusals_name_the_index_and_what_was_found() {
        let staged = StagedStateExistsSnafu {
            index: sample_index(),
            version: IndexVersion::FIRST,
        }
        .build();
        assert_eq!(
            staged.to_string(),
            "index example/notes holds interrupted staged state for version 1; nothing was written"
        );

        let held = WriterHeldSnafu.build();
        assert_eq!(
            held.to_string(),
            "this thread already holds the backend's lifecycle writer (a WriterGuard, or a \
             Prepared or Staged operation, not yet dropped); nothing was written"
        );

        let stored = VersionStoredSnafu {
            index: sample_index(),
            version: IndexVersion::FIRST,
        }
        .build();
        assert_eq!(
            stored.to_string(),
            "index example/notes already stores a payload for version 1 that no staging marker \
             names; nothing was written"
        );

        let conflict = OperationConflictSnafu {
            conflict: Box::new(sample_conflict()),
        }
        .build();
        let message = conflict.to_string();
        assert!(message.contains("op-1"), "{message}");
        assert!(message.contains("example/notes"), "{message}");
        assert!(message.contains(&"0a".repeat(32)), "{message}");
        assert!(message.contains(&"0b".repeat(32)), "{message}");

        let detail = sample_conflict();
        assert_eq!(detail.index(), &sample_index());
        assert_eq!(detail.key(), &sample_key());
        assert_ne!(detail.recorded(), detail.requested());
    }

    #[test]
    fn error_size_stays_within_the_result_large_err_threshold() {
        // WHY: clippy's `result_large_err` fires on every function returning
        // `Result<_, HeuremaError>` once the error exceeds 128 bytes, across
        // heurema, atmis, thesauros, and their tests, and CI denies warnings.
        // A variant with a large payload must box it; this test fails here
        // first instead of the lint firing workspace-wide.
        let size = std::mem::size_of::<HeuremaError>();
        assert!(
            size <= 128,
            "HeuremaError is {size} bytes; box the new payload"
        );
    }

    #[test]
    fn invalid_identifier_display_names_kind_value_and_reason() {
        let error = InvalidIdentifierSnafu {
            kind: IdentifierKind::OwnerNamespace,
            value: "Example",
            reason: "has 'E' at byte 0",
        }
        .build();
        assert_eq!(
            error.to_string(),
            r#"invalid owner namespace "Example": has 'E' at byte 0"#
        );
    }

    #[test]
    fn lifecycle_refusals_display_what_was_refused() {
        let cases = [
            (
                FamilyMismatchSnafu {
                    expected: SnapshotFamily::Vector,
                    actual: SnapshotFamily::Fts,
                }
                .build(),
                "index family mismatch: expected Vector, got Fts",
            ),
            (
                DuplicateMemberSnafu {
                    member: "TestMember(7)",
                }
                .build(),
                "member TestMember(7) appears more than once in one operation",
            ),
            (
                EmptyBatchSnafu {
                    transition: LifecycleTransition::Remove,
                }
                .build(),
                "Remove operation names no members",
            ),
            (
                TransitionNotPermittedSnafu {
                    index: sample_index(),
                    transition: LifecycleTransition::Insert,
                    state: IndexStateKind::Absent,
                }
                .build(),
                "Insert is not permitted on index example/notes in state Absent",
            ),
            (
                RecordMismatchSnafu {
                    expected: sample_index(),
                    actual: other_index(),
                }
                .build(),
                "operation on index example/notes was checked against the record of index \
                 example/drafts",
            ),
            (
                UnencodableOperationSnafu {
                    reason: "a map key encodes as a sequence",
                }
                .build(),
                "operation cannot be encoded: a map key encodes as a sequence",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected, "{error:?}");
        }
    }
}
