//! Error types for Heurēma.

use std::fmt;
use std::sync::Arc;

use crate::SnapshotFamily;
use crate::lifecycle::{IdentifierKind, IndexIdentity, IndexStateKind, LifecycleTransition};

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

    /// WHY: a snapshot whose format version this build does not read may be
    /// valid for a newer build. It is its own variant, in the `Unsupported`
    /// category, so a consumer never treats it as corrupt and overwrites it.
    #[snafu(display(
        "index snapshot format version {found} is not supported; this build reads version {supported}"
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
    /// violating an engine invariant, are corrupt rather than a backend I/O
    /// failure; a consumer must rebuild them, and retrying the read cannot
    /// help.
    #[snafu(display("corrupt index snapshot: {source}"))]
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

    /// WHY: an operation's digest is computed over a canonical encoding that
    /// admits no floating-point numbers and only string or integer map keys.
    /// A consumer member identity, provenance, or retention value outside
    /// that grammar is the caller's input to fix, not a storage failure.
    #[snafu(display("operation has no canonical encoding: {reason}"))]
    UnencodableOperation {
        /// Why the encoder refused the operation.
        reason: String,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },
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
    /// invariant, or belong to another index family. Retrying the read cannot
    /// succeed; the stored state needs a rebuild.
    Corrupt,
    /// The persistence backend failed to encode, write, or read bytes; the
    /// error's source chain carries the backend's cause. Whether a retry can
    /// succeed depends on that cause.
    Storage,
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
            | Self::UnencodableOperation { .. } => ErrorCategory::Refused,
            Self::IndexNotFound { .. } => ErrorCategory::NotFound,
            Self::NotYetImplemented { .. } | Self::UnsupportedSnapshotVersion { .. } => {
                ErrorCategory::Unsupported
            }
            Self::SnapshotFormat { .. } | Self::CorruptSnapshot { .. } => ErrorCategory::Corrupt,
            Self::Persistence { .. } => ErrorCategory::Storage,
        }
    }
}

#[cfg(test)]
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
            UnencodableOperationSnafu { reason: "f64" }.build(),
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
            HeuremaError::UnencodableOperation { .. } => {
                ("UnencodableOperation", ErrorCategory::Refused)
            }
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
                "SnapshotFormat",
                "TransitionNotPermitted",
                "UnencodableOperation",
                "UnsupportedSnapshotVersion",
            ]),
            "every variant is sampled"
        );
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
                UnencodableOperationSnafu {
                    reason: "a map key encodes as a sequence",
                }
                .build(),
                "operation has no canonical encoding: a map key encodes as a sequence",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected, "{error:?}");
        }
    }
}
