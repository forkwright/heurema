//! Error types for Heurēma.

use std::fmt;
use std::sync::Arc;

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
    /// WHY: The fresh HNSW implementation must reproduce krites's strict
    /// vector dimension checks instead of silently accepting malformed query
    /// vectors.
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
    /// need to remain in the same error chain for callers.
    #[snafu(display("persistence backend error: {source}"))]
    Persistence {
        /// Backend-specific source error.
        source: PersistenceSource,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },

    /// WHY: Phase 1 commits the public API before Phase 2's fresh HNSW/BM25
    /// implementations land.
    #[snafu(display("not yet implemented: {feature}"))]
    NotYetImplemented {
        /// Feature not yet implemented; lands in Phase 2.
        feature: String,
        /// Error creation location.
        #[snafu(implicit)]
        location: snafu::Location,
    },
}
