//! HNSW vector index contract and Phase 1 stub type.

use std::hash::Hash;

use crate::HeuremaError;

mod stub;

pub use stub::HnswIndex;

const DEFAULT_EF_CONSTRUCTION: usize = 50;
const DEFAULT_M_NEIGHBOURS: usize = 16;

/// WHY: Krites supports these distance modes today; Heurēma keeps them in the
/// public API so extraction does not change query-engine semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VectorDistance {
    /// Squared Euclidean distance.
    L2,
    /// Cosine distance.
    Cosine,
    /// Inner-product distance.
    InnerProduct,
}

/// WHY: HNSW construction parameters must be explicit because graph quality,
/// recall, and storage shape depend on them.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HnswConfig {
    /// Vector dimensionality accepted by the index.
    pub dimensions: usize,
    /// Distance function used for ranking neighbors.
    pub distance: VectorDistance,
    /// Candidate beam width used while building graph connections.
    pub ef_construction: usize,
    /// Target neighbor count from the HNSW paper's `m` parameter.
    pub m_neighbours: usize,
}

impl HnswConfig {
    /// WHY: Consumers need a minimal config constructor that preserves krites's
    /// current defaults for early API smoke tests.
    #[must_use]
    pub const fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            distance: VectorDistance::L2,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
            m_neighbours: DEFAULT_M_NEIGHBOURS,
        }
    }
}

/// WHY: Vector engines need a stable contract for insert, kNN query, removal,
/// and size inspection independent of SQL or Datalog storage.
pub trait VectorIndex {
    /// Identifier stored alongside each vector.
    type Id: Eq + Hash + Clone;

    /// WHY: Insert mirrors krites `hnsw_put`: a consumer supplies an owned ID
    /// plus the vector bytes to index.
    ///
    /// # Errors
    ///
    /// Implementations must return [`HeuremaError::DimensionMismatch`] when
    /// `vector.len()` differs from the dimensionality the index was
    /// configured with, before mutating any index state.
    fn insert(&mut self, id: Self::Id, vector: &[f32]) -> Result<(), HeuremaError>;

    /// WHY: Query mirrors krites `hnsw_knn`: consumers ask for top-k IDs and
    /// distances without receiving engine-owned tuples.
    ///
    /// # Errors
    ///
    /// Implementations must return [`HeuremaError::DimensionMismatch`] when
    /// `vector.len()` differs from the dimensionality the index was
    /// configured with.
    fn query(&self, vector: &[f32], k: usize) -> Result<Vec<(Self::Id, f32)>, HeuremaError>;

    /// WHY: Remove mirrors krites `hnsw_remove`: deleting a base row must also
    /// delete the corresponding vector graph entry.
    ///
    /// Removal is idempotent: removing an `id` that is not in the index is a
    /// successful no-op, so row-deletion cleanup can retry safely.
    fn remove(&mut self, id: &Self::Id) -> Result<(), HeuremaError>;

    /// WHY: Consumers need index cardinality for adaptive exact-vs-HNSW search.
    fn len(&self) -> usize;

    /// WHY: `len` callers need a non-arithmetic emptiness predicate for clean
    /// clippy-compliant APIs.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
