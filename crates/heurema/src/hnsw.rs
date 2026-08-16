//! HNSW vector index contract and Phase 1 stub type.

use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::HeuremaError;

mod stub;

pub use stub::HnswIndex;

const DEFAULT_EF_CONSTRUCTION: usize = 50;
const DEFAULT_M_NEIGHBOURS: usize = 16;

/// WHY: Krites supports these distance modes today; Heurēma keeps them in the
/// public API so its independently-written implementation preserves the
/// query-engine semantics krites already established.
///
/// WHY `Serialize` + `Deserialize`: a `PersistenceBackend` adapter encodes an
/// index's `HnswConfig` as part of its snapshot bytes; see `persistence.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
///
/// WHY `Serialize` + `Deserialize`: see [`VectorDistance`]. `new` fills
/// defaults for the unset fields but validates nothing, so every field
/// combination deserializes to a valid value; this is a pure data-transfer
/// type, not a validated newtype, and the plain derive is the correct form
/// per `RUST.md` § Serde validation. `deny_unknown_fields` still applies —
/// bytes decoding as a different concrete config shape under a shared
/// `PersistenceBackend` snapshot name is a real failure mode
/// (`crates/thesauros/tests/persistence_fjall.rs` exercises it), and a
/// closed schema turns a silent partial-match into an explicit decode error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    ///
    /// WHY: `Ord` is the bound [`crate::rrf`] fuses under. Requiring it here
    /// keeps the advertised hybrid path — query an index, fuse the result —
    /// compilable for a generic consumer that knows only this trait.
    type Id: Ord + Hash + Clone;

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
    /// Ranking contract — implementations must satisfy all of it, because
    /// [`crate::rrf`] reads *position* as the authoritative rank and ignores
    /// the returned score entirely:
    ///
    /// - **Ordering is normative.** Element 0 is the best match, and each
    ///   subsequent element is no better than its predecessor. A result vector
    ///   in any other order still type-checks but silently changes what
    ///   fusion computes.
    /// - **Score polarity is ascending-is-better** — these are distances, so a
    ///   smaller `f32` is a closer match, and the sequence is non-decreasing.
    ///   The score is advisory: it is carried for display and thresholding,
    ///   never used to re-derive rank.
    /// - **IDs are unique** within one result vector.
    /// - **Ties are stable.** Equal scores must be ordered by ascending `Id`,
    ///   so repeating a query over unchanged index state returns an identical
    ///   vector.
    /// - **At most `k`** elements are returned; fewer is valid when the index
    ///   holds fewer candidates.
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
