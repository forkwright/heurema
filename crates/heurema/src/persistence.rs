//! Persistence backend contract for Heurēma indexes.

use crate::{FtsIndex, HeuremaError, VectorIndex};

/// WHY: Persistence remains outside HNSW and BM25 algorithms so consumers can
/// choose fjall, in-memory, or engine-owned storage without changing indexes.
pub trait PersistenceBackend {
    /// WHY: Vector indexes need named durable snapshots that can be owned by a
    /// query engine catalog.
    ///
    /// Saving to a `name` that already holds a snapshot replaces it:
    /// last-write-wins, never an error. Catalog owners that need
    /// create-only semantics must check existence before saving.
    fn save_vector_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: VectorIndex;

    /// WHY: Query engines need to load a concrete vector index type from their
    /// catalog entry.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::IndexNotFound`] when no snapshot exists under
    /// `name`, and [`HeuremaError::Persistence`] on storage failure.
    fn load_vector_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: VectorIndex;

    /// WHY: FTS indexes need the same named snapshot lifecycle as vector
    /// indexes so hybrid search storage stays coherent.
    ///
    /// Saving to a `name` that already holds a snapshot replaces it:
    /// last-write-wins, never an error. Catalog owners that need
    /// create-only semantics must check existence before saving.
    fn save_fts_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: FtsIndex;

    /// WHY: Query engines need to load a concrete FTS index type from their
    /// catalog entry.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::IndexNotFound`] when no snapshot exists under
    /// `name`, and [`HeuremaError::Persistence`] on storage failure.
    fn load_fts_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: FtsIndex;
}
