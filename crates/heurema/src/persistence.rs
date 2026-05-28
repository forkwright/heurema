//! Persistence backend contract for Heurēma indexes.

use std::hash::Hash;

use crate::{FtsIndex, HeuremaError, VectorIndex};

/// WHY: Persistence remains outside HNSW and BM25 algorithms so consumers can
/// choose fjall, in-memory, or engine-owned storage without changing indexes.
pub trait PersistenceBackend {
    /// WHY: Vector indexes need named durable snapshots that can be owned by a
    /// query engine catalog.
    fn save_vector_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: VectorIndex;

    /// WHY: Query engines need to load a concrete vector index type from their
    /// catalog entry.
    fn load_vector_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: VectorIndex;

    /// WHY: FTS indexes need the same named snapshot lifecycle as vector
    /// indexes so hybrid search storage stays coherent.
    fn save_fts_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: FtsIndex;

    /// WHY: Query engines need to load a concrete FTS index type from their
    /// catalog entry.
    fn load_fts_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: FtsIndex,
        I::Id: Eq + Hash + Clone;
}
