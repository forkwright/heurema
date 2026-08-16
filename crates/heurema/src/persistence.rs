//! Persistence backend contract for Heurēma indexes.

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{FtsIndex, HeuremaError, VectorIndex};

/// WHY: Persistence remains outside HNSW and BM25 algorithms so consumers can
/// choose fjall, in-memory, or engine-owned storage without changing indexes.
///
/// WARNING: Phase 1 declared every method here generic over only
/// `I: VectorIndex` / `I: FtsIndex`. Phase 3, landing this trait's first
/// implementations (`atmis`, `thesauros`), adds `Serialize` to
/// the save methods and `DeserializeOwned` to the load methods. This is a
/// signature change, not an addition beside the old one: neither
/// `VectorIndex` nor `FtsIndex` exposes a constructor, so `load_vector_index`
/// could never build a value of a caller-chosen `I` under the original
/// bound, and `Ok` from either Phase 1 load method was structurally
/// unreachable. `DeserializeOwned` supplies the missing construction path;
/// `Serialize` is its mirror on save. The bound lives at the persistence
/// boundary only. `VectorIndex` and `FtsIndex` themselves are unchanged.
pub trait PersistenceBackend {
    /// WHY: Vector indexes need named durable snapshots that can be owned by a
    /// query engine catalog.
    ///
    /// Saving to a `name` that already holds a snapshot replaces it:
    /// last-write-wins, never an error. Catalog owners that need
    /// create-only semantics must check existence before saving.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::Persistence`] when the backend cannot encode
    /// `idx` or fails to write the encoded snapshot.
    fn save_vector_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: VectorIndex + Serialize;

    /// WHY: Query engines need to load a concrete vector index type from their
    /// catalog entry.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::IndexNotFound`] when no snapshot exists under
    /// `name`, and [`HeuremaError::Persistence`] on storage failure or when
    /// the stored snapshot does not decode as `I`.
    fn load_vector_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: VectorIndex + DeserializeOwned;

    /// WHY: FTS indexes need the same named snapshot lifecycle as vector
    /// indexes so hybrid search storage stays coherent.
    ///
    /// Saving to a `name` that already holds a snapshot replaces it:
    /// last-write-wins, never an error. Catalog owners that need
    /// create-only semantics must check existence before saving.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::Persistence`] when the backend cannot encode
    /// `idx` or fails to write the encoded snapshot.
    fn save_fts_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: FtsIndex + Serialize;

    /// WHY: Query engines need to load a concrete FTS index type from their
    /// catalog entry.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::IndexNotFound`] when no snapshot exists under
    /// `name`, and [`HeuremaError::Persistence`] on storage failure or when
    /// the stored snapshot does not decode as `I`.
    fn load_fts_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: FtsIndex + DeserializeOwned;
}
