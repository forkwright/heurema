//! # Heurēma
//!
//! Shared vector, full-text, persistence, and rank-fusion search primitives for
//! the forkwright fleet.
//!
//! This crate owns the index contracts that query engines wrap, and stays
//! agnostic of any consumer-specific language surface — pinax owns SQL.
//!
//! Datalog is not a separate repo's concern: heurēma owns that engine as
//! `akolouthia`, in this workspace, because separating an engine from the
//! indexes it queries puts a cross-repo seam on the hottest path. `mneme` is
//! the memory policy layer above heurēma, not a second engine.

#![deny(missing_docs)]

mod error;

/// WHY: Full-text search needs a shared trait boundary before BM25 extraction
/// moves out of krites.
pub mod fts;
/// WHY: HNSW vector search needs one fleet implementation and one correctness
/// proof instead of per-consumer graph implementations.
pub mod hnsw;
/// WHY: Persistence stays pluggable so query engines can choose in-memory,
/// fjall-backed, or engine-owned storage without changing index APIs.
pub mod persistence;
/// WHY: Hybrid search consumers need rank fusion without depending on krites.
pub mod rrf;

pub use error::{HeuremaError, PersistenceSource};
pub use fts::{Bm25Index, FtsConfig, FtsIndex, TokenizerConfig};
pub use hnsw::{HnswConfig, HnswIndex, VectorDistance, VectorIndex};
pub use persistence::PersistenceBackend;
pub use rrf::{DEFAULT_RRF_K_CONSTANT, rrf, rrf_with_default};
