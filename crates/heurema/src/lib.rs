//! # Heurēma
//!
//! Shared vector, full-text, persistence, and rank-fusion search primitives for
//! the forkwright fleet.
//!
//! This crate owns the index contracts that query engines wrap, and stays
//! agnostic of any consumer-specific language surface — pinax owns SQL.
//!
//! Datalog is not a separate repo's concern: heurēma owns that engine as
//! `akolouthia`, planned for this workspace and not yet built, because
//! separating an engine from the indexes it queries puts a cross-repo seam on
//! the hottest path. `mneme` is
//! the memory policy layer above heurēma, not a second engine.

#![deny(missing_docs)]

mod error;

/// WHY: Full-text search needs one shared trait boundary; `Bm25Index` is
/// heurēma's fresh BM25 implementation of it.
pub mod fts;
/// WHY: HNSW vector search needs one fleet implementation and one correctness
/// proof instead of per-consumer graph implementations.
pub mod hnsw;
/// WHY: The durable retrieval lifecycle's contract (index identity, operation
/// identity, transitions, records, and the storage those records live in)
/// must be readable from types before any durable write depends on it, and
/// consumers and adapters bind that one vocabulary.
pub mod lifecycle;
/// WHY: Persistence stays pluggable so query engines can choose in-memory,
/// fjall-backed, or engine-owned storage without changing index APIs.
pub mod persistence;
/// WHY: Hybrid search consumers need rank fusion without depending on krites.
pub mod rrf;

pub use error::{ErrorCategory, HeuremaError, OperationConflictDetail, PersistenceSource};
pub use fts::{Bm25Index, FtsConfig, FtsIndex, TokenizerConfig};
pub use hnsw::{HnswConfig, HnswIndex, VectorDistance, VectorIndex};
pub use lifecycle::{
    CheckedOperation, IdentifierKind, IndexChange, IndexConfig, IndexIdentity, IndexMember,
    IndexName, IndexRecord, IndexState, IndexStateKind, IndexVersion, LifecycleOperation,
    LifecycleTransition, MemberContent, MemberIdentity, OperationDigest, OperationIdentity,
    OperationKey, OwnerNamespace, ProvenanceReference, RetentionReference, ValidatedOperation,
};
pub use lifecycle::{
    DestroyWrite, LIFECYCLE_FORMAT_VERSION, LifecycleBackend, PublishWrite, QuarantineWrite,
    QuarantinedEntry, StageWrite, StagedEntry,
};
pub use persistence::{
    PersistenceBackend, SNAPSHOT_FORMAT_VERSION, SnapshotEnvelope, SnapshotFamily,
    decode_snapshot_payload,
};
pub use rrf::{DEFAULT_RRF_K_CONSTANT, rrf, rrf_with_default};
