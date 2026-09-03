//! Conformance oracle for the HNSW and BM25 surfaces (heurema#29): property
//! tests derived from the published algorithm definitions plus the ranking
//! contracts documented on `VectorIndex::query` and `FtsIndex::query`. Tests
//! blocked on the Phase 2 implementations carry `#[ignore]`;
//! `tests/oracle/PARITY.md` defines the red/green contract and what "reaches
//! parity" means operationally.

mod bm25;
mod hnsw;
mod support;
