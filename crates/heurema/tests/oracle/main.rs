//! Conformance oracle for the HNSW and BM25 surfaces (heurema#29): property
//! tests derived from the published algorithm definitions plus the ranking
//! contracts documented on `VectorIndex::query` and `FtsIndex::query`. Every
//! case runs live against the fresh engines; `tests/oracle/PARITY.md` defines
//! the green contract and what "reaches parity" means operationally.

mod bm25;
mod bm25_formula;
mod hnsw;
mod support;
