# heurēma

*εὕρημα - a finding, a discovery. Root of "eureka." Search indices are the means by which a system finds what it didn't know it was holding.*

Shared vector, full-text, persistence, and rank-fusion search primitives for the forkwright fleet.

Heurēma owns the public contracts for HNSW vector search, BM25 full-text search, reciprocal-rank fusion, and pluggable persistence. Query engines such as `pinax` and `mneme` wrap these primitives in their own SQL or Datalog surfaces instead of growing separate HNSW or BM25 implementations.

The discovery primitives are found once and shared, not reinvented by every consumer.

## Status

Phase 1: scaffold + API + RRF. HNSW and FTS implementations are typed stubs until Phase 2 extracts the canonical code from aletheia's `krites` crate. Phase 1 commits the public API so consumers can build against the trait shape today.

## API surface

```rust
use heurema::{
    Bm25Index, FtsConfig, FtsIndex,
    HnswConfig, HnswIndex, VectorIndex,
    PersistenceBackend,
    rrf, rrf_with_default, DEFAULT_RRF_K_CONSTANT,
};
```

- `VectorIndex` - insert / query / remove for ID-keyed vectors, plus `len` / `is_empty`.
- `FtsIndex` - insert / query / remove for ID-keyed documents, with BM25-style scores.
- `PersistenceBackend` - save / load named vector and FTS indexes; backend-agnostic.
- `rrf` / `rrf_with_default` - reciprocal-rank fusion with the paper-standard `k = 60`.

The API is deliberately engine-agnostic. Heurēma knows nothing about SQL, Datalog, or any consumer-owned query language; it provides the index contracts those engines wrap.

## Non-goals

Heurēma is not a vector database, query language, embedding-model host, or distributed search layer. Embedding models live in `logismos`; SQL routing lives in `pinax`; Datalog routing lives in `mneme`. Heurēma is the substrate those systems depend on.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
