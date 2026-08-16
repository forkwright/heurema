# heurēma

*εὕρημα - a finding, a discovery. Root of "eureka." Search indices are the means by which a system finds what it didn't know it was holding.*

A reservation for the fleet's shared vector (HNSW), full-text (BM25), and rank-fusion search primitives, plus a persistence-adapter substrate that is fully working today. HNSW and BM25 stay a committed trait surface pending a fresh implementation.

## What's real

`rrf` / `rrf_with_default` (reciprocal-rank fusion, `crates/heurema/src/rrf.rs`) is a complete,
tested implementation: `f64` accumulation narrowed to a public `f32` score, a documented total-order
tie-break (score descending, then `Id` ascending - deterministic regardless of input ranking order),
and intra-ranking dedup (a duplicated `Id` within one ranking contributes only its best rank). Covered
by `crates/heurema/tests/rrf_correctness.rs` and `index_rrf_composition.rs`.

`PersistenceBackend` (`crates/heurema/src/persistence.rs`) ships two implementations. `atmis`'s
`AtmisBackend` keeps snapshots in a `HashMap<String, Vec<u8>>` behind an `RwLock` and never touches
disk; every test in this repo can run against it without filesystem I/O. `thesauros`'s
`ThesaurosBackend` opens a fjall keyspace, splits vector and FTS snapshots into separate partitions, and
fsyncs (`fjall::PersistMode::SyncAll`) after every write, so a save that returns `Ok` is durable before
the caller observes it. Both encode through `serde_json`, so a caller-chosen index type needs
`Serialize` on save and `DeserializeOwned` on load; see `persistence.rs` for why the trait carries that
bound. Neither crate is named `heurema-*`: see the `WHY` comment on the workspace `Cargo.toml`
`[workspace.dependencies]` block for why (`NAMING.md` forbids that shape; both are independent GNOMON
coinages instead).

## What's a stub

`HnswIndex` (`src/hnsw/stub.rs`) and `Bm25Index` (`src/fts/stub.rs`) are concrete types that satisfy
the `VectorIndex` / `FtsIndex` trait bounds so downstream code compiles, but every `insert` / `query` /
`remove` returns `HeuremaError::NotYetImplemented`, and `len()` returns a field that is never
incremented, so there is no index behind either type yet.

This is deliberate, not drift: the trait surface *is* the design. Committing the API now lets `pinax`
and `mneme` build against the shape before the engines exist.

The engines will be **written fresh**, not extracted. aletheia's `krites` carries working HNSW and BM25,
but that code is vendored CozoDB under MPL-2.0, so moving it would relocate a provenance question rather
than resolve one. `krites` instead serves as a behavioural reference and its tests as a conformance
oracle, which is also the opportunity to fix what the vendored implementation got wrong.

Nothing here should be read as "heurēma provides HNSW/BM25 search" until those implementations land.

## API surface

```rust
use heurema::{
    Bm25Index, FtsConfig, FtsIndex,
    HnswConfig, HnswIndex, VectorIndex,
    PersistenceBackend,
    rrf, rrf_with_default, DEFAULT_RRF_K_CONSTANT,
};
use atmis::AtmisBackend;
use thesauros::ThesaurosBackend;
```

- `VectorIndex` - insert / query / remove for ID-keyed vectors, plus `len` / `is_empty`. Trait is real; `HnswIndex` is a stub.
- `FtsIndex` - insert / query / remove for ID-keyed documents, with BM25-style scores. Trait is real; `Bm25Index` is a stub.
- `PersistenceBackend` - save / load named vector and FTS indexes; backend-agnostic. Trait is real; `atmis`'s `AtmisBackend` and `thesauros`'s `ThesaurosBackend` both implement it.
- `rrf` / `rrf_with_default` - reciprocal-rank fusion with the paper-standard `k = 60`. Implemented and tested.

The API is deliberately engine-agnostic. Heurēma knows nothing about SQL, Datalog, or any consumer-owned query language; it provides the index contracts those engines wrap.

## Non-goals

Heurēma is not a vector database, embedding-model host, or distributed search layer. Embedding models live in `logismos`; SQL routing lives in `pinax`.

Datalog is the exception, and it is deliberate: heurēma owns the Datalog engine as `akolouthia`. Splitting the engine from the indexes it queries would put a cross-repo seam on the hottest path in the fleet, so the engine and the index contracts live in one workspace behind one facade, adapted per consumer by configuration. The `mneme` repo is the memory *policy* layer - factor sets, admission, lifecycle rules - sitting over heurēma, not a second engine.

## License

MPL-2.0. See [LICENSE](LICENSE).

The choice is load-bearing, because heurēma has to be consumable from both sides of the fleet at once.
`aletheia` is AGPL-3.0-or-later, and MPL §1.12 makes that a Secondary License, so §3.3 lets it consume
heurēma directly. `kanon`, `logismos` and `daimon` are PolyForm Shield, which cannot grant AGPL's §5(c)
and §13 terms - under an AGPL heurēma they could not have been consumers at all. MPL's file-level
copyleft reaches those repos without that conflict.

Two consequences worth stating so they are not undone by a later tidy-up. Exhibit B is deliberately not
attached to any file here; attaching it would make heurēma Incompatible With Secondary Licenses and cut
off aletheia. And `deny.toml` does not allow AGPL dependencies, since taking one would propagate terms
the Shield consumers cannot accept.
