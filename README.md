# heurēma

*εὕρημα - a finding, a discovery. Root of "eureka." Search indices are the means by which a system finds what it didn't know it was holding.*

Shared vector (HNSW), full-text (BM25), rank-fusion, and persistence-adapter primitives for the fleet.

## What's real

`HnswIndex` (`crates/heurema/src/hnsw/engine.rs`) is a safe, deterministic, in-memory HNSW graph that
implements `VectorIndex`. It persists its level-assignment state, bounds greedy and best-first
traversal, keeps reciprocal links plus a base-layer backbone cycle that degree pruning cannot remove,
repairs connectivity on replacement and removal, and validates dimensions, finite components,
configuration, and snapshot graph invariants. It landed in PR #50 (fresh navigable HNSW graph) and
meets the pinned brute-force recall floor in `crates/heurema/tests/oracle/hnsw.rs`.

`Bm25Index` (`crates/heurema/src/fts/bm25.rs`) is an in-memory BM25 engine that implements `FtsIndex`
for the `Simple` pipeline (`FtsConfig::simple()`): insert, replacement, removal, corpus statistics, and
ranked scores with stable ties. It landed in PR #48 (fresh Simple-pipeline BM25). Any other tokenizer
or filter pipeline returns `HeuremaError::NotYetImplemented` before it mutates state.

Both engines were written fresh in this repository; CLAUDE.md's Roadmap records the ruling. The
conformance oracle in `crates/heurema/tests/oracle/` runs every BM25 and HNSW case live; none is
ignored.

`rrf` / `rrf_with_default` (reciprocal-rank fusion, `crates/heurema/src/rrf.rs`) is a complete,
tested implementation: `f64` accumulation narrowed to a public `f32` score, a documented total-order
tie-break (score descending, then `Id` ascending - deterministic regardless of input ranking order),
and intra-ranking dedup (a duplicated `Id` within one ranking contributes only its best rank). Covered
by `crates/heurema/tests/rrf_correctness.rs` and `index_rrf_composition.rs`.

`PersistenceBackend` (`crates/heurema/src/persistence.rs`) ships two implementations. `atmis`'s
`AtmisBackend` keeps snapshots in a `HashMap<String, Vec<u8>>` behind an `RwLock` and never touches
disk; every test in this repo can run against it without filesystem I/O. `thesauros`'s
`ThesaurosBackend` opens a fjall database, keeps vector and FTS snapshots in separate keyspaces, and
fsyncs (`fjall::PersistMode::SyncAll`) after every write, so a save that returns `Ok` is durable before
the caller observes it. Both wrap each index in a versioned `SnapshotEnvelope` (format version plus
`SnapshotFamily`), encode it through `serde_json`, and load through `decode_snapshot_payload`, which
refuses a format version this build does not read (`HeuremaError::UnsupportedSnapshotVersion`) or the
wrong family (`HeuremaError::SnapshotFormat`) before the index's own decoder runs, and reports bytes
that do not decode as `HeuremaError::CorruptSnapshot` (PR #51, versioned individual snapshots). Each
error names its class through `HeuremaError::category()`. A caller-chosen index type needs
`Serialize` on save and `DeserializeOwned` on load; see `persistence.rs` for why the trait carries that
bound. Neither crate is named `heurema-*`: see the `WHY` comment on the workspace `Cargo.toml`
`[workspace.dependencies]` block for why (`NAMING.md` forbids that shape; both are independent GNOMON
coinages instead).

A snapshot is a whole-index save of one index, not a transaction: nothing here makes two indexes, or an
index and the records it was built from, become visible together.

## Status

Phase 01 (fresh HNSW and BM25 engines, each checked against an independent oracle) is complete.
Phase 02, the durable retrieval lifecycle (named indexes, staged writes, one atomic publish point,
recovery, and deletion rules), is in progress: its type vocabulary and pre-publish validation
(`heurema::lifecycle`), its storage layer (`LifecycleBackend`, implemented by both adapters), and the
driver that stages each operation's version and publishes it at one atomic point (`IndexLifecycle`)
have landed; recovery on reopen follows. The Datalog engine `akolouthia` comes after it. CLAUDE.md's
Roadmap carries the phase list.

## API surface

```rust
use heurema::{
    Bm25Index, FtsConfig, FtsIndex, TokenizerConfig,
    HnswConfig, HnswIndex, VectorDistance, VectorIndex,
    PersistenceBackend, SnapshotEnvelope, SnapshotFamily, SNAPSHOT_FORMAT_VERSION,
    decode_snapshot_payload,
    rrf, rrf_with_default, DEFAULT_RRF_K_CONSTANT,
    HeuremaError,
};
use atmis::AtmisBackend;
use thesauros::ThesaurosBackend;
```

- `VectorIndex` - insert / query / remove for ID-keyed vectors, plus `len` / `is_empty`. `HnswIndex` implements it as an in-memory HNSW graph.
- `FtsIndex` - insert / query / remove for ID-keyed documents, with BM25-style scores. `Bm25Index` implements the `Simple` pipeline.
- `PersistenceBackend` - save / load named vector and FTS indexes; backend-agnostic. `atmis`'s `AtmisBackend` and `thesauros`'s `ThesaurosBackend` both implement it.
- `SnapshotEnvelope` / `SnapshotFamily` / `decode_snapshot_payload` - the versioned per-index snapshot format both adapters share.
- `rrf` / `rrf_with_default` - reciprocal-rank fusion with the paper-standard `k = 60`.
- `lifecycle` - the Phase 02 durable-lifecycle vocabulary and its pre-publish validation: `IndexIdentity` (`OwnerNamespace` plus `IndexName`), `OperationKey` / `OperationIdentity` / `OperationDigest`, `IndexVersion`, `LifecycleOperation` / `IndexChange` / `LifecycleTransition`, `IndexRecord` / `IndexState`, and the `MemberIdentity` / `ProvenanceReference` / `RetentionReference` marker traits a consumer implements on its own types. `CheckedOperation::check` refuses an invalid operation without any index state and computes its digest (SHA-256 over a canonical encoding documented on `OperationDigest`); `CheckedOperation::permit` applies the permission table (`LifecycleTransition::is_permitted_from`) and the checks against the current record, yielding a `ValidatedOperation`. Both are pure functions. `HeuremaError::category()` sorts every error into an `ErrorCategory`.
- `LifecycleBackend` - the lifecycle's storage contract: object safe, opaque heurēma-encoded bytes under one `storage_key` grammar, and four writes (stage, publish, destroy, quarantine), each atomic and durable before `Ok`. Writes compare-and-set against the head they were computed from, never overwrite a stored payload or operation record, never stage for an operation key already recorded, and refuse while an index holds interrupted staged state. Every lifecycle over one backend shares its `WriterLock` (`LifecycleBackend::writer`). `AtmisBackend` applies each write under one mutex; `ThesaurosBackend` commits each as one fjall write batch with `PersistMode::SyncAll`, and refuses a value of 4 GiB or more (fjall stores at most 2^32 - 1 bytes) before writing; a stage also checks the head and operation record its publish will write.
- `IndexLifecycle` - the lifecycle driver over any `LifecycleBackend`. `apply` runs the stateless checks, takes the backend's `WriterLock`, reads the head before a replay lookup by key and digest, reads the active version, runs the permission and state checks, and builds the next version in memory, then stages that version and publishes it in one atomic write of the new head, the operation record, and the staging marker's removal; `prepare`, `Prepared::stage`, and `Staged::publish` run the same steps one at a time. An operation refused before it stages writes nothing, and one refused by a stateless check makes no backend call; a publish refused after its stage leaves the staged version as orphan staged state, which only a writer that bypasses the shared writer, or damage, causes. Another thread's operation waits for the writer; a thread already holding the writer (through a prepared or staged operation, or a `WriterGuard` it took) is refused with `WriterHeld`, never deadlocked. Validation also refuses a member identity, provenance, or retention value that does not read back as itself from its JSON encoding. The same key with the same content replays the recorded `PublishReceipt`; with other content it is `OperationConflict`. `index` returns the active version as a `PublishedIndex` whose members carry their `MemberEntry` (provenance, introducing version, superseded version) and whose `query_vector` / `query_text` return `IndexHit`s carrying identity, provenance, and version; a staged version is never readable. A version staged and never published refuses every later mutation of its index until recovery, which a later Phase 02 change adds. Every published version payload is kept until destroy, so storage grows with each operation.

The index API is engine-agnostic: heurēma knows nothing about SQL or any consumer-owned query language,
and it returns IDs and scores rather than consumer tuples.

## Non-goals

Heurēma is not a vector database, embedding-model host, or distributed search layer. Embedding models live in `logismos`; SQL routing lives in `pinax`.

Datalog is the exception, and it is deliberate: heurēma owns the fleet's Datalog engine, `akolouthia`, planned for this workspace (no crate exists yet). Splitting the engine from the indexes it queries would put a cross-repo seam on the hottest path in the fleet, so the engine and the index contracts belong in one workspace behind one facade. The `mneme` repo is the memory *policy* layer - factor sets, admission, lifecycle rules - sitting over heurēma, not a second engine.

## License

Apache-2.0 OR MIT, at your option. See [LICENSE](LICENSE).

heurēma is a library other fleet repositories depend on, so it carries the fleet's interop license
posture (PR #28, one license declaration rendered per repository): a permissive dual license that every
consumer can take whatever its own license is. `deny.toml` does not allow AGPL dependencies, since
taking one would propagate its terms to every consumer.
