<!--
scope: heurema repo conventions (fleet substrate: the heurema traits-and-engines crate plus its persistence-adapter sibling crates)
defers_to: ~/.claude/CLAUDE.md for operator principles; `crates/basanos/standards/STANDARDS.md` in `forkwright/kanon` for fleet-wide standards
tightens: per-crate CLAUDE.md under crates/heurema/ may narrow conventions
-->

# heurema

Shared search-index primitives for the forkwright fleet. One implementation, one proof, orthogonal to query engines.

## Standards

Universal fleet standards live in `crates/basanos/standards/` in `forkwright/kanon`. This repo inherits — it does not restate.

Particularly relevant:

- `PHILOSOPHY.md` — fleet philosophical SSOT (including §"Presence: attention as a moral act")
- `GNOMON.md` — naming layer test (L1–L4)
- `COHERENCE.md` — architectural quality tests
- `RUST.md` — Rust-specific standards
- `TESTING.md` — testing principles
- `FLEET-REPO-SETUP.md` — fleet repo conventions this repo conforms to

## Layout

```
Cargo.toml           # workspace root
crates/heurema/      # traits, HNSW + BM25 engines, RRF, snapshot envelope,
                     # durable-lifecycle vocabulary, validation, storage contract
  src/               # lib.rs, error.rs, fts.rs (+ fts/bm25.rs),
                     # hnsw.rs (+ hnsw/engine.rs), lifecycle.rs (+ lifecycle/
                     # backend.rs, digest.rs, identity.rs, member.rs,
                     # operation.rs, record.rs, validate.rs), persistence.rs, rrf.rs
  tests/             # api_smoke.rs, bm25_lifecycle.rs, bm25_snapshot.rs,
                     # index_rrf_composition.rs, lifecycle_digest.rs,
                     # lifecycle_validation.rs, oracle/,
                     # persistence_contract.rs, persistence_schema_closed.rs,
                     # rrf_correctness.rs
crates/atmis/        # in-memory Persistence/LifecycleBackend adapter (ἀτμίς, vapor)
  src/lib.rs         # AtmisBackend
  tests/             # persistence_memory.rs
crates/thesauros/    # fjall-backed Persistence/LifecycleBackend adapter (θησαυρός, storehouse)
  src/lib.rs         # ThesaurosBackend
  tests/             # persistence_fjall.rs, lifecycle_backend.rs (both adapters)
_llm/                # structured LLM corpus
.github/workflows/   # ci.yml, codeql.yml, gate-attestation.yml, release-please.yml,
                     # release-pr-checks.yml, security.yml
scripts/docs-drift.sh # CI guard: cited workspace paths exist, no stub-era wording
```

The workspace shape was chosen for exactly this: the two adapter sub-crates plug in as siblings with no restructure of `crates/heurema/`.

## Commands

```bash
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Patterns

- **Errors:** one `snafu` enum, `HeuremaError` (`#[non_exhaustive]`), with `#[snafu(implicit)] location` on every variant. Inside `heurema`, errors are mostly built through context selectors (`SnapshotFormatSnafu { .. }.build()`, `ensure!`). The selectors are unreachable outside the crate (`mod error` is private), so `persistence.rs`'s decode helper and the adapter crates construct variants directly with `location: std::panic::Location::caller()`. `PersistenceSource` type-erases only at the backend boundary. `HeuremaError::category()` maps every variant to one `ErrorCategory` through an exhaustive match with no `_` arm.
- **Lifecycle vocabulary:** `lifecycle` holds the Phase 02 types (index and operation identity, transitions, the index record) and the object-safe `LifecycleBackend` storage contract (`lifecycle/backend.rs`), whose adapters store opaque heurēma-encoded bytes under `storage_key` and enforce only structural rules: head compare-and-set, never overwriting, refusing while staged state exists. atmis applies each write under one mutex, thesauros as one `SyncAll` fjall batch; `tests/lifecycle_backend.rs` runs every case on both. Identifier newtypes construct only through `TryFrom` (never `From<&str>`) and deserialize through it. Member identity, provenance, and retention are consumer-owned newtypes implementing heurēma's marker traits, which are never blanket-implemented; heurēma defines no provenance or retention shape.
- **Lifecycle validation:** `CheckedOperation::check` (stateless checks plus the `OperationDigest`) and then `permit` (the permission table and checks against the current record) are pure functions; only a `ValidatedOperation` may be staged. Validation calls the engines' own checks (`check_vector`, `require_simple_pipeline`), so an operation it accepts is not refused on apply. The digest is SHA-256 over a private canonical serde encoder (`src/lifecycle/digest.rs`, grammar documented on `OperationDigest`), never over `serde_json` output: it refuses floats in consumer types and sorts map entries itself.
- **Traits:** `VectorIndex`, `FtsIndex`, `PersistenceBackend` carry the cross-engine contracts; `LifecycleBackend` carries the lifecycle's storage contract and is object safe. Default methods exist only where the override would be uniform across implementors (e.g., `is_empty`).
- **Unsupported configurations:** `HnswIndex` (`src/hnsw/engine.rs`) and `Bm25Index` (`src/fts/bm25.rs`) are real engines. `Bm25Index` implements only the argument-less `Simple` tokenizer with no filters; any other analyzer pipeline returns `HeuremaError::NotYetImplemented` before it mutates state.
- **Persistence adapters:** `AtmisBackend` and `ThesaurosBackend` both wrap each index in a versioned `SnapshotEnvelope` and encode through `serde_json`, never by cloning the live `I` — that byte-level round trip is what proves the encode/decode path a durable backend depends on. Loads go through `decode_snapshot_payload`, which refuses an unsupported format version or the wrong `SnapshotFamily` before the index decoder runs. A snapshot is one index's whole-state save, never a transaction. `PersistenceBackend`'s save methods bound `I: Serialize`, its load methods bound `I: DeserializeOwned` (`persistence.rs`); a caller-chosen index type needs both derives to satisfy an adapter.
- **No suppressions without `reason`:** `#[expect(lint, reason = "…")]` not `#[allow]`. The `reason` documents the invariant, not the lint name.
- **No `unsafe`:** workspace `unsafe_code = "forbid"`. HNSW follows the published algorithm; the implementation is written here, and this crate stays safe Rust end-to-end.

## Roadmap

Phases are numbered 00–04. Earlier docs numbered them 1–3: old Phase 1 (API, RRF, typed placeholder engines) is the pre-00 scaffold, old Phase 2 (fresh engines) is Phase 01, and old Phase 3 (the `atmis` / `thesauros` adapters) landed before Phase 01 and counts as part of that pre-00 scaffold. Those labels are retired.

- **Phase 00 — producer contract.** Ownership plus ranking, error, removal, durable-state, and `akolouthia` boundary semantics. The retrieval half is fixed by the trait docs and the oracle suite; the durable-state half lands with Phase 02.
- **Phase 01 — fresh retrieval engines. Complete.** BM25 for the `Simple` pipeline (PR #48), the navigable HNSW graph (PR #50), versioned individual snapshots (PR #51), and an independently written BM25 formula reference that `Bm25Index` must match (`tests/oracle/bm25_formula.rs`). HNSW and BM25 are written fresh here, permanently. `krites` is vendored CozoDB under MPL-2.0 (see `aletheia/crates/krites/NOTICE.md`), so lifting its code would relocate a provenance question rather than resolve one — but that licensing fact is the occasion for write-fresh, not the whole of the ruling. Aletheia's clean-room rewrite of `krites` (phase 05b, gated by `aletheia#5954` / `aletheia#6060` / ADR-007 on phase 05g) shipped in v0.35.0, closing the sequencing gate this entry used to track; the settled answer on the far side of that gate is that `krites` is **replaced, not relocated** — its engines retire by krites' own callers repointing at heurēma once heurēma reaches parity, never by heurēma inheriting krites' code. `krites` serves only as behavioural reference and its tests as a conformance oracle. This entry is the single statement of that; other repo docs point here rather than restate it.
- **Phase 02 — durable retrieval lifecycle. In progress.** A named index gains creation, replace/rebuild, atomic publish, reopen/recovery, deletion, and corruption rules. Writes stage under one operation identity and become visible at one publish point; no index version is visible without its canonical member identities and provenance mapping.
- **Phase 03 — `akolouthia`.** The Datalog producer beside the indexes: provenance-bearing fact and rule admission, stratified semi-naive derivation checked against an independent naive evaluator, and facts, derivations, and index membership published together through the Phase 02 lifecycle.
- **Phase 04 — consumer adapters and retirement.** pinax, mneme, and aletheia bind the producer contracts for their own semantics; each completed migration deletes the superseded `krites` path. No compatibility runtime survives.

The long-run goal is retirement, not coexistence: heurēma exists so the derived engines in `krites` can be deleted. A phase that ships a second implementation beside the vendored one, without retiring it, has doubled the maintenance surface and resolved nothing.

Planning (roadmap, state, and phase plans) lives in the fleet's private planning repository. Update it there, not here; public docs cite a decision's content, never a private path.

## Conventional commits

`<type>(<scope>): <description>` — `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`, `perf`. Scope is the crate name (`heurema`) or the repo file (`workspace`, `ci`).

A `Gate-Passed` trailer is advisory fleet-wide: `kanon gate --stamp` writes it as provenance, and never by hand. Merges are gated by this repository's required GitHub checks; a PR without the trailer gets the full `gate-attestation` build instead of a trailer check.

## Boundaries

- Always: keep the index API engine-agnostic — heurēma must not learn about SQL or consumer-specific tuple shapes. Datalog belongs here only as `akolouthia` (planned), heurēma's own engine, never as a consumer's query language.
- Ask first: changes to the public trait surface (`VectorIndex`, `FtsIndex`, `PersistenceBackend`) — those are the contracts pinax and mneme bind.
- Never: pull in C dependencies; ship `unsafe` blocks; depend on aletheia, kanon, or any consumer-side crate. The dependency direction is one-way out.
