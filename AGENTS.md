<!--
scope: heurema repo agent onboarding and dispatch conventions
defers_to: CLAUDE.md for repo coding conventions
tightens: phase-specific dispatch prompts may override defaults with justification
-->

# AGENTS.md

## Purpose

heurēma is a fleet substrate providing HNSW vector, BM25 full-text, persistence, and reciprocal-rank-fusion primitives: the `heurema` crate (traits, the HNSW and BM25 engines, RRF, the snapshot envelope) plus its `thesauros` (durable) and `atmis` (in-memory) `PersistenceBackend` sibling crates. Its intended consumers are `pinax` for SQL and `mneme`, the memory-*policy* layer (admission, retention, lifecycle rules) over heurēma's own Datalog engine, `akolouthia`, which is planned for this workspace and not yet built — mneme is not a second engine (see README.md's Non-goals) — and aletheia, whose `krites` engines retire as their callers repoint at heurēma.

Agents working here:

- maintain the fresh HNSW (`src/hnsw/engine.rs`) and BM25 (`src/fts/bm25.rs`) engines and build the Phase 02 durable retrieval lifecycle, per `CLAUDE.md`'s Roadmap section — `krites` is behavioural reference and conformance oracle only, permanently, never a code source;
- maintain the `thesauros` / `atmis` `PersistenceBackend` and `LifecycleBackend` adapters;
- fix CI / lint / gate failures;
- maintain the trait surface against consumer drift.

## Build

```bash
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

No system dependencies. Pure-Rust workspace (`heurema`, `atmis`, `thesauros`). Headless CI compatible. `scripts/docs-drift.sh` runs the cited-path and stub-wording checks locally.

## Standards

Inherited from `crates/basanos/standards/` in `forkwright/kanon`. Key documents: `PHILOSOPHY.md`, `GNOMON.md`, `COHERENCE.md`, `RUST.md`, `TESTING.md`, `FLEET-REPO-SETUP.md`.

Local CLAUDE.md narrows the standards to repo-specific patterns; read it before editing.

## Gate trailer

`Gate-Passed` is advisory fleet-wide. `kanon gate --stamp` writes it as provenance; never hand-write it. Merges are gated by this repository's required GitHub checks, and a PR without the trailer gets the full `gate-attestation` build.

## Cross-repo touchpoints

| Repo | Surface | Direction |
|------|---------|-----------|
| fleet private planning repository | planning artifacts (state, roadmap, vision, phase plans) | heurēma reads; updates happen there |
| `kanon/crates/basanos/standards/` | universal standards | heurēma reads; never edits from here |
| `aletheia/crates/krites/` | behavioural reference + conformance oracle, permanently — never a code source (see `CLAUDE.md`'s Roadmap section) | heurēma reads; aletheia owns the original |

Any change that touches the public API surface must also update the planning repository's heurēma state record and any consumer's pinned version.
