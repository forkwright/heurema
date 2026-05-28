<!--
scope: heurema repo agent onboarding and dispatch conventions
defers_to: CLAUDE.md for repo coding conventions
tightens: phase-specific dispatch prompts may override defaults with justification
-->

# AGENTS.md

## Purpose

heurēma is a single-crate fleet substrate providing HNSW vector, BM25 full-text, persistence, and reciprocal-rank-fusion primitives. Consumed externally by fleet query engines (`pinax` for SQL, `mneme` for Datalog) and by aletheia's `krites` memory stack (per ADR-003: aletheia's memory stack stays internal but is free to adopt heurēma when it chooses).

Agents working here:

- land Phase 2 HNSW and BM25 extractions from aletheia/krites;
- add persistence adapter sub-crates (`heurema-fjall`, `heurema-memory`);
- fix CI / lint / gate failures;
- maintain the trait surface against consumer drift.

## Build

```bash
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

No system dependencies. Pure-Rust crate. Headless CI compatible.

## Standards

Inherited from `~/dev/kanon/crates/basanos/standards/`. Key documents: `PHILOSOPHY.md`, `GNOMON.md`, `COHERENCE.md`, `RUST.md`, `TESTING.md`, `FLEET-REPO-SETUP.md`.

Local CLAUDE.md narrows the standards to repo-specific patterns; read it before editing.

## Gate trailer

Every PR commit carries `Gate-Passed: <sha>+<iso-8601-timestamp>` (or the legacy `Gate-Passed: kanon X.Y.Z` form for version-form evidence). The `gate-attestation` workflow blocks merges without it.

## Cross-repo touchpoints

| Repo | Surface | Direction |
|------|---------|-----------|
| `kanon/projects/heurema/` | planning artifacts (STATE.md, ROADMAP.md, vision.md, design.md) | heurēma reads; updates happen there |
| `kanon/crates/basanos/standards/` | universal standards | heurēma reads; never edits from here |
| `aletheia/crates/krites/` | Phase 2 source of HNSW + BM25 code | heurēma adopts; aletheia owns the original |

Any change that touches the public API surface must also update kanon's `projects/heurema/STATE.md` and any consumer's pinned version.
