<!--
scope: heurema repo conventions (single-crate fleet substrate, Phase 1)
defers_to: ~/.claude/CLAUDE.md for operator principles; ~/dev/kanon/crates/basanos/standards/STANDARDS.md for fleet-wide standards
tightens: per-crate CLAUDE.md under crates/heurema/ may narrow conventions
-->

# heurema

Shared search-index primitives for the forkwright fleet. One implementation, one proof, orthogonal to query engines.

## Standards

Universal fleet standards live in `~/dev/kanon/crates/basanos/standards/`. This repo inherits — it does not restate.

Particularly relevant:

- `PHILOSOPHY.md` — fleet philosophical SSOT (including §"Presence: attention as a moral act")
- `GNOMON.md` — naming layer test (L1–L4)
- `COHERENCE.md` — architectural quality tests
- `RUST.md` — Rust-specific standards
- `TESTING.md` — testing principles
- `FLEET-REPO-SETUP.md` — fleet repo conventions this repo conforms to

## Layout

```
Cargo.toml             # workspace root
crates/heurema/        # single crate, workspace member
  src/                 # lib.rs, error.rs, fts.rs (+ fts/stub.rs),
                       # hnsw.rs (+ hnsw/stub.rs), persistence.rs, rrf.rs
  tests/               # api_smoke.rs, rrf_correctness.rs
_llm/                  # structured LLM corpus
.github/workflows/     # release-please.yml, gate-attestation.yml
```

The workspace shape is intentional even at one crate: future adapter sub-crates (`heurema-fjall`, `heurema-memory`) plug in without restructure.

## Commands

```bash
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Patterns

- **Errors:** `snafu` with `.context()` propagation, `Location` on every variant, `PersistenceSource` type-erases only at the backend boundary.
- **Traits:** `VectorIndex`, `FtsIndex`, `PersistenceBackend` carry the cross-engine contracts. Default methods exist only where the override would be uniform across implementors (e.g., `is_empty`).
- **Stubs:** Phase 1 ships typed stubs for HNSW and BM25 that return `HeuremaError::NotYetImplemented { feature: "Phase 2: …" }`. Stubs preserve trait bounds so consumers compile and exercise the API shape before the krites extraction lands.
- **No suppressions without `reason`:** `#[expect(lint, reason = "…")]` not `#[allow]`. The `reason` documents the invariant, not the lint name.
- **No `unsafe`:** workspace `unsafe_code = "forbid"`. HNSW algorithms borrow from peer-reviewed implementations; this crate stays safe Rust end-to-end.

## Roadmap

- **Phase 1 (current):** API + RRF + stubs. Locked.
- **Phase 2:** extract HNSW and BM25 from aletheia's `krites` crate. Driven by aletheia Phase 05g.
- **Phase 3:** persistence adapters (`heurema-fjall`, `heurema-memory`) ship as sibling crates under this workspace.

Planning lives in kanon at `projects/heurema/` — STATE.md, ROADMAP.md, vision.md, design.md. Update there, not here.

## Conventional commits + Gate-Passed trailer

`<type>(<scope>): <description>` — `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`, `perf`. Scope is the crate name (`heurema`) or the repo file (`workspace`, `ci`).

Every PR commit carries the `Gate-Passed: <sha>+<iso-8601-timestamp>` trailer; the `gate-attestation` GitHub Action verifies at least one PR commit has it.

## Boundaries

- Always: keep the API engine-agnostic — heurēma must not learn about SQL, Datalog, or consumer-specific tuple shapes.
- Ask first: changes to the public trait surface (`VectorIndex`, `FtsIndex`, `PersistenceBackend`) — those are consumed by pinax and mneme.
- Never: pull in C dependencies; ship `unsafe` blocks; depend on aletheia, kanon, or any consumer-side crate. The dependency direction is one-way out.
