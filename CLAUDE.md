<!--
scope: heurema repo conventions (fleet substrate: the heurema trait crate plus its persistence-adapter sibling crates)
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
Cargo.toml           # workspace root
crates/heurema/      # trait + RRF crate, workspace member
  src/               # lib.rs, error.rs, fts.rs (+ fts/stub.rs),
                     # hnsw.rs (+ hnsw/stub.rs), persistence.rs, rrf.rs
  tests/             # api_smoke.rs, index_rrf_composition.rs,
                     # persistence_contract.rs, rrf_correctness.rs
crates/atmis/        # in-memory PersistenceBackend adapter (ἀτμίς, vapor)
  src/lib.rs         # AtmisBackend
  tests/             # persistence_memory.rs
crates/thesauros/    # fjall-backed PersistenceBackend adapter (θησαυρός, storehouse)
  src/lib.rs         # ThesaurosBackend
  tests/             # persistence_fjall.rs
_llm/                # structured LLM corpus
.github/workflows/   # release-please.yml, gate-attestation.yml, ci.yml, security.yml
```

The workspace shape was chosen for exactly this: the two adapter sub-crates plug in as siblings with no restructure of `crates/heurema/`.

## Commands

```bash
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Patterns

- **Errors:** `snafu` with `.context()` propagation, `Location` on every variant, `PersistenceSource` type-erases only at the backend boundary.
- **Traits:** `VectorIndex`, `FtsIndex`, `PersistenceBackend` carry the cross-engine contracts. Default methods exist only where the override would be uniform across implementors (e.g., `is_empty`).
- **Stubs:** Phase 1 ships typed stubs for HNSW and BM25 that return `HeuremaError::NotYetImplemented { feature: "Phase 2: …" }`. Stubs preserve trait bounds so consumers compile and exercise the API shape before the real engines land.
- **Persistence adapters:** `AtmisBackend` and `ThesaurosBackend` both encode through `serde_json`, never by cloning the live `I` — that byte-level round trip is what proves the encode/decode path a durable backend depends on. `PersistenceBackend`'s save methods bound `I: Serialize`, its load methods bound `I: DeserializeOwned` (`persistence.rs`); a caller-chosen index type needs both derives to satisfy an adapter.
- **No suppressions without `reason`:** `#[expect(lint, reason = "…")]` not `#[allow]`. The `reason` documents the invariant, not the lint name.
- **No `unsafe`:** workspace `unsafe_code = "forbid"`. HNSW follows the published algorithm; the implementation is written here, and this crate stays safe Rust end-to-end.

## Roadmap

- **Phase 1:** API + RRF + stubs. Locked.
- **Phase 2:** the source is gated on aletheia's krites clean-room rewrite (`aletheia#5954`, phase 05b) landing and then clearing its standalone-extraction gate (`aletheia#6060` / ADR-007, itself gated on phase 05g). Until that gate clears, `krites` is vendored CozoDB under MPL-2.0 (see `aletheia/crates/krites/NOTICE.md`) — moving it here would relocate a provenance question rather than resolve one, so HNSW and BM25 are written fresh here, with `krites` serving only as behavioural reference and its tests as a conformance oracle. Once the gate clears, the rewritten `krites` is standards-compliant and MPL-free and extraction becomes the live option. This entry tracks which side of the gate Phase 2 is on; other repo docs point here rather than restate it. Phase 3 was not gated on this phase and landed independently.
- **Phase 3:** `thesauros` and `atmis` implement `PersistenceBackend` as sibling crates under this workspace.

The long-run goal is retirement, not coexistence: heurēma exists so the derived engines in `krites` can be deleted. A phase that ships a second implementation beside the vendored one, without retiring it, has doubled the maintenance surface and resolved nothing.

Planning lives in kanon at `projects/heurema/` — STATE.md, ROADMAP.md, vision.md, design.md. Update there, not here.

## Conventional commits + Gate-Passed trailer

`<type>(<scope>): <description>` — `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`, `perf`. Scope is the crate name (`heurema`) or the repo file (`workspace`, `ci`).

Every PR commit carries the `Gate-Passed: <sha>+<iso-8601-timestamp>` trailer; the `gate-attestation` GitHub Action verifies at least one PR commit has it.

## Boundaries

- Always: keep the API engine-agnostic — heurēma must not learn about SQL, Datalog, or consumer-specific tuple shapes.
- Ask first: changes to the public trait surface (`VectorIndex`, `FtsIndex`, `PersistenceBackend`) — those are consumed by pinax and mneme.
- Never: pull in C dependencies; ship `unsafe` blocks; depend on aletheia, kanon, or any consumer-side crate. The dependency direction is one-way out.
