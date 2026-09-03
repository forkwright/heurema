# Parity harness

How Phase 2's HNSW and BM25 implementations are measured against this oracle
(heurema#29): what "reaches parity" means operationally, and how the suite
moves from its intended red state to green.

## Red state (now)

Every engine-dependent test in this directory carries
`#[ignore = "phase 2: ..."]` because the Phase 1 stubs answer
`HeuremaError::NotYetImplemented`. Three tests run live:

- `hnsw::dimension_mismatch_is_rejected_before_state_change` — the stub
  already honours the dimension-check contract, so the harness demonstrably
  executes against the real trait surface rather than only compiling.
- `bm25::stub_reports_not_yet_implemented_until_phase_2_lands` and
  `hnsw::stub_reports_not_yet_implemented_until_phase_2_lands` — tripwires
  that assert the stub's `NotYetImplemented` and go red the moment a real
  engine answers.

`cargo test -p heurema --test oracle` is green today with 20 ignored.
`cargo test -p heurema --test oracle -- --include-ignored` fails all 20 with
`NotYetImplemented` — that failure is the intended red state, not a defect.

## Green definition (Phase 2 exit)

An implementation reaches parity when all of the following hold in one
commit:

1. `cargo test -p heurema --test oracle` passes with zero ignored tests in
   this directory. Landing the engine and stripping every
   `#[ignore = "phase 2: ..."]` marker are the same commit; the two tripwires
   are deleted in it, which their red state forces.
2. The property assertions pass unmodified — no weakening an assertion to
   make an implementation pass. An implementation that cannot meet one (for
   example `length_normalization_prefers_the_shorter_document` under a
   `b = 0` choice) reopens the parity definition in its own PR rather than
   editing the oracle.
3. `hnsw::recall_against_brute_force_meets_floor` holds at the 0.90 floor on
   the pinned fixture.
4. Graph-internal invariants the trait cannot observe (entry-point
   reachability, level-distribution shape, greedy-descent step bounds) land
   as unit tests beside the engine in `src/hnsw/` / `src/fts/`. The oracle
   deliberately asserts only trait-observable behaviour; this clause is what
   keeps "parity" from meaning "the trait surface alone was checked".

During Phase 2 development the work-in-progress check is
`cargo test -p heurema --test oracle -- --include-ignored`.

## What parity does not mean

- **Not bit-exact score equality with krites.** The idf variant for terms in
  more than half the corpus and the k1/b constants are implementation
  choices; `OBSERVATIONS.md` names them unpinned.
- **Not fixture parity with krites.** No krites test content exists in this
  repo (heurema#29: a curated corpus is a compilation; extracted test
  material is derived and banned). Every fixture here is generated fresh from
  the pinned seeds in `support.rs`.

## Fixture governance

The seeds in `support.rs` are part of the parity contract: changing one
re-baselines every fixture that draws from it, so a seed change is called out
in its commit message and re-validated against the recall floor. Fixture
sizes stay small enough for the suite to run in milliseconds — the recall
fixture is the ceiling, not the template.

## Middle path, if a property cannot reach a behaviour class

Where a class in `OBSERVATIONS.md` genuinely cannot be expressed as a
property, author a fresh fixture by *running* krites at the pinned SHA and
recording only the input/output fact — never by reading its test files into
this repo. Any such fixture records the krites SHA it was observed against,
in the fixture itself.
