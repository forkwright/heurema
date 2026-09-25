# Parity harness

How heurēma's HNSW and BM25 implementations are measured against this oracle
(heurema#29): what "reaches parity" means operationally.

## Current state (oracle suite green since PR #50)

Every BM25 and HNSW oracle case runs live against the fresh engines: BM25 for
the `Simple` pipeline (PR #48) and the navigable HNSW graph (PR #50). No test in
this directory is ignored, and the two tripwires that asserted the pre-engine
placeholders' `NotYetImplemented` answer were deleted in the commits that landed
the engines, as the green definition below requires.

`cargo test -p heurema --test oracle` is the parity check.

HNSW graph-internal invariants are unit-tested beside the engine in
`src/hnsw/engine.rs` (clause 4 below). BM25's formula is pinned by
`bm25_formula`, which checks `Bm25Index` against an independently written `f64`
BM25 reference that recomputes every score from the raw live corpus over
seeded insert, replace, and remove workloads: IDs must match in order, scores
must agree within a derived `f32` bound, and every perturbed reference must
be caught. With it, Phase 01's exit is met.

## Green definition (Phase 01 exit)

An implementation reaches parity when all of the following hold in one
commit:

1. `cargo test -p heurema --test oracle` passes with zero ignored tests in
   this directory. Landing an engine and removing every ignore marker that
   waited on it were the same commit, and the tripwires were deleted in it.
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
   Query traversal has a documented internal expansion budget derived from
   `max(k, 8m)`, and construction uses `max(ef_construction, m)`. Those are
   explicit bounds on expanded candidates and upper-layer greedy steps, with
   separate adversarial-cycle tests and the unchanged recall floor as the
   acceptance pair. They are internal work budgets, not a claim that the
   standard HNSW `ef` retained-beam semantics are unchanged.

## What parity does not mean

- **Not bit-exact score equality with krites.** The idf variant for terms in
  more than half the corpus and the k1/b constants are implementation
  choices; `OBSERVATIONS.md` names them unpinned.
- **Not fixture parity with krites.** No krites test content exists in this
  repo (heurema#29: a curated corpus is a compilation; extracted test
  material is derived and banned). Every fixture here is generated fresh from
  the pinned seeds in `support.rs`.

## Fixture governance

The seeds in `support.rs`, and the inline workload seeds in `hnsw.rs` and
`bm25_formula.rs`, are part of the parity contract: changing one
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
