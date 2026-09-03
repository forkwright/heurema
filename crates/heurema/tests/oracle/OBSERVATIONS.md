# Oracle observations

The observation record for the conformance oracle (heurema#29). The property
tests in this directory assert the published HNSW and BM25 algorithm contracts
plus the ranking contracts on `VectorIndex::query` and `FtsIndex::query`; this
file records what was observed of the krites oracle, at which pin, and which
observed behaviour classes the property tests do not cover.

Nothing was extracted: no krites test body, fixture, or expected value was
read into this repo. A curated corpus is a compilation, and extracted test
material is treated as derived (heurema#29). Observation below means the
category-level shape of the pinned tree's test surface — file names, test
names, and which subsystems were still derived — never copied content.

## Pin

- Observed tree: `aletheia` (`krites`) commit
  `fa63f73f1bc54419d51fff7fc78d6d1233695467` (`origin/main`, 2026-08-25).
- Sovereignty ledger at the pin (`crates/krites/PROVENANCE.toml`):
  142 derived / 68 sovereign of 210 rows.
- FTS subsystem (`src/fts/`): 24 derived / 10 sovereign. The scoring core
  (`ast.rs`, `config.rs`, `indexing.rs`, `mod.rs`) and most tokenizers are
  still derived; the sovereign rows are the ascii-folding fold tables, the
  stop-word filter, and `error.rs`.
- Vector-search subsystem: `runtime/hnsw/search.rs`,
  `data/program/search/*`, and `query/ra/search.rs` are all still derived.
  One sovereign sibling (`runtime/hnsw_sovereign/search.rs`) exists with its
  own persistence tests; the derived engine is the behavioural witness this
  oracle is pinned against.

## Behaviour classes observed at the pin

Gathered from test and file names on the pinned tree's public test surface.
"Coverage" below names where each class lands in this repo.

### BM25 / FTS (from `src/fts/indexing.rs`, `tests/public_api_fts_io_misc.rs`)

| Class | Coverage |
|-------|----------|
| Degenerate inputs score zero (zero tf, zero df, empty corpus) | `bm25::empty_index_query_returns_no_results`, `bm25::no_match_query_returns_no_results` at trait level; the per-component zero cases are formula internals Phase 2 unit-tests in `src/fts/` |
| Typical input scores nonzero | implicit in every ranking assertion |
| Length normalization direction: longer document scores lower at equal tf | `bm25::length_normalization_prefers_the_shorter_document` |
| BM25 differs from plain tf-idf (tf saturation) | `bm25::term_frequency_saturates` |
| idf decreases in document frequency | `bm25::rare_term_outranks_common_term_on_the_same_document` |
| Index lifecycle: put / search / delete | `bm25::len_tracks_inserts`, `bm25::remove_is_idempotent_and_excludes_the_document` |
| Score-kind variant plumbing | not covered — an engine-API detail with no trait surface |
| Proximity (`fts_near`) term chaining | not covered — `FtsIndex::query` is a term bag; no proximity surface exists to pin |
| Tokenizer folding tables and stop words | not covered — the Phase 2 scope is the `Simple` pipeline (`FtsConfig::simple`); the tables there are derived material and would have to be regenerated fresh regardless |

### HNSW (from `runtime/hnsw*/`, `data/program/search/`, `tests/public_api_queries_and_hnsw.rs`)

| Class | Coverage |
|-------|----------|
| Distance functions: L2 correctness, cosine on identical vectors | `hnsw::self_query_returns_the_inserted_vector_first_at_zero_distance` |
| Cosine of a zero vector must not produce NaN | `hnsw::cosine_distance_with_a_zero_vector_stays_finite` |
| Mismatched query dtype errors | not applicable — the trait is `&[f32]`-only; the dimension-mismatch analogue is `hnsw::dimension_mismatch_is_rejected_before_state_change` (live against the stub) |
| Lifecycle: empty search, insert-and-search, delete, deleted IDs excluded, exact match is top result | `hnsw::empty_index_query_returns_no_results`, `hnsw::len_tracks_inserts`, `hnsw::self_query_...`, `hnsw::remove_is_idempotent_and_excludes_the_id_from_results` |
| Results ordered by ascending distance, ties stable | `hnsw::query_results_satisfy_the_ranking_contract`, `hnsw::equal_distance_ties_order_by_ascending_id`, `hnsw::smaller_k_is_a_prefix_of_larger_k` |
| Recall measured against exact top-k | `hnsw::recall_against_brute_force_meets_floor` (0.90 floor on a pinned 256-vector fixture) |
| Query termination / bounded result sets on a dense fixture | `hnsw::queries_terminate_and_stay_bounded_on_a_dense_fixture` |
| Random level distribution is non-degenerate | not trait-observable — graph-internal; Phase 2 unit-test territory in `src/hnsw/` |
| HNSW result-cache eviction and retention | not covered — an engine-internal choice; the trait exposes no cache |
| Storage-backed close/reopen preserves recall | not oracle territory — owned by the `PersistenceBackend` contract tests (`persistence_contract.rs`, the adapter test suites) |

## Deliberately unpinned

Parity is property-level, not bit-exact score equality with krites
(`tests/oracle/PARITY.md`). Two published-formula choices are implementation
decisions and stay unpinned: the idf variant for terms appearing in more than
half the corpus, and the k1/b parameter values. The tokenizer argument value
model (`TokenizerConfig::args`) is likewise unset until Phase 2.
