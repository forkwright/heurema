//! Proves `HnswIndex`/`Bm25Index` close their schema at the type level — the
//! same guarantee `HnswConfig`/`FtsConfig`/`TokenizerConfig` already carry.
//!
//! WHY the top-level type and not the nested config: `PersistenceBackend`'s
//! `save_vector_index`/`load_vector_index` (`persistence.rs`) encode and
//! decode `HnswIndex<Id>`/`Bm25Index<Id>` themselves, not their nested
//! `config` field alone. A snapshot decoding with an extra field silently
//! ignored is the same wrong-shape-under-a-shared-name hazard the config
//! types' `deny_unknown_fields` already guards against, one level up.

use heurema::{Bm25Index, HnswIndex};

#[test]
fn hnsw_index_rejects_unknown_top_level_fields() {
    // Every real field `HnswIndex<u64>` has, plus one it has never had.
    let json = r#"{
        "config": {
            "dimensions": 3,
            "distance": "L2",
            "ef_construction": 50,
            "m_neighbours": 16
        },
        "len": 0,
        "_id": null,
        "unexpected_extra_field": "wrong-index-family-marker"
    }"#;

    let decoded: Result<HnswIndex<u64>, _> = serde_json::from_str(json);
    assert!(
        decoded.is_err(),
        "an unknown top-level field on HnswIndex must be a decode error, not a \
         silent ignore — a snapshot from a different concrete VectorIndex type \
         that happens to be a field superset of HnswIndex would otherwise load \
         as a plausible-looking wrong value instead of failing loudly"
    );
}

#[test]
fn bm25_index_rejects_unknown_top_level_fields() {
    let json = r#"{
        "config": {
            "tokenizer": {"name": "Simple", "args": []},
            "filters": []
        },
        "len": 0,
        "_id": null,
        "unexpected_extra_field": "wrong-index-family-marker"
    }"#;

    let decoded: Result<Bm25Index<String>, _> = serde_json::from_str(json);
    assert!(
        decoded.is_err(),
        "an unknown top-level field on Bm25Index must be a decode error, not a \
         silent ignore, for the same reason as HnswIndex above"
    );
}
