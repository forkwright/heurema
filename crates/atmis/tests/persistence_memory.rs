//! Integration tests for [`atmis::AtmisBackend`] against the
//! `heurema::PersistenceBackend` contract.
//!
//! WHY these round trips prove the byte-level path, not a live clone: every
//! save here goes through `serde_json::to_vec` and every load through
//! `serde_json::from_slice` — `AtmisBackend` never keeps the caller's `&I`
//! or reaches back into it. A backend that instead stashed a `Clone` of the
//! live value would pass a save/load round trip without ever exercising the
//! encode/decode contract a durable backend depends on.

use serde::{Deserialize, Serialize};

use atmis::AtmisBackend;
use heurema::{
    Bm25Index, FtsConfig, FtsIndex, HeuremaError, HnswConfig, HnswIndex, PersistenceBackend,
    TokenizerConfig, VectorDistance, VectorIndex,
};

fn vector_config() -> HnswConfig {
    // WHY: `HnswConfig` is `#[non_exhaustive]`, so external crates cannot
    // build one with a struct expression (E0639); its public fields are
    // mutable post-construction, which is how a non-default fixture is
    // built from outside the defining crate.
    let mut config = HnswConfig::new(384);
    config.distance = VectorDistance::Cosine;
    config.ef_construction = 200;
    config.m_neighbours = 32;
    config
}

fn fts_config() -> FtsConfig {
    let mut config = FtsConfig::simple();
    config.tokenizer = TokenizerConfig::new("NGram", vec!["3".to_owned()]);
    config
        .filters
        .push(TokenizerConfig::new("Lowercase", Vec::new()));
    config
}

#[test]
fn save_and_load_vector_index_round_trips_through_json_bytes() -> Result<(), HeuremaError> {
    let backend = AtmisBackend::new();
    let index = HnswIndex::<u64>::new(vector_config());

    backend.save_vector_index("embeddings", &index)?;
    let loaded: HnswIndex<u64> = backend.load_vector_index("embeddings")?;

    assert_eq!(
        loaded.config(),
        &vector_config(),
        "the decoded config must match every field of the saved config, not just dimensions"
    );
    assert_eq!(loaded.len(), index.len(), "cardinality must round-trip");
    Ok(())
}

#[test]
fn save_and_load_fts_index_round_trips_through_json_bytes() -> Result<(), HeuremaError> {
    let backend = AtmisBackend::new();
    let index = Bm25Index::<String>::new(fts_config());

    backend.save_fts_index("documents", &index)?;
    let loaded: Bm25Index<String> = backend.load_fts_index("documents")?;

    assert_eq!(
        loaded.config(),
        &fts_config(),
        "the decoded config must carry the non-default tokenizer and filter list"
    );
    assert_eq!(loaded.len(), index.len(), "cardinality must round-trip");
    Ok(())
}

#[test]
fn load_vector_index_without_a_prior_save_is_index_not_found() {
    let backend = AtmisBackend::new();

    match backend.load_vector_index::<HnswIndex<u64>>("never-saved") {
        Err(HeuremaError::IndexNotFound { name, .. }) => {
            assert_eq!(
                name, "never-saved",
                "the error must name the missing snapshot"
            );
        }
        Ok(_) => panic!("a name with no prior save must not load"),
        Err(other) => panic!("expected IndexNotFound, got {other:?}"),
    }
}

#[test]
fn load_fts_index_without_a_prior_save_is_index_not_found() {
    let backend = AtmisBackend::new();

    match backend.load_fts_index::<Bm25Index<String>>("never-saved") {
        Err(HeuremaError::IndexNotFound { name, .. }) => {
            assert_eq!(
                name, "never-saved",
                "the error must name the missing snapshot"
            );
        }
        Ok(_) => panic!("a name with no prior save must not load"),
        Err(other) => panic!("expected IndexNotFound, got {other:?}"),
    }
}

#[test]
fn save_vector_index_is_idempotent() -> Result<(), HeuremaError> {
    let backend = AtmisBackend::new();
    let index = HnswIndex::<u64>::new(vector_config());

    backend.save_vector_index("embeddings", &index)?;
    let first: HnswIndex<u64> = backend.load_vector_index("embeddings")?;

    // WHY: TESTING.md's idempotency pattern — call the operation, capture
    // state, call it again with identical input, assert state is
    // unchanged.
    backend.save_vector_index("embeddings", &index)?;
    let second: HnswIndex<u64> = backend.load_vector_index("embeddings")?;

    assert_eq!(
        first.config(),
        second.config(),
        "replaying an identical save must not change the loaded snapshot"
    );
    Ok(())
}

#[test]
fn save_fts_index_is_idempotent() -> Result<(), HeuremaError> {
    let backend = AtmisBackend::new();
    let index = Bm25Index::<String>::new(fts_config());

    backend.save_fts_index("documents", &index)?;
    let first: Bm25Index<String> = backend.load_fts_index("documents")?;

    backend.save_fts_index("documents", &index)?;
    let second: Bm25Index<String> = backend.load_fts_index("documents")?;

    assert_eq!(
        first.config(),
        second.config(),
        "replaying an identical save must not change the loaded snapshot"
    );
    Ok(())
}

#[test]
fn save_vector_index_replaces_rather_than_merges() -> Result<(), HeuremaError> {
    let backend = AtmisBackend::new();
    let first_config = HnswConfig::new(3);
    let mut second_config = HnswConfig::new(3);
    second_config.dimensions = 768;

    backend.save_vector_index("catalog-entry", &HnswIndex::<u64>::new(first_config))?;
    backend.save_vector_index(
        "catalog-entry",
        &HnswIndex::<u64>::new(second_config.clone()),
    )?;
    let loaded: HnswIndex<u64> = backend.load_vector_index("catalog-entry")?;

    assert_eq!(
        loaded.config(),
        &second_config,
        "per the trait's documented contract, the second save must win outright — \
         a merge or an error would both violate last-write-wins"
    );
    Ok(())
}

/// WHY: a probe [`VectorIndex`] with a JSON shape incompatible with
/// [`HnswIndex`] — a required field `HnswIndex` never emits — proves
/// [`HeuremaError::Persistence`] is reachable from a genuine decode failure,
/// not only from the not-found path. `PersistenceBackend` carries no
/// type-tag on a saved snapshot, so loading the wrong concrete type under a
/// shared name is exactly this failure mode, not a hypothetical one.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProbeVectorIndex {
    marker: String,
}

impl VectorIndex for ProbeVectorIndex {
    type Id = u64;

    fn insert(&mut self, _id: Self::Id, _vector: &[f32]) -> Result<(), HeuremaError> {
        Ok(())
    }

    fn query(&self, _vector: &[f32], _k: usize) -> Result<Vec<(Self::Id, f32)>, HeuremaError> {
        Ok(Vec::new())
    }

    fn remove(&mut self, _id: &Self::Id) -> Result<(), HeuremaError> {
        Ok(())
    }

    fn len(&self) -> usize {
        0
    }
}

#[test]
fn load_vector_index_reports_persistence_error_on_shape_mismatch() -> Result<(), HeuremaError> {
    let backend = AtmisBackend::new();
    backend.save_vector_index("catalog-entry", &HnswIndex::<u64>::new(HnswConfig::new(3)))?;

    match backend.load_vector_index::<ProbeVectorIndex>("catalog-entry") {
        Err(HeuremaError::Persistence { .. }) => {}
        Ok(_) => {
            panic!("HnswIndex bytes must not decode as the incompatible ProbeVectorIndex shape")
        }
        Err(other) => panic!("expected Persistence, got {other:?}"),
    }
    Ok(())
}
