//! Integration tests for [`thesauros::ThesaurosBackend`] against the
//! `heurema::PersistenceBackend` contract.
//!
//! WHY every round trip here drops the backend and reopens the keyspace at
//! the same path before loading: `fjall::Keyspace::open` auto-recovery
//! deletes segments absent from the levels manifest, so the only way to
//! prove data survived is to actually close the process's handle and come
//! back through the disk, exactly as a crash-restart would. An in-process
//! load against the still-open backend would pass even if nothing had been
//! written to the journal at all.

use serde::{Deserialize, Serialize};

use heurema::{
    Bm25Index, FtsConfig, FtsIndex, HeuremaError, HnswConfig, HnswIndex, PersistenceBackend,
    TokenizerConfig, VectorDistance, VectorIndex,
};
use thesauros::ThesaurosBackend;

fn vector_config() -> HnswConfig {
    // WHY: `HnswConfig` is `#[non_exhaustive]` — struct-literal construction
    // from outside the defining crate is E0639. Its fields are public and
    // mutable post-construction, which is how a non-default fixture is
    // built here.
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
fn vector_index_survives_close_and_reopen() {
    let dir = tempfile::tempdir().expect("temp dir");

    {
        let backend = ThesaurosBackend::open(dir.path()).expect("open keyspace");
        backend
            .save_vector_index("embeddings", &HnswIndex::<u64>::new(vector_config()))
            .expect("save vector index");
    } // WHY: the backend, its Keyspace, and both PartitionHandles drop here —
    // the only handle onto this fjall database goes away before reopening.

    let reopened = ThesaurosBackend::open(dir.path()).expect("reopen keyspace");
    let loaded: HnswIndex<u64> = reopened
        .load_vector_index("embeddings")
        .expect("load vector index");

    assert_eq!(
        loaded.config(),
        &vector_config(),
        "every field of the saved config must survive a real close-and-reopen, \
         not just dimensions"
    );
}

#[test]
fn fts_index_survives_close_and_reopen() {
    let dir = tempfile::tempdir().expect("temp dir");

    {
        let backend = ThesaurosBackend::open(dir.path()).expect("open keyspace");
        backend
            .save_fts_index("documents", &Bm25Index::<String>::new(fts_config()))
            .expect("save fts index");
    }

    let reopened = ThesaurosBackend::open(dir.path()).expect("reopen keyspace");
    let loaded: Bm25Index<String> = reopened
        .load_fts_index("documents")
        .expect("load fts index");

    assert_eq!(
        loaded.config(),
        &fts_config(),
        "the non-default tokenizer and filter list must survive a real close-and-reopen"
    );
}

#[test]
fn load_vector_index_without_a_prior_save_is_index_not_found() {
    let dir = tempfile::tempdir().expect("temp dir");
    let backend = ThesaurosBackend::open(dir.path()).expect("open keyspace");

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
    let dir = tempfile::tempdir().expect("temp dir");
    let backend = ThesaurosBackend::open(dir.path()).expect("open keyspace");

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
fn save_vector_index_is_idempotent() {
    let dir = tempfile::tempdir().expect("temp dir");
    let backend = ThesaurosBackend::open(dir.path()).expect("open keyspace");
    let index = HnswIndex::<u64>::new(vector_config());

    backend
        .save_vector_index("embeddings", &index)
        .expect("first save");
    let first: HnswIndex<u64> = backend
        .load_vector_index("embeddings")
        .expect("load after first save");

    // WHY: TESTING.md's idempotency pattern — call the operation, capture
    // state, call it again with identical input, assert state is
    // unchanged.
    backend
        .save_vector_index("embeddings", &index)
        .expect("second save");
    let second: HnswIndex<u64> = backend
        .load_vector_index("embeddings")
        .expect("load after second save");

    assert_eq!(
        first.config(),
        second.config(),
        "replaying an identical save must not change the loaded snapshot"
    );
}

#[test]
fn save_fts_index_is_idempotent() {
    let dir = tempfile::tempdir().expect("temp dir");
    let backend = ThesaurosBackend::open(dir.path()).expect("open keyspace");
    let index = Bm25Index::<String>::new(fts_config());

    backend
        .save_fts_index("documents", &index)
        .expect("first save");
    let first: Bm25Index<String> = backend
        .load_fts_index("documents")
        .expect("load after first save");

    backend
        .save_fts_index("documents", &index)
        .expect("second save");
    let second: Bm25Index<String> = backend
        .load_fts_index("documents")
        .expect("load after second save");

    assert_eq!(
        first.config(),
        second.config(),
        "replaying an identical save must not change the loaded snapshot"
    );
}

#[test]
fn save_vector_index_replaces_rather_than_merges() {
    let dir = tempfile::tempdir().expect("temp dir");
    let backend = ThesaurosBackend::open(dir.path()).expect("open keyspace");
    let first_config = HnswConfig::new(3);
    let mut second_config = HnswConfig::new(3);
    second_config.dimensions = 768;

    backend
        .save_vector_index("catalog-entry", &HnswIndex::<u64>::new(first_config))
        .expect("first save");
    backend
        .save_vector_index(
            "catalog-entry",
            &HnswIndex::<u64>::new(second_config.clone()),
        )
        .expect("second save");
    let loaded: HnswIndex<u64> = backend
        .load_vector_index("catalog-entry")
        .expect("load after both saves");

    assert_eq!(
        loaded.config(),
        &second_config,
        "per the trait's documented contract, the second save must win outright — \
         a merge or an error would both violate last-write-wins"
    );
}

#[test]
fn open_reports_persistence_error_when_path_is_not_a_directory() {
    let dir = tempfile::tempdir().expect("temp dir");
    let blocked_path = dir.path().join("not-a-directory");
    std::fs::write(
        &blocked_path,
        b"occupies the path fjall would open as a keyspace",
    )
    .expect("write blocking file");

    match ThesaurosBackend::open(&blocked_path) {
        Err(HeuremaError::Persistence { .. }) => {}
        Ok(_) => panic!("fjall cannot open a keyspace at a path that is already a regular file"),
        Err(other) => panic!("expected Persistence, got {other:?}"),
    }
}

/// WHY: a probe [`VectorIndex`] with a JSON shape incompatible with
/// [`HnswIndex`] proves [`HeuremaError::Persistence`] is reachable from a
/// genuine decode failure on the fjall-backed path too, not only from the
/// not-found path or a directory-open failure.
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
fn load_vector_index_reports_persistence_error_on_shape_mismatch() {
    let dir = tempfile::tempdir().expect("temp dir");
    let backend = ThesaurosBackend::open(dir.path()).expect("open keyspace");
    backend
        .save_vector_index("catalog-entry", &HnswIndex::<u64>::new(HnswConfig::new(3)))
        .expect("save vector index");

    match backend.load_vector_index::<ProbeVectorIndex>("catalog-entry") {
        Err(HeuremaError::Persistence { .. }) => {}
        Ok(_) => {
            panic!("HnswIndex bytes must not decode as the incompatible ProbeVectorIndex shape")
        }
        Err(other) => panic!("expected Persistence, got {other:?}"),
    }
}
