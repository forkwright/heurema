//! Integration tests for [`thesauros::ThesaurosBackend`] against the
//! `heurema::PersistenceBackend` contract.
//!
//! WHY every round trip here drops the backend and reopens the database at
//! the same path before loading: `fjall::Database::open` auto-recovery
//! deletes segments absent from the levels manifest, so the only way to
//! prove data survived is to actually close the process's handle and come
//! back through the disk, exactly as a crash-restart would. An in-process
//! load against the still-open backend would pass even if nothing had been
//! written to the journal at all.

use serde::{Deserialize, Serialize};

use heurema::{
    Bm25Index, FtsConfig, HeuremaError, HnswConfig, HnswIndex, PersistenceBackend,
    PersistenceSource, TokenizerConfig, VectorDistance, VectorIndex,
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

// WHY: test setup (tempdir, blocking-file write) fails with `io::Error`,
// distinct from every backend call's `HeuremaError`. Routing both through
// `PersistenceSource` lets every test return `Result<(), HeuremaError>` and
// propagate with `?` instead of `.expect()` — `unwrap_used`/`expect_used`
// are workspace-lint warnings that `-D warnings` promotes to hard errors,
// and clippy does not exempt `#[test]` functions from them here.
fn io_error(source: std::io::Error) -> HeuremaError {
    HeuremaError::Persistence {
        source: PersistenceSource::new(source),
        location: std::panic::Location::caller(),
    }
}

#[test]
fn vector_index_survives_close_and_reopen() -> Result<(), HeuremaError> {
    let dir = tempfile::tempdir().map_err(io_error)?;

    {
        let backend = ThesaurosBackend::open(dir.path())?;
        backend.save_vector_index("embeddings", &HnswIndex::<u64>::new(vector_config()))?;
    } // WHY: the backend, its Database, and both Keyspaces drop here —
    // the only handle onto this fjall database goes away before reopening.

    let reopened = ThesaurosBackend::open(dir.path())?;
    let loaded: HnswIndex<u64> = reopened.load_vector_index("embeddings")?;

    assert_eq!(
        loaded.config(),
        &vector_config(),
        "every field of the saved config must survive a real close-and-reopen, \
         not just dimensions"
    );
    Ok(())
}

#[test]
fn fts_index_survives_close_and_reopen() -> Result<(), HeuremaError> {
    let dir = tempfile::tempdir().map_err(io_error)?;

    {
        let backend = ThesaurosBackend::open(dir.path())?;
        backend.save_fts_index("documents", &Bm25Index::<String>::new(fts_config()))?;
    }

    let reopened = ThesaurosBackend::open(dir.path())?;
    let loaded: Bm25Index<String> = reopened.load_fts_index("documents")?;

    assert_eq!(
        loaded.config(),
        &fts_config(),
        "the non-default tokenizer and filter list must survive a real close-and-reopen"
    );
    Ok(())
}

#[test]
fn load_vector_index_without_a_prior_save_is_index_not_found() -> Result<(), HeuremaError> {
    let dir = tempfile::tempdir().map_err(io_error)?;
    let backend = ThesaurosBackend::open(dir.path())?;

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
    Ok(())
}

#[test]
fn load_fts_index_without_a_prior_save_is_index_not_found() -> Result<(), HeuremaError> {
    let dir = tempfile::tempdir().map_err(io_error)?;
    let backend = ThesaurosBackend::open(dir.path())?;

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
    Ok(())
}

#[test]
fn save_vector_index_is_idempotent() -> Result<(), HeuremaError> {
    let dir = tempfile::tempdir().map_err(io_error)?;
    let backend = ThesaurosBackend::open(dir.path())?;
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
    let dir = tempfile::tempdir().map_err(io_error)?;
    let backend = ThesaurosBackend::open(dir.path())?;
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
    let dir = tempfile::tempdir().map_err(io_error)?;
    let backend = ThesaurosBackend::open(dir.path())?;
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

#[test]
fn open_reports_persistence_error_when_path_is_not_a_directory() -> Result<(), HeuremaError> {
    let dir = tempfile::tempdir().map_err(io_error)?;
    let blocked_path = dir.path().join("not-a-directory");
    std::fs::write(
        &blocked_path,
        b"occupies the path fjall would open as a keyspace",
    )
    .map_err(io_error)?;

    match ThesaurosBackend::open(&blocked_path) {
        Err(HeuremaError::Persistence { .. }) => {}
        Ok(_) => panic!("fjall cannot open a keyspace at a path that is already a regular file"),
        Err(other) => panic!("expected Persistence, got {other:?}"),
    }
    Ok(())
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
fn load_vector_index_reports_persistence_error_on_shape_mismatch() -> Result<(), HeuremaError> {
    let dir = tempfile::tempdir().map_err(io_error)?;
    let backend = ThesaurosBackend::open(dir.path())?;
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
