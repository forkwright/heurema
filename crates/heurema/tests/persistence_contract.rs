use std::cell::RefCell;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use heurema::{
    Bm25Index, FtsConfig, FtsIndex, HeuremaError, HnswConfig, HnswIndex, PersistenceBackend,
    PersistenceSource, VectorIndex,
};

#[derive(Debug, Default)]
struct LeafError;

impl fmt::Display for LeafError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("leaf failure")
    }
}

impl Error for LeafError {}

#[derive(Debug, Default)]
struct MidError {
    source: LeafError,
}

impl fmt::Display for MidError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("mid failure")
    }
}

impl Error for MidError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

// WHY: proves the trait is implementable outside the crate and that an
// adapter can construct the public error variants it is contracted to return.
#[derive(Debug, Default)]
struct RecordingBackend {
    saved: RefCell<BTreeSet<String>>,
}

impl RecordingBackend {
    fn not_found(name: &str) -> HeuremaError {
        HeuremaError::IndexNotFound {
            name: name.to_owned(),
            location: snafu::Location::default(),
        }
    }
}

impl PersistenceBackend for RecordingBackend {
    fn save_vector_index<I>(&self, name: &str, _idx: &I) -> Result<(), HeuremaError>
    where
        I: VectorIndex,
    {
        self.saved.borrow_mut().insert(name.to_owned());
        Ok(())
    }

    fn load_vector_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: VectorIndex,
    {
        Err(Self::not_found(name))
    }

    fn save_fts_index<I>(&self, name: &str, _idx: &I) -> Result<(), HeuremaError>
    where
        I: FtsIndex,
    {
        self.saved.borrow_mut().insert(name.to_owned());
        Ok(())
    }

    fn load_fts_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: FtsIndex,
    {
        Err(Self::not_found(name))
    }
}

#[derive(Debug, Default)]
struct FailingBackend;

impl FailingBackend {
    fn storage_failure() -> HeuremaError {
        HeuremaError::Persistence {
            source: PersistenceSource::new(MidError::default()),
            location: snafu::Location::default(),
        }
    }
}

impl PersistenceBackend for FailingBackend {
    fn save_vector_index<I>(&self, _name: &str, _idx: &I) -> Result<(), HeuremaError>
    where
        I: VectorIndex,
    {
        Err(Self::storage_failure())
    }

    fn load_vector_index<I>(&self, _name: &str) -> Result<I, HeuremaError>
    where
        I: VectorIndex,
    {
        Err(Self::storage_failure())
    }

    fn save_fts_index<I>(&self, _name: &str, _idx: &I) -> Result<(), HeuremaError>
    where
        I: FtsIndex,
    {
        Err(Self::storage_failure())
    }

    fn load_fts_index<I>(&self, _name: &str) -> Result<I, HeuremaError>
    where
        I: FtsIndex,
    {
        Err(Self::storage_failure())
    }
}

#[test]
fn backend_saves_are_name_keyed_for_both_index_families() -> Result<(), HeuremaError> {
    let backend = RecordingBackend::default();
    let vectors = HnswIndex::<u64>::new(HnswConfig::new(3));
    let documents = Bm25Index::<String>::new(FtsConfig::simple());

    backend.save_vector_index("embeddings", &vectors)?;
    backend.save_fts_index("documents", &documents)?;

    let saved = backend.saved.borrow();
    assert!(
        saved.contains("embeddings") && saved.contains("documents"),
        "backends receive the engine-owned snapshot names"
    );
    Ok(())
}

#[test]
fn missing_vector_index_loads_as_index_not_found() {
    let backend = RecordingBackend::default();

    match backend.load_vector_index::<HnswIndex<u64>>("missing") {
        Err(HeuremaError::IndexNotFound { name, .. }) => {
            assert_eq!(name, "missing", "the error must carry the missing name");
        }
        Ok(_) => panic!("an unknown name must not load"),
        Err(other) => panic!("expected IndexNotFound, got {other:?}"),
    }
}

#[test]
fn missing_fts_index_loads_as_index_not_found() {
    let backend = RecordingBackend::default();

    let Err(error) = backend.load_fts_index::<Bm25Index<String>>("missing") else {
        panic!("an unknown name must not load");
    };

    assert_eq!(
        error.to_string(),
        "index not found: missing",
        "the display must name the missing index"
    );
    match error {
        HeuremaError::IndexNotFound { name, .. } => {
            assert_eq!(name, "missing", "the error must carry the missing name");
        }
        other => panic!("expected IndexNotFound, got {other:?}"),
    }
}

#[test]
fn persistence_source_display_forwards_wrapped_message() {
    let source = PersistenceSource::new(MidError::default());

    assert_eq!(
        source.to_string(),
        "mid failure",
        "the boundary wrapper must present the wrapped error's message"
    );
}

#[test]
fn persistence_source_chain_skips_duplicate_hop() {
    let source = PersistenceSource::new(MidError::default());

    let Some(next) = source.source() else {
        panic!("the wrapped error's chain must continue past the wrapper");
    };
    assert_eq!(
        next.to_string(),
        "leaf failure",
        "the next hop must be the wrapped error's own source, not a repeat of its message"
    );
}

#[test]
fn persistence_source_chain_ends_for_sourceless_wrapped_error() {
    let source = PersistenceSource::new(LeafError);

    assert!(
        source.source().is_none(),
        "a wrapped error without a source terminates the chain at the wrapper"
    );
}

#[test]
fn persistence_variant_reports_backend_chain_without_duplicate_hop() {
    let backend = FailingBackend;
    let vectors = HnswIndex::<u64>::new(HnswConfig::new(3));

    let error = match backend.save_vector_index("embeddings", &vectors) {
        Ok(()) => panic!("the failing backend must fail"),
        Err(error) => error,
    };

    assert_eq!(
        error.to_string(),
        "persistence backend error: mid failure",
        "the persistence variant surfaces the backend message"
    );
    let Some(boundary) = error.source() else {
        panic!("the persistence variant must chain to its backend source");
    };
    assert_eq!(
        boundary.to_string(),
        "mid failure",
        "the boundary hop presents the backend error's message"
    );
    let Some(leaf) = boundary.source() else {
        panic!("the chain must continue into the backend error's own source");
    };
    assert_eq!(
        leaf.to_string(),
        "leaf failure",
        "each chain hop must add information instead of repeating the last message"
    );
    assert!(leaf.source().is_none(), "the chain terminates at the leaf");
}
