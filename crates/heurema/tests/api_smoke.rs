use heurema::{
    Bm25Index, FtsConfig, FtsIndex, HeuremaError, HnswConfig, HnswIndex, VectorDistance,
    VectorIndex,
};

#[test]
fn hnsw_config_new_pins_krites_parity_defaults() {
    let config = HnswConfig::new(768);

    assert_eq!(config.dimensions, 768, "dimensions must pass through");
    assert_eq!(
        config.distance,
        VectorDistance::L2,
        "krites default distance is L2"
    );
    assert_eq!(
        config.ef_construction, 50,
        "krites default construction beam width is 50"
    );
    assert_eq!(
        config.m_neighbours, 16,
        "krites default HNSW m parameter is 16"
    );
}

#[test]
fn fts_config_simple_pins_krites_default_tokenizer() {
    let config = FtsConfig::simple();

    assert_eq!(
        config.tokenizer.name, "Simple",
        "krites default tokenizer is Simple"
    );
    assert!(
        config.tokenizer.args.is_empty(),
        "the Simple tokenizer takes no arguments"
    );
    assert!(
        config.filters.is_empty(),
        "the simple pipeline applies no post-tokenization filters"
    );
}

#[test]
fn hnsw_stub_exposes_vector_index_contract() {
    let mut index = HnswIndex::<u64>::new(HnswConfig::new(3));

    assert!(index.is_empty(), "new HNSW stub should be empty");
    assert_not_yet(index.insert(7, &[1.0, 2.0, 3.0]));
    assert_not_yet(index.query(&[1.0, 2.0, 3.0], 10));
    assert_not_yet(index.remove(&7));
}

#[test]
fn hnsw_stub_reports_dimension_mismatch_before_implementation() {
    let mut index = HnswIndex::<u64>::new(HnswConfig::new(3));

    let error = match index.insert(7, &[1.0, 2.0]) {
        Ok(()) => panic!("short vector should fail dimension check"),
        Err(error) => error,
    };

    match error {
        HeuremaError::DimensionMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, 3, "expected dimension should come from config");
            assert_eq!(actual, 2, "actual dimension should come from input");
        }
        other => panic!("expected dimension mismatch, got {other:?}"),
    }
}

#[test]
fn fts_stub_exposes_bm25_index_contract() {
    let mut index = Bm25Index::<String>::new(FtsConfig::simple());
    let doc_id = String::from("doc:1");

    assert!(index.is_empty(), "new FTS stub should be empty");
    assert_not_yet(index.insert(doc_id.clone(), "search text"));
    assert_not_yet(index.query("search", 10));
    assert_not_yet(index.remove(&doc_id));
}

fn assert_not_yet<T>(result: Result<T, HeuremaError>) {
    match result {
        Err(HeuremaError::NotYetImplemented { feature, .. }) => {
            assert!(
                feature.contains("Phase 2"),
                "stub error should point callers at Phase 2"
            );
        }
        Ok(_) => panic!("expected NotYetImplemented, got Ok"),
        Err(other) => panic!("expected NotYetImplemented, got {other:?}"),
    }
}
