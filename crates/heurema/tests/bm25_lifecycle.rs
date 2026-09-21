//! BM25 lifecycle regressions independent of the formula oracle.

use heurema::{Bm25Index, FtsConfig, FtsIndex, HeuremaError, TokenizerConfig};

#[test]
fn replacement_removes_old_postings_before_new_document_is_visible() -> Result<(), HeuremaError> {
    let mut index = Bm25Index::<u64>::new(FtsConfig::simple());
    index.insert(7, "obsolete signal")?;
    index.insert(7, "current signal")?;

    assert_eq!(index.len(), 1, "replacement keeps one logical member");
    assert!(index.query("obsolete", 10)?.is_empty());
    assert_eq!(
        index.query("current", 10)?.first().map(|(id, _)| *id),
        Some(7)
    );
    Ok(())
}

#[test]
fn unsupported_pipeline_refuses_before_mutating_state() {
    let mut config = FtsConfig::simple();
    config.tokenizer = TokenizerConfig::new("NGram", vec!["3".to_owned()]);
    let mut index = Bm25Index::<u64>::new(config);

    assert!(matches!(
        index.insert(1, "must not enter the index"),
        Err(HeuremaError::NotYetImplemented { .. })
    ));
    assert!(
        index.is_empty(),
        "unsupported configuration does not mutate state"
    );
}
