//! Persistence must preserve ranking and reject inconsistent derived state.

use heurema::{Bm25Index, FtsConfig, FtsIndex, HeuremaError};
use serde_json::{Value, json};

fn populated() -> Result<Bm25Index<u64>, HeuremaError> {
    let mut index = Bm25Index::new(FtsConfig::simple());
    index.insert(1, "alpha alpha beta")?;
    index.insert(2, "alpha gamma")?;
    index.insert(3, "")?;
    Ok(index)
}

#[test]
fn snapshot_round_trip_preserves_rankings_and_later_mutations()
-> Result<(), Box<dyn std::error::Error>> {
    let mut original = populated()?;
    let bytes = serde_json::to_vec(&original)?;
    let mut restored: Bm25Index<u64> = serde_json::from_slice(&bytes)?;
    assert_eq!(
        original.query("alpha beta", 10)?,
        restored.query("alpha beta", 10)?
    );
    for index in [&mut original, &mut restored] {
        index.insert(1, "gamma gamma")?;
        index.remove(&2)?;
    }
    assert_eq!(original.query("gamma", 10)?, restored.query("gamma", 10)?);
    assert_eq!(original.len(), restored.len());
    Ok(())
}

#[test]
fn corrupt_snapshot_relationships_are_rejected_at_decode() -> Result<(), Box<dyn std::error::Error>>
{
    let original = serde_json::to_value(populated()?)?;
    let mutations: &[(&str, Value)] = &[
        ("/documents/1/length", json!(0)),
        ("/documents/1/counts/alpha", json!(0)),
        ("/documents/1/counts/alpha", json!(u64::MAX)),
        ("/total_document_terms", json!(0)),
        ("/postings/alpha/1", json!(100)),
        ("/postings/alpha", json!({"99": 2})),
        ("/postings", json!({})),
        ("/config/tokenizer/name", json!("NGram")),
    ];
    for (pointer, replacement) in mutations {
        let mut corrupt = original.clone();
        let Some(field) = corrupt.pointer_mut(pointer) else {
            panic!("invalid test pointer {pointer}");
        };
        *field = replacement.clone();
        assert!(
            serde_json::from_value::<Bm25Index<u64>>(corrupt).is_err(),
            "accepted corruption at {pointer}"
        );
    }
    Ok(())
}

#[test]
fn simple_pipeline_defines_case_and_unicode_tokenization() -> Result<(), HeuremaError> {
    let mut index = Bm25Index::new(FtsConfig::simple());
    index.insert(1, "CAFÉ—Alpha,42!")?;
    assert_eq!(index.query("café", 1)?[0].0, 1);
    assert_eq!(index.query("ALPHA", 1)?[0].0, 1);
    assert_eq!(index.query("42", 1)?[0].0, 1);
    assert!(
        index.query("cafe", 1)?.is_empty(),
        "Simple does not remove accents"
    );
    Ok(())
}
