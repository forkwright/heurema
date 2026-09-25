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

/// The snapshot text of `original` with the map at `pointer` written out
/// entry by entry, and, when `repeat` is set, its first entry written again,
/// with the same value, after its last.
///
/// WHY text: a `serde_json::Value` map cannot hold one key twice.
fn with_map_spliced(
    original: &Value,
    pointer: &str,
    repeat: bool,
) -> Result<String, serde_json::Error> {
    const SPLICE: &str = "spliced map";
    let mut value = original.clone();
    let Some(field) = value.pointer_mut(pointer) else {
        panic!("invalid test pointer {pointer}");
    };
    let Value::Object(map) = field.take() else {
        panic!("{pointer} is not a map");
    };
    let mut entries = map
        .iter()
        .map(|(key, value)| Ok(format!("{}:{value}", serde_json::to_string(key)?)))
        .collect::<Result<Vec<String>, serde_json::Error>>()?;
    let Some(first) = entries.first().cloned() else {
        panic!("{pointer} is empty");
    };
    if repeat {
        entries.push(first);
    }
    *field = json!(SPLICE);
    let text = serde_json::to_string(&value)?;
    let quoted = serde_json::to_string(SPLICE)?;
    assert_eq!(text.matches(&quoted).count(), 1, "{text}");
    Ok(text.replacen(&quoted, &format!("{{{}}}", entries.join(",")), 1))
}

#[test]
fn a_snapshot_map_naming_one_key_twice_is_rejected_at_decode()
-> Result<(), Box<dyn std::error::Error>> {
    let original = serde_json::to_value(populated()?)?;
    for pointer in [
        "/documents",
        "/documents/1/counts",
        "/postings",
        "/postings/alpha",
    ] {
        // NOTE: the repeat has the first entry's own value, so without the
        // refusal it would decode as the original index; the splice alone
        // does.
        let spliced = with_map_spliced(&original, pointer, false)?;
        serde_json::from_str::<Bm25Index<u64>>(&spliced)?;
        let repeated = with_map_spliced(&original, pointer, true)?;
        let Err(error) = serde_json::from_str::<Bm25Index<u64>>(&repeated) else {
            panic!("accepted a repeated key at {pointer}");
        };
        assert!(
            error.to_string().contains("names one key twice"),
            "{pointer}: {error}"
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
