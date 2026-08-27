//! BM25 properties: the published scoring formula's shape (idf decreasing in
//! document frequency, tf saturation, length normalization, per-term
//! additivity) plus the trait's ranking contract. Properties that depend on
//! an implementation choice the formula leaves open (idf variant for terms in
//! more than half the corpus, the k1/b parameter values) are named as
//! uncovered in `tests/oracle/OBSERVATIONS.md` rather than asserted here.

use heurema::{Bm25Index, FtsConfig, FtsIndex, HeuremaError};

fn index_of<S: AsRef<str>>(docs: &[S]) -> Result<Bm25Index<u64>, HeuremaError> {
    let mut index = Bm25Index::<u64>::new(FtsConfig::simple());
    for (id, doc) in docs.iter().enumerate() {
        index.insert(id as u64, doc.as_ref())?;
    }
    Ok(index)
}

fn score_of(results: &[(u64, f32)], id: u64) -> f32 {
    let Some((_, score)) = results.iter().find(|(result_id, _)| *result_id == id) else {
        panic!("doc {id} must appear in the results");
    };
    *score
}

fn assert_ranking_contract(results: &[(u64, f32)], k: usize) {
    assert!(results.len() <= k, "at most k results are returned");
    let mut seen = std::collections::BTreeSet::new();
    for (id, _) in results {
        assert!(seen.insert(*id), "result IDs are unique within one query");
    }
    for pair in results.windows(2) {
        let (a_id, a_score) = pair[0];
        let (b_id, b_score) = pair[1];
        assert!(
            a_score >= b_score,
            "scores are non-increasing: descending is better"
        );
        if a_score.to_bits() == b_score.to_bits() {
            assert!(a_id < b_id, "equal scores order by ascending ID");
        }
    }
}

#[test]
fn stub_reports_not_yet_implemented_until_phase_2_lands() {
    // WHY: tripwire. A real engine makes this fail, and the same commit must
    // delete it and strip every `#[ignore = "phase 2: ..."]` marker in this
    // module — otherwise the ignore markers outlive the phase silently.
    let index = Bm25Index::<u64>::new(FtsConfig::simple());

    let result = index.query("anything", 1);

    let Err(HeuremaError::NotYetImplemented { feature, .. }) = result else {
        panic!("the Phase 1 stub must report NotYetImplemented, got {result:?}");
    };
    assert!(
        feature.contains("Phase 2"),
        "the stub names the landing phase, got {feature}"
    );
}

#[test]
#[ignore = "phase 2: real BM25 engine lands (tests/oracle/PARITY.md)"]
fn empty_index_query_returns_no_results() -> Result<(), HeuremaError> {
    let index = Bm25Index::<u64>::new(FtsConfig::simple());

    let results = index.query("anything", 10)?;

    assert!(
        results.is_empty(),
        "an empty index holds no documents to return"
    );
    Ok(())
}

#[test]
#[ignore = "phase 2: real BM25 engine lands (tests/oracle/PARITY.md)"]
fn len_tracks_inserts() -> Result<(), HeuremaError> {
    let docs = ["one two", "three four", "five six"];
    let mut index = Bm25Index::<u64>::new(FtsConfig::simple());

    for (expected, doc) in docs.iter().enumerate() {
        index.insert(expected as u64, doc)?;
        assert_eq!(index.len(), expected + 1, "each insert adds one document");
    }
    Ok(())
}

#[test]
#[ignore = "phase 2: real BM25 engine lands (tests/oracle/PARITY.md)"]
fn no_match_query_returns_no_results() -> Result<(), HeuremaError> {
    let index = index_of(&["alpha beta", "gamma delta", "epsilon zeta"])?;

    let results = index.query("absentterm", 10)?;

    assert!(
        results.is_empty(),
        "the contract returns matching documents; a term absent from the corpus matches none"
    );
    Ok(())
}

#[test]
#[ignore = "phase 2: real BM25 engine lands (tests/oracle/PARITY.md)"]
fn query_results_satisfy_the_ranking_contract() -> Result<(), HeuremaError> {
    let mut index = Bm25Index::<u64>::new(FtsConfig::simple());
    index.insert(9, "alpha beta gamma")?;
    index.insert(2, "alpha beta gamma")?;
    index.insert(5, "alpha alpha alpha alpha")?;

    let first = index.query("alpha", 10)?;
    assert_ranking_contract(&first, 10);
    let position_of = |id: u64| first.iter().position(|(result_id, _)| *result_id == id);
    let (Some(pos_two), Some(pos_nine)) = (position_of(2), position_of(9)) else {
        panic!("both identical documents must appear in the results");
    };
    assert_eq!(
        score_of(&first, 2).to_bits(),
        score_of(&first, 9).to_bits(),
        "identical documents score identically"
    );
    assert_eq!(
        pos_two + 1,
        pos_nine,
        "tied IDs are adjacent and ascending, independent of insert order"
    );

    let second = index.query("alpha", 10)?;
    assert_eq!(
        first, second,
        "unchanged index state must answer identical queries identically"
    );

    let capped = index.query("alpha", 1)?;
    assert_eq!(capped.len(), 1, "at most k results are returned");
    Ok(())
}

#[test]
#[ignore = "phase 2: real BM25 engine lands (tests/oracle/PARITY.md)"]
fn rare_term_outranks_common_term_on_the_same_document() -> Result<(), HeuremaError> {
    // WHY: the published idf is strictly decreasing in document frequency
    // under both standard forms (Robertson's ln((N - df + 0.5)/(df + 0.5))
    // and the non-negative ln(1 + ...) variant), so with tf and document
    // length held equal the rarer term must score higher.
    let mut docs: Vec<String> = vec![String::from("common rare x0 y0")];
    for i in 1..10 {
        docs.push(format!("common x{i} y{i} z{i}"));
    }
    let index = index_of(&docs)?;

    let rare_score = score_of(&index.query("rare", 10)?, 0);
    let common_score = score_of(&index.query("common", 10)?, 0);

    assert!(
        rare_score > common_score,
        "idf decreases in document frequency: rare ({rare_score}) must outscore common ({common_score}) on the same document"
    );
    Ok(())
}

#[test]
#[ignore = "phase 2: real BM25 engine lands (tests/oracle/PARITY.md)"]
fn scores_are_non_negative_for_uncommon_terms() -> Result<(), HeuremaError> {
    // WHY: both published idf forms are non-negative for df <= N/2, and the
    // tf component is non-negative, so queries over uncommon terms never
    // score below zero. The sign for commoner terms is an idf-variant choice
    // and is deliberately uncovered (tests/oracle/OBSERVATIONS.md).
    let mut docs: Vec<String> = (0..10).map(|i| format!("f{i}a f{i}b f{i}c")).collect();
    docs[0] = String::from("seldom f0a f0b");
    docs[1] = String::from("seldom f1a f1b");
    docs[2] = String::from("seldom f2a f2b");
    let index = index_of(&docs)?;

    let results = index.query("seldom", 10)?;

    assert!(
        !results.is_empty(),
        "the query term is present in the corpus"
    );
    for (id, score) in &results {
        assert!(
            *score >= 0.0,
            "doc {id} scored {score}: uncommon-term scores are non-negative"
        );
    }
    Ok(())
}

#[test]
#[ignore = "phase 2: real BM25 engine lands (tests/oracle/PARITY.md)"]
fn term_frequency_saturates() -> Result<(), HeuremaError> {
    // WHY: the published tf component x(k1+1)/(x + k1*c) is strictly concave
    // in x, so each added occurrence scores less than the last. Equal
    // document lengths isolate tf from length normalization; filler docs keep
    // df("sat") at 3 of 8 so its idf is positive.
    let docs = [
        "sat p1 q1 r1",
        "sat sat p2 q2",
        "sat sat sat p3",
        "f1 f2 f3 f4",
        "f5 f6 f7 f8",
        "f9 f10 f11 f12",
        "f13 f14 f15 f16",
        "f17 f18 f19 f20",
    ];
    let index = index_of(&docs)?;

    let results = index.query("sat", 10)?;
    let s1 = score_of(&results, 0);
    let s2 = score_of(&results, 1);
    let s3 = score_of(&results, 2);

    assert!(s1 < s2, "a second occurrence scores higher: {s1} vs {s2}");
    assert!(s2 < s3, "a third occurrence scores higher: {s2} vs {s3}");
    assert!(
        (s2 - s1) > (s3 - s2),
        "saturation: the increment shrinks ({s1} -> {s2} -> {s3})"
    );
    Ok(())
}

#[test]
#[ignore = "phase 2: real BM25 engine lands (tests/oracle/PARITY.md)"]
fn length_normalization_prefers_the_shorter_document() -> Result<(), HeuremaError> {
    // WHY: the published length-normalization factor 1 - b + b*dl/avgdl
    // penalizes above-average document lengths for b > 0, so at equal term
    // frequency the shorter document scores higher. An implementation
    // shipping b = 0 would produce a tie here and must revisit the parity
    // definition (tests/oracle/PARITY.md) rather than weaken this assertion.
    let docs = [
        "len u1",
        "len u1 u2 u3 u4 u5 u6",
        "f1 f2 f3 f4",
        "f5 f6 f7 f8",
        "f9 f10 f11 f12",
        "f13 f14 f15 f16",
    ];
    let index = index_of(&docs)?;

    let results = index.query("len", 10)?;
    let short = score_of(&results, 0);
    let long = score_of(&results, 1);

    assert!(
        short > long,
        "at equal term frequency the shorter document scores higher: {short} vs {long}"
    );
    Ok(())
}

#[test]
#[ignore = "phase 2: real BM25 engine lands (tests/oracle/PARITY.md)"]
fn per_term_contributions_sum_for_multi_term_queries() -> Result<(), HeuremaError> {
    // WHY: the published score is a sum over query terms of idf * tf-norm, so
    // a two-term query scores the sum of the single-term scores on a document
    // containing both. Additivity pins the formula's shape without pinning
    // the implementation's parameter values.
    let docs = [
        "t1 t2 j1 j2",
        "f1 f2 f3 f4",
        "f5 f6 f7 f8",
        "f9 f10 f11 f12",
        "f13 f14 f15 f16",
        "f17 f18 f19 f20",
    ];
    let index = index_of(&docs)?;

    let both = score_of(&index.query("t1 t2", 10)?, 0);
    let sum = score_of(&index.query("t1", 10)?, 0) + score_of(&index.query("t2", 10)?, 0);

    assert!(
        (both - sum).abs() <= 1e-5 * sum,
        "multi-term score {both} is the sum of the single-term scores {sum}"
    );
    Ok(())
}

#[test]
#[ignore = "phase 2: real BM25 engine lands (tests/oracle/PARITY.md)"]
fn remove_is_idempotent_and_excludes_the_document() -> Result<(), HeuremaError> {
    let docs = [
        "alpha x0 y0",
        "beta x1 y1",
        "vanish x2 y2",
        "gamma x3 y3",
        "delta x4 y4",
        "epsilon x5 y5",
    ];
    let mut index = index_of(&docs)?;
    assert_eq!(index.len(), 6, "all inserts are counted");

    // WHY: removal is idempotent by contract so row-deletion cleanup can
    // retry safely; removing an absent ID is a successful no-op.
    index.remove(&2)?;
    index.remove(&2)?;
    index.remove(&999)?;
    assert_eq!(
        index.len(),
        5,
        "only the removal of a present ID changes cardinality"
    );

    let results = index.query("vanish", 10)?;
    assert!(
        results.is_empty(),
        "the only document containing the term was removed"
    );

    let results = index.query("alpha", 10)?;
    assert!(
        results.iter().all(|(id, _)| *id != 2),
        "a removed ID never appears in results"
    );
    Ok(())
}
