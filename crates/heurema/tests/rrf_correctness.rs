use heurema::{DEFAULT_RRF_K_CONSTANT, HeuremaError, rrf, rrf_with_default};

#[test]
fn reciprocal_rank_fusion_scores_known_fixture() -> Result<(), HeuremaError> {
    let rankings = vec![
        vec![("alpha", 0.98), ("beta", 0.72), ("gamma", 0.51)],
        vec![("beta", 12.0), ("delta", 9.0), ("alpha", 4.0)],
    ];

    let fused = rrf(&rankings, DEFAULT_RRF_K_CONSTANT)?;

    assert_eq!(fused[0].0, "beta", "beta appears high in both rankings");
    assert_eq!(fused[1].0, "alpha", "alpha appears in both rankings");
    assert_eq!(fused.len(), 4, "fusion keeps every unique result ID");

    let Some(beta_score) = fused
        .iter()
        .find(|(id, _)| *id == "beta")
        .map(|(_, score)| *score)
    else {
        panic!("beta score should be present");
    };
    let expected = (1.0 / 62.0) + (1.0 / 61.0);
    assert!(
        (beta_score - expected).abs() < f32::EPSILON,
        "beta score should equal sum of reciprocal ranks"
    );
    Ok(())
}

#[test]
fn default_constant_is_paper_standard_sixty() {
    assert_eq!(
        DEFAULT_RRF_K_CONSTANT.to_bits(),
        60.0_f32.to_bits(),
        "default dampening constant must stay the RRF paper's k = 60"
    );

    let fused = rrf_with_default(&[vec![("alpha", 1.0)]]);

    assert_eq!(fused.len(), 1, "single hit fuses to a single entry");
    assert_eq!(fused[0].0, "alpha", "the fused entry keeps its ID");
    let expected = 1.0_f32 / 61.0;
    assert!(
        (fused[0].1 - expected).abs() < f32::EPSILON,
        "a rank-1 hit under the default constant must score 1/(60+1)"
    );
}

#[test]
fn empty_rankings_return_empty_fusion() {
    let rankings: Vec<Vec<(&str, f32)>> = Vec::new();

    assert!(
        rrf_with_default(&rankings).is_empty(),
        "no rankings should produce no fused results"
    );
}

#[test]
fn tied_scores_order_by_ascending_id_regardless_of_arrangement() {
    // WHY: an ID present only in the vector ranking at rank 1 and an ID
    // present only in the BM25 ranking at rank 1 fuse to the exact same
    // score — the documented contract must break that tie by ascending ID.
    let vector_only = vec![("delta", 0.9)];
    let bm25_only = vec![("alpha", 11.0)];

    let fused = rrf_with_default(&[vector_only.clone(), bm25_only.clone()]);
    let ids: Vec<&str> = fused.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        ids,
        vec!["alpha", "delta"],
        "exact-score ties must order by ascending ID"
    );

    let swapped = rrf_with_default(&[bm25_only, vector_only]);
    assert_eq!(
        fused, swapped,
        "tie order must not depend on ranking arrangement"
    );
}

#[test]
fn tie_order_is_stable_across_repeated_fusion() {
    // WHY: the pre-fix implementation drained a HashMap, so tied entries
    // shuffled with the per-process hash seed; the contract now enforces one
    // canonical order for every invocation over the same input.
    let rankings = vec![
        vec![("kappa", 0.4), ("mu", 0.3)],
        vec![("iota", 0.8), ("lambda", 0.6)],
    ];

    let first = rrf_with_default(&rankings);
    let ids: Vec<&str> = first.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        ids,
        vec!["iota", "kappa", "lambda", "mu"],
        "rank-tied pairs must resolve score-descending then ID-ascending"
    );
    for _ in 0..100 {
        assert_eq!(
            rrf_with_default(&rankings),
            first,
            "repeated fusion of identical input must return identical order"
        );
    }
}

#[test]
fn duplicate_id_within_one_ranking_counts_best_rank_only() {
    let rankings = vec![vec![("alpha", 0.9), ("beta", 0.8), ("alpha", 0.2)]];

    let fused = rrf_with_default(&rankings);

    assert_eq!(fused.len(), 2, "a duplicated ID is one result, not two");
    assert_eq!(fused[0].0, "alpha", "alpha keeps its best rank");
    let expected_alpha = 1.0_f32 / 61.0;
    assert!(
        (fused[0].1 - expected_alpha).abs() < f32::EPSILON,
        "the duplicate at rank 3 must not add to alpha's rank-1 contribution"
    );
    let expected_beta = 1.0_f32 / 62.0;
    assert!(
        (fused[1].1 - expected_beta).abs() < f32::EPSILON,
        "beta keeps its ordinary rank-2 contribution"
    );
}

#[test]
fn same_id_across_rankings_still_accumulates() {
    let rankings = vec![vec![("alpha", 0.9)], vec![("alpha", 7.0)]];

    let fused = rrf_with_default(&rankings);

    assert_eq!(fused.len(), 1, "one ID across rankings fuses to one entry");
    let expected = 2.0_f32 / 61.0;
    assert!(
        (fused[0].1 - expected).abs() < f32::EPSILON,
        "cross-ranking occurrences must sum their reciprocal ranks"
    );
}

#[test]
fn invalid_k_constant_returns_typed_error() {
    let rankings = vec![vec![("alpha", 1.0)]];

    for bad in [0.0_f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        match rrf(&rankings, bad) {
            Err(HeuremaError::InvalidKConstant { k_constant, .. }) => {
                assert_eq!(
                    k_constant.to_bits(),
                    bad.to_bits(),
                    "the error must carry the rejected constant"
                );
            }
            Ok(_) => panic!("k_constant {bad} must be rejected"),
            Err(other) => panic!("expected InvalidKConstant for {bad}, got {other:?}"),
        }
    }
}
