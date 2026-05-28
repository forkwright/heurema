use heurema::{DEFAULT_RRF_K_CONSTANT, rrf, rrf_with_default};

#[test]
fn reciprocal_rank_fusion_scores_known_fixture() {
    let rankings = vec![
        vec![("alpha", 0.98), ("beta", 0.72), ("gamma", 0.51)],
        vec![("beta", 12.0), ("delta", 9.0), ("alpha", 4.0)],
    ];

    let fused = rrf(&rankings, DEFAULT_RRF_K_CONSTANT);

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
}

#[test]
fn default_helper_matches_standard_constant() {
    let rankings = vec![vec![("alpha", 1.0)], vec![("alpha", 0.5)]];

    assert_eq!(
        rrf_with_default(&rankings),
        rrf(&rankings, DEFAULT_RRF_K_CONSTANT),
        "default helper must use the paper-standard constant"
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
