//! Reciprocal-rank fusion utilities.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

/// WHY: The original RRF paper uses 60 as the standard rank dampening constant.
pub const DEFAULT_RRF_K_CONSTANT: f32 = 60.0;

/// WHY: Hybrid search consumers need a dependency-free way to combine vector,
/// BM25, and future ranked result lists.
#[must_use]
pub fn rrf<Id: Eq + Hash + Clone>(rankings: &[Vec<(Id, f32)>], k_constant: f32) -> Vec<(Id, f32)> {
    assert!(
        k_constant.is_finite() && k_constant > 0.0,
        "RRF k_constant must be finite and positive"
    );

    let mut scores: HashMap<Id, f32> = HashMap::new();
    for ranking in rankings {
        for (rank, (id, _score)) in ranking.iter().enumerate() {
            #[expect(
                clippy::as_conversions,
                reason = "RRF rank arithmetic is defined in the f32 public score domain"
            )]
            let one_based_rank = rank as f32 + 1.0;
            let contribution = 1.0 / (k_constant + one_based_rank);
            scores
                .entry(id.clone())
                .and_modify(|score| *score += contribution)
                .or_insert(contribution);
        }
    }

    let mut fused: Vec<_> = scores.into_iter().collect();
    fused.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    fused
}

/// WHY: Most consumers should use the paper-standard rank dampening constant
/// unless they have measured a different value for their corpus.
#[must_use]
pub fn rrf_with_default<Id: Eq + Hash + Clone>(rankings: &[Vec<(Id, f32)>]) -> Vec<(Id, f32)> {
    rrf(rankings, DEFAULT_RRF_K_CONSTANT)
}
