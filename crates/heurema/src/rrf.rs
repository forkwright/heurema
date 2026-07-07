//! Reciprocal-rank fusion utilities.

use std::collections::{BTreeMap, BTreeSet};

use snafu::ensure;

use crate::HeuremaError;
use crate::error::InvalidKConstantSnafu;

/// WHY: The original RRF paper uses 60 as the standard rank dampening constant.
pub const DEFAULT_RRF_K_CONSTANT: f32 = 60.0;

/// WHY: Hybrid search consumers need a dependency-free way to combine vector,
/// BM25, and future ranked result lists.
///
/// Ordering contract: results are sorted by fused score descending, and exact
/// score ties are broken by ascending `Id` — the same set of rankings always
/// fuses to the same order, independent of ranking arrangement and process.
/// Scores are accumulated and compared in `f64`, then narrowed to the `f32`
/// public score domain; two returned entries can therefore display equal `f32`
/// scores while their order was decided by distinct `f64` sums. An `Id`
/// duplicated within one ranking contributes only its best (lowest) rank; the
/// same `Id` across different rankings accumulates as usual.
///
/// # Errors
///
/// Returns [`HeuremaError::InvalidKConstant`] when `k_constant` is not finite
/// and positive.
#[must_use = "fusion has no side effects; the fused ranking is the result"]
pub fn rrf<Id: Ord + Clone>(
    rankings: &[Vec<(Id, f32)>],
    k_constant: f32,
) -> Result<Vec<(Id, f32)>, HeuremaError> {
    ensure!(
        k_constant.is_finite() && k_constant > 0.0,
        InvalidKConstantSnafu { k_constant }
    );
    Ok(fuse(rankings, f64::from(k_constant)))
}

/// WHY: Most consumers should use the paper-standard rank dampening constant
/// unless they have measured a different value for their corpus.
///
/// Infallible: [`DEFAULT_RRF_K_CONSTANT`] always passes the [`rrf`] constant
/// validation. The ordering contract is identical to [`rrf`].
#[must_use = "fusion has no side effects; the fused ranking is the result"]
pub fn rrf_with_default<Id: Ord + Clone>(rankings: &[Vec<(Id, f32)>]) -> Vec<(Id, f32)> {
    // INVARIANT: DEFAULT_RRF_K_CONSTANT is a finite positive literal, so the
    // rrf() validation can never reject it and fusion proceeds directly.
    fuse(rankings, f64::from(DEFAULT_RRF_K_CONSTANT))
}

fn fuse<Id: Ord + Clone>(rankings: &[Vec<(Id, f32)>], k_constant: f64) -> Vec<(Id, f32)> {
    let mut scores: BTreeMap<Id, f64> = BTreeMap::new();
    for ranking in rankings {
        let mut seen_in_ranking: BTreeSet<&Id> = BTreeSet::new();
        for (rank, (id, _score)) in ranking.iter().enumerate() {
            // WHY: a duplicated ID inside one ranking is one result listed
            // twice, not two results — only its best rank may contribute.
            if !seen_in_ranking.insert(id) {
                continue;
            }
            #[expect(
                clippy::as_conversions,
                reason = "rank is an in-memory result-list offset; every practical rank is exactly representable in the f64 mantissa"
            )]
            let one_based_rank = rank as f64 + 1.0;
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k_constant + one_based_rank);
        }
    }

    let mut fused: Vec<(Id, f64)> = scores.into_iter().collect();
    // INVARIANT: score descending then Id ascending — the documented
    // deterministic ordering contract; map keys are unique, so the comparator
    // is a total order and unstable sorting cannot reorder equal elements.
    fused.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    fused
        .into_iter()
        .map(|(id, score)| {
            #[expect(
                clippy::as_conversions,
                reason = "the public score domain is f32; ordering was already resolved on the f64 sums"
            )]
            let public_score = score as f32;
            (id, public_score)
        })
        .collect()
}
