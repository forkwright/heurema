//! Deterministic fixtures shared by the oracle property tests.

/// WHY: oracle fixtures must be reproducible run over run, so pseudo-random
/// data comes from a fixed-seed generator. The seed is part of the oracle
/// contract: changing one re-baselines every fixture that draws from it (see
/// `tests/oracle/PARITY.md`).
pub(super) struct XorShift64(u64);

impl XorShift64 {
    pub(super) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform draw in [0, 1) from the top 24 bits of the state.
    pub(super) fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }
}

pub(super) fn random_vector(rng: &mut XorShift64, dimensions: usize) -> Vec<f32> {
    (0..dimensions).map(|_| rng.next_f32()).collect()
}

pub(super) fn seeded_vectors(seed: u64, count: usize, dimensions: usize) -> Vec<(u64, Vec<f32>)> {
    let mut rng = XorShift64::new(seed);
    (0..count)
        .map(|id| (id as u64, random_vector(&mut rng, dimensions)))
        .collect()
}

fn l2_squared(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| (a - b) * (a - b)).sum()
}

/// Exact top-k by ascending squared-L2 distance with ties broken by ascending
/// ID: the ground truth the HNSW recall floor is measured against.
pub(super) fn brute_force_topk(vectors: &[(u64, Vec<f32>)], query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = vectors
        .iter()
        .map(|(id, vector)| (*id, l2_squared(vector, query)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}
