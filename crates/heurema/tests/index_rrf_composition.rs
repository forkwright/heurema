//! Conformance tests for composing the index traits with fusion.
//!
//! WHY: Vector search, full-text search, and RRF exist in one crate so that
//! hybrid search can be built from them. Testing each surface in isolation let
//! the advertised consumer path — query an index, fuse the result — become
//! uncompilable while every other test stayed green. These tests exercise the
//! composition itself, generically, so the boundary is what fails.

use std::collections::BTreeSet;

use heurema::{DEFAULT_RRF_K_CONSTANT, FtsIndex, HeuremaError, VectorIndex, rrf, rrf_with_default};

/// WHY: This is the regression guard. It knows nothing about any concrete
/// index — only the traits — so it compiles if and only if a generic consumer
/// can feed trait output straight into fusion. Narrowing `VectorIndex::Id`
/// below what `rrf` fuses under makes this fail to build.
fn fuse_vector_query<V: VectorIndex>(
    index: &V,
    vector: &[f32],
    k: usize,
) -> Result<Vec<(V::Id, f32)>, HeuremaError> {
    let ranked = index.query(vector, k)?;
    rrf(&[ranked], DEFAULT_RRF_K_CONSTANT)
}

/// WHY: The FTS half of the same guard; the two traits declare `Id`
/// independently, so one can regress without the other.
fn fuse_text_query<F: FtsIndex>(
    index: &F,
    query: &str,
    k: usize,
) -> Result<Vec<(F::Id, f32)>, HeuremaError> {
    let ranked = index.query(query, k)?;
    rrf(&[ranked], DEFAULT_RRF_K_CONSTANT)
}

/// WHY: The actual reason both traits coexist — fusing a vector ranking with a
/// text ranking requires their two `Id` types to meet fusion's bound at once.
fn fuse_hybrid<V, F>(
    vector_index: &V,
    text_index: &F,
    vector: &[f32],
    query: &str,
    k: usize,
) -> Result<Vec<(V::Id, f32)>, HeuremaError>
where
    V: VectorIndex,
    F: FtsIndex<Id = V::Id>,
{
    let vector_ranked = vector_index.query(vector, k)?;
    let text_ranked = text_index.query(query, k)?;
    rrf(&[vector_ranked, text_ranked], DEFAULT_RRF_K_CONSTANT)
}

/// A vector index that replays a scripted ranking.
///
/// WHY: Fusion reads position, not score, so a conformance fixture only needs
/// to control the order and contents of the returned vector.
struct ScriptedVectorIndex {
    ranking: Vec<(u32, f32)>,
    dimensions: usize,
}

impl VectorIndex for ScriptedVectorIndex {
    type Id = u32;

    fn insert(&mut self, id: Self::Id, vector: &[f32]) -> Result<(), HeuremaError> {
        if vector.len() != self.dimensions {
            return Ok(());
        }
        self.ranking.push((id, 0.0));
        Ok(())
    }

    fn query(&self, _vector: &[f32], k: usize) -> Result<Vec<(Self::Id, f32)>, HeuremaError> {
        Ok(self.ranking.iter().take(k).copied().collect())
    }

    fn remove(&mut self, id: &Self::Id) -> Result<(), HeuremaError> {
        self.ranking.retain(|(candidate, _)| candidate != id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.ranking.len()
    }
}

/// A full-text index that replays a scripted ranking.
struct ScriptedFtsIndex {
    ranking: Vec<(u32, f32)>,
}

impl FtsIndex for ScriptedFtsIndex {
    type Id = u32;

    fn insert(&mut self, id: Self::Id, _document: &str) -> Result<(), HeuremaError> {
        self.ranking.push((id, 0.0));
        Ok(())
    }

    fn query(&self, _query: &str, k: usize) -> Result<Vec<(Self::Id, f32)>, HeuremaError> {
        Ok(self.ranking.iter().take(k).copied().collect())
    }

    fn remove(&mut self, id: &Self::Id) -> Result<(), HeuremaError> {
        self.ranking.retain(|(candidate, _)| candidate != id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.ranking.len()
    }
}

#[test]
fn generic_consumer_fuses_vector_index_output() -> Result<(), HeuremaError> {
    let index = ScriptedVectorIndex {
        ranking: vec![(7, 0.10), (3, 0.42), (9, 0.88)],
        dimensions: 3,
    };

    let fused = fuse_vector_query(&index, &[0.0, 1.0, 0.0], 10)?;

    assert_eq!(
        fused.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![7, 3, 9],
        "fusing a single ranking preserves the producer's order"
    );
    Ok(())
}

#[test]
fn generic_consumer_fuses_fts_index_output() -> Result<(), HeuremaError> {
    let index = ScriptedFtsIndex {
        ranking: vec![(4, 12.5), (1, 8.0)],
    };

    let fused = fuse_text_query(&index, "canonical corpus", 10)?;

    assert_eq!(
        fused.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![4, 1],
        "fusing a single ranking preserves the producer's order"
    );
    Ok(())
}

#[test]
fn hybrid_consumer_fuses_both_index_kinds() -> Result<(), HeuremaError> {
    // WHY: opposite score polarities on purpose — distances ascend, BM25
    // scores descend. Fusion must be immune because it reads position.
    //
    // ID 2 is deliberately each engine's *second* hit and the only ID both
    // return: 1/62 + 1/62 against 1/61 for a single first place. Agreeing
    // engines outrank either engine's own top hit, by a factor of about two
    // rather than a margin that rounds away.
    let vector_index = ScriptedVectorIndex {
        ranking: vec![(1, 0.05), (2, 0.30), (3, 0.90)],
        dimensions: 2,
    };
    let text_index = ScriptedFtsIndex {
        ranking: vec![(4, 40.0), (2, 22.0), (5, 1.5)],
    };

    let fused = fuse_hybrid(&vector_index, &text_index, &[1.0, 0.0], "hybrid", 10)?;

    assert_eq!(
        fused.len(),
        5,
        "fusion keeps every unique ID from both sides"
    );
    assert_eq!(
        fused[0].0, 2,
        "the ID both engines return outranks either engine's own top hit"
    );
    assert_eq!(
        fused.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![2, 1, 4, 3, 5],
        "equal-rank IDs from the two producers tie and break by ascending ID"
    );

    let ids: BTreeSet<u32> = fused.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        ids,
        BTreeSet::from([1, 2, 3, 4, 5]),
        "no ID from either producer is dropped"
    );
    Ok(())
}

#[test]
fn fusion_is_immune_to_producer_score_polarity() -> Result<(), HeuremaError> {
    // WHY: the ranking contract makes score advisory. The same order carrying
    // inverted scores must fuse identically, or a producer's polarity choice
    // would silently change hybrid results.
    let ascending = ScriptedVectorIndex {
        ranking: vec![(1, 0.01), (2, 0.50), (3, 0.99)],
        dimensions: 1,
    };
    let descending = ScriptedVectorIndex {
        ranking: vec![(1, 0.99), (2, 0.50), (3, 0.01)],
        dimensions: 1,
    };

    let from_ascending = fuse_vector_query(&ascending, &[0.0], 10)?;
    let from_descending = fuse_vector_query(&descending, &[0.0], 10)?;

    assert_eq!(
        from_ascending, from_descending,
        "fusion reads rank position, so score polarity cannot affect the result"
    );
    Ok(())
}

#[test]
fn adversarial_duplicate_ids_contribute_only_their_best_rank() -> Result<(), HeuremaError> {
    // WHY: a producer violating ID uniqueness must not be able to inflate an
    // ID's fused score by listing it repeatedly.
    let offender = ScriptedVectorIndex {
        ranking: vec![(1, 0.1), (1, 0.2), (1, 0.3), (2, 0.4)],
        dimensions: 1,
    };
    let honest = ScriptedVectorIndex {
        ranking: vec![(1, 0.1), (2, 0.4)],
        dimensions: 1,
    };

    let fused_offender = fuse_vector_query(&offender, &[0.0], 10)?;

    assert_eq!(
        fused_offender.len(),
        2,
        "a repeated ID collapses to one fused entry"
    );
    assert_eq!(
        fused_offender[0].0, 1,
        "the duplicated ID keeps its best rank"
    );

    let honest_top = fuse_vector_query(&honest, &[0.0], 10)?[0].1;
    assert!(
        (fused_offender[0].1 - honest_top).abs() < f32::EPSILON,
        "duplication earns no score advantage over listing the ID once"
    );
    Ok(())
}

#[test]
fn adversarial_tied_scores_fuse_deterministically() -> Result<(), HeuremaError> {
    // WHY: an implementation returning every score equal still has a normative
    // order. Fusion must break exact ties by ascending ID rather than by
    // whatever order the map iterated.
    let index = ScriptedFtsIndex {
        ranking: vec![(9, 1.0), (4, 1.0), (7, 1.0)],
    };

    let first = fuse_text_query(&index, "tied", 10)?;
    let second = fuse_text_query(&index, "tied", 10)?;

    assert_eq!(
        first, second,
        "repeated fusion of stable input is identical"
    );
    assert_eq!(
        first.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![9, 4, 7],
        "distinct ranks outrank ID order; ties are only broken when scores match"
    );

    // Two rankings that are exact reverses give every ID the same fused score,
    // so the ascending-ID tiebreak is the only thing deciding order.
    let all_tied = rrf_with_default(&[vec![(9, 1.0), (4, 1.0)], vec![(4, 1.0), (9, 1.0)]]);
    assert_eq!(
        all_tied.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![4, 9],
        "a genuine score tie is broken by ascending ID"
    );
    Ok(())
}

#[test]
fn adversarial_unordered_producer_changes_fusion() -> Result<(), HeuremaError> {
    // WHY: this test documents why ordering is normative rather than advisory.
    // The same set of results in a different order fuses differently, so an
    // implementation that ignores the ordering contract silently corrupts
    // hybrid search instead of failing loudly.
    let ordered = ScriptedVectorIndex {
        ranking: vec![(1, 0.1), (2, 0.2), (3, 0.3)],
        dimensions: 1,
    };
    let shuffled = ScriptedVectorIndex {
        ranking: vec![(3, 0.3), (1, 0.1), (2, 0.2)],
        dimensions: 1,
    };

    let from_ordered = fuse_vector_query(&ordered, &[0.0], 10)?;
    let from_shuffled = fuse_vector_query(&shuffled, &[0.0], 10)?;

    assert_ne!(
        from_ordered.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        from_shuffled.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        "position is authoritative, so violating the ordering contract is observable"
    );
    Ok(())
}

#[test]
fn query_honours_the_k_bound() -> Result<(), HeuremaError> {
    let index = ScriptedVectorIndex {
        ranking: vec![(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)],
        dimensions: 1,
    };

    let fused = fuse_vector_query(&index, &[0.0], 2)?;

    assert_eq!(fused.len(), 2, "at most k results reach fusion");
    Ok(())
}
