//! HNSW properties: the paper's graph-search invariants observable through
//! `VectorIndex`, plus the trait's ranking contract. Graph-internal
//! invariants the trait cannot observe (entry-point reachability, layer
//! assignment) are named in `tests/oracle/OBSERVATIONS.md` as Phase 2 unit-
//! test territory, not oracle territory.

use heurema::{HeuremaError, HnswConfig, HnswIndex, VectorDistance, VectorIndex};

use super::support::{XorShift64, brute_force_topk, random_vector, seeded_vectors};

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
            a_score <= b_score,
            "distances are non-decreasing: ascending is better"
        );
        if a_score.to_bits() == b_score.to_bits() {
            assert!(a_id < b_id, "equal scores order by ascending ID");
        }
    }
}

#[test]
fn dimension_mismatch_is_rejected_before_state_change() {
    let mut index = HnswIndex::<u64>::new(HnswConfig::new(4));

    for wrong in [0_usize, 3, 5, 8] {
        let vector = vec![0.0_f32; wrong];
        match index.insert(1, &vector) {
            Err(HeuremaError::DimensionMismatch {
                expected, actual, ..
            }) => {
                assert_eq!(expected, 4, "expected dimension comes from the config");
                assert_eq!(actual, wrong, "actual dimension comes from the input");
            }
            Ok(()) => panic!("insert of a {wrong}-dimensional vector into a 4-dimensional index must fail"),
            Err(other) => panic!("the dimension check precedes every other failure mode, got {other:?}"),
        }
        match index.query(&vector, 1) {
            Err(HeuremaError::DimensionMismatch { .. }) => {}
            Ok(_) => panic!("query with a {wrong}-dimensional vector must fail"),
            Err(other) => panic!("the dimension check precedes every other failure mode, got {other:?}"),
        }
        assert!(
            index.is_empty(),
            "rejected inserts must not change index cardinality"
        );
    }
}

#[test]
fn stub_reports_not_yet_implemented_until_phase_2_lands() {
    // WHY: tripwire. A real engine makes this fail, and the same commit must
    // delete it and strip every `#[ignore = "phase 2: ..."]` marker in this
    // module — otherwise the ignore markers outlive the phase silently.
    let mut index = HnswIndex::<u64>::new(HnswConfig::new(4));

    let result = index.insert(1, &[0.0; 4]);

    let Err(HeuremaError::NotYetImplemented { feature, .. }) = result else {
        panic!("the Phase 1 stub must report NotYetImplemented, got {result:?}");
    };
    assert!(
        feature.contains("Phase 2"),
        "the stub names the landing phase, got {feature}"
    );
}

#[test]
#[ignore = "phase 2: real HNSW engine lands (tests/oracle/PARITY.md)"]
fn empty_index_query_returns_no_results() -> Result<(), HeuremaError> {
    let index = HnswIndex::<u64>::new(HnswConfig::new(8));

    let results = index.query(&[0.0; 8], 10)?;

    assert!(
        results.is_empty(),
        "an empty index holds no candidates to return"
    );
    Ok(())
}

#[test]
#[ignore = "phase 2: real HNSW engine lands (tests/oracle/PARITY.md)"]
fn len_tracks_inserts() -> Result<(), HeuremaError> {
    let vectors = seeded_vectors(0x5EED_000B, 5, 4);
    let mut index = HnswIndex::<u64>::new(HnswConfig::new(4));

    for (expected, (id, vector)) in vectors.iter().enumerate() {
        index.insert(*id, vector)?;
        assert_eq!(index.len(), expected + 1, "each insert adds one entry");
    }
    Ok(())
}

#[test]
#[ignore = "phase 2: real HNSW engine lands (tests/oracle/PARITY.md)"]
fn query_results_satisfy_the_ranking_contract() -> Result<(), HeuremaError> {
    let vectors = seeded_vectors(0x5EED_0008, 64, 8);
    let mut index = HnswIndex::<u64>::new(HnswConfig::new(8));
    for (id, vector) in &vectors {
        index.insert(*id, vector)?;
    }

    let query = random_vector(&mut XorShift64::new(0x5EED_0009), 8);
    let first = index.query(&query, 10)?;
    assert_ranking_contract(&first, 10);

    let second = index.query(&query, 10)?;
    assert_eq!(
        first, second,
        "unchanged index state must answer identical queries identically"
    );
    Ok(())
}

#[test]
#[ignore = "phase 2: real HNSW engine lands (tests/oracle/PARITY.md)"]
fn smaller_k_is_a_prefix_of_larger_k() -> Result<(), HeuremaError> {
    let vectors = seeded_vectors(0x5EED_0006, 64, 8);
    let mut index = HnswIndex::<u64>::new(HnswConfig::new(8));
    for (id, vector) in &vectors {
        index.insert(*id, vector)?;
    }

    let query = random_vector(&mut XorShift64::new(0x5EED_0007), 8);
    let top_five = index.query(&query, 5)?;
    let top_ten = index.query(&query, 10)?;

    assert_eq!(
        &top_ten[..5],
        top_five.as_slice(),
        "ordering is normative and ties are stable, so top-5 is a prefix of top-10"
    );
    Ok(())
}

#[test]
#[ignore = "phase 2: real HNSW engine lands (tests/oracle/PARITY.md)"]
fn equal_distance_ties_order_by_ascending_id() -> Result<(), HeuremaError> {
    let mut index = HnswIndex::<u64>::new(HnswConfig::new(4));
    index.insert(9, &[1.0, 2.0, 3.0, 4.0])?;
    index.insert(2, &[1.0, 2.0, 3.0, 4.0])?;

    let results = index.query(&[1.0, 2.0, 3.0, 4.0], 2)?;

    let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        ids,
        vec![2, 9],
        "equal distances resolve by ascending ID so repeated queries are identical"
    );
    Ok(())
}

#[test]
#[ignore = "phase 2: real HNSW engine lands (tests/oracle/PARITY.md)"]
fn self_query_returns_the_inserted_vector_first_at_zero_distance() -> Result<(), HeuremaError> {
    let vectors = seeded_vectors(0x5EED_0005, 16, 8);

    for distance in [VectorDistance::L2, VectorDistance::Cosine] {
        let mut config = HnswConfig::new(8);
        config.distance = distance;
        let mut index = HnswIndex::<u64>::new(config);
        for (id, vector) in &vectors {
            index.insert(*id, vector)?;
        }

        let (target_id, target) = &vectors[7];
        let results = index.query(target, 4)?;
        let Some((best_id, best_distance)) = results.first() else {
            panic!("self query must return at least the queried vector ({distance:?})");
        };
        assert_eq!(
            best_id, target_id,
            "the queried vector is its own nearest neighbour ({distance:?})"
        );
        assert!(
            best_distance.abs() < 1e-6,
            "self distance is zero for L2 and cosine ({distance:?}), got {best_distance}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "phase 2: real HNSW engine lands (tests/oracle/PARITY.md)"]
fn cosine_distance_with_a_zero_vector_stays_finite() -> Result<(), HeuremaError> {
    // WHY: cosine distance divides by vector norms, so the zero vector is the
    // formula's one degenerate input; the pinned oracle observes no-NaN for
    // this case (tests/oracle/OBSERVATIONS.md). Whatever value convention the
    // implementation picks, every returned distance must be finite and the
    // ranking contract must still hold.
    let vectors = seeded_vectors(0x5EED_000C, 16, 8);
    let mut config = HnswConfig::new(8);
    config.distance = VectorDistance::Cosine;
    let mut index = HnswIndex::<u64>::new(config);
    for (id, vector) in &vectors {
        index.insert(*id, vector)?;
    }

    let zero = vec![0.0_f32; 8];
    index.insert(1000, &zero)?;
    let results = index.query(&zero, 4)?;

    for (_, distance) in &results {
        assert!(
            distance.is_finite(),
            "cosine distances involving a zero vector must be finite, got {distance}"
        );
    }
    assert_ranking_contract(&results, 4);
    Ok(())
}

#[test]
#[ignore = "phase 2: real HNSW engine lands (tests/oracle/PARITY.md)"]
fn recall_against_brute_force_meets_floor() -> Result<(), HeuremaError> {
    // WHY 0.90: at 256 points with the paper's default m = 16 and
    // ef_construction = 50 the graph is dense relative to N, so exact recall
    // is the common case; the floor leaves slack for the paper's probabilistic
    // guarantee without letting a degraded graph pass. The fixture and its
    // seeds are part of the parity definition (tests/oracle/PARITY.md).
    const RECALL_FLOOR: f32 = 0.90;
    const K: usize = 10;

    let vectors = seeded_vectors(0x5EED_0001, 256, 16);
    let mut index = HnswIndex::<u64>::new(HnswConfig::new(16));
    for (id, vector) in &vectors {
        index.insert(*id, vector)?;
    }

    let mut query_rng = XorShift64::new(0x5EED_0002);
    let queries: Vec<Vec<f32>> = (0..8).map(|_| random_vector(&mut query_rng, 16)).collect();

    let mut total = 0.0_f32;
    for query in &queries {
        let approximate: Vec<u64> = index
            .query(query, K)?
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let exact = brute_force_topk(&vectors, query, K);
        let hits = approximate.iter().filter(|id| exact.contains(id)).count();
        total += hits as f32 / K as f32;
    }
    let recall = total / queries.len() as f32;

    assert!(
        recall >= RECALL_FLOOR,
        "recall@{K} {recall} against exact brute force must meet the {RECALL_FLOOR} floor"
    );
    Ok(())
}

#[test]
#[ignore = "phase 2: real HNSW engine lands (tests/oracle/PARITY.md)"]
fn queries_terminate_and_stay_bounded_on_a_dense_fixture() -> Result<(), HeuremaError> {
    // WHY: greedy-descent termination is observable through the trait only as
    // "query returns"; this fixture's 32 queries over 512 vectors completing
    // are the termination evidence, and the at-most-k bound is asserted per
    // query.
    let vectors = seeded_vectors(0x5EED_0003, 512, 8);
    let mut index = HnswIndex::<u64>::new(HnswConfig::new(8));
    for (id, vector) in &vectors {
        index.insert(*id, vector)?;
    }

    let mut rng = XorShift64::new(0x5EED_0004);
    for _ in 0..32 {
        let query = random_vector(&mut rng, 8);
        let results = index.query(&query, 7)?;
        assert!(results.len() <= 7, "at most k results are returned");
    }
    Ok(())
}

#[test]
#[ignore = "phase 2: real HNSW engine lands (tests/oracle/PARITY.md)"]
fn remove_is_idempotent_and_excludes_the_id_from_results() -> Result<(), HeuremaError> {
    let vectors = seeded_vectors(0x5EED_000A, 32, 8);
    let mut index = HnswIndex::<u64>::new(HnswConfig::new(8));
    for (id, vector) in &vectors {
        index.insert(*id, vector)?;
    }
    assert_eq!(index.len(), 32, "all inserts are counted");

    // WHY: removal is idempotent by contract so row-deletion cleanup can
    // retry safely; removing an absent ID is a successful no-op.
    index.remove(&7)?;
    index.remove(&7)?;
    index.remove(&999)?;
    assert_eq!(
        index.len(),
        31,
        "only the removal of a present ID changes cardinality"
    );

    let query = vectors[7].1.clone();
    let results = index.query(&query, 31)?;
    assert!(
        results.iter().all(|(id, _)| *id != 7),
        "a removed ID never appears in results"
    );

    for id in 0..32_u64 {
        index.remove(&id)?;
    }
    assert!(index.is_empty(), "removing every entry empties the index");
    let results = index.query(&query, 10)?;
    assert!(results.is_empty(), "an emptied index returns no results");
    Ok(())
}
