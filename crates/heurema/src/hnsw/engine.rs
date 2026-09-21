//! Safe in-memory HNSW graph for the published vector contract.

#[cfg(test)]
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::HeuremaError;
use crate::error::{
    DimensionMismatchSnafu, DistanceNotRepresentableSnafu, InvalidHnswConfigSnafu,
    InvalidVectorSnafu,
};
use crate::hnsw::{HnswConfig, VectorDistance, VectorIndex};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(
    serialize = "Id: Serialize + Ord",
    deserialize = "Id: Ord + Deserialize<'de>"
))]
struct Node<Id> {
    vector: Vec<f32>,
    level: usize,
    neighbours: Vec<BTreeSet<Id>>,
    // A separate reciprocal base-layer cycle. Normal HNSW links can be pruned
    // to their degree limits; the two backbone links cannot, so local mutation
    // never silently disconnects traversal or create a high-degree hub.
    backbone: BTreeSet<Id>,
}

/// A hierarchical navigable small-world graph.
///
/// Nodes retain their vectors because both construction and search compare a
/// query with graph-local candidates. The ordered maps and sets make every
/// equal-distance choice reproducible, including after a snapshot round trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(try_from = "RawHnswIndex<Id>")]
#[serde(bound(
    serialize = "Id: Serialize + Ord",
    deserialize = "Id: Ord + Clone + Deserialize<'de>"
))]
pub struct HnswIndex<Id> {
    config: HnswConfig,
    nodes: BTreeMap<Id, Node<Id>>,
    entry_point: Option<Id>,
    max_level: usize,
    level_state: u64,
    #[cfg(test)]
    #[serde(skip)]
    distance_evaluations: Cell<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(deserialize = "Id: Ord + Clone + Deserialize<'de>"))]
struct RawHnswIndex<Id> {
    config: HnswConfig,
    nodes: BTreeMap<Id, Node<Id>>,
    entry_point: Option<Id>,
    max_level: usize,
    level_state: u64,
}

impl<Id: Ord + Clone> TryFrom<RawHnswIndex<Id>> for HnswIndex<Id> {
    type Error = &'static str;

    fn try_from(raw: RawHnswIndex<Id>) -> Result<Self, Self::Error> {
        validate_config(&raw.config)?;
        if raw.level_state == 0 {
            return Err("HNSW snapshot PRNG state must be non-zero");
        }
        let expected_entry = select_entry(&raw.nodes);
        if raw.entry_point != expected_entry {
            return Err("HNSW snapshot entry point disagrees with graph levels");
        }
        let expected_level = raw.nodes.values().map(|node| node.level).max().unwrap_or(0);
        if raw.max_level != expected_level {
            return Err("HNSW snapshot max level disagrees with graph nodes");
        }
        for (id, node) in &raw.nodes {
            if node.vector.len() != raw.config.dimensions
                || !node.vector.iter().all(|component| component.is_finite())
                || node.level > 16
                || node.neighbours.len() != node.level + 1
            {
                return Err("HNSW snapshot node shape disagrees with configuration");
            }
            for (level, neighbours) in node.neighbours.iter().enumerate() {
                if neighbours.len() > degree_limit(&raw.config, level) {
                    return Err("HNSW snapshot exceeds a normal-link degree limit");
                }
                for neighbour in neighbours {
                    if neighbour == id {
                        return Err("HNSW snapshot has a self-neighbour");
                    }
                    let Some(other) = raw.nodes.get(neighbour) else {
                        return Err("HNSW snapshot has a dangling neighbour");
                    };
                    if other.level < level || !other.neighbours[level].contains(id) {
                        return Err("HNSW snapshot has a non-reciprocal neighbour");
                    }
                }
            }
            for neighbour in &node.backbone {
                if neighbour == id {
                    return Err("HNSW snapshot has a self backbone link");
                }
                let Some(other) = raw.nodes.get(neighbour) else {
                    return Err("HNSW snapshot has a dangling backbone link");
                };
                if !other.backbone.contains(id) {
                    return Err("HNSW snapshot has a non-reciprocal backbone link");
                }
            }
        }
        if !raw.nodes.is_empty() {
            let expected_degree = match raw.nodes.len() {
                1 => 0,
                2 => 1,
                _ => 2,
            };
            if raw
                .nodes
                .values()
                .any(|node| node.backbone.len() != expected_degree)
            {
                return Err("HNSW snapshot backbone does not have its bounded cycle degree");
            }
            let expected_edges = match raw.nodes.len() {
                1 => 0,
                2 => 1,
                count => count,
            };
            let edges = raw
                .nodes
                .values()
                .map(|node| node.backbone.len())
                .sum::<usize>()
                / 2;
            if edges != expected_edges {
                return Err("HNSW snapshot backbone is not a cycle");
            }
            let Some(start) = raw.nodes.keys().next().cloned() else {
                return Err("HNSW snapshot has no backbone root");
            };
            let mut seen = BTreeSet::new();
            let mut pending = vec![start];
            while let Some(id) = pending.pop() {
                if seen.insert(id.clone()) {
                    pending.extend(raw.nodes[&id].backbone.iter().cloned());
                }
            }
            if seen.len() != raw.nodes.len() {
                return Err("HNSW snapshot backbone is disconnected");
            }
        }
        Ok(Self {
            config: raw.config,
            nodes: raw.nodes,
            entry_point: raw.entry_point,
            max_level: raw.max_level,
            level_state: raw.level_state,
            #[cfg(test)]
            distance_evaluations: Cell::new(0),
        })
    }
}

impl<Id> HnswIndex<Id> {
    /// Create an empty graph using the supplied construction configuration.
    #[must_use]
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: BTreeMap::new(),
            entry_point: None,
            max_level: 0,
            // A fixed, persisted seed makes level selection reproducible for
            // a given mutation history without coupling duplicate vectors to
            // a shared level.
            level_state: 0x9e37_79b9_7f4a_7c15,
            #[cfg(test)]
            distance_evaluations: Cell::new(0),
        }
    }

    /// Return the graph construction configuration.
    #[must_use]
    pub const fn config(&self) -> &HnswConfig {
        &self.config
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<(), HeuremaError> {
        validate_config(&self.config).map_err(|reason| {
            InvalidHnswConfigSnafu {
                reason: reason.to_owned(),
            }
            .build()
        })?;
        if vector.len() != self.config.dimensions {
            Err(DimensionMismatchSnafu {
                expected: self.config.dimensions,
                actual: vector.len(),
            }
            .build())
        } else if !vector.iter().all(|component| component.is_finite()) {
            Err(InvalidVectorSnafu {
                reason: "all components must be finite".to_owned(),
            }
            .build())
        } else {
            Ok(())
        }
    }
}

impl<Id> HnswIndex<Id>
where
    Id: Ord + Hash + Clone,
{
    fn distance(&self, left: &[f32], right: &[f32]) -> f64 {
        #[cfg(test)]
        self.distance_evaluations
            .set(self.distance_evaluations.get().saturating_add(1));
        match self.config.distance {
            VectorDistance::L2 => left
                .iter()
                .zip(right)
                .map(|(a, b)| {
                    let difference = f64::from(*a) - f64::from(*b);
                    difference * difference
                })
                .sum(),
            VectorDistance::InnerProduct => -left
                .iter()
                .zip(right)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>(),
            VectorDistance::Cosine => {
                let (dot, left_norm, right_norm) = left.iter().zip(right).fold(
                    (0.0_f64, 0.0_f64, 0.0_f64),
                    |(dot, left_norm, right_norm), (a, b)| {
                        let left = f64::from(*a);
                        let right = f64::from(*b);
                        (
                            dot + left * right,
                            left_norm + left * left,
                            right_norm + right * right,
                        )
                    },
                );
                if left_norm == 0.0 && right_norm == 0.0 {
                    0.0
                } else if left_norm == 0.0 || right_norm == 0.0 {
                    1.0
                } else {
                    1.0 - dot / (left_norm.sqrt() * right_norm.sqrt())
                }
            }
        }
    }

    fn node_distance(&self, vector: &[f32], id: &Id) -> Option<f64> {
        self.nodes
            .get(id)
            .map(|node| self.distance(vector, &node.vector))
    }

    fn compare_ids_by_distance(&self, vector: &[f32], left: &Id, right: &Id) -> std::cmp::Ordering {
        self.node_distance(vector, left)
            .unwrap_or(f64::INFINITY)
            .total_cmp(&self.node_distance(vector, right).unwrap_or(f64::INFINITY))
            .then_with(|| left.cmp(right))
    }

    fn next_level(&mut self) -> usize {
        self.level_state ^= self.level_state << 13;
        self.level_state ^= self.level_state >> 7;
        self.level_state ^= self.level_state << 17;
        let mut state = self.level_state;
        let mut level = 0;
        while level < 16 && state.trailing_zeros() >= 2 {
            level += 1;
            state = state.rotate_right(2);
        }
        level
    }

    fn greedy_at_level(&self, vector: &[f32], mut current: Id, level: usize) -> Id {
        loop {
            let Some(node) = self.nodes.get(&current) else {
                return current;
            };
            let mut next = current.clone();
            for neighbour in &node.neighbours[level] {
                if self
                    .compare_ids_by_distance(vector, neighbour, &next)
                    .is_lt()
                {
                    next = neighbour.clone();
                }
            }
            if level == 0 {
                for neighbour in &node.backbone {
                    if self
                        .compare_ids_by_distance(vector, neighbour, &next)
                        .is_lt()
                    {
                        next = neighbour.clone();
                    }
                }
            }
            if next == current {
                return current;
            }
            current = next;
        }
    }

    fn search_layer(&self, vector: &[f32], entries: Vec<Id>, level: usize, ef: usize) -> Vec<Id> {
        let mut visited = BTreeSet::new();
        let mut frontier = entries;
        let mut results = Vec::new();
        while !frontier.is_empty() {
            frontier.sort_by(|left, right| self.compare_ids_by_distance(vector, left, right));
            let current = frontier.remove(0);
            if !visited.insert(current.clone()) {
                continue;
            }
            if results.len() >= ef
                && let Some(worst) = results.last()
                && self
                    .compare_ids_by_distance(vector, &current, worst)
                    .is_gt()
            {
                break;
            }
            results.push(current.clone());
            results.sort_by(|left, right| self.compare_ids_by_distance(vector, left, right));
            if results.len() > ef {
                results.pop();
            }
            if let Some(node) = self.nodes.get(&current) {
                for neighbour in &node.neighbours[level] {
                    if !visited.contains(neighbour) {
                        frontier.push(neighbour.clone());
                    }
                }
                if level == 0 {
                    for neighbour in &node.backbone {
                        if !visited.contains(neighbour) {
                            frontier.push(neighbour.clone());
                        }
                    }
                }
            }
        }
        results
    }

    fn degree_limit(&self, level: usize) -> usize {
        degree_limit(&self.config, level)
    }

    fn prune(&mut self, id: &Id, level: usize) {
        let Some(vector) = self.nodes.get(id).map(|node| node.vector.clone()) else {
            return;
        };
        let limit = self.degree_limit(level);
        let mut ranked: Vec<Id> = self.nodes[id].neighbours[level].iter().cloned().collect();
        ranked.sort_by(|left, right| self.compare_ids_by_distance(&vector, left, right));
        let retained: BTreeSet<Id> = ranked.into_iter().take(limit).collect();
        let removed: Vec<Id> = self.nodes[id].neighbours[level]
            .difference(&retained)
            .cloned()
            .collect();
        let Some(node) = self.nodes.get_mut(id) else {
            return;
        };
        node.neighbours[level] = retained;
        for other in removed {
            if let Some(node) = self.nodes.get_mut(&other) {
                node.neighbours[level].remove(id);
            }
        }
    }

    fn link(&mut self, left: &Id, right: &Id, level: usize) {
        if left == right {
            return;
        }
        let Some(source) = self.nodes.get_mut(left) else {
            return;
        };
        source.neighbours[level].insert(right.clone());
        let Some(target) = self.nodes.get_mut(right) else {
            return;
        };
        target.neighbours[level].insert(left.clone());
        self.prune(left, level);
        self.prune(right, level);
    }

    fn reselect_entry(&mut self) {
        self.max_level = self
            .nodes
            .values()
            .map(|node| node.level)
            .max()
            .unwrap_or(0);
        self.entry_point = select_entry(&self.nodes);
    }

    fn connect_backbone(&mut self, left: &Id, right: &Id) {
        if left == right {
            return;
        }
        if let Some(node) = self.nodes.get_mut(left) {
            node.backbone.insert(right.clone());
        }
        if let Some(node) = self.nodes.get_mut(right) {
            node.backbone.insert(left.clone());
        }
    }

    fn disconnect_backbone(&mut self, left: &Id, right: &Id) {
        if let Some(node) = self.nodes.get_mut(left) {
            node.backbone.remove(right);
        }
        if let Some(node) = self.nodes.get_mut(right) {
            node.backbone.remove(left);
        }
    }

    fn splice_backbone(&mut self, id: &Id, anchor: &Id) {
        let neighbours: Vec<Id> = self.nodes[anchor].backbone.iter().cloned().collect();
        match neighbours.as_slice() {
            // The first insertion was the singleton cycle. The second forms
            // its single undirected edge; every later insertion splices one
            // deterministic edge, keeping all backbone degrees at most two.
            [] => self.connect_backbone(id, anchor),
            // A two-node undirected cycle has one stored edge. Preserve it
            // while adding the third side of the first true simple cycle.
            [successor] => {
                self.connect_backbone(anchor, id);
                self.connect_backbone(id, successor);
            }
            [successor, ..] => {
                self.disconnect_backbone(anchor, successor);
                self.connect_backbone(anchor, id);
                self.connect_backbone(id, successor);
            }
        }
    }

    fn remove_node(&mut self, id: &Id) {
        let Some(node) = self.nodes.remove(id) else {
            return;
        };
        for (level, neighbours) in node.neighbours.into_iter().enumerate() {
            let neighbours: Vec<Id> = neighbours.into_iter().collect();
            for neighbour in &neighbours {
                if let Some(other) = self.nodes.get_mut(neighbour) {
                    other.neighbours[level].remove(id);
                }
            }
            // Deleting a bridge must not silently partition its former
            // neighbourhood. Reconnect the affected local graph before the
            // ordinary degree pruning selects its closest navigable links.
            for (offset, left) in neighbours.iter().enumerate() {
                for right in neighbours.iter().skip(offset + 1) {
                    if self.nodes.get(left).is_some_and(|node| node.level >= level)
                        && self
                            .nodes
                            .get(right)
                            .is_some_and(|node| node.level >= level)
                    {
                        self.link(left, right, level);
                    }
                }
            }
        }
        // Bypass the removed vertex in the reciprocal cycle. This is O(1): a
        // valid backbone has at most two incident edges at every size.
        let anchors: Vec<Id> = node.backbone.into_iter().collect();
        for anchor in &anchors {
            if let Some(other) = self.nodes.get_mut(anchor) {
                other.backbone.remove(id);
            }
        }
        if let [left, right] = anchors.as_slice() {
            self.connect_backbone(left, right);
        }
        self.reselect_entry();
    }
}

impl<Id> VectorIndex for HnswIndex<Id>
where
    Id: Ord + Hash + Clone,
{
    type Id = Id;

    fn insert(&mut self, id: Self::Id, vector: &[f32]) -> Result<(), HeuremaError> {
        self.validate_vector(vector)?;
        self.remove_node(&id);
        let level = self.next_level();
        // `remove_node` reanchors a replacement's former backbone before we
        // select this entry, so it is always a surviving graph vertex.
        let surviving_entry = self.entry_point.clone();
        self.nodes.insert(
            id.clone(),
            Node {
                vector: vector.to_vec(),
                level,
                neighbours: (0..=level).map(|_| BTreeSet::new()).collect(),
                backbone: BTreeSet::new(),
            },
        );
        let Some(mut entry) = surviving_entry else {
            self.reselect_entry();
            return Ok(());
        };
        // Normal links are degree-pruned HNSW shortcuts. The separate cycle
        // edge is never pruned and splices locally, preserving a bounded base
        // traversal through insertion, replacement, and removal.
        self.splice_backbone(&id, &entry);
        for current_level in ((level + 1)..=self.max_level).rev() {
            entry = self.greedy_at_level(vector, entry, current_level);
        }
        for current_level in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(
                vector,
                vec![entry.clone()],
                current_level,
                self.config
                    .ef_construction
                    .max(self.config.m_neighbours)
                    .max(1),
            );
            let count = self.degree_limit(current_level);
            for candidate in candidates.into_iter().take(count) {
                self.link(&id, &candidate, current_level);
            }
            entry = self.greedy_at_level(vector, entry, current_level);
        }
        self.reselect_entry();
        Ok(())
    }

    fn query(&self, vector: &[f32], k: usize) -> Result<Vec<(Self::Id, f32)>, HeuremaError> {
        self.validate_vector(vector)?;
        if k == 0 || self.nodes.is_empty() {
            return Ok(Vec::new());
        }
        let Some(mut entry) = self.entry_point.clone() else {
            return Ok(Vec::new());
        };
        for level in (1..=self.max_level).rev() {
            entry = self.greedy_at_level(vector, entry, level);
        }
        let candidates = self.search_layer(vector, vec![entry], 0, self.query_beam(k));
        candidates
            .into_iter()
            .take(k)
            .filter_map(|id| {
                self.node_distance(vector, &id)
                    .map(|distance| (id, distance))
            })
            .map(|(id, distance)| {
                let score = distance as f32;
                if score.is_finite() {
                    Ok((id, score))
                } else {
                    Err(DistanceNotRepresentableSnafu {
                        reason: format!("{distance} cannot be returned by VectorIndex::query"),
                    }
                    .build())
                }
            })
            .collect()
    }

    fn remove(&mut self, id: &Self::Id) -> Result<(), HeuremaError> {
        self.remove_node(id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl<Id> HnswIndex<Id>
where
    Id: Ord + Hash + Clone,
{
    // `ef_construction` controls insertion only. The trait has no query-ef
    // parameter, so search derives a stable bounded beam from k and the
    // graph degree instead of silently reusing construction policy.
    fn query_beam(&self, k: usize) -> usize {
        // The public configuration deliberately exposes no ef-search knob.
        // Eight adjacency widths retains a bounded graph walk while giving
        // the default M=16 graph enough frontier to meet the pinned recall
        // fixture without borrowing `ef_construction` as query policy.
        k.max(self.config.m_neighbours.saturating_mul(8)).max(1)
    }

    #[cfg(test)]
    fn reset_distance_evaluations(&self) {
        self.distance_evaluations.set(0);
    }

    #[cfg(test)]
    fn distance_evaluations(&self) -> usize {
        self.distance_evaluations.get()
    }
}

fn select_entry<Id: Ord + Clone>(nodes: &BTreeMap<Id, Node<Id>>) -> Option<Id> {
    let max_level = nodes.values().map(|node| node.level).max().unwrap_or(0);
    nodes
        .iter()
        .filter(|(_, node)| node.level == max_level)
        .map(|(id, _)| id.clone())
        .next()
}

fn validate_config(config: &HnswConfig) -> Result<(), &'static str> {
    if config.dimensions == 0 {
        Err("dimensions must be non-zero")
    } else if config.m_neighbours == 0 {
        Err("m_neighbours must be non-zero")
    } else if config.ef_construction == 0 {
        Err("ef_construction must be non-zero")
    } else {
        Ok(())
    }
}

fn degree_limit(config: &HnswConfig, level: usize) -> usize {
    if level == 0 {
        config.m_neighbours.saturating_mul(2)
    } else {
        config.m_neighbours
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests need concise fixture failures")]
mod tests {
    use super::*;

    fn fixture_vector(id: u64) -> Vec<f32> {
        (0..4)
            .map(|axis| ((id.wrapping_mul(17) + axis * 23) % 97) as f32 / 97.0)
            .collect()
    }

    fn exact_top(index: &HnswIndex<u64>, query: &[f32], k: usize) -> Vec<u64> {
        let mut scored: Vec<(u64, f64)> = index
            .nodes
            .iter()
            .map(|(id, node)| (*id, index.distance(query, &node.vector)))
            .collect();
        scored.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        scored.into_iter().take(k).map(|(id, _)| id).collect()
    }

    fn assert_base_reachable(index: &HnswIndex<u64>) {
        let mut seen = BTreeSet::new();
        let mut pending = vec![index.entry_point.expect("non-empty graph")];
        while let Some(id) = pending.pop() {
            if !seen.insert(id) {
                continue;
            }
            pending.extend(index.nodes[&id].neighbours[0].iter().copied());
            pending.extend(index.nodes[&id].backbone.iter().copied());
        }
        assert_eq!(seen.len(), index.len(), "base graph must remain navigable");
    }

    #[test]
    fn base_graph_is_reachable_from_the_entry_point() {
        let mut index = HnswIndex::<u64>::new(HnswConfig::new(4));
        for id in 0..128_u64 {
            index.insert(id, &fixture_vector(id)).expect("valid insert");
        }
        assert_base_reachable(&index);
        assert!(
            index
                .nodes
                .values()
                .all(|node| node.neighbours[0].len() <= 32),
            "normal HNSW links must remain within their base 2m degree limit"
        );
        let mut levels = [0_usize; 4];
        for node in index.nodes.values() {
            levels[node.level.min(3)] += 1;
        }
        assert!(
            levels[0] > levels[1] && levels[1] > levels[2],
            "levels follow the persisted geometric PRNG distribution"
        );
    }

    #[test]
    fn replacement_and_entry_removal_repair_graph_and_preserve_recall() {
        let mut index = HnswIndex::<u64>::new(HnswConfig::new(4));
        for id in 0..96_u64 {
            index.insert(id, &fixture_vector(id)).expect("valid insert");
        }
        let former_entry = index.entry_point.expect("non-empty graph");
        index.remove(&former_entry).expect("idempotent removal");
        index
            .insert(40, &[0.03, 0.11, 0.29, 0.47])
            .expect("replacement");
        assert_base_reachable(&index);
        for query_id in [3_u64, 17, 40, 71] {
            let query = fixture_vector(query_id);
            let expected = exact_top(&index, &query, 5);
            let actual: Vec<u64> = index
                .query(&query, 5)
                .expect("valid query")
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let recall = actual.iter().filter(|id| expected.contains(id)).count() as f32 / 5.0;
            assert!(
                recall >= 0.8,
                "repair must retain useful graph recall after deletion/replacement"
            );
        }
    }

    #[test]
    fn replacing_the_current_entry_reanchors_before_graph_search_and_round_trip() {
        let mut index = HnswIndex::<u64>::new(HnswConfig::new(4));
        for id in 0..96_u64 {
            index.insert(id, &fixture_vector(id)).expect("valid insert");
        }
        let entry = index.entry_point.expect("populated graph has an entry");
        index
            .insert(entry, &[0.07, 0.19, 0.31, 0.43])
            .expect("entry replacement reconnects through a surviving entry");
        assert_base_reachable(&index);
        assert!(
            index.nodes.values().all(|node| node.backbone.len() == 2),
            "replacement must retain a degree-two cycle: {:?}",
            index
                .nodes
                .iter()
                .map(|(id, node)| (*id, node.backbone.len()))
                .collect::<Vec<_>>()
        );
        let restored: HnswIndex<u64> =
            serde_json::from_slice(&serde_json::to_vec(&index).expect("serializable reanchor"))
                .expect("reanchored snapshot reopens");
        assert_base_reachable(&restored);
        assert_eq!(
            index
                .query(&[0.07, 0.19, 0.31, 0.43], 5)
                .expect("original remains queryable"),
            restored
                .query(&[0.07, 0.19, 0.31, 0.43], 5)
                .expect("restored graph remains queryable"),
        );
    }

    #[test]
    fn deleting_a_backbone_cycle_vertex_rejoins_its_two_local_neighbours() {
        let mut index = HnswIndex::<u64>::new(HnswConfig::new(2));
        for id in 0..4 {
            index.nodes.insert(
                id,
                Node {
                    vector: vec![id as f32, 0.0],
                    level: 0,
                    neighbours: vec![BTreeSet::new()],
                    backbone: BTreeSet::new(),
                },
            );
        }
        index.connect_backbone(&0, &1);
        index.connect_backbone(&1, &2);
        index.connect_backbone(&2, &3);
        index.connect_backbone(&3, &0);
        index.reselect_entry();
        index.remove(&0).expect("removal is valid");
        assert_base_reachable(&index);
        let result_ids: Vec<u64> = index
            .query(&[3.0, 0.0], 3)
            .expect("backbone is part of graph traversal")
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(result_ids.first(), Some(&3));
    }

    #[test]
    fn duplicate_vectors_keep_fixed_k_query_distance_work_bounded_as_corpus_grows() {
        fn build(count: u64) -> HnswIndex<u64> {
            let mut config = HnswConfig::new(4);
            config.m_neighbours = 2;
            config.ef_construction = 8;
            let mut index = HnswIndex::new(config);
            for id in 0..count {
                index
                    .insert(id, &[1.0, 1.0, 1.0, 1.0])
                    .expect("duplicate finite vector");
            }
            index
        }

        let small = build(32);
        let large = build(512);
        assert!(
            large.nodes.values().all(|node| node.backbone.len() <= 2),
            "the persistent guard must never form an entry hub"
        );
        let small_entry = small.entry_point;
        let large_entry = large.entry_point;
        small.reset_distance_evaluations();
        small.query(&[1.0, 1.0, 1.0, 1.0], 4).expect("query");
        let small_work = small.distance_evaluations();
        large.reset_distance_evaluations();
        large.query(&[1.0, 1.0, 1.0, 1.0], 4).expect("query");
        let large_work = large.distance_evaluations();
        assert_eq!(
            small.entry_point, small_entry,
            "query does not mutate its entry"
        );
        assert_eq!(
            large.entry_point, large_entry,
            "query does not mutate its entry"
        );
        assert!(
            large_work <= 4_096,
            "fixed M/k query must inspect a bounded graph frontier, got {large_work} distances"
        );
        assert!(
            large_work <= small_work.saturating_mul(5),
            "16x duplicate corpus must retain sublinear fixed-M/k query work: {small_work} -> {large_work}"
        );
    }

    #[test]
    fn invalid_vectors_and_snapshots_are_rejected_before_use() {
        let mut index = HnswIndex::<u64>::new(HnswConfig::new(2));
        assert!(matches!(
            index.insert(1, &[f32::NAN, 0.0]),
            Err(HeuremaError::InvalidVector { .. })
        ));
        assert!(
            index.is_empty(),
            "non-finite insert cannot mutate graph state"
        );
        index.insert(1, &[0.0, 1.0]).expect("finite vector");
        let mut value = serde_json::to_value(&index).expect("serializable graph");
        value["entry_point"] = serde_json::json!(999_u64);
        assert!(
            serde_json::from_value::<HnswIndex<u64>>(value).is_err(),
            "dangling entry point must not load"
        );
    }

    #[test]
    fn finite_extreme_vectors_use_f64_ranking_and_reject_unrepresentable_scores() {
        let mut index = HnswIndex::<u64>::new(HnswConfig::new(2));
        index
            .insert(1, &[f32::MAX, f32::MAX])
            .expect("finite coordinates remain indexable");
        index.insert(2, &[0.0, 0.0]).expect("finite coordinates");
        assert!(matches!(
            index.query(&[0.0, 0.0], 2),
            Err(HeuremaError::DistanceNotRepresentable { .. })
        ));

        let mut cosine_config = HnswConfig::new(2);
        cosine_config.distance = VectorDistance::Cosine;
        let mut cosine = HnswIndex::<u64>::new(cosine_config);
        cosine
            .insert(1, &[f32::MAX, f32::MAX])
            .expect("finite coordinates remain indexable");
        cosine
            .insert(2, &[f32::MAX, -f32::MAX])
            .expect("finite coordinates remain indexable");
        let scores = cosine
            .query(&[f32::MAX, f32::MAX], 2)
            .expect("f64 cosine arithmetic remains finite");
        assert!(scores.iter().all(|(_, score)| score.is_finite()));
    }

    #[test]
    fn snapshot_rejects_zero_prng_state_disconnected_backbone_and_overdegree_links() {
        let mut config = HnswConfig::new(2);
        config.m_neighbours = 1;
        let mut index = HnswIndex::<u64>::new(config);
        for id in 0..6 {
            index.insert(id, &[id as f32, 0.0]).expect("valid insert");
        }
        let mut zero_state = serde_json::to_value(&index).expect("serializable graph");
        zero_state["level_state"] = serde_json::json!(0_u64);
        assert!(serde_json::from_value::<HnswIndex<u64>>(zero_state).is_err());

        let mut disconnected = serde_json::to_value(&index).expect("serializable graph");
        disconnected["nodes"]["0"]["backbone"] = serde_json::json!([]);
        assert!(serde_json::from_value::<HnswIndex<u64>>(disconnected).is_err());

        let mut overdegree = serde_json::to_value(&index).expect("serializable graph");
        overdegree["nodes"]["0"]["neighbours"][0] = serde_json::json!([1, 2, 3, 4, 5]);
        for id in 1..6 {
            overdegree["nodes"][id.to_string()]["neighbours"][0]
                .as_array_mut()
                .expect("neighbour array")
                .push(serde_json::json!(0));
        }
        assert!(serde_json::from_value::<HnswIndex<u64>>(overdegree).is_err());
    }
}
