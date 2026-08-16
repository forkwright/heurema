//! Phase 1 HNSW stub implementation.

use std::hash::Hash;
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::HeuremaError;
use crate::error::{DimensionMismatchSnafu, NotYetImplementedSnafu};
use crate::hnsw::{HnswConfig, VectorIndex};

const HNSW_PHASE_2_FEATURE: &str = "Phase 2: HNSW not yet implemented";

/// WHY: Phase 1 needs a concrete HNSW type so downstream trait bounds compile
/// before Phase 2's fresh graph implementation lands.
///
/// WHY `Serialize` + `Deserialize`: a `PersistenceBackend` adapter needs to
/// encode and reconstruct a whole `HnswIndex<Id>` snapshot; see
/// `persistence.rs`. `_id: PhantomData<Id>` carries no bytes either way, so
/// the round-tripped state today is `config` + `len`, matching Phase 1's
/// stub reality where no `insert` has ever mutated `len` past zero.
///
/// WHY `deny_unknown_fields`: `save_vector_index`/`load_vector_index`
/// (`persistence.rs`) encode and decode this type itself, not only its
/// nested `config` field — `HnswConfig` already closes its own schema for
/// exactly this reason (`hnsw.rs`), but that guard stops at the field
/// boundary. A snapshot from a different concrete `VectorIndex` type that
/// happens to be a field superset of `HnswIndex` would otherwise decode
/// here silently instead of failing loudly, the same wrong-shape-under-a-
/// shared-name hazard one level up (`persistence_schema_closed.rs`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HnswIndex<Id> {
    config: HnswConfig,
    len: usize,
    _id: PhantomData<Id>,
}

impl<Id> HnswIndex<Id> {
    /// WHY: Consumers need to instantiate the Phase 1 API surface in tests and
    /// design spikes without pulling in krites internals.
    #[must_use]
    pub const fn new(config: HnswConfig) -> Self {
        Self {
            config,
            len: 0,
            _id: PhantomData,
        }
    }

    /// WHY: Query engines need to inspect the index configuration they passed
    /// through persistence and planning layers.
    #[must_use]
    pub const fn config(&self) -> &HnswConfig {
        &self.config
    }

    fn not_ready() -> HeuremaError {
        NotYetImplementedSnafu {
            feature: HNSW_PHASE_2_FEATURE.to_owned(),
        }
        .build()
    }
}

impl<Id> VectorIndex for HnswIndex<Id>
where
    Id: Ord + Hash + Clone,
{
    type Id = Id;

    fn insert(&mut self, _id: Self::Id, vector: &[f32]) -> Result<(), HeuremaError> {
        if vector.len() != self.config.dimensions {
            return Err(DimensionMismatchSnafu {
                expected: self.config.dimensions,
                actual: vector.len(),
            }
            .build());
        }
        Err(Self::not_ready())
    }

    fn query(&self, vector: &[f32], _k: usize) -> Result<Vec<(Self::Id, f32)>, HeuremaError> {
        if vector.len() != self.config.dimensions {
            return Err(DimensionMismatchSnafu {
                expected: self.config.dimensions,
                actual: vector.len(),
            }
            .build());
        }
        Err(Self::not_ready())
    }

    fn remove(&mut self, _id: &Self::Id) -> Result<(), HeuremaError> {
        Err(Self::not_ready())
    }

    fn len(&self) -> usize {
        self.len
    }
}
