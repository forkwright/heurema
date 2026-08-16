//! Phase 1 BM25 FTS stub implementation.

use std::hash::Hash;
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::HeuremaError;
use crate::error::NotYetImplementedSnafu;
use crate::fts::{FtsConfig, FtsIndex};

const FTS_PHASE_2_FEATURE: &str = "Phase 2: BM25 FTS not yet implemented";

/// WHY: Phase 1 needs a concrete BM25 type so downstream trait bounds compile
/// before the krites FTS implementation is extracted.
///
/// WHY `Serialize` + `Deserialize`: see [`crate::hnsw::HnswIndex`]; the same
/// reasoning applies here (`_id: PhantomData<Id>` carries no bytes, so
/// today's round-tripped state is `config` + `len`).
///
/// WHY `deny_unknown_fields`: see [`crate::hnsw::HnswIndex`] — the same
/// top-level closed-schema reasoning applies here, one level above
/// `FtsConfig`'s own guard.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bm25Index<Id> {
    config: FtsConfig,
    len: usize,
    _id: PhantomData<Id>,
}

impl<Id> Bm25Index<Id> {
    /// WHY: Consumers need to instantiate the Phase 1 API surface in tests and
    /// design spikes without pulling in krites internals.
    #[must_use]
    pub const fn new(config: FtsConfig) -> Self {
        Self {
            config,
            len: 0,
            _id: PhantomData,
        }
    }

    /// WHY: Query engines need to inspect the analyzer configuration they pass
    /// through persistence and planning layers.
    #[must_use]
    pub const fn config(&self) -> &FtsConfig {
        &self.config
    }

    fn not_ready() -> HeuremaError {
        NotYetImplementedSnafu {
            feature: FTS_PHASE_2_FEATURE.to_owned(),
        }
        .build()
    }
}

impl<Id> FtsIndex for Bm25Index<Id>
where
    Id: Ord + Hash + Clone,
{
    type Id = Id;

    fn insert(&mut self, _id: Self::Id, _document: &str) -> Result<(), HeuremaError> {
        Err(Self::not_ready())
    }

    fn query(&self, _query: &str, _k: usize) -> Result<Vec<(Self::Id, f32)>, HeuremaError> {
        Err(Self::not_ready())
    }

    fn remove(&mut self, _id: &Self::Id) -> Result<(), HeuremaError> {
        Err(Self::not_ready())
    }

    fn len(&self) -> usize {
        self.len
    }
}
