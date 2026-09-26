//! A published version as a reader sees it: its member table and queries
//! whose every hit carries the member's identity and provenance.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::encoding::{VersionEngine, VersionPayload, corrupt};
use super::identity::{IndexIdentity, IndexVersion};
use super::member::{MemberIdentity, ProvenanceReference};
use super::operation::IndexConfig;
use crate::error::FamilyMismatchSnafu;
use crate::{FtsIndex, HeuremaError, SnapshotFamily, VectorIndex};

/// One member's entry in a published version's member table.
///
/// WHY `introduced` and `supersedes`: an entry is written by exactly one
/// operation, the one that published `introduced`. When that operation
/// replaced an earlier entry for the same identity, `supersedes` names the
/// version that introduced the earlier one, which stays readable in the
/// versions before `introduced`, so a replaced fact is linked, never
/// silently overwritten.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(serialize = "P: Serialize", deserialize = "P: ProvenanceReference")
)]
#[non_exhaustive]
pub struct MemberEntry<P> {
    /// Where the member's content came from, as the consumer recorded it.
    pub provenance: P,
    /// The version whose operation wrote this entry.
    pub introduced: IndexVersion,
    /// The version that introduced the entry this one replaced; `None` when
    /// the identity was not a member before.
    pub supersedes: Option<IndexVersion>,
}

/// One query result: the member's identity, its score, its provenance, and
/// the version it was read from.
///
/// WHY the provenance and version travel with the hit: a hit without them
/// could not be traced back to its source, and heurēma publishes no version
/// whose members lack either.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct IndexHit<M, P> {
    /// The member's identity.
    pub id: M,
    /// The engine's score: for a vector index a distance (smaller is
    /// closer), for a full-text index a BM25 score (larger is better). The
    /// engine's ranking contract ([`VectorIndex::query`],
    /// [`FtsIndex::query`]) orders the hits.
    pub score: f32,
    /// The member's provenance in the version the hit was read from.
    pub provenance: P,
    /// The published version the hit was read from.
    pub version: IndexVersion,
}

/// One published version of a named index, read whole: its configuration,
/// its engine, and its member table.
///
/// A reader gets one only through
/// [`IndexLifecycle::index`](super::IndexLifecycle::index), which resolves
/// the index's head and then reads the immutable payload the head names, so
/// the value is exactly one published version and never a staged one.
#[derive(Debug, Clone)]
pub struct PublishedIndex<M, P> {
    identity: IndexIdentity,
    version: IndexVersion,
    config: IndexConfig,
    engine: VersionEngine<M>,
    members: BTreeMap<M, MemberEntry<P>>,
}

impl<M, P> PublishedIndex<M, P> {
    /// A reader's view of a decoded payload.
    pub(super) fn from_payload(payload: VersionPayload<M, P>) -> Self {
        Self {
            identity: payload.index,
            version: payload.version,
            config: payload.config,
            engine: payload.engine,
            members: payload.members,
        }
    }

    /// The index this version belongs to.
    #[must_use]
    pub const fn identity(&self) -> &IndexIdentity {
        &self.identity
    }

    /// The published version this value holds.
    #[must_use]
    pub const fn version(&self) -> IndexVersion {
        self.version
    }

    /// The configuration the version was built under.
    #[must_use]
    pub const fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// The number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the version has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Every member and its entry, in ascending identity order.
    pub fn members(&self) -> impl Iterator<Item = (&M, &MemberEntry<P>)> {
        self.members.iter()
    }
}

impl<M: MemberIdentity, P: ProvenanceReference> PublishedIndex<M, P> {
    /// The entry of member `id`, or `None` when `id` is not a member.
    #[must_use]
    pub fn member(&self, id: &M) -> Option<&MemberEntry<P>> {
        self.members.get(id)
    }

    /// The `k` members nearest `vector`, under the engine's ranking contract
    /// ([`VectorIndex::query`]), each with its provenance.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::FamilyMismatch`] on a full-text index, and otherwise
    /// the engine's own query errors ([`HeuremaError::DimensionMismatch`],
    /// [`HeuremaError::InvalidVector`],
    /// [`HeuremaError::DistanceNotRepresentable`]).
    pub fn query_vector(
        &self,
        vector: &[f32],
        k: usize,
    ) -> Result<Vec<IndexHit<M, P>>, HeuremaError> {
        match &self.engine {
            VersionEngine::Vector(engine) => self.hits(engine.query(vector, k)?),
            VersionEngine::Fts(_) => FamilyMismatchSnafu {
                expected: SnapshotFamily::Fts,
                actual: SnapshotFamily::Vector,
            }
            .fail(),
        }
    }

    /// The `k` members that best match `text`, under the engine's ranking
    /// contract ([`FtsIndex::query`]), each with its provenance.
    ///
    /// # Errors
    ///
    /// [`HeuremaError::FamilyMismatch`] on a vector index, and otherwise the
    /// engine's own query errors.
    pub fn query_text(&self, text: &str, k: usize) -> Result<Vec<IndexHit<M, P>>, HeuremaError> {
        match &self.engine {
            VersionEngine::Fts(engine) => self.hits(engine.query(text, k)?),
            VersionEngine::Vector(_) => FamilyMismatchSnafu {
                expected: SnapshotFamily::Vector,
                actual: SnapshotFamily::Fts,
            }
            .fail(),
        }
    }

    /// Pairs each ranked identity with its entry's provenance.
    ///
    /// INVARIANT: the payload decoder refused any version whose engine holds
    /// a member its table does not name, so every ranked identity has an
    /// entry; a miss is reported as corruption rather than dropped.
    fn hits(&self, ranked: Vec<(M, f32)>) -> Result<Vec<IndexHit<M, P>>, HeuremaError> {
        ranked
            .into_iter()
            .map(|(id, score)| {
                let Some(entry) = self.members.get(&id) else {
                    return Err(corrupt(format!(
                        "version {} of index {} ranked a member its table does not name",
                        self.version, self.identity
                    )));
                };
                Ok(IndexHit {
                    provenance: entry.provenance.clone(),
                    id,
                    score,
                    version: self.version,
                })
            })
            .collect()
    }
}
