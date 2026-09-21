//! Persistence backend contract for Heurēma indexes.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::SnapshotFormatSnafu;
use crate::{FtsIndex, HeuremaError, VectorIndex};

/// Current on-disk and in-memory snapshot envelope version.
pub const SNAPSHOT_FORMAT_VERSION: u16 = 1;

/// The index family whose payload an adapter snapshot contains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum SnapshotFamily {
    /// A [`VectorIndex`] payload.
    Vector,
    /// An [`FtsIndex`] payload.
    Fts,
}

/// Versioned adapter payload that forms the individual-index persistence seam.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotEnvelope<T> {
    format_version: u16,
    family: SnapshotFamily,
    payload: T,
}

impl<T> SnapshotEnvelope<T> {
    /// Wrap one index payload using the current format.
    #[must_use]
    pub const fn new(family: SnapshotFamily, payload: T) -> Self {
        Self {
            format_version: SNAPSHOT_FORMAT_VERSION,
            family,
            payload,
        }
    }

    /// Return the payload only when its version and family match the caller.
    ///
    /// Older unversioned bytes are rejected rather than guessed at; a future
    /// migration can add an explicit decoder without making a partial state
    /// look like a valid current index.
    pub fn into_payload(self, expected: SnapshotFamily) -> Result<T, HeuremaError> {
        if self.format_version != SNAPSHOT_FORMAT_VERSION {
            return Err(SnapshotFormatSnafu {
                reason: format!("format version {} is unsupported", self.format_version),
            }
            .build());
        }
        if self.family != expected {
            return Err(SnapshotFormatSnafu {
                reason: format!("expected {expected:?} snapshot, found {:?}", self.family),
            }
            .build());
        }
        Ok(self.payload)
    }
}

/// WHY: Persistence remains outside HNSW and BM25 algorithms so consumers can
/// choose fjall, in-memory, or engine-owned storage without changing indexes.
///
/// WARNING: Phase 1 declared every method here generic over only
/// `I: VectorIndex` / `I: FtsIndex`. Phase 3, landing this trait's first
/// implementations (`atmis`, `thesauros`), adds `Serialize` to
/// the save methods and `DeserializeOwned` to the load methods. This is a
/// signature change, not an addition beside the old one: neither
/// `VectorIndex` nor `FtsIndex` exposes a constructor, so `load_vector_index`
/// could never build a value of a caller-chosen `I` under the original
/// bound, and `Ok` from either Phase 1 load method was structurally
/// unreachable. `DeserializeOwned` supplies the missing construction path;
/// `Serialize` is its mirror on save. The bound lives at the persistence
/// boundary only. `VectorIndex` and `FtsIndex` themselves are unchanged.
pub trait PersistenceBackend {
    /// WHY: Vector indexes need named durable snapshots that can be owned by a
    /// query engine catalog.
    ///
    /// Saving to a `name` that already holds a snapshot replaces it:
    /// last-write-wins, never an error. Catalog owners that need
    /// create-only semantics must check existence before saving.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::Persistence`] when the backend cannot encode
    /// `idx` or fails to write the encoded snapshot.
    fn save_vector_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: VectorIndex + Serialize;

    /// WHY: Query engines need to load a concrete vector index type from their
    /// catalog entry.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::IndexNotFound`] when no snapshot exists under
    /// `name`, [`HeuremaError::SnapshotFormat`] for a decoded envelope with an
    /// unsupported version or wrong family, and [`HeuremaError::Persistence`]
    /// on storage failure or when the stored bytes do not decode as `I`.
    fn load_vector_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: VectorIndex + DeserializeOwned;

    /// WHY: FTS indexes need the same named snapshot lifecycle as vector
    /// indexes so hybrid search storage stays coherent.
    ///
    /// Saving to a `name` that already holds a snapshot replaces it:
    /// last-write-wins, never an error. Catalog owners that need
    /// create-only semantics must check existence before saving.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::Persistence`] when the backend cannot encode
    /// `idx` or fails to write the encoded snapshot.
    fn save_fts_index<I>(&self, name: &str, idx: &I) -> Result<(), HeuremaError>
    where
        I: FtsIndex + Serialize;

    /// WHY: Query engines need to load a concrete FTS index type from their
    /// catalog entry.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::IndexNotFound`] when no snapshot exists under
    /// `name`, [`HeuremaError::SnapshotFormat`] for a decoded envelope with an
    /// unsupported version or wrong family, and [`HeuremaError::Persistence`]
    /// on storage failure or when the stored bytes do not decode as `I`.
    fn load_fts_index<I>(&self, name: &str) -> Result<I, HeuremaError>
    where
        I: FtsIndex + DeserializeOwned;
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests need concise envelope assertions")]
mod tests {
    use super::*;

    #[test]
    fn envelope_accepts_only_the_current_version_and_requested_family() {
        let accepted = SnapshotEnvelope::new(SnapshotFamily::Vector, 7_u64)
            .into_payload(SnapshotFamily::Vector)
            .expect("current vector envelope");
        assert_eq!(accepted, 7);

        let future = SnapshotEnvelope {
            format_version: SNAPSHOT_FORMAT_VERSION + 1,
            family: SnapshotFamily::Vector,
            payload: 7_u64,
        };
        assert!(matches!(
            future.into_payload(SnapshotFamily::Vector),
            Err(HeuremaError::SnapshotFormat { .. })
        ));

        let wrong_family = SnapshotEnvelope::new(SnapshotFamily::Fts, 7_u64);
        assert!(matches!(
            wrong_family.into_payload(SnapshotFamily::Vector),
            Err(HeuremaError::SnapshotFormat { .. })
        ));
    }

    #[test]
    fn unversioned_or_torn_bytes_cannot_decode_as_a_valid_envelope() {
        let unversioned = br#"{\"config\":{},\"nodes\":{}}"#;
        let torn = br#"{\"format_version\":1,\"family\":\"Vector\""#;
        assert!(
            serde_json::from_slice::<SnapshotEnvelope<serde_json::Value>>(unversioned).is_err()
        );
        assert!(serde_json::from_slice::<SnapshotEnvelope<serde_json::Value>>(torn).is_err());
    }
}
