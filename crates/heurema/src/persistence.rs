//! Persistence backend contract for Heurēma indexes.

use serde::de::{DeserializeOwned, IgnoredAny};
use serde::{Deserialize, Serialize};

use crate::error::{SnapshotFormatSnafu, UnsupportedSnapshotVersionSnafu};
use crate::{FtsIndex, HeuremaError, PersistenceSource, VectorIndex};

/// Current on-disk and in-memory snapshot envelope version.
pub const SNAPSHOT_FORMAT_VERSION: u16 = 1;

/// The index family whose payload an adapter snapshot contains.
///
/// The same discriminant names a lifecycle index's family
/// ([`IndexConfig::family`](crate::IndexConfig::family)), and so which
/// [`MemberContent`](crate::MemberContent) the index accepts.
///
/// WHY `#[non_exhaustive]`: a new index family must be addable without
/// breaking every consumer that matches on this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotHeader {
    format_version: u16,
    family: SnapshotFamily,
    // Deserialize this as ignored data so format policy is decided before a
    // caller-chosen engine type observes or validates its payload.
    #[serde(rename = "payload")]
    _payload: IgnoredAny,
}

fn validate_header(
    format_version: u16,
    family: SnapshotFamily,
    expected: SnapshotFamily,
) -> Result<(), HeuremaError> {
    if format_version != SNAPSHOT_FORMAT_VERSION {
        return Err(UnsupportedSnapshotVersionSnafu {
            found: format_version,
            supported: SNAPSHOT_FORMAT_VERSION,
        }
        .build());
    }
    if family != expected {
        return Err(SnapshotFormatSnafu {
            reason: format!("expected {expected:?} snapshot, found {family:?}"),
        }
        .build());
    }
    Ok(())
}

#[track_caller]
fn decode_error(source: serde_json::Error) -> HeuremaError {
    HeuremaError::CorruptSnapshot {
        source: PersistenceSource::new(source),
        location: std::panic::Location::caller(),
    }
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
        validate_header(self.format_version, self.family, expected)?;
        Ok(self.payload)
    }
}

/// Decode one adapter snapshot after validating its format header.
///
/// The header deliberately uses [`IgnoredAny`] for `payload`, so an
/// unsupported version ([`HeuremaError::UnsupportedSnapshotVersion`]) or a
/// wrong family ([`HeuremaError::SnapshotFormat`]) is refused before a
/// concrete index deserializer receives an incompatible payload. Bytes that
/// do not decode, including unversioned, torn, or invariant-violating
/// snapshots, are [`HeuremaError::CorruptSnapshot`].
pub fn decode_snapshot_payload<T>(bytes: &[u8], expected: SnapshotFamily) -> Result<T, HeuremaError>
where
    T: DeserializeOwned,
{
    let header: SnapshotHeader = serde_json::from_slice(bytes).map_err(decode_error)?;
    validate_header(header.format_version, header.family, expected)?;
    serde_json::from_slice::<SnapshotEnvelope<T>>(bytes)
        .map_err(decode_error)?
        .into_payload(expected)
}

/// WHY: Persistence remains outside HNSW and BM25 algorithms so consumers can
/// choose fjall, in-memory, or engine-owned storage without changing indexes.
///
/// WARNING: The trait originally declared every method here generic over
/// only `I: VectorIndex` / `I: FtsIndex`. Its first implementations
/// (`atmis`, `thesauros`) added `Serialize` to
/// the save methods and `DeserializeOwned` to the load methods. This is a
/// signature change, not an addition beside the old one: neither
/// `VectorIndex` nor `FtsIndex` exposes a constructor, so `load_vector_index`
/// could never build a value of a caller-chosen `I` under the original
/// bound, and `Ok` from either original load method was structurally
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
    /// `name`, [`HeuremaError::UnsupportedSnapshotVersion`] for an envelope
    /// whose format version this build does not read,
    /// [`HeuremaError::SnapshotFormat`] for an envelope of the wrong family,
    /// [`HeuremaError::CorruptSnapshot`] when the stored bytes do not decode
    /// as `I`, and [`HeuremaError::Persistence`] on storage failure.
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
    /// `name`, [`HeuremaError::UnsupportedSnapshotVersion`] for an envelope
    /// whose format version this build does not read,
    /// [`HeuremaError::SnapshotFormat`] for an envelope of the wrong family,
    /// [`HeuremaError::CorruptSnapshot`] when the stored bytes do not decode
    /// as `I`, and [`HeuremaError::Persistence`] on storage failure.
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
        let refusal = future
            .into_payload(SnapshotFamily::Vector)
            .expect_err("a future format version is refused");
        assert!(
            matches!(
                refusal,
                HeuremaError::UnsupportedSnapshotVersion { found, supported, .. }
                    if found == SNAPSHOT_FORMAT_VERSION + 1 && supported == SNAPSHOT_FORMAT_VERSION
            ),
            "a future version is unsupported, not corrupt: {refusal:?}"
        );
        assert_eq!(
            refusal.category(),
            crate::ErrorCategory::Unsupported,
            "bytes from a newer build must not be classed as corrupt"
        );

        let wrong_family = SnapshotEnvelope::new(SnapshotFamily::Fts, 7_u64);
        let refusal = wrong_family
            .into_payload(SnapshotFamily::Vector)
            .expect_err("a wrong family is refused");
        assert!(
            matches!(refusal, HeuremaError::SnapshotFormat { .. }),
            "a wrong family is a format refusal: {refusal:?}"
        );
        assert_eq!(
            refusal.category(),
            crate::ErrorCategory::Corrupt,
            "wrong family"
        );
    }

    #[test]
    fn unversioned_or_torn_bytes_cannot_decode_as_a_valid_envelope() {
        let unversioned = br#"{"config":{},"nodes":{}}"#;
        let torn = br#"{"format_version":1,"family":"Vector""#;
        for (label, bytes) in [("unversioned", &unversioned[..]), ("torn", &torn[..])] {
            let refusal =
                decode_snapshot_payload::<serde_json::Value>(bytes, SnapshotFamily::Vector)
                    .expect_err("undecodable bytes are refused");
            assert!(
                matches!(refusal, HeuremaError::CorruptSnapshot { .. }),
                "{label} bytes are corrupt, not a backend failure: {refusal:?}"
            );
            assert_eq!(refusal.category(), crate::ErrorCategory::Corrupt, "{label}");
        }
    }
}
