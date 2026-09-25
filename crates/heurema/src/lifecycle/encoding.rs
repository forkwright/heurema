//! The byte encodings of the four records the lifecycle stores through a
//! [`LifecycleBackend`](super::LifecycleBackend): heads, version payloads,
//! staging markers, and operation records.
//!
//! Every record is a `serde_json` object that carries `format_version`
//! ([`LIFECYCLE_FORMAT_VERSION`]) and refuses unknown fields. Decoding reads
//! the format version first, so a record written by a newer build is
//! reported as [`HeuremaError::UnsupportedSnapshotVersion`] and left alone
//! instead of being taken for damage. Bytes that do not decode, or that
//! decode into a record contradicting the key it was stored under or its own
//! invariants, are [`HeuremaError::CorruptSnapshot`], never
//! [`HeuremaError::Persistence`]: the backend returned them intact, and a
//! retry would read the same bytes.
//!
//! WHY private: a consumer reads these records through the lifecycle's
//! typed API, never as bytes, so the encodings can change with the format
//! version without breaking a public type.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use snafu::IntoError;

use super::apply::MemberChange;
use super::backend::LIFECYCLE_FORMAT_VERSION;
use super::identity::{IndexIdentity, IndexVersion, OperationIdentity, OperationKey};
use super::member::{MemberContent, MemberIdentity, ProvenanceReference, RetentionReference};
use super::operation::{IndexConfig, LifecycleTransition};
use super::published::MemberEntry;
use super::record::{IndexRecord, IndexState};
use crate::error::{
    CorruptSnapshotSnafu, FamilyMismatchSnafu, UnencodableOperationSnafu,
    UnsupportedSnapshotVersionSnafu,
};
use crate::{
    Bm25Index, FtsIndex, HeuremaError, HnswIndex, PersistenceSource, SnapshotFamily, VectorIndex,
};

/// The first field every lifecycle record carries, read before the rest.
///
/// WHY no `deny_unknown_fields`: this reads only the version, so a record
/// from a newer build, whose other fields this build does not know, still
/// reports its version instead of failing as corrupt.
#[derive(Deserialize)]
struct FormatHeader {
    format_version: u16,
}

/// The stored form of an index's head: its [`IndexRecord`].
#[derive(Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(serialize = "R: Serialize", deserialize = "R: RetentionReference")
)]
struct HeadBytes<R> {
    format_version: u16,
    record: IndexRecord<R>,
}

/// The engine of one version, of the family its configuration names.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "M: Serialize + Ord", deserialize = "M: MemberIdentity"))]
pub(super) enum VersionEngine<M> {
    /// An HNSW vector engine.
    Vector(HnswIndex<M>),
    /// A BM25 full-text engine.
    Fts(Bm25Index<M>),
}

impl<M: MemberIdentity> VersionEngine<M> {
    /// An engine with no members, built under `config`.
    pub(super) fn empty(config: &IndexConfig) -> Self {
        match config {
            IndexConfig::Vector(hnsw) => Self::Vector(HnswIndex::new(hnsw.clone())),
            IndexConfig::Fts(fts) => Self::Fts(Bm25Index::new(fts.clone())),
        }
    }

    /// The family this engine indexes.
    pub(super) const fn family(&self) -> SnapshotFamily {
        match self {
            Self::Vector(_) => SnapshotFamily::Vector,
            Self::Fts(_) => SnapshotFamily::Fts,
        }
    }

    /// Whether this engine was built under exactly `config`.
    fn is_built_under(&self, config: &IndexConfig) -> bool {
        match (self, config) {
            (Self::Vector(engine), IndexConfig::Vector(hnsw)) => engine.config() == hnsw,
            (Self::Fts(engine), IndexConfig::Fts(fts)) => engine.config() == fts,
            (Self::Vector(_), IndexConfig::Fts(_)) | (Self::Fts(_), IndexConfig::Vector(_)) => {
                false
            }
        }
    }

    /// Whether this engine holds exactly the members `table` names.
    ///
    /// INVARIANT: both sides iterate in ascending `M` order, so element-wise
    /// equality is set equality.
    fn holds_exactly<P>(&self, table: &BTreeMap<M, MemberEntry<P>>) -> bool {
        match self {
            Self::Vector(engine) => engine.member_ids().eq(table.keys()),
            Self::Fts(engine) => engine.member_ids().eq(table.keys()),
        }
    }

    /// Insert or replace `id` with `content`, through the engine's own
    /// checks.
    pub(super) fn insert(&mut self, id: M, content: &MemberContent) -> Result<(), HeuremaError> {
        match (self, content) {
            (Self::Vector(engine), MemberContent::Vector(vector)) => {
                VectorIndex::insert(engine, id, vector)
            }
            (Self::Fts(engine), MemberContent::Document(document)) => {
                FtsIndex::insert(engine, id, document)
            }
            (engine, content) => FamilyMismatchSnafu {
                expected: engine.family(),
                actual: content.family(),
            }
            .fail(),
        }
    }

    /// Remove `id`; an absent identity is a no-op.
    pub(super) fn remove(&mut self, id: &M) -> Result<(), HeuremaError> {
        match self {
            Self::Vector(engine) => VectorIndex::remove(engine, id),
            Self::Fts(engine) => FtsIndex::remove(engine, id),
        }
    }
}

/// One version of a named index: its engine and its member table, which
/// maps every member's identity to its provenance.
///
/// Constructed only by [`VersionPayload::new`] or decoded through
/// [`RawVersionPayload`]'s `TryFrom`, which refuses a payload whose engine
/// and member table disagree. A version's members therefore never lack
/// their identities or their provenance.
#[derive(Debug, Clone, Serialize)]
#[serde(bound(serialize = "M: Serialize + Ord, P: Serialize"))]
pub(super) struct VersionPayload<M, P> {
    format_version: u16,
    /// The index this version belongs to.
    pub(super) index: IndexIdentity,
    /// This version's number.
    pub(super) version: IndexVersion,
    /// The version this one succeeds; `None` for version 1.
    pub(super) predecessor: Option<IndexVersion>,
    /// The operation that published this version.
    pub(super) operation: OperationIdentity,
    /// The configuration the engine was built under.
    pub(super) config: IndexConfig,
    /// The engine, holding exactly the members of `members`.
    pub(super) engine: VersionEngine<M>,
    /// Every member's identity, provenance, and introducing version.
    pub(super) members: BTreeMap<M, MemberEntry<P>>,
}

impl<M, P> VersionPayload<M, P> {
    /// A payload for `version`, whose predecessor is the version before it.
    pub(super) fn new(
        index: IndexIdentity,
        version: IndexVersion,
        operation: OperationIdentity,
        config: IndexConfig,
        engine: VersionEngine<M>,
        members: BTreeMap<M, MemberEntry<P>>,
    ) -> Self {
        Self {
            format_version: LIFECYCLE_FORMAT_VERSION,
            index,
            version,
            predecessor: predecessor_of(version),
            operation,
            config,
            engine,
            members,
        }
    }
}

/// A version payload as stored, before its invariants are checked.
#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "M: MemberIdentity, P: ProvenanceReference")
)]
struct RawVersionPayload<M, P> {
    format_version: u16,
    index: IndexIdentity,
    version: IndexVersion,
    predecessor: Option<IndexVersion>,
    operation: OperationIdentity,
    config: IndexConfig,
    engine: VersionEngine<M>,
    members: BTreeMap<M, MemberEntry<P>>,
}

impl<M: MemberIdentity, P> TryFrom<RawVersionPayload<M, P>> for VersionPayload<M, P> {
    type Error = HeuremaError;

    /// Refuses, as [`HeuremaError::CorruptSnapshot`], a payload whose engine
    /// was built under another configuration, whose engine members differ
    /// from its member table, whose predecessor is not the version before
    /// it, or whose member entries name a version after this one.
    fn try_from(raw: RawVersionPayload<M, P>) -> Result<Self, Self::Error> {
        if raw.format_version != LIFECYCLE_FORMAT_VERSION {
            return UnsupportedSnapshotVersionSnafu {
                found: raw.format_version,
                supported: LIFECYCLE_FORMAT_VERSION,
            }
            .fail();
        }
        if !raw.engine.is_built_under(&raw.config) {
            return Err(corrupt(
                "version payload engine was built under another configuration",
            ));
        }
        if !raw.engine.holds_exactly(&raw.members) {
            return Err(corrupt(
                "version payload engine members differ from its member table",
            ));
        }
        if raw.predecessor != predecessor_of(raw.version) {
            return Err(corrupt(
                "version payload predecessor is not the version before it",
            ));
        }
        let entries_precede_the_version = raw.members.values().all(|entry| {
            entry.introduced <= raw.version
                && entry
                    .supersedes
                    .is_none_or(|superseded| superseded < entry.introduced)
        });
        if !entries_precede_the_version {
            return Err(corrupt(
                "version payload member entry names a version after the one it belongs to",
            ));
        }
        Ok(Self {
            format_version: raw.format_version,
            index: raw.index,
            version: raw.version,
            predecessor: raw.predecessor,
            operation: raw.operation,
            config: raw.config,
            engine: raw.engine,
            members: raw.members,
        })
    }
}

/// The marker of a staged, unpublished version: which operation staged it.
///
/// WHY it names the operation and transition: recovery reports what an
/// interrupted operation was, and the marker is all that survives of it
/// besides the payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StagingMarker {
    format_version: u16,
    index: IndexIdentity,
    version: IndexVersion,
    operation: OperationIdentity,
    transition: LifecycleTransition,
}

impl StagingMarker {
    /// The marker of `version`, staged by `operation`.
    pub(super) const fn new(
        index: IndexIdentity,
        version: IndexVersion,
        operation: OperationIdentity,
        transition: LifecycleTransition,
    ) -> Self {
        Self {
            format_version: LIFECYCLE_FORMAT_VERSION,
            index,
            version,
            operation,
            transition,
        }
    }
}

/// The audit record of one published operation.
///
/// WHY the full linkage now: `from`, `previous_config`, and `changes`
/// explain every transition (which version it started from, which
/// configuration a rebuild replaced, and each member it inserted, replaced,
/// removed, dropped, or retired with that member's previous entry), so a
/// later read API needs no format change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        serialize = "M: Serialize, P: Serialize, R: Serialize",
        deserialize = "M: MemberIdentity, P: ProvenanceReference, R: RetentionReference"
    )
)]
pub(super) struct OperationRecord<M, P, R> {
    format_version: u16,
    /// The index the operation changed.
    pub(super) index: IndexIdentity,
    /// The operation's key and digest.
    pub(super) operation: OperationIdentity,
    /// The transition it performed.
    pub(super) transition: LifecycleTransition,
    /// The version it was applied to; `None` for Create.
    pub(super) from: Option<IndexVersion>,
    /// The state it left the index in.
    pub(super) resulting: IndexState<R>,
    /// The configuration a Rebuild replaced; `None` for every other
    /// transition.
    pub(super) previous_config: Option<IndexConfig>,
    /// Every member it changed, in ascending identity order.
    pub(super) changes: Vec<MemberChange<M, P>>,
}

impl<M, P, R> OperationRecord<M, P, R> {
    /// The record of `operation`, applied to `from`, leaving `resulting`.
    pub(super) const fn new(
        index: IndexIdentity,
        operation: OperationIdentity,
        transition: LifecycleTransition,
        from: Option<IndexVersion>,
        resulting: IndexState<R>,
        previous_config: Option<IndexConfig>,
        changes: Vec<MemberChange<M, P>>,
    ) -> Self {
        Self {
            format_version: LIFECYCLE_FORMAT_VERSION,
            index,
            operation,
            transition,
            from,
            resulting,
            previous_config,
            changes,
        }
    }
}

/// Encodes a head record.
pub(super) fn encode_head<R: Serialize + Clone>(
    record: &IndexRecord<R>,
) -> Result<Vec<u8>, HeuremaError> {
    encode(&HeadBytes {
        format_version: LIFECYCLE_FORMAT_VERSION,
        record: record.clone(),
    })
}

/// Encodes a version payload, a staging marker, or an operation record.
///
/// WHY a refusal, not a storage failure: validation already refused every
/// value the canonical digest encoder cannot write, so a value `serde_json`
/// still refuses is the caller's input to fix.
pub(super) fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, HeuremaError> {
    serde_json::to_vec(value).map_err(|error| {
        UnencodableOperationSnafu {
            reason: error.to_string(),
        }
        .build()
    })
}

/// Decodes the head stored for `index`.
pub(super) fn decode_head<R: RetentionReference>(
    bytes: &[u8],
    index: &IndexIdentity,
) -> Result<IndexRecord<R>, HeuremaError> {
    check_format(bytes)?;
    let head: HeadBytes<R> = serde_json::from_slice(bytes).map_err(decode_error)?;
    if head.record.identity != *index {
        return Err(corrupt(format!(
            "head stored for index {index} names index {}",
            head.record.identity
        )));
    }
    Ok(head.record)
}

/// Decodes the payload stored for `version` of `index`.
pub(super) fn decode_payload<M: MemberIdentity, P: ProvenanceReference>(
    bytes: &[u8],
    index: &IndexIdentity,
    version: IndexVersion,
) -> Result<VersionPayload<M, P>, HeuremaError> {
    check_format(bytes)?;
    let raw: RawVersionPayload<M, P> = serde_json::from_slice(bytes).map_err(decode_error)?;
    let payload = VersionPayload::try_from(raw)?;
    if payload.index != *index || payload.version != version {
        return Err(corrupt(format!(
            "payload stored for version {version} of index {index} is version {} of index {}",
            payload.version, payload.index
        )));
    }
    Ok(payload)
}

/// Decodes the record stored for operation `key` on `index`.
pub(super) fn decode_operation<M, P, R>(
    bytes: &[u8],
    index: &IndexIdentity,
    key: &OperationKey,
) -> Result<OperationRecord<M, P, R>, HeuremaError>
where
    M: MemberIdentity,
    P: ProvenanceReference,
    R: RetentionReference,
{
    check_format(bytes)?;
    let record: OperationRecord<M, P, R> = serde_json::from_slice(bytes).map_err(decode_error)?;
    if record.index != *index || record.operation.key != *key {
        return Err(corrupt(format!(
            "operation record stored for key {key} of index {index} is key {} of index {}",
            record.operation.key, record.index
        )));
    }
    Ok(record)
}

/// The version before `version`; `None` for version 1.
fn predecessor_of(version: IndexVersion) -> Option<IndexVersion> {
    IndexVersion::try_from(version.get() - 1).ok()
}

/// Refuses a record whose format version this build does not read.
fn check_format(bytes: &[u8]) -> Result<(), HeuremaError> {
    let header: FormatHeader = serde_json::from_slice(bytes).map_err(decode_error)?;
    if header.format_version != LIFECYCLE_FORMAT_VERSION {
        return UnsupportedSnapshotVersionSnafu {
            found: header.format_version,
            supported: LIFECYCLE_FORMAT_VERSION,
        }
        .fail();
    }
    Ok(())
}

#[track_caller]
fn decode_error(source: serde_json::Error) -> HeuremaError {
    CorruptSnapshotSnafu.into_error(PersistenceSource::new(source))
}

/// A stored record that decodes but contradicts its key or its own
/// invariants.
#[track_caller]
pub(super) fn corrupt(reason: impl Into<String>) -> HeuremaError {
    CorruptSnapshotSnafu.into_error(PersistenceSource::new(InconsistentRecord(reason.into())))
}

/// Why a decoded lifecycle record was refused.
#[derive(Debug)]
struct InconsistentRecord(String);

impl fmt::Display for InconsistentRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for InconsistentRecord {}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests need concise payload fixtures")]
mod tests {
    use super::*;
    use crate::lifecycle::test_placeholders::{PlaceholderProvenance, TestMember};
    use crate::{ErrorCategory, HnswConfig, IndexName, OperationDigest, OwnerNamespace};

    fn notes() -> IndexIdentity {
        IndexIdentity::new(
            OwnerNamespace::try_from("example").expect("namespace"),
            IndexName::try_from("notes").expect("name"),
        )
    }

    fn version(value: u64) -> IndexVersion {
        IndexVersion::try_from(value).expect("version")
    }

    fn identity() -> OperationIdentity {
        OperationIdentity {
            key: OperationKey::try_from("insert-1").expect("key"),
            digest: OperationDigest::try_from("0a".repeat(32)).expect("digest"),
        }
    }

    fn entry(provenance: u32, introduced: u64) -> MemberEntry<PlaceholderProvenance> {
        MemberEntry {
            provenance: PlaceholderProvenance(provenance),
            introduced: version(introduced),
            supersedes: None,
        }
    }

    /// Version 2 of a two-dimensional vector index holding members 1 and 2.
    fn two_member_payload() -> VersionPayload<TestMember, PlaceholderProvenance> {
        let config = IndexConfig::Vector(HnswConfig::new(2));
        let mut engine = VersionEngine::empty(&config);
        let mut members = BTreeMap::new();
        for (id, vector) in [(1, [1.0, 0.0]), (2, [0.0, 1.0])] {
            engine
                .insert(TestMember(id), &MemberContent::Vector(vector.to_vec()))
                .expect("insert");
            members.insert(TestMember(id), entry(7, 2));
        }
        VersionPayload::new(notes(), version(2), identity(), config, engine, members)
    }

    fn decode(
        bytes: &[u8],
    ) -> Result<VersionPayload<TestMember, PlaceholderProvenance>, HeuremaError> {
        decode_payload(bytes, &notes(), version(2))
    }

    fn assert_corrupt<T: fmt::Debug>(result: Result<T, HeuremaError>, what: &str) {
        let error = result.expect_err(what);
        assert!(
            matches!(error, HeuremaError::CorruptSnapshot { .. }),
            "{what}: {error:?}"
        );
        assert_eq!(error.category(), ErrorCategory::Corrupt, "{what}");
    }

    /// Re-encodes `payload` after `tamper` edits its JSON form.
    fn tampered(
        payload: &VersionPayload<TestMember, PlaceholderProvenance>,
        tamper: impl FnOnce(&mut serde_json::Value),
    ) -> Vec<u8> {
        let mut value = serde_json::to_value(payload).expect("payload encodes");
        tamper(&mut value);
        serde_json::to_vec(&value).expect("tampered payload encodes")
    }

    #[test]
    fn version_payload_round_trips() {
        let payload = two_member_payload();
        let bytes = encode(&payload).expect("payload encodes");
        let decoded = decode(&bytes).expect("payload decodes");
        assert_eq!(decoded.members, payload.members);
        assert_eq!(decoded.predecessor, Some(version(1)));
        assert_eq!(decoded.config, payload.config);
    }

    #[test]
    fn version_payload_with_mismatched_member_table_is_rejected_on_decode() {
        let payload = two_member_payload();

        let missing_member = tampered(&payload, |value| {
            value["members"]
                .as_object_mut()
                .expect("member table")
                .remove("2");
        });
        assert_corrupt(
            decode(&missing_member),
            "an engine member absent from the table",
        );

        let extra_member = tampered(&payload, |value| {
            let entry = value["members"]["1"].clone();
            value["members"]
                .as_object_mut()
                .expect("member table")
                .insert("3".to_owned(), entry);
        });
        assert_corrupt(
            decode(&extra_member),
            "a table member absent from the engine",
        );

        let renamed_member = tampered(&payload, |value| {
            let table = value["members"].as_object_mut().expect("member table");
            let entry = table.remove("2").expect("member 2");
            table.insert("9".to_owned(), entry);
        });
        assert_corrupt(
            decode(&renamed_member),
            "a table naming another member than the engine holds",
        );
    }

    #[test]
    fn version_payload_contradicting_its_config_predecessor_or_key_is_rejected() {
        let payload = two_member_payload();

        let other_config = tampered(&payload, |value| {
            value["config"]["Vector"]["dimensions"] = serde_json::json!(3);
        });
        assert_corrupt(
            decode(&other_config),
            "a config the engine was not built under",
        );

        let skipped_predecessor = tampered(&payload, |value| {
            value["predecessor"] = serde_json::Value::Null;
        });
        assert_corrupt(
            decode(&skipped_predecessor),
            "a version 2 with no predecessor",
        );

        let later_entry = tampered(&payload, |value| {
            value["members"]["1"]["introduced"] = serde_json::json!(3);
        });
        assert_corrupt(
            decode(&later_entry),
            "an entry introduced after its version",
        );

        let bytes = encode(&payload).expect("payload encodes");
        assert_corrupt(
            decode_payload::<TestMember, PlaceholderProvenance>(&bytes, &notes(), version(3)),
            "a payload stored under another version's key",
        );
    }

    #[test]
    fn undecodable_or_future_records_are_corrupt_or_unsupported_never_storage() {
        assert_corrupt(decode(b"{\"format_version\":1"), "torn bytes");
        assert_corrupt(decode(b"[1,2,3]"), "bytes of another shape");

        let payload = two_member_payload();
        let future = tampered(&payload, |value| {
            value["format_version"] = serde_json::json!(LIFECYCLE_FORMAT_VERSION + 1);
            value["field_from_a_newer_build"] = serde_json::json!(true);
        });
        let error = decode(&future).expect_err("a newer format is refused");
        assert!(
            matches!(
                error,
                HeuremaError::UnsupportedSnapshotVersion { found, supported, .. }
                    if found == LIFECYCLE_FORMAT_VERSION + 1
                        && supported == LIFECYCLE_FORMAT_VERSION
            ),
            "{error:?}"
        );
        assert_eq!(error.category(), ErrorCategory::Unsupported);
    }
}
