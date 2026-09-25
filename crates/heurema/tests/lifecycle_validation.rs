//! Pre-publish lifecycle validation: one test per refusal class, the
//! permission table cell by cell, and the documented check order.
//!
//! `CheckedOperation::check` takes no index state and `permit` takes only
//! the record the caller passes, so every refusal here happens in a pure
//! function, before any adapter could be called. A missing provenance or
//! retention reference is not a runtime refusal class: it does not compile,
//! as the `compile_fail` doctests on `IndexMember` and `IndexChange` show.

use std::fmt;

use heurema::{
    CheckedOperation, ErrorCategory, FtsConfig, HeuremaError, HnswConfig, IdentifierKind,
    IndexChange, IndexConfig, IndexIdentity, IndexMember, IndexName, IndexRecord, IndexState,
    IndexStateKind, IndexVersion, LifecycleOperation, LifecycleTransition, MemberContent,
    MemberIdentity, OperationKey, OwnerNamespace, ProvenanceReference, RetentionReference,
    SnapshotFamily, TokenizerConfig, ValidatedOperation,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// test-local placeholder; heurēma defines no provenance shape
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct TestMember(u64);

impl MemberIdentity for TestMember {}

/// test-local placeholder; heurēma defines no provenance shape
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PlaceholderProvenance(u32);

impl ProvenanceReference for PlaceholderProvenance {}

/// test-local placeholder; heurēma defines no provenance shape
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PlaceholderRetention(u32);

impl RetentionReference for PlaceholderRetention {}

/// test-local placeholder; heurēma defines no provenance shape. A member
/// identity that serializes as a struct, which cannot key a snapshot map.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct CompositeMember {
    shard: u32,
    local: u64,
}

impl MemberIdentity for CompositeMember {}

/// test-local placeholder; heurēma defines no provenance shape. A member
/// identity that serializes as a two-element sequence.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct PairMember(u32, u32);

impl MemberIdentity for PairMember {}

/// test-local placeholder; heurēma defines no provenance shape. A member
/// identity that serializes as a string.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct NamedMember(String);

impl MemberIdentity for NamedMember {}

/// test-local placeholder; heurēma defines no provenance shape. An untagged
/// identity with an integer and a string variant: every value encodes as an
/// integer or a string, but serde_json writes an integer key as its digits
/// and reads those digits back as the string variant.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(untagged)]
enum MixedMember {
    Number(u64),
    Text(String),
}

impl MemberIdentity for MixedMember {}

/// test-local placeholder; heurēma defines no provenance shape. Provenance
/// whose serialized form holds a float, which has no canonical encoding.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct WeightedProvenance(u32);

impl Serialize for WeightedProvenance {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(f64::from(self.0) / 1000.0)
    }
}

impl ProvenanceReference for WeightedProvenance {}

/// test-local placeholder; heurēma defines no provenance shape. An identity
/// written as an integer but read back only from a string: it survives a
/// JSON object key, where serde_json writes the integer as its digits, and
/// fails as a JSON value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DigitsMember(u64);

impl Serialize for DigitsMember {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for DigitsMember {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let digits = String::deserialize(deserializer)?;
        digits.parse().map(Self).map_err(serde::de::Error::custom)
    }
}

impl MemberIdentity for DigitsMember {}

/// test-local placeholder; heurēma defines no provenance shape.
/// [`DigitsMember`]'s twin, written as a string, so it reads back both ways.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct TextMember(u64);

impl Serialize for TextMember {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de> Deserialize<'de> for TextMember {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let digits = String::deserialize(deserializer)?;
        digits.parse().map(Self).map_err(serde::de::Error::custom)
    }
}

impl MemberIdentity for TextMember {}

/// test-local placeholder; heurēma defines no provenance shape. Empty tags
/// are skipped when written, and with no serde default they cannot be read
/// back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TaggedProvenance {
    source: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,
}

impl ProvenanceReference for TaggedProvenance {}

/// test-local placeholder; heurēma defines no provenance shape. JSON writes
/// `Some(None)` as `null`, which reads back as `None`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ReviewedProvenance {
    #[expect(
        clippy::option_option,
        reason = "the nested option is the consumer shape serde_json cannot round-trip, which this placeholder exists to refuse"
    )]
    reviewed: Option<Option<u32>>,
}

impl ProvenanceReference for ReviewedProvenance {}

/// test-local placeholder; heurēma defines no retention shape. The same
/// shape as [`TaggedProvenance`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TaggedRetention {
    source: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,
}

impl RetentionReference for TaggedRetention {}

type Change = IndexChange<TestMember, PlaceholderProvenance, PlaceholderRetention>;
type ChangeOf<M, P = PlaceholderProvenance> = IndexChange<M, P, PlaceholderRetention>;
type Record = IndexRecord<PlaceholderRetention>;

fn index() -> Result<IndexIdentity, HeuremaError> {
    Ok(IndexIdentity::new(
        OwnerNamespace::try_from("example")?,
        IndexName::try_from("notes")?,
    ))
}

fn operation<M: MemberIdentity, P: ProvenanceReference>(
    change: ChangeOf<M, P>,
) -> Result<LifecycleOperation<M, P, PlaceholderRetention>, HeuremaError> {
    Ok(LifecycleOperation::new(
        index()?,
        OperationKey::try_from("op-1")?,
        change,
    ))
}

fn check<M: MemberIdentity, P: ProvenanceReference>(
    change: ChangeOf<M, P>,
) -> Result<CheckedOperation<M, P, PlaceholderRetention>, HeuremaError> {
    CheckedOperation::check(operation(change)?)
}

/// [`check`] for an operation whose retention type is `R`.
fn check_with<M: MemberIdentity, P: ProvenanceReference, R: RetentionReference>(
    change: IndexChange<M, P, R>,
) -> Result<CheckedOperation<M, P, R>, HeuremaError> {
    CheckedOperation::check(LifecycleOperation::new(
        index()?,
        OperationKey::try_from("op-1")?,
        change,
    ))
}

fn tagged(
    id: u64,
    tags: &[&str],
    content: MemberContent,
) -> IndexMember<TestMember, TaggedProvenance> {
    IndexMember::new(
        TestMember(id),
        TaggedProvenance {
            source: 7,
            tags: tags.iter().map(|&tag| tag.to_owned()).collect(),
        },
        content,
    )
}

fn assert_unencodable(error: &HeuremaError, reason_starts: &str) {
    let HeuremaError::UnencodableOperation { reason, .. } = error else {
        panic!("unexpected {error:?}");
    };
    assert!(reason.starts_with(reason_starts), "{error}");
    assert_eq!(error.category(), ErrorCategory::Refused, "{error}");
}

fn vector(id: u64, components: &[f32]) -> IndexMember<TestMember, PlaceholderProvenance> {
    IndexMember::new(
        TestMember(id),
        PlaceholderProvenance(7),
        MemberContent::Vector(components.to_vec()),
    )
}

fn document(id: u64, text: &str) -> IndexMember<TestMember, PlaceholderProvenance> {
    IndexMember::new(
        TestMember(id),
        PlaceholderProvenance(7),
        MemberContent::Document(text.to_owned()),
    )
}

fn vector_config(dimensions: usize) -> IndexConfig {
    IndexConfig::Vector(HnswConfig::new(dimensions))
}

fn active(config: IndexConfig) -> Result<Record, HeuremaError> {
    Ok(IndexRecord::new(
        index()?,
        config,
        IndexState::Active {
            version: IndexVersion::FIRST,
        },
    ))
}

fn destroyed(config: IndexConfig) -> Result<Record, HeuremaError> {
    Ok(IndexRecord::new(
        index()?,
        config,
        IndexState::Destroyed {
            last_version: IndexVersion::try_from(3)?,
            retention: PlaceholderRetention(9),
            operation: OperationKey::try_from("destroy-1")?,
        },
    ))
}

/// The error a refused step returned.
fn refused<T: fmt::Debug>(result: Result<T, HeuremaError>) -> HeuremaError {
    match result {
        Ok(value) => panic!("expected a refusal, got {value:?}"),
        Err(error) => error,
    }
}

fn assert_family_mismatch(
    error: &HeuremaError,
    expected_family: SnapshotFamily,
    actual_family: SnapshotFamily,
) {
    let HeuremaError::FamilyMismatch {
        expected, actual, ..
    } = error
    else {
        panic!("unexpected {error:?}");
    };
    assert_eq!(
        (*expected, *actual),
        (expected_family, actual_family),
        "{error}"
    );
}

fn assert_dimension_mismatch(error: &HeuremaError, want: usize, got: usize) {
    let HeuremaError::DimensionMismatch {
        expected, actual, ..
    } = error
    else {
        panic!("unexpected {error:?}");
    };
    assert_eq!((*expected, *actual), (want, got), "{error}");
}

fn unsupported_pipelines() -> Vec<FtsConfig> {
    let mut ngram = FtsConfig::simple();
    ngram.tokenizer = TokenizerConfig::new("NGram", vec!["3".to_owned()]);
    let mut simple_with_args = FtsConfig::simple();
    simple_with_args.tokenizer = TokenizerConfig::new("Simple", vec!["x".to_owned()]);
    let mut filtered = FtsConfig::simple();
    filtered.filters = vec![TokenizerConfig::new("Lowercase", Vec::new())];
    vec![ngram, simple_with_args, filtered]
}

fn invalid_hnsw_configs() -> Vec<(HnswConfig, &'static str)> {
    let zero_dimensions = HnswConfig::new(0);
    let mut zero_m = HnswConfig::new(2);
    zero_m.m_neighbours = 0;
    let mut zero_ef = HnswConfig::new(2);
    zero_ef.ef_construction = 0;
    vec![
        (zero_dimensions, "dimensions"),
        (zero_m, "m_neighbours"),
        (zero_ef, "ef_construction"),
    ]
}

#[test]
fn insert_with_wrong_dimensions_is_refused() -> Result<(), HeuremaError> {
    let current = active(vector_config(3))?;
    for components in [&[0.0, 1.0][..], &[0.0, 1.0, 2.0, 3.0][..]] {
        let checked = check(Change::Insert {
            members: vec![vector(1, &[0.0, 1.0, 2.0]), vector(2, components)],
        })?;
        let error = refused(checked.permit(Some(&current)));
        assert_dimension_mismatch(&error, 3, components.len());
    }
    Ok(())
}

#[test]
fn rebuild_with_wrong_dimensions_is_refused_without_state() {
    let error = refused(check(Change::Rebuild {
        config: vector_config(3),
        members: vec![vector(1, &[0.0, 1.0, 2.0]), vector(2, &[0.0, 1.0])],
    }));
    assert_dimension_mismatch(&error, 3, 2);
}

#[test]
fn non_finite_vector_is_refused() {
    for component in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let inserted = refused(check(Change::Insert {
            members: vec![vector(1, &[0.0, component])],
        }));
        let rebuilt = refused(check(Change::Rebuild {
            config: vector_config(2),
            members: vec![vector(1, &[component, 0.0])],
        }));
        for error in [inserted, rebuilt] {
            assert!(
                matches!(error, HeuremaError::InvalidVector { .. }),
                "{component} is refused as an invalid vector: {error:?}"
            );
        }
    }
}

#[test]
fn invalid_hnsw_config_is_refused_at_create() {
    for (config, field) in invalid_hnsw_configs() {
        let created = refused(check(Change::Create {
            config: IndexConfig::Vector(config.clone()),
        }));
        let rebuilt = refused(check(Change::Rebuild {
            config: IndexConfig::Vector(config),
            members: Vec::new(),
        }));
        for error in [created, rebuilt] {
            let HeuremaError::InvalidHnswConfig { reason, .. } = &error else {
                panic!("unexpected {error:?}");
            };
            assert!(reason.contains(field), "{field} is named: {reason}");
        }
    }
}

#[test]
fn unsupported_fts_pipeline_is_refused_at_create() {
    for config in unsupported_pipelines() {
        let created = refused(check(Change::Create {
            config: IndexConfig::Fts(config.clone()),
        }));
        let rebuilt = refused(check(Change::Rebuild {
            config: IndexConfig::Fts(config.clone()),
            members: vec![document(1, "text")],
        }));
        for error in [created, rebuilt] {
            assert!(
                matches!(error, HeuremaError::NotYetImplemented { .. }),
                "{config:?} is refused as unsupported: {error:?}"
            );
            assert_eq!(error.category(), ErrorCategory::Unsupported, "{error}");
        }
    }
}

#[test]
fn document_into_vector_index_is_refused() -> Result<(), HeuremaError> {
    let current = active(vector_config(2))?;
    let checked = check(Change::Insert {
        members: vec![document(1, "text")],
    })?;
    let error = refused(checked.permit(Some(&current)));
    assert_family_mismatch(&error, SnapshotFamily::Vector, SnapshotFamily::Fts);

    let fts = active(IndexConfig::Fts(FtsConfig::simple()))?;
    let checked = check(Change::Insert {
        members: vec![vector(1, &[0.0, 1.0])],
    })?;
    let error = refused(checked.permit(Some(&fts)));
    assert_family_mismatch(&error, SnapshotFamily::Fts, SnapshotFamily::Vector);

    let error = refused(check(Change::Rebuild {
        config: vector_config(2),
        members: vec![document(1, "text")],
    }));
    assert_family_mismatch(&error, SnapshotFamily::Vector, SnapshotFamily::Fts);
    Ok(())
}

#[test]
fn mixed_family_insert_is_refused() {
    // WHY both orders: members are checked in ascending identity order, so
    // the refusal names the same families however the caller listed them.
    for members in [
        vec![vector(1, &[0.0, 1.0]), document(2, "text")],
        vec![document(2, "text"), vector(1, &[0.0, 1.0])],
    ] {
        let error = refused(check(Change::Insert { members }));
        assert_family_mismatch(&error, SnapshotFamily::Vector, SnapshotFamily::Fts);
    }
}

#[test]
fn rebuild_cannot_change_the_index_family() -> Result<(), HeuremaError> {
    let vector_index = active(vector_config(2))?;
    let checked = check(Change::Rebuild {
        config: IndexConfig::Fts(FtsConfig::simple()),
        members: vec![document(1, "text")],
    })?;
    let error = refused(checked.permit(Some(&vector_index)));
    assert_family_mismatch(&error, SnapshotFamily::Vector, SnapshotFamily::Fts);

    let same_family = check(Change::Rebuild {
        config: vector_config(4),
        members: vec![vector(1, &[0.0, 1.0, 2.0, 3.0])],
    })?;
    let rebuilt = same_family.permit(Some(&vector_index))?;
    assert_eq!(
        rebuilt.from_state(),
        IndexStateKind::Active,
        "a rebuild may change the configuration within one family"
    );
    Ok(())
}

#[test]
fn duplicate_member_in_one_batch_is_refused() {
    let inserted = refused(check(Change::Insert {
        members: vec![
            vector(1, &[0.0, 1.0]),
            vector(2, &[1.0, 0.0]),
            vector(1, &[1.0, 1.0]),
        ],
    }));
    let removed = refused(check(Change::Remove {
        members: vec![TestMember(4), TestMember(1), TestMember(4)],
    }));
    let rebuilt = refused(check(Change::Rebuild {
        config: vector_config(2),
        members: vec![vector(3, &[0.0, 1.0]), vector(3, &[0.0, 1.0])],
    }));
    for (error, id) in [(inserted, 1), (removed, 4), (rebuilt, 3)] {
        let HeuremaError::DuplicateMember { member, .. } = &error else {
            panic!("unexpected {error:?}");
        };
        assert_eq!(member, &format!("TestMember({id})"), "{error}");
    }
}

#[test]
fn empty_insert_and_remove_batches_are_refused() -> Result<(), HeuremaError> {
    let inserted = refused(check(Change::Insert {
        members: Vec::new(),
    }));
    let removed = refused(check(Change::Remove {
        members: Vec::new(),
    }));
    for (error, transition) in [
        (inserted, LifecycleTransition::Insert),
        (removed, LifecycleTransition::Remove),
    ] {
        let HeuremaError::EmptyBatch {
            transition: refused_transition,
            ..
        } = &error
        else {
            panic!("unexpected {error:?}");
        };
        assert_eq!(*refused_transition, transition, "{error}");
    }

    let emptied = check(Change::Rebuild {
        config: vector_config(2),
        members: Vec::new(),
    })?;
    assert_eq!(
        emptied.operation().transition(),
        LifecycleTransition::Rebuild,
        "a rebuild to an empty member set names its whole new state and is accepted"
    );
    Ok(())
}

#[test]
fn member_identity_that_is_not_a_json_key_is_refused() -> Result<(), HeuremaError> {
    let composite = refused(check(ChangeOf::<CompositeMember>::Remove {
        members: vec![CompositeMember { shard: 1, local: 2 }],
    }));
    let pair = refused(check(ChangeOf::<PairMember>::Insert {
        members: vec![IndexMember::new(
            PairMember(1, 2),
            PlaceholderProvenance(7),
            MemberContent::Vector(vec![0.0, 1.0]),
        )],
    }));
    for (error, encoded_as) in [(composite, "a map or struct"), (pair, "a sequence")] {
        let HeuremaError::InvalidIdentifier { kind, reason, .. } = &error else {
            panic!("unexpected {error:?}");
        };
        assert_eq!(*kind, IdentifierKind::MemberIdentity, "{error}");
        assert!(
            reason.starts_with(&format!("encodes as {encoded_as};")),
            "{error}"
        );
    }

    let named = check(ChangeOf::<NamedMember>::Remove {
        members: vec![NamedMember("doc-7".to_owned())],
    })?;
    assert_eq!(
        named.identity().key.as_str(),
        "op-1",
        "a string identity is accepted"
    );
    Ok(())
}

#[test]
fn member_identity_that_does_not_survive_a_json_key_round_trip_is_refused()
-> Result<(), HeuremaError> {
    let error = refused(check(ChangeOf::<MixedMember>::Remove {
        members: vec![
            MixedMember::Text("doc-7".to_owned()),
            MixedMember::Number(7),
        ],
    }));
    let HeuremaError::InvalidIdentifier {
        kind,
        value,
        reason,
        ..
    } = &error
    else {
        panic!("unexpected {error:?}");
    };
    assert_eq!(*kind, IdentifierKind::MemberIdentity, "{error}");
    assert_eq!(value, "Number(7)", "{error}");
    assert!(
        reason.starts_with(r#"reads back from the JSON object {"7":0} as a different identity"#),
        "{error}"
    );

    let text_only = check(ChangeOf::<MixedMember>::Remove {
        members: vec![MixedMember::Text("doc-7".to_owned())],
    })?;
    assert_eq!(
        text_only.identity().key.as_str(),
        "op-1",
        "a value of the same type that round-trips is accepted"
    );
    Ok(())
}

#[test]
fn member_identity_that_does_not_read_back_from_a_json_value_is_refused() -> Result<(), HeuremaError>
{
    let digits = |id| {
        IndexMember::new(
            DigitsMember(id),
            PlaceholderProvenance(7),
            MemberContent::Vector(vec![0.0, 1.0]),
        )
    };
    let inserted = refused(check(ChangeOf::<DigitsMember>::Insert {
        members: vec![digits(1)],
    }));
    let removed = refused(check(ChangeOf::<DigitsMember>::Remove {
        members: vec![DigitsMember(1)],
    }));
    for error in [inserted, removed] {
        let HeuremaError::InvalidIdentifier { kind, reason, .. } = &error else {
            panic!("unexpected {error:?}");
        };
        assert_eq!(*kind, IdentifierKind::MemberIdentity, "{error}");
        assert!(
            reason.starts_with("cannot be read back from its JSON value"),
            "{error}"
        );
    }

    let text = |id| {
        IndexMember::new(
            TextMember(id),
            PlaceholderProvenance(7),
            MemberContent::Vector(vec![0.0, 1.0]),
        )
    };
    check(ChangeOf::<TextMember>::Insert {
        members: vec![text(1)],
    })?;
    check(ChangeOf::<TextMember>::Remove {
        members: vec![TextMember(1)],
    })?;
    Ok(())
}

#[test]
fn provenance_that_does_not_read_back_as_itself_is_refused() -> Result<(), HeuremaError> {
    let point = || MemberContent::Vector(vec![0.0, 1.0]);
    let empty_tags = [
        check(ChangeOf::<TestMember, TaggedProvenance>::Insert {
            members: vec![tagged(1, &[], point())],
        }),
        check(ChangeOf::<TestMember, TaggedProvenance>::Rebuild {
            config: vector_config(2),
            members: vec![tagged(1, &[], point())],
        }),
    ];
    for result in empty_tags {
        assert_unencodable(&refused(result), "provenance of member TestMember(1)");
    }
    check(ChangeOf::<TestMember, TaggedProvenance>::Insert {
        members: vec![tagged(1, &["a"], point())],
    })?;
    check(ChangeOf::<TestMember, TaggedProvenance>::Rebuild {
        config: vector_config(2),
        members: vec![tagged(1, &["a"], point())],
    })?;

    let reviewed = |reviewed| {
        vec![IndexMember::new(
            TestMember(1),
            ReviewedProvenance { reviewed },
            point(),
        )]
    };
    assert_unencodable(
        &refused(check(ChangeOf::<TestMember, ReviewedProvenance>::Insert {
            members: reviewed(Some(None)),
        })),
        "provenance of member TestMember(1)",
    );
    for kept in [Some(Some(1)), None] {
        check(ChangeOf::<TestMember, ReviewedProvenance>::Insert {
            members: reviewed(kept),
        })?;
    }
    Ok(())
}

#[test]
fn retention_that_does_not_read_back_as_itself_is_refused() -> Result<(), HeuremaError> {
    let destroy = |tags: Vec<String>| {
        check_with(
            IndexChange::<TestMember, PlaceholderProvenance, TaggedRetention>::Destroy {
                retention: TaggedRetention { source: 9, tags },
            },
        )
    };
    assert_unencodable(&refused(destroy(Vec::new())), "retention ");
    destroy(vec!["kept".to_owned()])?;
    Ok(())
}

#[test]
fn record_of_another_index_is_refused_before_the_permission_table() -> Result<(), HeuremaError> {
    let other = IndexIdentity::new(
        OwnerNamespace::try_from("example")?,
        IndexName::try_from("drafts")?,
    );
    let mut foreign = active(vector_config(2))?;
    foreign.identity = other.clone();

    let error = refused(
        check(Change::Insert {
            members: vec![vector(1, &[0.0, 1.0])],
        })?
        .permit(Some(&foreign)),
    );
    let HeuremaError::RecordMismatch {
        expected, actual, ..
    } = &error
    else {
        panic!("unexpected {error:?}");
    };
    assert_eq!(expected, &index()?, "{error}");
    assert_eq!(actual, &other, "{error}");

    // Create is forbidden on the foreign record's Destroyed state; the
    // mismatch is reported first, because that state is not this index's.
    let mut foreign_destroyed = destroyed(vector_config(2))?;
    foreign_destroyed.identity = other;
    let error = refused(
        check(Change::Create {
            config: vector_config(2),
        })?
        .permit(Some(&foreign_destroyed)),
    );
    assert!(
        matches!(error, HeuremaError::RecordMismatch { .. }),
        "{error:?}"
    );
    Ok(())
}

#[test]
fn unencodable_provenance_is_refused() {
    let error = refused(check(ChangeOf::<TestMember, WeightedProvenance>::Insert {
        members: vec![IndexMember::new(
            TestMember(1),
            WeightedProvenance(500),
            MemberContent::Vector(vec![0.0, 1.0]),
        )],
    }));
    let HeuremaError::UnencodableOperation { reason, .. } = &error else {
        panic!("unexpected {error:?}");
    };
    assert!(reason.contains("f64"), "{error}");
}

/// Columns: Absent, Active, Destroyed. Written from the lifecycle's
/// permission table, independently of `is_permitted_from`.
const PERMISSION_TABLE: [(LifecycleTransition, [bool; 3]); 7] = [
    (LifecycleTransition::Create, [true, false, false]),
    (LifecycleTransition::Insert, [false, true, false]),
    (LifecycleTransition::Remove, [false, true, false]),
    (LifecycleTransition::Rebuild, [false, true, false]),
    (LifecycleTransition::Destroy, [false, true, false]),
    (LifecycleTransition::Publish, [true, true, false]),
    (LifecycleTransition::Recover, [true, true, true]),
];

const STATES: [IndexStateKind; 3] = [
    IndexStateKind::Absent,
    IndexStateKind::Active,
    IndexStateKind::Destroyed,
];

/// A valid change requesting `transition`, or `None` for the steps only the
/// lifecycle takes.
fn change_for(transition: LifecycleTransition) -> Option<Change> {
    match transition {
        LifecycleTransition::Create => Some(Change::Create {
            config: vector_config(2),
        }),
        LifecycleTransition::Insert => Some(Change::Insert {
            members: vec![vector(1, &[0.0, 1.0])],
        }),
        LifecycleTransition::Remove => Some(Change::Remove {
            members: vec![TestMember(1)],
        }),
        LifecycleTransition::Rebuild => Some(Change::Rebuild {
            config: vector_config(2),
            members: vec![vector(1, &[0.0, 1.0])],
        }),
        LifecycleTransition::Destroy => Some(Change::Destroy {
            retention: PlaceholderRetention(9),
        }),
        LifecycleTransition::Publish | LifecycleTransition::Recover => None,
        other => panic!("unexpected {other:?}"),
    }
}

fn record_for(state: IndexStateKind) -> Result<Option<Record>, HeuremaError> {
    match state {
        IndexStateKind::Absent => Ok(None),
        IndexStateKind::Active => active(vector_config(2)).map(Some),
        IndexStateKind::Destroyed => destroyed(vector_config(2)).map(Some),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn every_cell_of_the_permission_table_is_enforced() -> Result<(), HeuremaError> {
    let mut permitted_cells = 0;
    let mut permit_calls = 0;
    for (transition, row) in PERMISSION_TABLE {
        for (state, permitted) in STATES.into_iter().zip(row) {
            assert_eq!(
                transition.is_permitted_from(state),
                permitted,
                "{transition:?} from {state:?}"
            );
            permitted_cells += usize::from(permitted);

            let Some(change) = change_for(transition) else {
                continue;
            };
            permit_calls += 1;
            let current = record_for(state)?;
            let outcome = check(change)?.permit(current.as_ref());
            if permitted {
                let validated: ValidatedOperation<_, _, _> = outcome?;
                assert_eq!(
                    validated.from_state(),
                    state,
                    "{transition:?} from {state:?}"
                );
            } else {
                let error = refused(outcome);
                let HeuremaError::TransitionNotPermitted {
                    index: refused_index,
                    transition: refused_transition,
                    state: refused_state,
                    ..
                } = &error
                else {
                    panic!("unexpected {error:?}");
                };
                assert_eq!(
                    (refused_index, *refused_transition, *refused_state),
                    (&index()?, transition, state),
                    "{error}"
                );
            }
        }
    }
    assert_eq!(permitted_cells, 10, "the table permits ten of 21 cells");
    assert_eq!(
        permit_calls, 15,
        "every caller transition is permitted or refused from every state"
    );
    Ok(())
}

#[test]
fn family_is_checked_for_every_member_before_any_dimension() -> Result<(), HeuremaError> {
    // Member 1, first in identity order, has the wrong dimension; member 2
    // has the wrong family. Both input orders meet FamilyMismatch.
    let wrong_dimension = vector(1, &[0.0, 1.0, 2.0]);
    let wrong_family = document(2, "text");
    for members in [
        vec![wrong_dimension.clone(), wrong_family.clone()],
        vec![wrong_family.clone(), wrong_dimension.clone()],
    ] {
        let rebuilt = refused(check(Change::Rebuild {
            config: vector_config(2),
            members: members.clone(),
        }));
        assert_family_mismatch(&rebuilt, SnapshotFamily::Vector, SnapshotFamily::Fts);

        // A mixed Insert is refused by the stateless check, before permit
        // sees the current configuration.
        let inserted = refused(check(Change::Insert { members }));
        assert_family_mismatch(&inserted, SnapshotFamily::Vector, SnapshotFamily::Fts);
    }

    // Against the current configuration, a single-family batch of the other
    // family is refused for its family, whatever its vectors' dimensions.
    // (A batch mixing families never reaches permit: check refuses it.)
    let fts_index = active(IndexConfig::Fts(FtsConfig::simple()))?;
    let checked = check(Change::Insert {
        members: vec![vector(1, &[0.0, 1.0, 2.0]), vector(2, &[0.0])],
    })?;
    let error = refused(checked.permit(Some(&fts_index)));
    assert_family_mismatch(&error, SnapshotFamily::Fts, SnapshotFamily::Vector);
    Ok(())
}

#[test]
fn checks_run_in_the_documented_order() {
    // Duplicate (step 2) before a non-finite vector (step 4).
    let error = refused(check(Change::Insert {
        members: vec![vector(1, &[f32::NAN, 0.0]), vector(1, &[0.0, 0.0])],
    }));
    assert!(
        matches!(error, HeuremaError::DuplicateMember { .. }),
        "{error:?}"
    );

    // Member identity shape (step 3) before a non-finite vector (step 4).
    let error = refused(check(ChangeOf::<PairMember>::Insert {
        members: vec![IndexMember::new(
            PairMember(1, 2),
            PlaceholderProvenance(7),
            MemberContent::Vector(vec![f32::NAN]),
        )],
    }));
    assert!(
        matches!(error, HeuremaError::InvalidIdentifier { .. }),
        "{error:?}"
    );

    // A non-finite vector (step 4) before the configuration (step 5).
    let error = refused(check(Change::Rebuild {
        config: IndexConfig::Vector(HnswConfig::new(0)),
        members: vec![vector(1, &[f32::NAN])],
    }));
    assert!(
        matches!(error, HeuremaError::InvalidVector { .. }),
        "{error:?}"
    );

    // The configuration (step 5) before member family (step 6).
    let error = refused(check(Change::Rebuild {
        config: IndexConfig::Vector(HnswConfig::new(0)),
        members: vec![document(1, "text")],
    }));
    assert!(
        matches!(error, HeuremaError::InvalidHnswConfig { .. }),
        "{error:?}"
    );

    // A non-finite vector (step 4) before mixed families (step 7).
    let error = refused(check(Change::Insert {
        members: vec![vector(1, &[f32::INFINITY]), document(2, "text")],
    }));
    assert!(
        matches!(error, HeuremaError::InvalidVector { .. }),
        "{error:?}"
    );

    // Mixed families (step 7) before a provenance that does not read back
    // (step 9).
    let error = refused(check(ChangeOf::<TestMember, TaggedProvenance>::Insert {
        members: vec![
            tagged(1, &[], MemberContent::Vector(vec![0.0])),
            tagged(2, &[], MemberContent::Document("text".to_owned())),
        ],
    }));
    assert!(
        matches!(error, HeuremaError::FamilyMismatch { .. }),
        "{error:?}"
    );
}

#[test]
fn permitted_operation_keeps_its_identity_and_source_state() -> Result<(), HeuremaError> {
    let current = active(vector_config(2))?;
    let checked = check(Change::Insert {
        members: vec![vector(1, &[0.0, 1.0])],
    })?;
    let identity = checked.identity().clone();
    let operation = checked.operation().clone();

    let validated = checked.permit(Some(&current))?;
    assert_eq!(
        validated.identity(),
        &identity,
        "permit keeps the identity check computed"
    );
    assert_eq!(
        validated.operation(),
        &operation,
        "permit keeps the operation"
    );
    assert_eq!(validated.from_state(), IndexStateKind::Active);
    Ok(())
}

#[test]
fn refusals_are_categorised_as_refused() -> Result<(), HeuremaError> {
    let vector_index = active(vector_config(3))?;
    let refusals = [
        refused(check(Change::Insert {
            members: Vec::new(),
        })),
        refused(check(Change::Remove {
            members: vec![TestMember(1), TestMember(1)],
        })),
        refused(check(ChangeOf::<CompositeMember>::Remove {
            members: vec![CompositeMember { shard: 1, local: 2 }],
        })),
        refused(check(Change::Insert {
            members: vec![vector(1, &[f32::NAN])],
        })),
        refused(check(Change::Create {
            config: vector_config(0),
        })),
        refused(check(Change::Insert {
            members: vec![vector(1, &[0.0]), document(2, "text")],
        })),
        refused(check(Change::Rebuild {
            config: vector_config(3),
            members: vec![vector(1, &[0.0])],
        })),
        refused(
            check(Change::Insert {
                members: vec![vector(1, &[0.0])],
            })?
            .permit(Some(&vector_index)),
        ),
        refused(
            check(Change::Insert {
                members: vec![vector(1, &[0.0])],
            })?
            .permit(None),
        ),
        refused(check(ChangeOf::<MixedMember>::Remove {
            members: vec![MixedMember::Number(7)],
        })),
        refused(
            check(Change::Insert {
                members: vec![vector(1, &[0.0, 1.0])],
            })?
            .permit(Some(&{
                let mut foreign = active(vector_config(2))?;
                foreign.identity = IndexIdentity::new(
                    OwnerNamespace::try_from("example")?,
                    IndexName::try_from("drafts")?,
                );
                foreign
            })),
        ),
        refused(check(ChangeOf::<TestMember, WeightedProvenance>::Insert {
            members: vec![IndexMember::new(
                TestMember(1),
                WeightedProvenance(500),
                MemberContent::Document("text".to_owned()),
            )],
        })),
        refused(check(ChangeOf::<DigitsMember>::Remove {
            members: vec![DigitsMember(1)],
        })),
        refused(check(ChangeOf::<TestMember, TaggedProvenance>::Insert {
            members: vec![tagged(1, &[], MemberContent::Vector(vec![0.0]))],
        })),
    ];
    for error in &refusals {
        assert_eq!(error.category(), ErrorCategory::Refused, "{error:?}");
    }
    // WHY: an analyzer pipeline this build does not implement is a missing
    // capability, not a malformed input, so it keeps its own category.
    let unsupported = refused(check(Change::Create {
        config: IndexConfig::Fts(unsupported_pipelines().remove(0)),
    }));
    assert_eq!(
        unsupported.category(),
        ErrorCategory::Unsupported,
        "{unsupported:?}"
    );
    Ok(())
}
