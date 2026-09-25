//! The lifecycle operation digest: known answers built by hand from the byte
//! grammar documented on `OperationDigest`, and the properties that make it a
//! usable idempotency identity (order independence, content sensitivity,
//! refusal of values without a canonical encoding).
//!
//! The canonical bytes here are written from the documented grammar, not by
//! calling heurēma's encoder, so a change to the encoder that the grammar does
//! not describe fails these tests. The hard-coded digests were computed a
//! second way, by a separate script implementing the same grammar.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use heurema::{
    CheckedOperation, ErrorCategory, FtsConfig, HeuremaError, HnswConfig, IndexChange, IndexConfig,
    IndexIdentity, IndexMember, IndexName, LifecycleOperation, MemberContent, MemberIdentity,
    OperationDigest, OperationKey, OwnerNamespace, ProvenanceReference, RetentionReference,
};
use serde::{Deserialize, Serialize, Serializer};
use sha2::{Digest, Sha256};

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

/// test-local placeholder; heurēma defines no provenance shape. Wraps a map
/// and emits its entries in ascending key order.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct AscendingProvenance(BTreeMap<String, u32>);

impl Serialize for AscendingProvenance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter())
    }
}

impl ProvenanceReference for AscendingProvenance {}

/// test-local placeholder; heurēma defines no provenance shape. Wraps a map
/// and emits its entries in descending key order.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct DescendingProvenance(BTreeMap<String, u32>);

impl Serialize for DescendingProvenance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().rev())
    }
}

impl ProvenanceReference for DescendingProvenance {}

/// test-local placeholder; heurēma defines no provenance shape. Wraps a
/// `HashMap`, whose iteration order differs from one instance to the next.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct HashedProvenance(HashMap<String, u32>);

impl ProvenanceReference for HashedProvenance {}

/// test-local placeholder; heurēma defines no provenance shape. Serializes
/// as a float.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct WeightedProvenance(u32);

impl Serialize for WeightedProvenance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(f64::from(self.0) / 1000.0)
    }
}

impl ProvenanceReference for WeightedProvenance {}

/// test-local placeholder; heurēma defines no provenance shape. Holds a
/// float inside a map value.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct NanScoredProvenance;

impl Serialize for NanScoredProvenance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map([("score", f32::NAN)])
    }
}

impl ProvenanceReference for NanScoredProvenance {}

/// test-local placeholder; heurēma defines no provenance shape. A retention
/// reference that serializes as a float.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct FloatRetention;

impl Serialize for FloatRetention {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(0.25)
    }
}

impl RetentionReference for FloatRetention {}

/// test-local placeholder; heurēma defines no provenance shape. A map keyed
/// by pairs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PairKeyedProvenance(BTreeMap<(u8, u8), u32>);

impl ProvenanceReference for PairKeyedProvenance {}

/// test-local placeholder; heurēma defines no provenance shape. A map keyed
/// by bools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct BoolKeyedProvenance(BTreeMap<bool, u32>);

impl ProvenanceReference for BoolKeyedProvenance {}

/// test-local placeholder; heurēma defines no provenance shape. A map keyed
/// by integers, which the grammar accepts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct IntegerKeyedProvenance(BTreeMap<u64, u32>);

impl ProvenanceReference for IntegerKeyedProvenance {}

type Change = IndexChange<TestMember, PlaceholderProvenance, PlaceholderRetention>;
type ChangeOf<P, R> = IndexChange<TestMember, P, R>;

/// The domain prefix, as `OperationDigest`'s rustdoc states it.
const DOMAIN: &[u8] = b"heurema.lifecycle.operation.v1\n";

// The grammar's items, written from `OperationDigest`'s rustdoc.

fn count(value: usize) -> Vec<u8> {
    (value as u64).to_be_bytes().to_vec()
}

fn string(value: &str) -> Vec<u8> {
    [vec![0x40], count(value.len()), value.as_bytes().to_vec()].concat()
}

fn u32_item(value: u32) -> Vec<u8> {
    [vec![0x12], value.to_be_bytes().to_vec()].concat()
}

fn u64_item(value: u64) -> Vec<u8> {
    [vec![0x13], value.to_be_bytes().to_vec()].concat()
}

fn seq(items: &[Vec<u8>]) -> Vec<u8> {
    [vec![0x50], count(items.len()), items.concat()].concat()
}

/// A map or struct with string keys. The entries must be listed in the
/// documented order (ascending encoded key bytes: shorter keys first, then
/// bytewise), and this helper checks that they are, so a mistake in a
/// hand-written expectation fails loudly here rather than as a digest
/// mismatch.
fn map(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
    for pair in entries.windows(2) {
        assert!(
            string(pair[0].0) < string(pair[1].0),
            "expected entries out of documented order: {:?} before {:?}",
            pair[0].0,
            pair[1].0
        );
    }
    let body: Vec<u8> = entries
        .iter()
        .flat_map(|(key, value)| [string(key), value.clone()].concat())
        .collect();
    [vec![0x51], count(entries.len()), body].concat()
}

fn variant(name: &str, payload: Vec<u8>) -> Vec<u8> {
    [vec![0x60], string(name), payload].concat()
}

fn index_item() -> Vec<u8> {
    map(&[("name", string("notes")), ("namespace", string("example"))])
}

fn operation_item(transition: &str, change: Vec<u8>) -> Vec<u8> {
    map(&[
        ("index", index_item()),
        ("change", change),
        ("transition", string(transition)),
    ])
}

fn vector_member_item(id: u64, provenance: Vec<u8>, bits: &[u32]) -> Vec<u8> {
    let components: Vec<Vec<u8>> = bits.iter().map(|bits| u32_item(*bits)).collect();
    map(&[
        ("id", u64_item(id)),
        ("content", variant("vector_bits", seq(&components))),
        ("provenance", provenance),
    ])
}

fn sha256(item: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    hasher.update(item);
    hasher.finalize().into()
}

fn index() -> Result<IndexIdentity, HeuremaError> {
    Ok(IndexIdentity::new(
        OwnerNamespace::try_from("example")?,
        IndexName::try_from("notes")?,
    ))
}

fn digest_of<P: ProvenanceReference, R: RetentionReference>(
    key: &str,
    change: ChangeOf<P, R>,
) -> Result<OperationDigest, HeuremaError> {
    let operation = LifecycleOperation::new(index()?, OperationKey::try_from(key)?, change);
    Ok(CheckedOperation::check(operation)?.identity().digest)
}

fn vector(
    id: u64,
    provenance: u32,
    components: &[f32],
) -> IndexMember<TestMember, PlaceholderProvenance> {
    IndexMember::new(
        TestMember(id),
        PlaceholderProvenance(provenance),
        MemberContent::Vector(components.to_vec()),
    )
}

/// Member 1, vector `[0.0, 1.0]`, with the given provenance.
fn member_with<P: ProvenanceReference>(provenance: P) -> IndexMember<TestMember, P> {
    IndexMember::new(
        TestMember(1),
        provenance,
        MemberContent::Vector(vec![0.0, 1.0]),
    )
}

fn refusal_reason<P: ProvenanceReference, R: RetentionReference>(change: ChangeOf<P, R>) -> String {
    let error = match digest_of("refused", change) {
        Ok(digest) => panic!("expected a refusal, got digest {digest}"),
        Err(error) => error,
    };
    assert_eq!(error.category(), ErrorCategory::Refused, "{error:?}");
    let HeuremaError::UnencodableOperation { reason, .. } = error else {
        panic!("unexpected {error:?}");
    };
    reason
}

/// One known answer: an operation's change, its canonical item written by
/// hand from the grammar, and its digest as computed independently.
struct KnownAnswer {
    name: &'static str,
    change: Change,
    item: Vec<u8>,
    digest: &'static str,
}

fn known_insert() -> KnownAnswer {
    KnownAnswer {
        name: "insert",
        // Members are listed out of order; the canonical form sorts them by
        // id. Components are written as bit patterns: 0.5, 2.0, 1.0, -0.0.
        change: Change::Insert {
            members: vec![vector(2, 8, &[1.0, -0.0]), vector(1, 7, &[0.5, 2.0])],
        },
        item: operation_item(
            "Insert",
            map(&[(
                "members",
                seq(&[
                    vector_member_item(1, u32_item(7), &[0x3f00_0000, 0x4000_0000]),
                    vector_member_item(2, u32_item(8), &[0x3f80_0000, 0x8000_0000]),
                ]),
            )]),
        ),
        digest: "9a819419041aacedd407cf83b78c0ac0b78940f729075709c28c9db8137ef2b2",
    }
}

fn known_create() -> KnownAnswer {
    KnownAnswer {
        name: "create",
        change: Change::Create {
            config: IndexConfig::Vector(HnswConfig::new(4)),
        },
        item: operation_item(
            "Create",
            map(&[(
                "config",
                variant(
                    "Vector",
                    map(&[
                        ("distance", string("L2")),
                        ("dimensions", u64_item(4)),
                        ("m_neighbours", u64_item(16)),
                        ("ef_construction", u64_item(50)),
                    ]),
                ),
            )]),
        ),
        digest: "87fe6306436b94d07970809c3d9b3fbb8803f26bfa164f2779045fe12ac4df2e",
    }
}

fn known_rebuild() -> KnownAnswer {
    let config = variant(
        "Fts",
        map(&[
            ("filters", seq(&[])),
            (
                "tokenizer",
                map(&[("args", seq(&[])), ("name", string("Simple"))]),
            ),
        ]),
    );
    let member = map(&[
        ("id", u64_item(3)),
        ("content", variant("document", string("Hello world"))),
        ("provenance", u32_item(9)),
    ]);
    KnownAnswer {
        name: "rebuild",
        change: Change::Rebuild {
            config: IndexConfig::Fts(FtsConfig::simple()),
            members: vec![IndexMember::new(
                TestMember(3),
                PlaceholderProvenance(9),
                MemberContent::Document("Hello world".to_owned()),
            )],
        },
        item: operation_item(
            "Rebuild",
            map(&[("config", config), ("members", seq(&[member]))]),
        ),
        digest: "32b1d29a7aaa8e9fe032d9df6c0c8096dae3836925a045b5c107ca4289d9aa9a",
    }
}

fn known_remove() -> KnownAnswer {
    KnownAnswer {
        name: "remove",
        change: Change::Remove {
            members: vec![TestMember(9), TestMember(4)],
        },
        item: operation_item(
            "Remove",
            map(&[("members", seq(&[u64_item(4), u64_item(9)]))]),
        ),
        digest: "64cafa5dd784a6072b52f631426ce25500d147ee123cf246eaf6595c6ee249b2",
    }
}

fn known_destroy() -> KnownAnswer {
    KnownAnswer {
        name: "destroy",
        change: Change::Destroy {
            retention: PlaceholderRetention(5),
        },
        item: operation_item("Destroy", map(&[("retention", u32_item(5))])),
        digest: "94bd1963641778aaf6b57da893d491aef7c37957feb17ce7be01a9c5c8a9e1af",
    }
}

#[test]
fn digest_matches_the_documented_canonical_bytes() -> Result<(), HeuremaError> {
    let cases = [
        known_insert(),
        known_create(),
        known_rebuild(),
        known_remove(),
        known_destroy(),
    ];
    for KnownAnswer {
        name,
        change,
        item,
        digest: known,
    } in cases
    {
        let digest = digest_of("kat", change)?;
        assert_eq!(
            digest.as_bytes(),
            &sha256(&item),
            "{name}: digest is SHA-256 of the prefix and the documented bytes"
        );
        assert_eq!(digest.to_string(), known, "{name}: pinned digest");
    }
    Ok(())
}

#[test]
fn member_order_does_not_change_the_digest() -> Result<(), HeuremaError> {
    let members = [
        vector(3, 1, &[0.0, 1.0]),
        vector(1, 2, &[1.0, 0.0]),
        vector(2, 3, &[1.0, 1.0]),
    ];
    let mut reversed = members.to_vec();
    reversed.reverse();

    let insert = digest_of(
        "k",
        Change::Insert {
            members: members.to_vec(),
        },
    )?;
    let insert_reversed = digest_of("k", Change::Insert { members: reversed })?;
    assert_eq!(insert, insert_reversed, "insert members");

    let rebuild = |members: Vec<_>| {
        digest_of(
            "k",
            Change::Rebuild {
                config: IndexConfig::Vector(HnswConfig::new(2)),
                members,
            },
        )
    };
    let mut rotated = members.to_vec();
    rotated.rotate_left(1);
    assert_eq!(
        rebuild(members.to_vec())?,
        rebuild(rotated)?,
        "rebuild members"
    );

    let remove = |ids: [u64; 3]| {
        digest_of(
            "k",
            Change::Remove {
                members: ids.into_iter().map(TestMember).collect(),
            },
        )
    };
    assert_eq!(remove([3, 1, 2])?, remove([1, 2, 3])?, "removed identities");
    Ok(())
}

#[test]
fn map_entry_order_in_provenance_does_not_change_the_digest() -> Result<(), HeuremaError> {
    let entries = [("alpha", 1_u32), ("beta", 2), ("gamma", 3)];
    let tree: BTreeMap<String, u32> = entries
        .iter()
        .map(|(key, value)| ((*key).to_owned(), *value))
        .collect();

    let ascending = digest_of(
        "k",
        ChangeOf::<_, PlaceholderRetention>::Insert {
            members: vec![member_with(AscendingProvenance(tree.clone()))],
        },
    )?;
    let descending = digest_of(
        "k",
        ChangeOf::<_, PlaceholderRetention>::Insert {
            members: vec![member_with(DescendingProvenance(tree))],
        },
    )?;
    assert_eq!(ascending, descending, "opposite emission orders");

    // WHY many maps: each `HashMap` has its own random hasher, so these
    // iterate in different orders. The test first proves that they do, so
    // the digest comparison below is not vacuous.
    let mut orders = BTreeSet::new();
    for rotation in 0..32 {
        let mut hashed = HashMap::new();
        for (key, value) in entries.iter().cycle().skip(rotation % 3).take(3) {
            hashed.insert((*key).to_owned(), *value);
        }
        orders.insert(hashed.keys().cloned().collect::<Vec<_>>());
        let digest = digest_of(
            "k",
            ChangeOf::<_, PlaceholderRetention>::Insert {
                members: vec![member_with(HashedProvenance(hashed))],
            },
        )?;
        assert_eq!(digest, ascending, "HashMap iteration order {rotation}");
    }
    assert!(
        orders.len() > 1,
        "the HashMaps iterated in only one order, so they proved nothing: {orders:?}"
    );

    // The documented order is neither source order: "beta" (4 bytes) sorts
    // before "alpha" and "gamma" (5 bytes each).
    let item = operation_item(
        "Insert",
        map(&[(
            "members",
            seq(&[map(&[
                ("id", u64_item(1)),
                (
                    "content",
                    variant("vector_bits", seq(&[u32_item(0), u32_item(0x3f80_0000)])),
                ),
                (
                    "provenance",
                    map(&[
                        ("beta", u32_item(2)),
                        ("alpha", u32_item(1)),
                        ("gamma", u32_item(3)),
                    ]),
                ),
            ])]),
        )]),
    );
    assert_eq!(ascending.as_bytes(), &sha256(&item), "documented map order");
    assert_eq!(
        ascending.to_string(),
        "95e26ef97814cbf356543f666f0ff429b7ddd5a4115bf6cab005e646a1f3c072",
        "pinned digest"
    );
    Ok(())
}

#[test]
fn same_key_different_content_yields_different_digests() -> Result<(), HeuremaError> {
    let mut seen = BTreeMap::new();
    let variants = content_variants();
    let total = variants.len();
    for (name, change) in variants {
        let digest = digest_of("same-key", change)?.to_string();
        if let Some(previous) = seen.insert(digest, name) {
            panic!("{name} and {previous} share one digest");
        }
    }
    assert_eq!(seen.len(), total, "every content change changes the digest");

    let other_index = LifecycleOperation::new(
        IndexIdentity::new(
            OwnerNamespace::try_from("example")?,
            IndexName::try_from("other")?,
        ),
        OperationKey::try_from("same-key")?,
        base_insert(),
    );
    let other_index = CheckedOperation::check(other_index)?.identity().digest;
    assert_ne!(
        other_index,
        digest_of("same-key", base_insert())?,
        "the index is part of the operation"
    );

    // WHY: the key is not hashed. Identity is key plus digest, so the same
    // content under two keys has one digest and two identities.
    let first = CheckedOperation::check(LifecycleOperation::new(
        index()?,
        OperationKey::try_from("first-key")?,
        base_insert(),
    ))?;
    let second = CheckedOperation::check(LifecycleOperation::new(
        index()?,
        OperationKey::try_from("second-key")?,
        base_insert(),
    ))?;
    assert_eq!(
        first.identity().digest,
        second.identity().digest,
        "the key is not part of the digest"
    );
    assert_ne!(
        first.identity(),
        second.identity(),
        "the key is part of the identity"
    );
    Ok(())
}

fn base_insert() -> Change {
    Change::Insert {
        members: vec![vector(1, 7, &[0.0, 1.0])],
    }
}

/// The base insert and changes that each differ from it, or from each
/// other, in one part of their content.
fn content_variants() -> Vec<(&'static str, Change)> {
    vec![
        ("base", base_insert()),
        (
            "component",
            Change::Insert {
                members: vec![vector(1, 7, &[0.0, 2.0])],
            },
        ),
        (
            "provenance",
            Change::Insert {
                members: vec![vector(1, 8, &[0.0, 1.0])],
            },
        ),
        (
            "member id",
            Change::Insert {
                members: vec![vector(2, 7, &[0.0, 1.0])],
            },
        ),
        (
            "extra member",
            Change::Insert {
                members: vec![vector(1, 7, &[0.0, 1.0]), vector(2, 7, &[0.0, 1.0])],
            },
        ),
        (
            "document content",
            Change::Insert {
                members: vec![IndexMember::new(
                    TestMember(1),
                    PlaceholderProvenance(7),
                    MemberContent::Document("0 1".to_owned()),
                )],
            },
        ),
        (
            "remove transition",
            Change::Remove {
                members: vec![TestMember(1)],
            },
        ),
        (
            "rebuild transition",
            Change::Rebuild {
                config: IndexConfig::Vector(HnswConfig::new(2)),
                members: vec![vector(1, 7, &[0.0, 1.0])],
            },
        ),
        (
            "create config",
            Change::Create {
                config: IndexConfig::Vector(HnswConfig::new(2)),
            },
        ),
        (
            "create other config",
            Change::Create {
                config: IndexConfig::Vector(HnswConfig::new(3)),
            },
        ),
        (
            "destroy",
            Change::Destroy {
                retention: PlaceholderRetention(7),
            },
        ),
        (
            "destroy other retention",
            Change::Destroy {
                retention: PlaceholderRetention(8),
            },
        ),
    ]
}

#[test]
fn negative_and_positive_zero_are_distinct_operations() -> Result<(), HeuremaError> {
    let positive = digest_of(
        "k",
        Change::Insert {
            members: vec![vector(1, 7, &[0.0, 1.0])],
        },
    )?;
    let negative = digest_of(
        "k",
        Change::Insert {
            members: vec![vector(1, 7, &[-0.0, 1.0])],
        },
    )?;
    assert_ne!(
        positive, negative,
        "components are hashed by bit pattern, so -0.0 and 0.0 differ"
    );
    Ok(())
}

#[test]
fn float_in_consumer_provenance_is_refused() {
    let reasons = [
        (
            refusal_reason(ChangeOf::<_, PlaceholderRetention>::Insert {
                members: vec![member_with(WeightedProvenance(500))],
            }),
            "f64",
        ),
        (
            refusal_reason(ChangeOf::<_, PlaceholderRetention>::Insert {
                members: vec![member_with(NanScoredProvenance)],
            }),
            "f32",
        ),
        (
            refusal_reason(ChangeOf::<PlaceholderProvenance, _>::Destroy {
                retention: FloatRetention,
            }),
            "f32",
        ),
    ];
    for (reason, float) in reasons {
        assert!(
            reason.contains(float) && reason.contains("no canonical encoding"),
            "{reason}"
        );
    }
}

#[test]
fn non_string_non_integer_map_key_is_refused() -> Result<(), HeuremaError> {
    let pair_keyed = refusal_reason(ChangeOf::<_, PlaceholderRetention>::Insert {
        members: vec![member_with(PairKeyedProvenance(BTreeMap::from([(
            (1, 2),
            3,
        )])))],
    });
    assert_eq!(
        pair_keyed,
        "a map key encodes as a sequence; map keys must encode as a string or an integer"
    );
    let bool_keyed = refusal_reason(ChangeOf::<_, PlaceholderRetention>::Insert {
        members: vec![member_with(BoolKeyedProvenance(BTreeMap::from([(
            true, 3,
        )])))],
    });
    assert_eq!(
        bool_keyed,
        "a map key encodes as a bool; map keys must encode as a string or an integer"
    );

    digest_of(
        "k",
        ChangeOf::<_, PlaceholderRetention>::Insert {
            members: vec![member_with(IntegerKeyedProvenance(BTreeMap::from([(
                7, 3,
            )])))],
        },
    )?;
    Ok(())
}
