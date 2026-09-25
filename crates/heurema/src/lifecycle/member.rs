//! Consumer-owned member identity, provenance, and retention references, and
//! the members an operation carries.

use std::fmt;
use std::hash::Hash;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::SnapshotFamily;

/// The identity of one member of a named index, as the consumer names it.
///
/// The consumer implements this marker on its own newtype. heurēma defines no
/// member, provenance, or retention shape; it only requires that the
/// consumer's type can be ordered, hashed, and encoded.
///
/// Contract: an identity must serialize as a single JSON string or integer,
/// because index engines and version payloads key their maps by it, and its
/// encoding must be injective and consistent with its `Eq` and `Ord`. Two
/// distinct identities that encode alike would collapse into one map key and
/// lose a member without an error. It must also read back as itself from a
/// JSON object key and from a JSON value. serde_json writes an integer key
/// as its decimal digits, so an untagged enum that reads digits back as a
/// string variant breaks the first, and a type that reads back only from a
/// string breaks the second. Validation refuses an identity that does not
/// encode as a string or integer, or that does not survive either round
/// trip, with
/// [`HeuremaError::InvalidIdentifier`](crate::HeuremaError::InvalidIdentifier).
/// Injectivity and agreement with `Ord` cannot be checked from one value and
/// are the implementor's to keep.
///
/// WHY not blanket-implemented: a blanket impl would let a bare `u64` or
/// `String` stand in for a member identity. Without one, the orphan rule
/// stops a consumer implementing this heurēma trait for a type it does not
/// own, so every identity is a type the consumer names deliberately.
pub trait MemberIdentity: Ord + Hash + Clone + fmt::Debug + Serialize + DeserializeOwned {}

/// A reference to where one member's content came from, as the consumer
/// records it.
///
/// The consumer implements this marker on its own newtype. heurēma defines no
/// provenance shape: it stores and returns the consumer's value, and never
/// interprets it.
///
/// Contract: equal values must serialize identically, because an
/// operation's digest hashes what `Serialize` emits (the grammar is on
/// [`OperationDigest`](crate::OperationDigest)). Map entries are sorted
/// before hashing, but a sequence is hashed in the order it is emitted: hold
/// a set as a `BTreeSet` or a sorted `Vec`, never a `HashSet`. Floats, and
/// serde_json's `RawValue`, are refused with
/// [`HeuremaError::UnencodableOperation`](crate::HeuremaError::UnencodableOperation).
///
/// A value must also read back as itself, because heurēma stores it through
/// `serde_json`: `serde_json::from_slice(&serde_json::to_vec(&v)?)` must
/// equal `v`. A field skipped when serializing needs a serde default, and an
/// `Option<Option<T>>` loses `Some(None)`. Validation refuses a value that
/// does not read back with
/// [`HeuremaError::UnencodableOperation`](crate::HeuremaError::UnencodableOperation).
/// A stored record nests the value a few levels deep, so its own nesting
/// must stay well under serde_json's recursion limit of 128.
///
/// WHY not blanket-implemented: provenance is required on every member, and
/// a blanket impl would let `()` or a bare `String` satisfy that requirement
/// while recording nothing. Without one, the orphan rule makes the consumer
/// wrap its provenance in a type it owns.
pub trait ProvenanceReference: Clone + Eq + fmt::Debug + Serialize + DeserializeOwned {}

/// A reference to the retention decision that authorizes destroying an
/// index, as the consumer records it.
///
/// The consumer implements this marker on its own newtype. heurēma defines no
/// retention shape: it stores the consumer's value with the destroyed index's
/// record, and never interprets it.
///
/// Contract: as for [`ProvenanceReference`], equal values must serialize
/// identically, since the Destroy operation's digest hashes this value, and
/// a value must read back as itself from its JSON encoding.
///
/// WHY not blanket-implemented: physical deletion requires a retention
/// decision, and a blanket impl would let `()` stand in for one. Without one,
/// the orphan rule makes the consumer name the decision in a type it owns.
///
/// # Examples
///
/// ```
/// # use heurema::{
/// #     IndexChange, IndexIdentity, IndexName, LifecycleOperation, LifecycleTransition,
/// #     MemberIdentity, OperationKey, OwnerNamespace, ProvenanceReference, RetentionReference,
/// # };
/// # use serde::{Deserialize, Serialize};
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// # struct TestMember(u64);
/// # impl MemberIdentity for TestMember {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderProvenance(u32);
/// # impl ProvenanceReference for PlaceholderProvenance {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderRetention(u32);
/// # impl RetentionReference for PlaceholderRetention {}
/// # let index = IndexIdentity::new(OwnerNamespace::try_from("example")?, IndexName::try_from("notes")?);
/// # let key = OperationKey::try_from("destroy-notes")?;
/// let retention = PlaceholderRetention(3);
/// let destroy = LifecycleOperation::<TestMember, PlaceholderProvenance, _>::new(
///     index,
///     key,
///     IndexChange::Destroy { retention },
/// );
/// assert_eq!(destroy.transition(), LifecycleTransition::Destroy);
/// # Ok::<(), heurema::HeuremaError>(())
/// ```
///
/// `()` cannot stand in for a retention decision. This is the example above
/// with only the retention value changed:
///
/// ```compile_fail
/// # use heurema::{
/// #     IndexChange, IndexIdentity, IndexName, LifecycleOperation, LifecycleTransition,
/// #     MemberIdentity, OperationKey, OwnerNamespace, ProvenanceReference, RetentionReference,
/// # };
/// # use serde::{Deserialize, Serialize};
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// # struct TestMember(u64);
/// # impl MemberIdentity for TestMember {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderProvenance(u32);
/// # impl ProvenanceReference for PlaceholderProvenance {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderRetention(u32);
/// # impl RetentionReference for PlaceholderRetention {}
/// # let index = IndexIdentity::new(OwnerNamespace::try_from("example")?, IndexName::try_from("notes")?);
/// # let key = OperationKey::try_from("destroy-notes")?;
/// let retention = ();
/// let destroy = LifecycleOperation::<TestMember, PlaceholderProvenance, _>::new(
///     index,
///     key,
///     IndexChange::Destroy { retention },
/// );
/// assert_eq!(destroy.transition(), LifecycleTransition::Destroy);
/// # Ok::<(), heurema::HeuremaError>(())
/// ```
pub trait RetentionReference: Clone + Eq + fmt::Debug + Serialize + DeserializeOwned {}

/// What one member contributes to its index: a vector or a document.
///
/// WHY: one operation can target either index family, and the content's
/// variant is what ties a member to the family it may join.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MemberContent {
    /// A vector for a [`SnapshotFamily::Vector`] index.
    Vector(Vec<f32>),
    /// A document body for a [`SnapshotFamily::Fts`] index.
    Document(String),
}

impl MemberContent {
    /// The index family this content can join.
    #[must_use]
    pub const fn family(&self) -> SnapshotFamily {
        match self {
            Self::Vector(_) => SnapshotFamily::Vector,
            Self::Document(_) => SnapshotFamily::Fts,
        }
    }
}

/// One member an operation inserts: its identity, its provenance, and its
/// content.
///
/// WHY: no index version may hold a member without identity and provenance,
/// so the three travel together and provenance is a required field, not an
/// option. Construction outside this crate goes through [`IndexMember::new`]
/// or deserialization; both require the marker traits.
///
/// # Examples
///
/// ```
/// # use heurema::{IndexMember, MemberContent, MemberIdentity, ProvenanceReference};
/// # use serde::{Deserialize, Serialize};
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// # struct TestMember(u64);
/// # impl MemberIdentity for TestMember {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderProvenance(u32);
/// # impl ProvenanceReference for PlaceholderProvenance {}
/// let member = IndexMember::new(
///     TestMember(1),
///     PlaceholderProvenance(7),
///     MemberContent::Document("first note".to_owned()),
/// );
/// assert_eq!(member.id, TestMember(1));
/// ```
///
/// Leaving the provenance out does not compile. This is the example above
/// with the provenance argument removed:
///
/// ```compile_fail
/// # use heurema::{IndexMember, MemberContent, MemberIdentity, ProvenanceReference};
/// # use serde::{Deserialize, Serialize};
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// # struct TestMember(u64);
/// # impl MemberIdentity for TestMember {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderProvenance(u32);
/// # impl ProvenanceReference for PlaceholderProvenance {}
/// let member = IndexMember::new(
///     TestMember(1),
///     MemberContent::Document("first note".to_owned()),
/// );
/// assert_eq!(member.id, TestMember(1));
/// ```
///
/// Passing `None` does not compile either: provenance is required, not
/// optional. This is the first example with the provenance value replaced by
/// `None`; it starts compiling if the field ever becomes an `Option`:
///
/// ```compile_fail
/// # use heurema::{IndexMember, MemberContent, MemberIdentity, ProvenanceReference};
/// # use serde::{Deserialize, Serialize};
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// # struct TestMember(u64);
/// # impl MemberIdentity for TestMember {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderProvenance(u32);
/// # impl ProvenanceReference for PlaceholderProvenance {}
/// let member = IndexMember::<TestMember, PlaceholderProvenance>::new(
///     TestMember(1),
///     None,
///     MemberContent::Document("first note".to_owned()),
/// );
/// assert_eq!(member.id, TestMember(1));
/// ```
///
/// Nor does a bare `String` standing in for provenance, because `String` is
/// not a [`ProvenanceReference`], so `IndexMember<TestMember, String>` has no
/// constructor. This is the first example with only the provenance value
/// changed:
///
/// ```compile_fail
/// # use heurema::{IndexMember, MemberContent, MemberIdentity, ProvenanceReference};
/// # use serde::{Deserialize, Serialize};
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// # struct TestMember(u64);
/// # impl MemberIdentity for TestMember {}
/// # /// test-local placeholder; heurēma defines no provenance shape
/// # #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// # struct PlaceholderProvenance(u32);
/// # impl ProvenanceReference for PlaceholderProvenance {}
/// let member = IndexMember::new(
///     TestMember(1),
///     String::from("7"),
///     MemberContent::Document("first note".to_owned()),
/// );
/// assert_eq!(member.id, TestMember(1));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        serialize = "M: Serialize, P: Serialize",
        deserialize = "M: MemberIdentity, P: ProvenanceReference"
    )
)]
#[non_exhaustive]
pub struct IndexMember<M, P> {
    /// The member's identity within its index.
    pub id: M,
    /// Where the member's content came from.
    pub provenance: P,
    /// The vector or document the member contributes.
    pub content: MemberContent,
}

impl<M: MemberIdentity, P: ProvenanceReference> IndexMember<M, P> {
    /// WHY: the marker bounds live on the constructor, so a member cannot be
    /// built from a type that was never declared an identity or provenance.
    #[must_use]
    pub const fn new(id: M, provenance: P, content: MemberContent) -> Self {
        Self {
            id,
            provenance,
            content,
        }
    }
}
