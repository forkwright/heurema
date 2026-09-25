//! The canonical encoding of a lifecycle operation and its SHA-256 digest.
//!
//! The byte grammar is documented on [`OperationDigest`], the public type this
//! module computes; this module implements that grammar as a private
//! `serde::Serializer`.
//!
//! WHY a private serializer rather than hashing `serde_json` output: the
//! digest must be the same in every build that links heurēma, and serde_json's
//! output is decided by the consumer's whole dependency graph. Any crate there
//! can unify serde_json's `preserve_order` feature on, which turns a
//! `HashMap` inside provenance into per-process random key order; the float
//! formatter can change within serde_json 1.x; and `to_value` maps NaN and
//! both infinities to `null`, so two different provenance values could share
//! one digest. This encoder sorts map entries itself, refuses floats, and
//! depends on nothing but serde's data model. serde_json's own types are
//! held to the same rule: with `arbitrary_precision` unified on, a
//! `serde_json::Number` serializes as a private struct holding its decimal
//! text, and the encoder writes an integer there as the same item the default
//! build writes, so that feature cannot change a digest either.
//!
//! What the encoder cannot make canonical is a consumer type whose own
//! `Serialize` output differs between equal values. A sequence is hashed in
//! the order it is emitted, because sorting it would merge two operations
//! that differ only in order, so a `HashSet` inside provenance or retention
//! would give one value a different digest from one process to the next. The
//! marker traits' contracts rule such types out.

use std::fmt;

use serde::Serialize;
use serde::ser::{self, SerializeMap, SerializeStruct};
use sha2::{Digest, Sha256};

use super::identity::{IndexIdentity, OperationDigest};
use super::member::{
    IndexMember, MemberContent, MemberIdentity, ProvenanceReference, RetentionReference,
};
use super::operation::{IndexChange, IndexConfig, LifecycleOperation, LifecycleTransition};
use crate::HeuremaError;
use crate::error::UnencodableOperationSnafu;

/// The domain prefix hashed before every version-1 canonical encoding.
///
/// WHY: the version in the prefix separates encodings; a later grammar gets a
/// new prefix, so its digests can never collide with version-1 digests.
const DOMAIN: &[u8] = b"heurema.lifecycle.operation.v1\n";

/// Item tags, one per row of the grammar table on [`OperationDigest`].
mod tag {
    pub(super) const UNIT: u8 = 0x00;
    pub(super) const NONE: u8 = 0x01;
    pub(super) const SOME: u8 = 0x02;
    pub(super) const FALSE: u8 = 0x03;
    pub(super) const TRUE: u8 = 0x04;
    pub(super) const U8: u8 = 0x10;
    pub(super) const U16: u8 = 0x11;
    pub(super) const U32: u8 = 0x12;
    pub(super) const U64: u8 = 0x13;
    pub(super) const U128: u8 = 0x14;
    pub(super) const I8: u8 = 0x18;
    pub(super) const I16: u8 = 0x19;
    pub(super) const I32: u8 = 0x1a;
    pub(super) const I64: u8 = 0x1b;
    pub(super) const I128: u8 = 0x1c;
    pub(super) const STR: u8 = 0x40;
    pub(super) const BYTES: u8 = 0x41;
    pub(super) const SEQ: u8 = 0x50;
    pub(super) const MAP: u8 = 0x51;
    pub(super) const VARIANT: u8 = 0x60;
}

/// Why the encoder refused a value.
#[derive(Debug)]
pub(super) struct EncodeError(String);

impl fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for EncodeError {}

impl ser::Error for EncodeError {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Self(message.to_string())
    }
}

/// Encodes `value` as one canonical item.
pub(super) fn encode<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, EncodeError> {
    let mut out = Vec::new();
    value.serialize(Encoder { out: &mut out })?;
    Ok(out)
}

/// What kind of value an encoded item holds, for refusal messages.
fn describe(item: &[u8]) -> &'static str {
    match item.first() {
        Some(&tag::UNIT) => "a unit",
        Some(&tag::NONE) => "an absent option",
        Some(&tag::SOME) => "a present option",
        Some(&(tag::FALSE | tag::TRUE)) => "a bool",
        Some(&(tag::U8..=tag::U128 | tag::I8..=tag::I128)) => "an integer",
        Some(&tag::STR) => "a string",
        Some(&tag::BYTES) => "a byte string",
        Some(&tag::SEQ) => "a sequence",
        Some(&tag::MAP) => "a map or struct",
        Some(&tag::VARIANT) => "an enum variant with data",
        Some(_) | None => "an unrecognised item",
    }
}

/// Whether an encoded item may serve as a map key: a string or an integer.
fn is_key(item: &[u8]) -> bool {
    matches!(
        item.first(),
        Some(&(tag::STR | tag::U8..=tag::U128 | tag::I8..=tag::I128))
    )
}

/// Why `id` cannot identify a member, or `None` when it encodes as a string
/// or an integer.
///
/// WHY: engine snapshots key their members by identity, and a key must be a
/// string or an integer there as it is here.
pub(super) fn member_identity_refusal<M: Serialize>(id: &M) -> Option<String> {
    match encode(id) {
        Err(error) => Some(error.0),
        Ok(item) if is_key(&item) => None,
        Ok(item) => Some(format!(
            "encodes as {}; a member identity must encode as a string or an integer, \
             because engine snapshots key members by it",
            describe(&item)
        )),
    }
}

/// The digest of `operation`: SHA-256 over the domain prefix and the
/// operation's canonical item.
///
/// INVARIANT: the caller has already refused a batch that names one member
/// twice. Members are sorted by identity here, and a duplicate would leave
/// the order of the two entries to the caller's input order.
pub(super) fn operation_digest<M, P, R>(
    operation: &LifecycleOperation<M, P, R>,
) -> Result<OperationDigest, HeuremaError>
where
    M: MemberIdentity,
    P: ProvenanceReference,
    R: RetentionReference,
{
    let item = canonical_bytes(operation).map_err(|error| {
        UnencodableOperationSnafu {
            reason: error.to_string(),
        }
        .build()
    })?;
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    hasher.update(&item);
    Ok(OperationDigest::from_bytes(hasher.finalize().into()))
}

/// The canonical item of `operation`, without the domain prefix.
pub(super) fn canonical_bytes<M, P, R>(
    operation: &LifecycleOperation<M, P, R>,
) -> Result<Vec<u8>, EncodeError>
where
    M: MemberIdentity,
    P: ProvenanceReference,
    R: RetentionReference,
{
    encode(&CanonicalOperation::new(operation))
}

/// The encoded shape of an operation, documented on [`OperationDigest`].
#[derive(Serialize)]
struct CanonicalOperation<'a, M, P, R> {
    index: &'a IndexIdentity,
    transition: LifecycleTransition,
    change: CanonicalChange<'a, M, P, R>,
}

/// WHY untagged: `transition` already names the change, so each change
/// encodes as its fields alone.
#[derive(Serialize)]
#[serde(untagged)]
enum CanonicalChange<'a, M, P, R> {
    Create {
        config: &'a IndexConfig,
    },
    Insert {
        members: Vec<CanonicalMember<'a, M, P>>,
    },
    Remove {
        members: Vec<&'a M>,
    },
    Rebuild {
        config: &'a IndexConfig,
        members: Vec<CanonicalMember<'a, M, P>>,
    },
    Destroy {
        retention: &'a R,
    },
}

#[derive(Serialize)]
struct CanonicalMember<'a, M, P> {
    id: &'a M,
    provenance: &'a P,
    content: CanonicalContent<'a>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum CanonicalContent<'a> {
    VectorBits(VectorBits<'a>),
    Document(&'a str),
}

/// A vector written as the IEEE 754 bit patterns of its components.
///
/// WHY: bit patterns are exact and formatter-independent, and they keep
/// `-0.0` distinct from `0.0`. The encoder refuses every float, so this is
/// the only path by which a heurēma vector reaches the digest.
struct VectorBits<'a>(&'a [f32]);

impl Serialize for VectorBits<'_> {
    fn serialize<S: ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(|component| component.to_bits()))
    }
}

impl<'a, M: Ord, P, R> CanonicalOperation<'a, M, P, R> {
    fn new(operation: &'a LifecycleOperation<M, P, R>) -> Self {
        let change = match &operation.change {
            IndexChange::Create { config } => CanonicalChange::Create { config },
            IndexChange::Insert { members } => CanonicalChange::Insert {
                members: sorted_members(members),
            },
            IndexChange::Remove { members } => {
                let mut ids: Vec<&M> = members.iter().collect();
                ids.sort_unstable();
                CanonicalChange::Remove { members: ids }
            }
            IndexChange::Rebuild { config, members } => CanonicalChange::Rebuild {
                config,
                members: sorted_members(members),
            },
            IndexChange::Destroy { retention } => CanonicalChange::Destroy { retention },
        };
        Self {
            index: &operation.index,
            transition: operation.change.transition(),
            change,
        }
    }
}

fn sorted_members<M: Ord, P>(members: &[IndexMember<M, P>]) -> Vec<CanonicalMember<'_, M, P>> {
    let mut sorted: Vec<CanonicalMember<'_, M, P>> = members
        .iter()
        .map(|member| CanonicalMember {
            id: &member.id,
            provenance: &member.provenance,
            content: match &member.content {
                MemberContent::Vector(vector) => CanonicalContent::VectorBits(VectorBits(vector)),
                MemberContent::Document(document) => CanonicalContent::Document(document),
            },
        })
        .collect();
    sorted.sort_unstable_by(|left, right| left.id.cmp(right.id));
    sorted
}

/// Writes one item into `out`.
///
/// INVARIANT: a `Serialize` impl can only produce `Ok` by calling exactly one
/// serializer method (or `end` on the compound one returns), and every method
/// here writes a tag, so every successful call appends exactly one item.
struct Encoder<'a> {
    out: &'a mut Vec<u8>,
}

/// A length or count as the grammar writes it: a `u64`, big-endian.
fn count(len: usize) -> Result<[u8; 8], EncodeError> {
    u64::try_from(len)
        .map(u64::to_be_bytes)
        .map_err(|_| EncodeError(format!("a length of {len} does not fit the grammar's u64")))
}

/// `what` names the refused value with its article, such as "an f32".
fn float_refusal(what: &str) -> EncodeError {
    EncodeError(format!(
        "a consumer value contains {what}; floating-point numbers have no canonical \
         encoding, so member identity, provenance, and retention types must encode them as \
         integers or strings"
    ))
}

/// The struct name serde_json gives a `Number` when its `arbitrary_precision`
/// feature is on; the struct's one field, of the same name, holds the
/// number's decimal text.
///
/// WARNING: this is serde_json's private serialization token. If serde_json
/// renames it, such a number encodes as a one-entry map again, and the
/// `serde_json_numbers_encode_alike_with_and_without_arbitrary_precision`
/// test no longer describes serde_json.
const SERDE_JSON_NUMBER: &str = "$serde_json::private::Number";

/// The struct name serde_json gives a `RawValue`; its one field holds
/// unparsed JSON text.
const SERDE_JSON_RAW_VALUE: &str = "$serde_json::private::RawValue";

/// The item a serde_json `Number` with decimal text `text` gets.
///
/// WHY: without `arbitrary_precision`, serde_json holds a number as a `u64`
/// when it is a non-negative integer that fits one, as an `i64` when it is a
/// negative integer that fits one, and as an `f64` otherwise, `-0` included;
/// it serializes each through the matching serializer method. Writing the
/// text the same way keeps the feature from changing a digest. The text must
/// be exactly the integer's decimal form, so `01` or `+1` is refused rather
/// than read as `1`.
fn serde_json_number_item(text: &str) -> Result<Vec<u8>, EncodeError> {
    if let Ok(value) = text.parse::<u64>()
        && value.to_string() == text
    {
        return encode(&value);
    }
    if let Ok(value) = text.parse::<i64>()
        && value < 0
        && value.to_string() == text
    {
        return encode(&value);
    }
    Err(float_refusal(&format!(
        "the serde_json number {text}, which is not a u64 or a negative i64 and so is a float \
         in serde_json's default build"
    )))
}

/// The UTF-8 text of an encoded `0x40` item, or `None` for any other item.
fn decoded_str(item: &[u8]) -> Option<&str> {
    let (&tag::STR, rest) = item.split_first()? else {
        return None;
    };
    let (_, bytes) = rest.split_first_chunk::<8>()?;
    std::str::from_utf8(bytes).ok()
}

impl Encoder<'_> {
    fn fixed(self, tag: u8, payload: &[u8]) {
        self.out.push(tag);
        self.out.extend_from_slice(payload);
    }

    fn length_prefixed(self, tag: u8, payload: &[u8]) -> Result<(), EncodeError> {
        let length = count(payload.len())?;
        self.out.push(tag);
        self.out.extend_from_slice(&length);
        self.out.extend_from_slice(payload);
        Ok(())
    }

    /// Writes the `0x60` tag and the variant name that open a variant item.
    fn open_variant(&mut self, variant: &str) -> Result<(), EncodeError> {
        let length = count(variant.len())?;
        self.out.push(tag::VARIANT);
        self.out.push(tag::STR);
        self.out.extend_from_slice(&length);
        self.out.extend_from_slice(variant.as_bytes());
        Ok(())
    }
}

impl<'a> ser::Serializer for Encoder<'a> {
    type Ok = ();
    type Error = EncodeError;
    type SerializeSeq = Seq<'a>;
    type SerializeTuple = Seq<'a>;
    type SerializeTupleStruct = Seq<'a>;
    type SerializeTupleVariant = Seq<'a>;
    type SerializeMap = Map<'a>;
    type SerializeStruct = Struct<'a>;
    type SerializeStructVariant = Map<'a>;

    fn serialize_bool(self, value: bool) -> Result<(), EncodeError> {
        self.fixed(if value { tag::TRUE } else { tag::FALSE }, &[]);
        Ok(())
    }

    fn serialize_i8(self, value: i8) -> Result<(), EncodeError> {
        self.fixed(tag::I8, &value.to_be_bytes());
        Ok(())
    }

    fn serialize_i16(self, value: i16) -> Result<(), EncodeError> {
        self.fixed(tag::I16, &value.to_be_bytes());
        Ok(())
    }

    fn serialize_i32(self, value: i32) -> Result<(), EncodeError> {
        self.fixed(tag::I32, &value.to_be_bytes());
        Ok(())
    }

    fn serialize_i64(self, value: i64) -> Result<(), EncodeError> {
        self.fixed(tag::I64, &value.to_be_bytes());
        Ok(())
    }

    fn serialize_i128(self, value: i128) -> Result<(), EncodeError> {
        self.fixed(tag::I128, &value.to_be_bytes());
        Ok(())
    }

    fn serialize_u8(self, value: u8) -> Result<(), EncodeError> {
        self.fixed(tag::U8, &value.to_be_bytes());
        Ok(())
    }

    fn serialize_u16(self, value: u16) -> Result<(), EncodeError> {
        self.fixed(tag::U16, &value.to_be_bytes());
        Ok(())
    }

    fn serialize_u32(self, value: u32) -> Result<(), EncodeError> {
        self.fixed(tag::U32, &value.to_be_bytes());
        Ok(())
    }

    fn serialize_u64(self, value: u64) -> Result<(), EncodeError> {
        self.fixed(tag::U64, &value.to_be_bytes());
        Ok(())
    }

    fn serialize_u128(self, value: u128) -> Result<(), EncodeError> {
        self.fixed(tag::U128, &value.to_be_bytes());
        Ok(())
    }

    fn serialize_f32(self, _value: f32) -> Result<(), EncodeError> {
        Err(float_refusal("an f32"))
    }

    fn serialize_f64(self, _value: f64) -> Result<(), EncodeError> {
        Err(float_refusal("an f64"))
    }

    fn serialize_char(self, value: char) -> Result<(), EncodeError> {
        let mut buffer = [0_u8; 4];
        self.serialize_str(value.encode_utf8(&mut buffer))
    }

    fn serialize_str(self, value: &str) -> Result<(), EncodeError> {
        self.length_prefixed(tag::STR, value.as_bytes())
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<(), EncodeError> {
        self.length_prefixed(tag::BYTES, value)
    }

    fn serialize_none(self) -> Result<(), EncodeError> {
        self.fixed(tag::NONE, &[]);
        Ok(())
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), EncodeError> {
        self.out.push(tag::SOME);
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<(), EncodeError> {
        self.fixed(tag::UNIT, &[]);
        Ok(())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<(), EncodeError> {
        self.serialize_unit()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
    ) -> Result<(), EncodeError> {
        self.serialize_str(variant)
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<(), EncodeError> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        mut self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), EncodeError> {
        self.open_variant(variant)?;
        value.serialize(self)
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Seq<'a>, EncodeError> {
        Ok(Seq::new(self.out))
    }

    fn serialize_tuple(self, _len: usize) -> Result<Seq<'a>, EncodeError> {
        Ok(Seq::new(self.out))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Seq<'a>, EncodeError> {
        Ok(Seq::new(self.out))
    }

    fn serialize_tuple_variant(
        mut self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Seq<'a>, EncodeError> {
        self.open_variant(variant)?;
        Ok(Seq::new(self.out))
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Map<'a>, EncodeError> {
        Ok(Map::new(self.out))
    }

    fn serialize_struct(self, name: &'static str, _len: usize) -> Result<Struct<'a>, EncodeError> {
        match name {
            SERDE_JSON_NUMBER => Ok(Struct::SerdeJsonNumber {
                out: self.out,
                text: None,
            }),
            SERDE_JSON_RAW_VALUE => Err(EncodeError(
                "a consumer value contains a serde_json RawValue; unparsed JSON text has no \
                 canonical encoding"
                    .to_owned(),
            )),
            _ => Ok(Struct::Fields(Map::new(self.out))),
        }
    }

    fn serialize_struct_variant(
        mut self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Map<'a>, EncodeError> {
        self.open_variant(variant)?;
        Ok(Map::new(self.out))
    }
}

/// A `0x50` item being written: elements are encoded into `body` as they
/// arrive, and the tag and count are written ahead of them at `end`.
///
/// WHY buffered: `serialize_seq` may be called without a length, and the
/// count precedes the elements.
///
/// PERF: this body, and each map's entries, are copied once more into the
/// enclosing item, so a deeply nested byte is copied once per level and the
/// whole canonical item is held in memory before it is hashed. Streaming
/// heurēma's own levels into the hasher is tracked in #58; the grammar does
/// not change.
struct Seq<'a> {
    out: &'a mut Vec<u8>,
    elements: usize,
    body: Vec<u8>,
}

impl<'a> Seq<'a> {
    const fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            elements: 0,
            body: Vec::new(),
        }
    }

    fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), EncodeError> {
        value.serialize(Encoder {
            out: &mut self.body,
        })?;
        self.elements += 1;
        Ok(())
    }

    fn finish(self) -> Result<(), EncodeError> {
        let elements = count(self.elements)?;
        self.out.push(tag::SEQ);
        self.out.extend_from_slice(&elements);
        self.out.extend_from_slice(&self.body);
        Ok(())
    }
}

impl ser::SerializeSeq for Seq<'_> {
    type Ok = ();
    type Error = EncodeError;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), EncodeError> {
        self.element(value)
    }

    fn end(self) -> Result<(), EncodeError> {
        self.finish()
    }
}

impl ser::SerializeTuple for Seq<'_> {
    type Ok = ();
    type Error = EncodeError;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), EncodeError> {
        self.element(value)
    }

    fn end(self) -> Result<(), EncodeError> {
        self.finish()
    }
}

impl ser::SerializeTupleStruct for Seq<'_> {
    type Ok = ();
    type Error = EncodeError;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), EncodeError> {
        self.element(value)
    }

    fn end(self) -> Result<(), EncodeError> {
        self.finish()
    }
}

impl ser::SerializeTupleVariant for Seq<'_> {
    type Ok = ();
    type Error = EncodeError;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), EncodeError> {
        self.element(value)
    }

    fn end(self) -> Result<(), EncodeError> {
        self.finish()
    }
}

/// A `0x51` item being written: each entry is encoded on its own, and the
/// entries are sorted by their key bytes at `end`.
///
/// WHY: sorting here, on the encoded keys, is what makes the digest
/// independent of the order the source map iterates in.
struct Map<'a> {
    out: &'a mut Vec<u8>,
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    pending_key: Option<Vec<u8>>,
}

impl<'a> Map<'a> {
    const fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            entries: Vec::new(),
            pending_key: None,
        }
    }

    fn key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), EncodeError> {
        let item = encode(key)?;
        if !is_key(&item) {
            return Err(EncodeError(format!(
                "a map key encodes as {}; map keys must encode as a string or an integer",
                describe(&item)
            )));
        }
        self.pending_key = Some(item);
        Ok(())
    }

    fn value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), EncodeError> {
        let Some(key) = self.pending_key.take() else {
            return Err(EncodeError(
                "a map emitted a value before its key".to_owned(),
            ));
        };
        self.entries.push((key, encode(value)?));
        Ok(())
    }

    fn finish(mut self) -> Result<(), EncodeError> {
        if self.pending_key.is_some() {
            return Err(EncodeError(
                "a map emitted a key without its value".to_owned(),
            ));
        }
        self.entries
            .sort_unstable_by(|left, right| left.0.cmp(&right.0));
        if self.entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(EncodeError("a map emits the same key twice".to_owned()));
        }
        let entries = count(self.entries.len())?;
        self.out.push(tag::MAP);
        self.out.extend_from_slice(&entries);
        for (key, value) in &self.entries {
            self.out.extend_from_slice(key);
            self.out.extend_from_slice(value);
        }
        Ok(())
    }
}

impl SerializeMap for Map<'_> {
    type Ok = ();
    type Error = EncodeError;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), EncodeError> {
        self.key(key)
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), EncodeError> {
        self.value(value)
    }

    fn end(self) -> Result<(), EncodeError> {
        self.finish()
    }
}

/// A struct being written: an ordinary struct is a `0x51` item of its
/// fields; serde_json's arbitrary-precision number is the integer item of
/// its text.
enum Struct<'a> {
    Fields(Map<'a>),
    SerdeJsonNumber {
        out: &'a mut Vec<u8>,
        text: Option<String>,
    },
}

impl SerializeStruct for Struct<'_> {
    type Ok = ();
    type Error = EncodeError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        name: &'static str,
        value: &T,
    ) -> Result<(), EncodeError> {
        match self {
            Self::Fields(map) => {
                map.key(name)?;
                map.value(value)
            }
            Self::SerdeJsonNumber { text, .. } => {
                let item = encode(value)?;
                match (name, decoded_str(&item), text.is_none()) {
                    (SERDE_JSON_NUMBER, Some(decimal), true) => {
                        *text = Some(decimal.to_owned());
                        Ok(())
                    }
                    _ => Err(EncodeError(
                        "a serde_json number did not serialize as one decimal string".to_owned(),
                    )),
                }
            }
        }
    }

    fn end(self) -> Result<(), EncodeError> {
        match self {
            Self::Fields(map) => map.finish(),
            Self::SerdeJsonNumber { out, text } => {
                let Some(text) = text else {
                    return Err(EncodeError(
                        "a serde_json number did not serialize as one decimal string".to_owned(),
                    ));
                };
                out.extend_from_slice(&serde_json_number_item(&text)?);
                Ok(())
            }
        }
    }
}

impl ser::SerializeStructVariant for Map<'_> {
    type Ok = ();
    type Error = EncodeError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        name: &'static str,
        value: &T,
    ) -> Result<(), EncodeError> {
        self.key(name)?;
        self.value(value)
    }

    fn end(self) -> Result<(), EncodeError> {
        self.finish()
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests need concise encoding assertions")]
mod tests {
    use std::collections::BTreeMap;

    use serde::{Serialize, Serializer};

    use super::*;

    /// A length or count as the grammar table on `OperationDigest` states
    /// it: a `u64`, big-endian.
    fn len(value: u64) -> Vec<u8> {
        value.to_be_bytes().to_vec()
    }

    fn string(value: &str) -> Vec<u8> {
        let length = u64::try_from(value.len()).expect("a test string's length fits a u64");
        [vec![0x40], len(length), value.as_bytes().to_vec()].concat()
    }

    fn item<T: Serialize + ?Sized>(value: &T) -> Vec<u8> {
        encode(value).expect("value encodes")
    }

    fn refusal<T: Serialize + ?Sized>(value: &T) -> String {
        encode(value).expect_err("value is refused").to_string()
    }

    #[derive(Serialize)]
    struct Unit;

    #[derive(Serialize)]
    struct Newtype(u8);

    #[derive(Serialize)]
    struct Pair(u8, u8);

    #[derive(Serialize)]
    struct Fields {
        second: u8,
        first: u8,
    }

    #[derive(Serialize)]
    enum Shape {
        Bare,
        Wrapped(u8),
        Tuple(u8, u8),
        Named { field: u8 },
    }

    struct RawBytes(&'static [u8]);

    impl Serialize for RawBytes {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_bytes(self.0)
        }
    }

    /// Emits one key twice, which no Rust map can hold but a hand-written
    /// `Serialize` impl can.
    struct RepeatedKey;

    impl Serialize for RepeatedKey {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("key", &1_u8)?;
            map.serialize_entry("key", &2_u8)?;
            map.end()
        }
    }

    #[test]
    fn every_serde_data_model_kind_encodes_to_its_documented_item() {
        let cases: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
            ("unit", item(&()), vec![0x00]),
            ("unit struct", item(&Unit), vec![0x00]),
            ("none", item(&None::<u8>), vec![0x01]),
            ("some", item(&Some(7_u8)), vec![0x02, 0x10, 0x07]),
            ("false", item(&false), vec![0x03]),
            ("true", item(&true), vec![0x04]),
            ("u8", item(&0xab_u8), vec![0x10, 0xab]),
            ("u16", item(&0x0102_u16), vec![0x11, 0x01, 0x02]),
            ("u32", item(&1_u32), vec![0x12, 0, 0, 0, 1]),
            ("u64", item(&1_u64), [vec![0x13], len(1)].concat()),
            (
                "u128",
                item(&1_u128),
                [vec![0x14], vec![0; 15], vec![1]].concat(),
            ),
            ("i8", item(&-1_i8), vec![0x18, 0xff]),
            ("i16", item(&-2_i16), vec![0x19, 0xff, 0xfe]),
            ("i32", item(&-1_i32), vec![0x1a, 0xff, 0xff, 0xff, 0xff]),
            ("i64", item(&-1_i64), [vec![0x1b], vec![0xff; 8]].concat()),
            (
                "i128",
                item(&-1_i128),
                [vec![0x1c], vec![0xff; 16]].concat(),
            ),
            ("char", item(&'é'), string("é")),
            ("str", item("ab"), string("ab")),
            (
                "bytes",
                item(&RawBytes(&[9, 8])),
                [vec![0x41], len(2), vec![9, 8]].concat(),
            ),
            (
                "seq",
                item(&vec![1_u8, 2]),
                [vec![0x50], len(2), vec![0x10, 1, 0x10, 2]].concat(),
            ),
            (
                "tuple",
                item(&(1_u8, "a")),
                [vec![0x50], len(2), vec![0x10, 1], string("a")].concat(),
            ),
            ("newtype struct", item(&Newtype(5)), vec![0x10, 5]),
            (
                "tuple struct",
                item(&Pair(1, 2)),
                [vec![0x50], len(2), vec![0x10, 1, 0x10, 2]].concat(),
            ),
            ("unit variant", item(&Shape::Bare), string("Bare")),
            (
                "newtype variant",
                item(&Shape::Wrapped(3)),
                [vec![0x60], string("Wrapped"), vec![0x10, 3]].concat(),
            ),
            (
                "tuple variant",
                item(&Shape::Tuple(1, 2)),
                [
                    vec![0x60],
                    string("Tuple"),
                    vec![0x50],
                    len(2),
                    vec![0x10, 1, 0x10, 2],
                ]
                .concat(),
            ),
            (
                "struct variant",
                item(&Shape::Named { field: 4 }),
                [
                    vec![0x60],
                    string("Named"),
                    vec![0x51],
                    len(1),
                    string("field"),
                    vec![0x10, 4],
                ]
                .concat(),
            ),
            (
                "struct, fields sorted by encoded name",
                item(&Fields {
                    second: 2,
                    first: 1,
                }),
                [
                    vec![0x51],
                    len(2),
                    string("first"),
                    vec![0x10, 1],
                    string("second"),
                    vec![0x10, 2],
                ]
                .concat(),
            ),
        ];
        for (kind, actual, expected) in cases {
            assert_eq!(actual, expected, "{kind}");
        }
    }

    #[test]
    fn map_entries_sort_by_encoded_key_bytes() {
        // WHY: a string key item leads with its length, so shorter keys sort
        // first and equal lengths sort bytewise.
        let strings = BTreeMap::from([("bb", 1_u8), ("c", 2), ("a", 3)]);
        assert_eq!(
            item(&strings),
            [
                vec![0x51],
                len(3),
                string("a"),
                vec![0x10, 3],
                string("c"),
                vec![0x10, 2],
                string("bb"),
                vec![0x10, 1],
            ]
            .concat(),
            "string keys sort by length, then bytes"
        );

        let integers = BTreeMap::from([(256_u64, 1_u8), (1, 2)]);
        assert_eq!(
            item(&integers),
            [
                vec![0x51],
                len(2),
                vec![0x13],
                len(1),
                vec![0x10, 2],
                vec![0x13],
                len(256),
                vec![0x10, 1],
            ]
            .concat(),
            "same-width unsigned keys sort numerically"
        );
    }

    #[test]
    fn floats_are_refused_wherever_they_appear() {
        let refusals = [
            refusal(&1.5_f32),
            refusal(&1.5_f64),
            refusal(&f64::NAN),
            refusal(&Some(0.0_f64)),
            refusal(&vec![0.5_f64]),
            refusal(&BTreeMap::from([("weight", 0.5_f64)])),
            refusal(&(1_u8, -0.0_f32)),
        ];
        for message in refusals {
            assert!(
                message.contains("floating-point numbers have no canonical encoding"),
                "{message}"
            );
        }
    }

    #[test]
    fn map_keys_must_encode_as_strings_or_integers() {
        #[derive(Serialize, PartialEq, Eq, PartialOrd, Ord)]
        enum Key {
            Named,
        }

        let accepted = [
            item(&BTreeMap::from([('k', 1_u8)])),
            item(&BTreeMap::from([(Key::Named, 1_u8)])),
            item(&BTreeMap::from([(-3_i8, 1_u8)])),
            item(&BTreeMap::from([(u128::MAX, 1_u8)])),
        ];
        assert!(
            accepted.iter().all(|map| map.first() == Some(&0x51)),
            "char, unit variant, and integer keys are accepted"
        );

        let refused = [
            (refusal(&BTreeMap::from([(true, 1_u8)])), "a bool"),
            (refusal(&BTreeMap::from([((), 1_u8)])), "a unit"),
            (
                refusal(&BTreeMap::from([(Some(1_u8), 1_u8)])),
                "a present option",
            ),
            (
                refusal(&BTreeMap::from([((1_u8, 2_u8), 1_u8)])),
                "a sequence",
            ),
            (
                refusal(&BTreeMap::from([(BTreeMap::from([(1_u8, 1_u8)]), 1_u8)])),
                "a map or struct",
            ),
        ];
        for (message, kind) in refused {
            assert_eq!(
                message,
                format!(
                    "a map key encodes as {kind}; map keys must encode as a string or an integer"
                )
            );
        }
    }

    /// Serializes as serde_json serializes a `Number` when its
    /// `arbitrary_precision` feature is on.
    struct ArbitraryPrecisionNumber(&'static str);

    impl Serialize for ArbitraryPrecisionNumber {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut number = serializer.serialize_struct(SERDE_JSON_NUMBER, 1)?;
            number.serialize_field(SERDE_JSON_NUMBER, self.0)?;
            number.end()
        }
    }

    /// Serializes as serde_json serializes a `RawValue`.
    struct RawJson(&'static str);

    impl Serialize for RawJson {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut raw = serializer.serialize_struct(SERDE_JSON_RAW_VALUE, 1)?;
            raw.serialize_field(SERDE_JSON_RAW_VALUE, self.0)?;
            raw.end()
        }
    }

    #[test]
    fn serde_json_numbers_encode_alike_with_and_without_arbitrary_precision() {
        // Each text is parsed by serde_json as this build links it, and also
        // written in arbitrary_precision's form; both must give one outcome.
        let texts = [
            "0",
            "7",
            "18446744073709551615",
            "-3",
            "-9223372036854775808",
            "-0",
            "1.5",
            "1e3",
            "18446744073709551616",
            "-9223372036854775809",
        ];
        for text in texts {
            let parsed: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
            let default_build = encode(&parsed).map_err(|error| error.to_string());
            let arbitrary =
                encode(&ArbitraryPrecisionNumber(text)).map_err(|error| error.to_string());
            match (default_build, arbitrary) {
                (Ok(default_item), Ok(arbitrary_item)) => {
                    assert_eq!(default_item, arbitrary_item, "{text}");
                }
                (Err(default_refusal), Err(arbitrary_refusal)) => {
                    for message in [default_refusal, arbitrary_refusal] {
                        assert!(
                            message.contains("floating-point numbers have no canonical encoding"),
                            "{text}: {message}"
                        );
                    }
                }
                (default_build, arbitrary) => {
                    panic!(
                        "{text}: default build {default_build:?}, arbitrary precision {arbitrary:?}"
                    )
                }
            }
        }
        assert_eq!(item(&ArbitraryPrecisionNumber("7")), item(&7_u64));
        assert_eq!(item(&ArbitraryPrecisionNumber("-3")), item(&-3_i64));

        // Not serde_json's spelling of any integer, so not read as one.
        for text in ["01", "+1", "-01", ""] {
            let message = refusal(&ArbitraryPrecisionNumber(text));
            assert!(
                message.contains("floating-point numbers"),
                "{text:?}: {message}"
            );
        }
    }

    #[test]
    fn serde_json_raw_json_text_is_refused() {
        let message = refusal(&RawJson("{\"b\":1,\"a\":2}"));
        assert!(message.contains("serde_json RawValue"), "{message}");
    }

    #[test]
    fn sequences_keep_their_emission_order() {
        assert_ne!(
            item(&vec![1_u8, 2]),
            item(&vec![2_u8, 1]),
            "a sequence is ordered data; only map entries are sorted"
        );
    }

    #[test]
    fn a_map_that_emits_one_key_twice_is_refused() {
        assert_eq!(refusal(&RepeatedKey), "a map emits the same key twice");
    }

    #[test]
    fn member_identity_must_encode_as_a_string_or_an_integer() {
        assert_eq!(member_identity_refusal(&7_u64), None);
        assert_eq!(member_identity_refusal(&"doc-7"), None);
        assert_eq!(member_identity_refusal(&Newtype(7)), None);

        let sequence = member_identity_refusal(&Pair(1, 2)).expect("a pair is refused");
        assert!(sequence.starts_with("encodes as a sequence;"), "{sequence}");
        let float = member_identity_refusal(&0.5_f64).expect("a float is refused");
        assert!(float.contains("floating-point"), "{float}");
    }
}
