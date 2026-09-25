//! Identifiers for named indexes, their versions, and the operations that
//! change them.

use std::fmt;
use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use crate::HeuremaError;
use crate::error::InvalidIdentifierSnafu;

/// Which lifecycle identifier a [`HeuremaError::InvalidIdentifier`] refused.
///
/// WHY: one refusal variant serves every identifier, so the error names the
/// identifier's kind instead of multiplying near-identical variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IdentifierKind {
    /// An [`OwnerNamespace`].
    OwnerNamespace,
    /// An [`IndexName`].
    IndexName,
    /// An [`OperationKey`].
    OperationKey,
    /// An [`OperationDigest`].
    OperationDigest,
    /// An [`IndexVersion`].
    IndexVersion,
    /// A consumer's [`MemberIdentity`](crate::lifecycle::MemberIdentity)
    /// value.
    MemberIdentity,
}

impl fmt::Display for IdentifierKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OwnerNamespace => "owner namespace",
            Self::IndexName => "index name",
            Self::OperationKey => "operation key",
            Self::OperationDigest => "operation digest",
            Self::IndexVersion => "index version",
            Self::MemberIdentity => "member identity",
        })
    }
}

/// WHY: a refused value is echoed into the error message, and a caller-sized
/// value must not make the error arbitrarily large.
const REPORTED_VALUE_CHARS: usize = 64;

#[track_caller]
fn refusal(kind: IdentifierKind, value: &str, reason: String) -> HeuremaError {
    InvalidIdentifierSnafu {
        kind,
        value: value.chars().take(REPORTED_VALUE_CHARS).collect::<String>(),
        reason,
    }
    .build()
}

/// The character grammar of one string identifier:
/// `[lead][lead punctuation]{0,max_len-1}`, where `lead` is ASCII digits and
/// lowercase letters, plus uppercase letters when `uppercase` is set.
struct Grammar {
    kind: IdentifierKind,
    max_len: usize,
    uppercase: bool,
    punctuation: &'static str,
}

impl Grammar {
    fn pattern(&self) -> String {
        let lead = if self.uppercase {
            "A-Za-z0-9"
        } else {
            "a-z0-9"
        };
        format!(
            "[{lead}][{lead}{punctuation}]{{0,{rest}}}",
            punctuation = self.punctuation,
            rest = self.max_len - 1,
        )
    }

    const fn leads(&self, character: char) -> bool {
        character.is_ascii_digit()
            || character.is_ascii_lowercase()
            || (self.uppercase && character.is_ascii_uppercase())
    }

    fn follows(&self, character: char) -> bool {
        self.leads(character) || (character.is_ascii() && self.punctuation.contains(character))
    }

    #[track_caller]
    fn check(&self, value: &str) -> Result<(), HeuremaError> {
        if value.is_empty() {
            return Err(refusal(
                self.kind,
                value,
                format!("is empty; expected {}", self.pattern()),
            ));
        }
        if value.len() > self.max_len {
            return Err(refusal(
                self.kind,
                value,
                format!("is {} bytes; the limit is {}", value.len(), self.max_len),
            ));
        }
        for (offset, character) in value.char_indices() {
            let allowed = if offset == 0 {
                self.leads(character)
            } else {
                self.follows(character)
            };
            if !allowed {
                return Err(refusal(
                    self.kind,
                    value,
                    format!(
                        "has {character:?} at byte {offset}; expected {}",
                        self.pattern()
                    ),
                ));
            }
        }
        Ok(())
    }
}

const OWNER_NAMESPACE: Grammar = Grammar {
    kind: IdentifierKind::OwnerNamespace,
    max_len: OwnerNamespace::MAX_LEN,
    uppercase: false,
    punctuation: "._-",
};

const INDEX_NAME: Grammar = Grammar {
    kind: IdentifierKind::IndexName,
    max_len: IndexName::MAX_LEN,
    uppercase: false,
    punctuation: "._-",
};

const OPERATION_KEY: Grammar = Grammar {
    kind: IdentifierKind::OperationKey,
    max_len: OperationKey::MAX_LEN,
    uppercase: true,
    punctuation: "._:-",
};

/// Implements the shared surface of a validated string identifier: `TryFrom`
/// as the only constructor, `as_str`, `AsRef<str>`, `Display`, and the
/// conversion back into `String` that serde's `into` uses.
///
/// WHY: the three string identifiers differ only in their grammar, so their
/// conversions are written once and cannot drift apart.
macro_rules! string_identifier {
    ($name:ident, $grammar:expr) => {
        impl $name {
            /// The identifier as a string slice.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl TryFrom<String> for $name {
            type Error = HeuremaError;

            #[track_caller]
            fn try_from(value: String) -> Result<Self, Self::Error> {
                $grammar.check(&value)?;
                Ok(Self(value))
            }
        }

        impl TryFrom<&str> for $name {
            type Error = HeuremaError;

            #[track_caller]
            fn try_from(value: &str) -> Result<Self, Self::Error> {
                $grammar.check(value)?;
                Ok(Self(value.to_owned()))
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

/// The consumer that owns a named index.
///
/// Grammar: `[a-z0-9][a-z0-9._-]{0,63}`. The value is ASCII, lowercase, and
/// contains no `/`, so it is safe inside a storage key and `namespace/name`
/// splits one way only.
///
/// WHY: an index name is unique only within its owner; the namespace keeps two
/// consumers' indexes from colliding under one shared backend.
///
/// The only constructor is `TryFrom`; there is no infallible `From<&str>`,
/// because a conversion that can refuse must say so. Deserialization runs the
/// same check.
///
/// ```
/// use heurema::OwnerNamespace;
///
/// let _namespace = OwnerNamespace::try_from("example");
/// ```
///
/// ```compile_fail
/// use heurema::OwnerNamespace;
///
/// let _namespace = OwnerNamespace::from("example");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
#[repr(transparent)]
pub struct OwnerNamespace(String);

impl OwnerNamespace {
    /// Longest accepted namespace, in bytes.
    pub const MAX_LEN: usize = 64;
}

string_identifier!(OwnerNamespace, OWNER_NAMESPACE);

/// The name of one index within its [`OwnerNamespace`].
///
/// Grammar: `[a-z0-9][a-z0-9._-]{0,127}`: ASCII, lowercase, no `/`.
///
/// WHY: the name becomes part of every storage key for the index, so a `/`
/// inside it would make `namespace/name/...` keys ambiguous.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
#[repr(transparent)]
pub struct IndexName(String);

impl IndexName {
    /// Longest accepted index name, in bytes.
    pub const MAX_LEN: usize = 128;
}

string_identifier!(IndexName, INDEX_NAME);

/// A caller-supplied idempotency key for one lifecycle operation.
///
/// Grammar: `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`, so ULIDs and URN-shaped keys
/// (`urn:uuid:...`) fit. Whitespace and `/` are refused.
///
/// WHY: the caller, not heurēma, knows when two requests are the same
/// operation (a retry after a lost acknowledgement, say). The caller keeps
/// each key unique per [`IndexIdentity`] and never reuses it, including
/// after the index is destroyed.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
#[repr(transparent)]
pub struct OperationKey(String);

impl OperationKey {
    /// Longest accepted key, in bytes.
    pub const MAX_LEN: usize = 128;
}

string_identifier!(OperationKey, OPERATION_KEY);

/// The identity of one named index: its owner and its name.
///
/// Displays as `namespace/name`.
///
/// WHY: every lifecycle record, operation, and storage key is addressed by
/// this pair, so it is one type rather than two loose strings that could be
/// swapped.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct IndexIdentity {
    /// The consumer that owns the index.
    pub namespace: OwnerNamespace,
    /// The index's name within that namespace.
    pub name: IndexName,
}

impl IndexIdentity {
    /// WHY: both parts are already validated, so pairing them cannot fail.
    #[must_use]
    pub const fn new(namespace: OwnerNamespace, name: IndexName) -> Self {
        Self { namespace, name }
    }
}

impl fmt::Display for IndexIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.namespace, self.name)
    }
}

/// One published version of a named index, counted from 1.
///
/// Create publishes [`IndexVersion::FIRST`]; each later mutation publishes the
/// [`successor`](IndexVersion::successor) of the active version. Zero is not a
/// version.
///
/// WHY: versions order an index's history, and a published version's payload
/// never changes, so a reader holding a version number holds one fixed state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u64", into = "u64")]
#[repr(transparent)]
pub struct IndexVersion(NonZeroU64);

impl IndexVersion {
    /// The version Create publishes.
    pub const FIRST: Self = Self(NonZeroU64::MIN);

    /// The version number.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// The next version.
    ///
    /// # Errors
    ///
    /// Returns [`HeuremaError::InvalidIdentifier`] when the counter is
    /// exhausted (`u64::MAX` has no successor).
    #[track_caller]
    pub fn successor(self) -> Result<Self, HeuremaError> {
        match self.0.checked_add(1) {
            Some(next) => Ok(Self(next)),
            None => Err(refusal(
                IdentifierKind::IndexVersion,
                &self.to_string(),
                "version counter exhausted".to_owned(),
            )),
        }
    }
}

impl TryFrom<u64> for IndexVersion {
    type Error = HeuremaError;

    #[track_caller]
    fn try_from(value: u64) -> Result<Self, Self::Error> {
        let Some(version) = NonZeroU64::new(value) else {
            return Err(refusal(
                IdentifierKind::IndexVersion,
                "0",
                "versions start at 1".to_owned(),
            ));
        };
        Ok(Self(version))
    }
}

impl From<IndexVersion> for u64 {
    fn from(value: IndexVersion) -> Self {
        value.get()
    }
}

impl fmt::Display for IndexVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// The SHA-256 digest of one operation's canonical encoding.
///
/// Displays, serializes, and parses as exactly 64 lowercase hex digits; any
/// other spelling is refused, so one digest has one text form.
///
/// WHY: an [`OperationKey`] alone cannot tell a retry from a different
/// operation reusing the key. Pairing the key with a digest of the
/// operation's content can. This type is the digest's shape only; nothing in
/// this crate computes one yet. A digest can be parsed from its text form (a
/// recorded identity must be decodable), but heurēma never accepts a
/// caller-supplied digest for an operation it applies: it computes the digest
/// from the operation itself.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
#[repr(transparent)]
pub struct OperationDigest([u8; 32]);

impl OperationDigest {
    const HEX_LEN: usize = 64;

    /// The digest's 32 bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

const fn lowercase_hex_nibble(digit: u8) -> Option<u8> {
    match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        _ => None,
    }
}

impl TryFrom<&str> for OperationDigest {
    type Error = HeuremaError;

    #[track_caller]
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.len() != Self::HEX_LEN {
            return Err(refusal(
                IdentifierKind::OperationDigest,
                value,
                format!(
                    "is {} bytes; expected exactly {} lowercase hex digits",
                    value.len(),
                    Self::HEX_LEN
                ),
            ));
        }
        // INVARIANT: the length check above makes this loop see at most 64
        // characters, and any non-ASCII character is refused when reached,
        // so every nibble slot is written before the bytes are assembled.
        let mut nibbles = [0_u8; 64];
        for ((offset, character), nibble) in value.char_indices().zip(nibbles.iter_mut()) {
            let Some(parsed) = u8::try_from(character).ok().and_then(lowercase_hex_nibble) else {
                return Err(refusal(
                    IdentifierKind::OperationDigest,
                    value,
                    format!("has {character:?} at byte {offset}; expected lowercase hex [0-9a-f]"),
                ));
            };
            *nibble = parsed;
        }
        let mut bytes = [0_u8; 32];
        let (pairs, _) = nibbles.as_chunks::<2>();
        for (byte, &[high, low]) in bytes.iter_mut().zip(pairs) {
            *byte = (high << 4) | low;
        }
        Ok(Self(bytes))
    }
}

impl TryFrom<String> for OperationDigest {
    type Error = HeuremaError;

    #[track_caller]
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::try_from(value.as_str())
    }
}

impl From<OperationDigest> for String {
    fn from(value: OperationDigest) -> Self {
        value.to_string()
    }
}

impl fmt::Display for OperationDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for OperationDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("OperationDigest")
            .field(&format_args!("{self}"))
            .finish()
    }
}

/// The identity of one lifecycle operation: the caller's key plus the digest
/// of what the operation does.
///
/// WHY: the same key with the same digest is a replay of one operation; the
/// same key with a different digest is a different operation reusing a key,
/// which must be refused rather than applied. There is no public
/// constructor. A recorded identity can be decoded, but heurēma never takes
/// an identity from a caller for an operation it applies: it computes the
/// digest from the operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct OperationIdentity {
    /// The caller-supplied idempotency key.
    pub key: OperationKey,
    /// The digest of the operation's canonical encoding.
    pub digest: OperationDigest,
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "tests need concise identifier assertions"
)]
mod tests {
    use super::*;
    use crate::ErrorCategory;

    fn assert_refused<T: fmt::Debug>(result: Result<T, HeuremaError>, kind: IdentifierKind) {
        let error = result.expect_err("value must be refused");
        let HeuremaError::InvalidIdentifier { kind: refused, .. } = &error else {
            panic!("expected InvalidIdentifier, got {error:?}");
        };
        assert_eq!(*refused, kind, "{error}");
        assert_eq!(error.category(), ErrorCategory::Refused, "{error}");
    }

    #[test]
    fn owner_namespace_accepts_the_identifier_grammar() {
        let longest = "a".repeat(OwnerNamespace::MAX_LEN);
        for value in [
            "a",
            "0",
            "example",
            "fleet.search",
            "a-b_c.d",
            "9lives",
            longest.as_str(),
        ] {
            let namespace = OwnerNamespace::try_from(value).expect("grammar-conformant namespace");
            assert_eq!(namespace.as_str(), value);
            assert_eq!(namespace.as_ref(), value);
            assert_eq!(namespace.to_string(), value);
            assert_eq!(
                OwnerNamespace::try_from(value.to_owned()).expect("owned input"),
                namespace
            );
            assert_eq!(String::from(namespace), value);
        }
    }

    #[test]
    fn owner_namespace_refuses_empty_uppercase_slash_and_overlong_values() {
        let overlong = "a".repeat(OwnerNamespace::MAX_LEN + 1);
        for value in [
            "",
            "Example",
            "exAmple",
            "a/b",
            "/a",
            "a/",
            overlong.as_str(),
            "-a",
            ".a",
            "_a",
            "a b",
            "é",
            "a\u{0}",
        ] {
            assert_refused(
                OwnerNamespace::try_from(value),
                IdentifierKind::OwnerNamespace,
            );
        }

        let error = OwnerNamespace::try_from("a/b").expect_err("slash is refused");
        let message = error.to_string();
        assert!(
            message.contains("[a-z0-9][a-z0-9._-]{0,63}"),
            "refusal names the grammar: {message}"
        );

        let far_overlong = "a".repeat(1_000);
        let error = OwnerNamespace::try_from(far_overlong.as_str()).expect_err("overlong");
        let HeuremaError::InvalidIdentifier { value, reason, .. } = &error else {
            panic!("expected InvalidIdentifier, got {error:?}");
        };
        assert_eq!(value.chars().count(), REPORTED_VALUE_CHARS);
        assert!(reason.contains("1000 bytes"), "{reason}");
    }

    #[test]
    fn index_name_refuses_a_slash_so_storage_keys_stay_unambiguous() {
        for value in ["notes/v2", "/notes", "notes/", "/"] {
            assert_refused(IndexName::try_from(value), IdentifierKind::IndexName);
        }
        let dotted = IndexName::try_from("notes.v2").expect("dots are allowed");
        assert_eq!(dotted.as_str(), "notes.v2");

        let longest = "n".repeat(IndexName::MAX_LEN);
        assert!(IndexName::try_from(longest.as_str()).is_ok());
        let overlong = "n".repeat(IndexName::MAX_LEN + 1);
        assert_refused(
            IndexName::try_from(overlong.as_str()),
            IdentifierKind::IndexName,
        );
    }

    #[test]
    fn operation_key_accepts_ulid_and_urn_shaped_keys() {
        let longest = "K".repeat(OperationKey::MAX_LEN);
        for value in [
            "01J8ZQ6D4X5K2M9N7P3R8T1V0W",
            "urn:uuid:f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
            "import.2026-09-25:batch_7",
            "k",
            longest.as_str(),
        ] {
            let key = OperationKey::try_from(value).expect("ULID or URN-shaped key");
            assert_eq!(key.as_str(), value);
        }
    }

    #[test]
    fn operation_key_refuses_empty_whitespace_and_overlong_keys() {
        let overlong = "K".repeat(OperationKey::MAX_LEN + 1);
        for value in [
            "",
            " ",
            "a b",
            "key\n",
            "\tkey",
            " key",
            "key ",
            ":urn",
            "a/b",
            "ké",
            overlong.as_str(),
        ] {
            assert_refused(OperationKey::try_from(value), IdentifierKind::OperationKey);
        }
    }

    #[test]
    fn identifiers_deserialize_only_through_try_from() {
        let namespace: OwnerNamespace =
            serde_json::from_str(r#""example""#).expect("valid namespace decodes");
        assert_eq!(
            serde_json::to_string(&namespace).expect("namespace encodes"),
            r#""example""#
        );

        assert!(serde_json::from_str::<OwnerNamespace>(r#""Example""#).is_err());
        assert!(serde_json::from_str::<OwnerNamespace>(r#""""#).is_err());
        assert!(serde_json::from_str::<IndexName>(r#""a/b""#).is_err());
        assert!(serde_json::from_str::<OperationKey>(r#""a b""#).is_err());
        assert!(serde_json::from_str::<IndexVersion>("0").is_err());
        assert!(
            serde_json::from_str::<OperationDigest>(&format!(r#""{}""#, "A".repeat(64))).is_err()
        );
        assert!(
            serde_json::from_str::<IndexIdentity>(r#"{"namespace":"Example","name":"notes"}"#)
                .is_err()
        );
        assert!(
            serde_json::from_str::<IndexIdentity>(
                r#"{"namespace":"example","name":"notes","extra":1}"#
            )
            .is_err()
        );

        let identity: IndexIdentity =
            serde_json::from_str(r#"{"namespace":"example","name":"notes"}"#)
                .expect("valid identity decodes");
        assert_eq!(identity.to_string(), "example/notes");
    }

    #[test]
    fn index_version_refuses_zero_and_orders_numerically() {
        assert_refused(IndexVersion::try_from(0), IdentifierKind::IndexVersion);
        assert_eq!(IndexVersion::FIRST.get(), 1);
        assert_eq!(
            IndexVersion::try_from(1).expect("one is a version"),
            IndexVersion::FIRST
        );

        let mut versions =
            [10_u64, 9, 1, 2].map(|value| IndexVersion::try_from(value).expect("non-zero version"));
        versions.sort();
        assert_eq!(versions.map(IndexVersion::get), [1, 2, 9, 10]);
        assert_eq!(
            serde_json::to_string(&versions[3]).expect("version encodes"),
            "10"
        );
        assert_eq!(u64::from(versions[3]), 10);
        assert_eq!(versions[3].to_string(), "10");
    }

    #[test]
    fn index_version_successor_refuses_overflow() {
        let second = IndexVersion::FIRST.successor().expect("1 has a successor");
        assert_eq!(second.get(), 2);

        let last = IndexVersion::try_from(u64::MAX).expect("u64::MAX is a version");
        let error = last.successor().expect_err("u64::MAX has no successor");
        let HeuremaError::InvalidIdentifier {
            kind: IdentifierKind::IndexVersion,
            reason,
            ..
        } = &error
        else {
            panic!("expected an exhausted-counter refusal, got {error:?}");
        };
        assert!(reason.contains("exhausted"), "{reason}");
    }

    #[test]
    fn operation_digest_round_trips_lowercase_hex_and_refuses_other_shapes() {
        let text = "00ff10a0".repeat(8);
        let digest = OperationDigest::try_from(text.as_str()).expect("lowercase hex digest");
        assert_eq!(digest.to_string(), text);
        assert_eq!(digest.as_bytes()[..4], [0x00, 0xff, 0x10, 0xa0]);
        assert_eq!(
            OperationDigest::try_from(text.clone()).expect("owned input"),
            digest
        );
        assert_eq!(String::from(digest), text);

        let encoded = serde_json::to_string(&digest).expect("digest encodes");
        assert_eq!(encoded, format!(r#""{text}""#));
        let decoded: OperationDigest = serde_json::from_str(&encoded).expect("digest decodes");
        assert_eq!(decoded, digest);
        assert_eq!(format!("{digest:?}"), format!("OperationDigest({text})"));

        let short = &text[..63];
        let long = format!("{text}0");
        let uppercase = text.to_uppercase();
        let non_hex = format!("g{}", &text[1..]);
        let prefixed = format!("0x{}", &text[2..]);
        let non_ascii = format!("é{}", &text[2..]);
        for value in [
            "",
            short,
            long.as_str(),
            uppercase.as_str(),
            non_hex.as_str(),
            prefixed.as_str(),
            non_ascii.as_str(),
        ] {
            assert_refused(
                OperationDigest::try_from(value),
                IdentifierKind::OperationDigest,
            );
        }
    }

    #[test]
    fn index_identity_displays_namespace_slash_name() {
        let identity = IndexIdentity::new(
            OwnerNamespace::try_from("example").expect("namespace"),
            IndexName::try_from("notes.v2").expect("name"),
        );
        assert_eq!(identity.to_string(), "example/notes.v2");
    }
}
