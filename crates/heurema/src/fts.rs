//! BM25 full-text index contract and Phase 1 stub type.

use std::hash::Hash;

use crate::HeuremaError;

mod stub;

pub use stub::Bm25Index;

/// WHY: Krites models tokenizers and filters as named components with argument
/// lists; Heurēma keeps that shape without importing krites `DataValue`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct TokenizerConfig {
    /// Tokenizer or filter name, such as `Simple`, `NGram`, or `Stemmer`.
    pub name: String,
    /// String-encoded arguments owned by the eventual backend adapter.
    pub args: Vec<String>,
}

impl TokenizerConfig {
    /// WHY: Consumers need a small constructor for analyzer pipelines while the
    /// Phase 2 extraction decides the final argument value model.
    #[must_use]
    pub fn new(name: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            name: name.into(),
            args,
        }
    }
}

/// WHY: FTS configuration must preserve the tokenizer-plus-filter pipeline
/// model used by krites today.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FtsConfig {
    /// Base tokenizer used to split documents and queries.
    pub tokenizer: TokenizerConfig,
    /// Ordered filters applied after tokenization.
    pub filters: Vec<TokenizerConfig>,
}

impl FtsConfig {
    /// WHY: `Simple` is the krites default tokenizer used by basic FTS index
    /// smoke paths.
    #[must_use]
    pub fn simple() -> Self {
        Self {
            tokenizer: TokenizerConfig::new("Simple", Vec::new()),
            filters: Vec::new(),
        }
    }
}

/// WHY: Full-text engines need a stable contract for document indexing, BM25
/// query, removal, and size inspection independent of relation storage.
pub trait FtsIndex {
    /// Identifier stored alongside each indexed document.
    ///
    /// WHY: `Ord` is the bound [`crate::rrf`] fuses under. Requiring it here
    /// keeps the advertised hybrid path — query an index, fuse the result —
    /// compilable for a generic consumer that knows only this trait.
    type Id: Ord + Hash + Clone;

    /// WHY: Insert mirrors krites `put_fts_index_item`: consumers supply a
    /// document body and an engine-owned ID.
    fn insert(&mut self, id: Self::Id, document: &str) -> Result<(), HeuremaError>;

    /// WHY: Query mirrors krites `fts_search`: consumers ask for ranked IDs and
    /// BM25-style scores without receiving engine-owned tuples.
    ///
    /// Ranking contract — implementations must satisfy all of it, because
    /// [`crate::rrf`] reads *position* as the authoritative rank and ignores
    /// the returned score entirely:
    ///
    /// - **Ordering is normative.** Element 0 is the best match, and each
    ///   subsequent element is no better than its predecessor. A result vector
    ///   in any other order still type-checks but silently changes what
    ///   fusion computes.
    /// - **Score polarity is descending-is-better** — these are BM25-style
    ///   relevance scores, so a larger `f32` is a better match, and the
    ///   sequence is non-increasing. Note this is the opposite polarity to
    ///   [`crate::VectorIndex::query`], which returns distances; fusion is
    ///   immune to the difference precisely because it reads position rather
    ///   than score. The score is advisory: it is carried for display and
    ///   thresholding, never used to re-derive rank.
    /// - **IDs are unique** within one result vector.
    /// - **Ties are stable.** Equal scores must be ordered by ascending `Id`,
    ///   so repeating a query over unchanged index state returns an identical
    ///   vector.
    /// - **At most `k`** elements are returned; fewer is valid when the index
    ///   holds fewer matching documents.
    ///
    /// # Errors
    ///
    /// Implementations return [`HeuremaError`] when the backing engine cannot
    /// service the query.
    fn query(&self, query: &str, k: usize) -> Result<Vec<(Self::Id, f32)>, HeuremaError>;

    /// WHY: Remove gives consumers the same lifecycle hook as vector indexes
    /// when a source row or fact disappears.
    ///
    /// Removal is idempotent: removing an `id` that is not in the index is a
    /// successful no-op, so row-deletion cleanup can retry safely.
    fn remove(&mut self, id: &Self::Id) -> Result<(), HeuremaError>;

    /// WHY: Consumers need document cardinality for scoring and maintenance
    /// checks.
    fn len(&self) -> usize;

    /// WHY: `len` callers need a non-arithmetic emptiness predicate for clean
    /// clippy-compliant APIs.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
