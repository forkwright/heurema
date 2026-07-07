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
    type Id: Eq + Hash + Clone;

    /// WHY: Insert mirrors krites `put_fts_index_item`: consumers supply a
    /// document body and an engine-owned ID.
    fn insert(&mut self, id: Self::Id, document: &str) -> Result<(), HeuremaError>;

    /// WHY: Query mirrors krites `fts_search`: consumers ask for ranked IDs and
    /// BM25-style scores without receiving engine-owned tuples.
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
