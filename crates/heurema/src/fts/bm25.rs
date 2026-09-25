//! Fresh BM25 implementation for the published full-text contract.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::HeuremaError;
use crate::error::NotYetImplementedSnafu;
use crate::fts::{FtsConfig, FtsIndex};

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;
const SIMPLE_TOKENIZER: &str = "Simple";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DocumentTerms {
    counts: BTreeMap<String, usize>,
    length: usize,
}

/// An in-memory BM25 index for the configured `Simple` tokenizer pipeline.
///
/// `Simple` splits on every character for which [`char::is_alphanumeric`] is
/// false and discards the empty pieces; only then does it lowercase each
/// remaining token, as a whole, with [`str::to_lowercase`]. Context-dependent
/// mappings such as the Greek final sigma therefore resolve within the token,
/// and a lowercase mapping that yields a non-alphanumeric character (`İ`
/// becomes `i` plus U+0307) keeps it inside the token. The pipeline applies no
/// accent folding or stop-word filtering. Queries use the same pipeline.
///
/// # Scoring
///
/// A query scores each live document `d` against the distinct terms `t` of
/// the query that occur in `d`:
///
/// ```text
/// score(d) = sum over t of idf(t) * tf(t, d) * (k1 + 1)
///                          / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
///
/// idf(t)   = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
/// ```
///
/// - `k1 = 1.2` and `b = 0.75`.
/// - `N` is the number of live documents and `n(t)` the number of live
///   documents containing `t` at least once. The idf is the non-negative
///   variant of the Robertson/Spärck Jones weight: it is positive for every
///   `n(t)` in `1..=N`, including terms present in more than half the corpus.
/// - `tf(t, d)` counts the occurrences of token `t` in `d`. `|d|` is the
///   number of tokens `d` yields, repeats included, and `avgdl` is the sum of
///   `|d|` over the live documents divided by `N`. A document that yields no
///   tokens still counts toward `N` and toward `avgdl`'s denominator.
/// - A term repeated in the query contributes once: query-term frequency
///   carries no weight.
/// - The results are exactly the live documents containing at least one
///   query term, so a query that yields no tokens returns nothing. Their
///   order and truncation to `k` follow [`FtsIndex::query`]. Scores are
///   computed in `f32`.
///
/// Documents retain their token counts while postings retain per-term document
/// frequencies. Replacing or removing an ID first removes its old contribution,
/// so a completed mutation always leaves those two views in agreement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(try_from = "RawBm25Index<Id>")]
#[serde(bound(
    serialize = "Id: Serialize",
    deserialize = "Id: Ord + Deserialize<'de>"
))]
pub struct Bm25Index<Id> {
    config: FtsConfig,
    documents: BTreeMap<Id, DocumentTerms>,
    postings: BTreeMap<String, BTreeMap<Id, usize>>,
    total_document_terms: usize,
}

/// Snapshot bytes cross the same invariant boundary as index mutations.
/// Derived postings and lengths must agree with the document term counts;
/// accepting each field independently could silently change rankings on load.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(bound(deserialize = "Id: Ord + Deserialize<'de>"))]
struct RawBm25Index<Id> {
    config: FtsConfig,
    documents: BTreeMap<Id, DocumentTerms>,
    postings: BTreeMap<String, BTreeMap<Id, usize>>,
    total_document_terms: usize,
}

impl<Id: Ord> TryFrom<RawBm25Index<Id>> for Bm25Index<Id> {
    type Error = &'static str;

    fn try_from(raw: RawBm25Index<Id>) -> Result<Self, Self::Error> {
        let mut total = 0usize;
        let mut expected_postings: BTreeMap<&str, BTreeMap<&Id, usize>> = BTreeMap::new();
        for (id, document) in &raw.documents {
            let mut length = 0usize;
            for (term, count) in &document.counts {
                if term.is_empty() || *count == 0 {
                    return Err("BM25 snapshot has an empty term or zero term frequency");
                }
                length = length
                    .checked_add(*count)
                    .ok_or("BM25 document length overflow")?;
                expected_postings
                    .entry(term)
                    .or_default()
                    .insert(id, *count);
            }
            if length != document.length {
                return Err("BM25 document length disagrees with its term counts");
            }
            total = total
                .checked_add(length)
                .ok_or("BM25 corpus length overflow")?;
        }
        if total != raw.total_document_terms {
            return Err("BM25 corpus length disagrees with its documents");
        }
        if raw.postings.len() != expected_postings.len()
            || expected_postings.iter().any(|(term, expected)| {
                raw.postings.get(*term).is_none_or(|actual| {
                    actual.len() != expected.len()
                        || expected
                            .iter()
                            .any(|(id, count)| actual.get(*id) != Some(count))
                })
            })
        {
            return Err("BM25 postings disagree with document term counts");
        }
        let index = Self {
            config: raw.config,
            documents: raw.documents,
            postings: raw.postings,
            total_document_terms: total,
        };
        if !index.documents.is_empty() && index.simple_pipeline().is_err() {
            return Err("BM25 populated snapshot uses an unsupported analyzer");
        }
        Ok(index)
    }
}

impl<Id> Bm25Index<Id> {
    /// Create an empty index with the supplied analyzer configuration.
    #[must_use]
    pub fn new(config: FtsConfig) -> Self {
        Self {
            config,
            documents: BTreeMap::new(),
            postings: BTreeMap::new(),
            total_document_terms: 0,
        }
    }

    /// Return the configured analyzer pipeline.
    #[must_use]
    pub const fn config(&self) -> &FtsConfig {
        &self.config
    }

    fn simple_pipeline(&self) -> Result<(), HeuremaError> {
        if self.config.tokenizer.name == SIMPLE_TOKENIZER
            && self.config.tokenizer.args.is_empty()
            && self.config.filters.is_empty()
        {
            return Ok(());
        }
        Err(NotYetImplementedSnafu {
            feature: "BM25 tokenizer/filter pipeline beyond Simple".to_owned(),
        }
        .build())
    }
}

impl<Id> Bm25Index<Id>
where
    Id: Ord + Hash + Clone,
{
    fn remove_document(&mut self, id: &Id) {
        let Some(document) = self.documents.remove(id) else {
            return;
        };
        self.total_document_terms = self.total_document_terms.saturating_sub(document.length);
        for term in document.counts.into_keys() {
            let remove_posting = self.postings.get_mut(&term).is_some_and(|posting| {
                posting.remove(id);
                posting.is_empty()
            });
            if remove_posting {
                self.postings.remove(&term);
            }
        }
    }

    fn insert_document(&mut self, id: Id, document: DocumentTerms) {
        self.total_document_terms += document.length;
        for (term, count) in &document.counts {
            self.postings
                .entry(term.clone())
                .or_default()
                .insert(id.clone(), *count);
        }
        self.documents.insert(id, document);
    }
}

impl<Id> FtsIndex for Bm25Index<Id>
where
    Id: Ord + Hash + Clone,
{
    type Id = Id;

    fn insert(&mut self, id: Self::Id, document: &str) -> Result<(), HeuremaError> {
        self.simple_pipeline()?;
        let terms = document_terms(document);
        self.remove_document(&id);
        self.insert_document(id, terms);
        Ok(())
    }

    fn query(&self, query: &str, k: usize) -> Result<Vec<(Self::Id, f32)>, HeuremaError> {
        self.simple_pipeline()?;
        if k == 0 || self.documents.is_empty() {
            return Ok(Vec::new());
        }

        let document_count = self.documents.len() as f32;
        let average_length = self.total_document_terms as f32 / document_count;
        let query_terms: BTreeSet<String> = tokenize(query).into_iter().collect();
        let mut scores: BTreeMap<Id, f32> = BTreeMap::new();

        for term in query_terms {
            let Some(posting) = self.postings.get(&term) else {
                continue;
            };
            let document_frequency = posting.len() as f32;
            let idf = (1.0
                + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                .ln();
            for (id, term_frequency) in posting {
                let Some(document) = self.documents.get(id) else {
                    continue;
                };
                let length_ratio = document.length as f32 / average_length;
                let denominator =
                    *term_frequency as f32 + BM25_K1 * (1.0 - BM25_B + BM25_B * length_ratio);
                let contribution = idf * (*term_frequency as f32 * (BM25_K1 + 1.0)) / denominator;
                *scores.entry(id.clone()).or_default() += contribution;
            }
        }

        let mut ranked: Vec<(Id, f32)> = scores.into_iter().collect();
        ranked.sort_by(|(left_id, left_score), (right_id, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_id.cmp(right_id))
        });
        ranked.truncate(k);
        Ok(ranked)
    }

    fn remove(&mut self, id: &Self::Id) -> Result<(), HeuremaError> {
        self.simple_pipeline()?;
        self.remove_document(id);
        Ok(())
    }

    fn len(&self) -> usize {
        self.documents.len()
    }
}

fn document_terms(document: &str) -> DocumentTerms {
    let mut counts = BTreeMap::new();
    let mut length = 0;
    for token in tokenize(document) {
        *counts.entry(token).or_default() += 1;
        length += 1;
    }
    DocumentTerms { counts, length }
}

fn tokenize(input: &str) -> Vec<String> {
    input
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}
