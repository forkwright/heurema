//! BM25 against an independently written formula reference.
//!
//! The reference here is written from the published formula (the
//! Robertson/Spärck Jones relevance weight, as combined into BM25 in
//! Robertson and Zaragoza 2009, "The Probabilistic Relevance Framework: BM25
//! and Beyond") and from the scoring contract stated on `Bm25Index`. It shares
//! no code and no data structure with the engine: at every checkpoint it
//! re-tokenizes the live corpus from the raw document texts and recomputes
//! N, avgdl, every document frequency, and every score in `f64`, so it never
//! sees the engine's incremental postings. Seeded workloads drive inserts,
//! replacements, and removals (of present and absent IDs) through
//! `Bm25Index<u64>`, and every generated query's full result list must agree
//! with the reference up to the `f32` narrowing bound documented on
//! `tolerance`. Perturbed references run over the same workloads prove the
//! comparison can fail.

use std::collections::{BTreeMap, BTreeSet};

use heurema::{Bm25Index, FtsConfig, FtsIndex, HeuremaError};

use super::support::XorShift64;

/// Which idf form the reference applies.
#[derive(Clone, Copy, Debug)]
enum Idf {
    /// `ln(1 + (N - n + 0.5) / (n + 0.5))`, the documented engine form.
    NonNegative,
    /// `ln((N - n + 0.5) / (n + 0.5))`, the Robertson/Spärck Jones weight
    /// without the `1 +`, negative for terms in more than half the corpus.
    Classic,
}

/// How a term repeated within one query is weighted.
#[derive(Clone, Copy, Debug)]
enum QueryTerms {
    /// Each distinct term contributes once, as documented.
    Distinct,
    /// Each occurrence contributes again.
    EveryOccurrence,
}

/// What `|d|` counts.
#[derive(Clone, Copy, Debug)]
enum DocumentLength {
    /// Tokens, repeats included, as documented.
    Tokens,
    /// Distinct terms.
    DistinctTerms,
}

/// Whether documents yielding no tokens enter the corpus statistics.
#[derive(Clone, Copy, Debug)]
enum EmptyDocuments {
    /// They count toward N and avgdl, as documented.
    Counted,
    /// They are left out of N and avgdl.
    Excluded,
}

/// How text becomes terms.
#[derive(Clone, Copy, Debug)]
enum Tokenizer {
    /// Split on non-alphanumerics, then lowercase each token, as documented.
    Simple,
    /// Split on non-alphanumerics without lowercasing.
    CaseSensitive,
    /// Lowercase the whole text, then split.
    LowercaseFirst,
}

/// How inserts and removals change the live corpus.
#[derive(Clone, Copy, Debug)]
enum Lifecycle {
    /// An insert on a live ID replaces its document and a removal drops it,
    /// as documented.
    Faithful,
    /// An insert on a live ID keeps the first document.
    KeepFirst,
    /// Removals leave the corpus unchanged.
    IgnoreRemoval,
}

/// A BM25 reference: the formula parameters and the corpus rules it scores
/// under.
#[derive(Clone, Copy, Debug)]
struct Reference {
    k1: f64,
    b: f64,
    idf: Idf,
    query_terms: QueryTerms,
    document_length: DocumentLength,
    empty_documents: EmptyDocuments,
    tokenizer: Tokenizer,
    lifecycle: Lifecycle,
}

/// The reference exactly as `Bm25Index`'s rustdoc states the contract.
const CONTRACT: Reference = Reference {
    k1: 1.2,
    b: 0.75,
    idf: Idf::NonNegative,
    query_terms: QueryTerms::Distinct,
    document_length: DocumentLength::Tokens,
    empty_documents: EmptyDocuments::Counted,
    tokenizer: Tokenizer::Simple,
    lifecycle: Lifecycle::Faithful,
};

/// One deviation from the contract per entry. Each names a documented clause
/// the generated workloads must be able to tell apart from the contract.
const PERTURBATIONS: [(&str, Reference); 10] = [
    (
        "k1 = 1.25",
        Reference {
            k1: 1.25,
            ..CONTRACT
        },
    ),
    ("b = 0.7", Reference { b: 0.7, ..CONTRACT }),
    (
        "the classic idf without the 1 +",
        Reference {
            idf: Idf::Classic,
            ..CONTRACT
        },
    ),
    (
        "every repeated query term contributing",
        Reference {
            query_terms: QueryTerms::EveryOccurrence,
            ..CONTRACT
        },
    ),
    (
        "document length counting distinct terms",
        Reference {
            document_length: DocumentLength::DistinctTerms,
            ..CONTRACT
        },
    ),
    (
        "token-less documents left out of N and avgdl",
        Reference {
            empty_documents: EmptyDocuments::Excluded,
            ..CONTRACT
        },
    ),
    (
        "tokens that keep their case",
        Reference {
            tokenizer: Tokenizer::CaseSensitive,
            ..CONTRACT
        },
    ),
    (
        "text lowercased before it is split",
        Reference {
            tokenizer: Tokenizer::LowercaseFirst,
            ..CONTRACT
        },
    ),
    (
        "an insert on a live ID keeping the first document",
        Reference {
            lifecycle: Lifecycle::KeepFirst,
            ..CONTRACT
        },
    ),
    (
        "removals ignored",
        Reference {
            lifecycle: Lifecycle::IgnoreRemoval,
            ..CONTRACT
        },
    ),
];

/// f32 unit roundoff, 2^-24.
const F32_UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0;

/// One live document as the reference sees it, derived afresh from its text.
struct Document {
    id: u64,
    counts: BTreeMap<String, usize>,
    tokens: usize,
}

/// The formula inputs a document's score is a function of: its token count
/// and its term frequency for each distinct query term, in term order.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Profile {
    tokens: usize,
    frequencies: Vec<usize>,
}

/// A document the reference scores for a query.
struct Hit {
    id: u64,
    score: f64,
    /// Bound on how far the engine's `f32` score may sit from `score`.
    tolerance: f64,
    profile: Profile,
}

/// Maximal runs of alphanumeric characters, found by one scan over the
/// character boundaries of `text`.
fn alphanumeric_runs(text: &str) -> Vec<&str> {
    let mut runs = Vec::new();
    let mut start = None;
    for (offset, character) in text.char_indices() {
        match (start, character.is_alphanumeric()) {
            (None, true) => start = Some(offset),
            (Some(begin), false) => {
                runs.push(&text[begin..offset]);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(begin) = start {
        runs.push(&text[begin..]);
    }
    runs
}

/// Bound on |engine score - reference score| for a document matching
/// `matched` distinct query terms with reference score `score`.
///
/// WHY: the engine evaluates the documented formula in `f32` and the
/// reference in `f64`, so they agree only up to `f32` rounding. With
/// u = 2^-24 and every count exactly representable in `f32`, the engine's
/// first-order error is bounded term by term:
/// - idf: `ln(1 + x)` is taken of the rounded sum `1 + x`, which costs an
///   absolute u (plus under u from rounding `x`) however small idf is, and the
///   logarithm adds at most 2u relative. As n(t) nears N, `x` and idf both
///   approach zero, so this error is absolute rather than relative: at most
///   2u * (1 + idf).
/// - the tf and length factor `w` takes about eight roundings of positive
///   quantities (the constants 1.2 and 2.2 among them), and the product and
///   quotient two more: 12u relative on the contribution with the
///   logarithm's 2u, plus the idf's absolute 2u scaled by `w <= k1 + 1`.
/// - summing `m` positive contributions adds (m - 1)u relative.
///
/// Together: `(12 + m) * u * score + 4.4 * u * m`. The bound
/// `32u * (score + m)` covers both parts with room to spare for the at most
/// `MAX_QUERY_TOKENS` (5) distinct terms of a generated query: 32 against 17
/// on the relative part, 32 against 4.4 per term on the absolute part. The
/// perturbed references in `PERTURBATIONS` move scores by orders of magnitude
/// more. The reference's own `f64` error is near 2^-50 relative, negligible
/// here.
fn tolerance(score: f64, matched: usize) -> f64 {
    32.0 * F32_UNIT_ROUNDOFF * (score.abs() + matched as f64)
}

impl Reference {
    fn tokenize(&self, text: &str) -> Vec<String> {
        match self.tokenizer {
            Tokenizer::Simple => alphanumeric_runs(text)
                .into_iter()
                .map(str::to_lowercase)
                .collect(),
            Tokenizer::CaseSensitive => alphanumeric_runs(text)
                .into_iter()
                .map(str::to_owned)
                .collect(),
            Tokenizer::LowercaseFirst => alphanumeric_runs(&text.to_lowercase())
                .into_iter()
                .map(str::to_owned)
                .collect(),
        }
    }

    /// The live corpus as documents, rebuilt from the raw texts.
    fn snapshot(&self, corpus: &BTreeMap<u64, String>) -> Vec<Document> {
        corpus
            .iter()
            .map(|(id, text)| {
                let mut counts = BTreeMap::new();
                let mut tokens = 0;
                for token in self.tokenize(text) {
                    *counts.entry(token).or_insert(0) += 1;
                    tokens += 1;
                }
                Document {
                    id: *id,
                    counts,
                    tokens,
                }
            })
            .filter(|document| match self.empty_documents {
                EmptyDocuments::Counted => true,
                EmptyDocuments::Excluded => document.tokens > 0,
            })
            .collect()
    }

    fn length(&self, document: &Document) -> f64 {
        match self.document_length {
            DocumentLength::Tokens => document.tokens as f64,
            DocumentLength::DistinctTerms => document.counts.len() as f64,
        }
    }

    fn idf(&self, live: f64, containing: f64) -> f64 {
        let odds = (live - containing + 0.5) / (containing + 0.5);
        match self.idf {
            Idf::NonNegative => odds.ln_1p(),
            Idf::Classic => odds.ln(),
        }
    }

    /// Every document containing a query term, scored document-at-a-time and
    /// ranked by descending score, then ascending ID.
    fn rank(&self, documents: &[Document], query: &str) -> Vec<Hit> {
        let mut weights: BTreeMap<String, f64> = BTreeMap::new();
        for term in self.tokenize(query) {
            let weight = weights.entry(term).or_insert(0.0);
            *weight = match self.query_terms {
                QueryTerms::Distinct => 1.0,
                QueryTerms::EveryOccurrence => *weight + 1.0,
            };
        }
        let live = documents.len() as f64;
        let average = documents
            .iter()
            .map(|document| self.length(document))
            .sum::<f64>()
            / live;
        let terms: Vec<(&str, f64, f64)> = weights
            .iter()
            .map(|(term, weight)| {
                let containing = documents
                    .iter()
                    .filter(|document| document.counts.contains_key(term))
                    .count();
                (term.as_str(), *weight, self.idf(live, containing as f64))
            })
            .collect();

        let mut hits = Vec::new();
        for document in documents {
            let saturation = self.k1 * (1.0 - self.b + self.b * self.length(document) / average);
            let mut score = 0.0;
            let mut matched = 0;
            let mut frequencies = Vec::with_capacity(terms.len());
            for &(term, weight, idf) in &terms {
                let frequency = document.counts.get(term).copied().unwrap_or(0);
                frequencies.push(frequency);
                if frequency > 0 {
                    let tf = frequency as f64;
                    score += weight * idf * tf * (self.k1 + 1.0) / (tf + saturation);
                    matched += 1;
                }
            }
            if matched > 0 {
                hits.push(Hit {
                    id: document.id,
                    score,
                    tolerance: tolerance(score, matched),
                    profile: Profile {
                        tokens: document.tokens,
                        frequencies,
                    },
                });
            }
        }
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        hits
    }
}

/// Checks one query's engine answers against the reference ranking,
/// describing the first rule broken.
///
/// `engine_all` is the engine's answer with `k` at least the live document
/// count, and `engine_top` its answer for `k`.
fn compare(
    reference: &[Hit],
    engine_all: &[(u64, f32)],
    engine_top: &[(u64, f32)],
    k: usize,
) -> Result<(), String> {
    // WHY: which documents match is integer set logic, never a float
    // question, so result counts compare exactly: every document holding a
    // query term is returned, capped at k.
    let due = k.min(reference.len());
    if engine_top.len() != due {
        return Err(format!(
            "the k = {k} answer has {} results, but {} documents match so {due} are due",
            engine_top.len(),
            reference.len()
        ));
    }
    if engine_all.len() != reference.len() {
        return Err(format!(
            "the untruncated answer has {} results, but {} documents match",
            engine_all.len(),
            reference.len()
        ));
    }
    // WHY: a score is a function of the index state alone, so the top-k
    // answer is the head of the full ranking, bit for bit. Checking this
    // exactly leaves the tolerance to govern only the full ranking below.
    for (position, (top, all)) in engine_top.iter().zip(engine_all).enumerate() {
        if top.0 != all.0 || top.1.to_bits() != all.1.to_bits() {
            return Err(format!(
                "position {position} of the k = {k} answer is {top:?}, but the untruncated ranking has {all:?}"
            ));
        }
    }
    check_ranking(reference, engine_all)
}

/// Checks a full engine ranking against the reference's scores and order.
fn check_ranking(reference: &[Hit], engine: &[(u64, f32)]) -> Result<(), String> {
    let by_id: BTreeMap<u64, &Hit> = reference.iter().map(|hit| (hit.id, hit)).collect();
    let mut seen = BTreeSet::new();
    let mut twins: BTreeMap<&Profile, (u64, f32)> = BTreeMap::new();
    let mut ceiling: Option<(u64, f64)> = None;
    let mut previous: Option<(u64, f32)> = None;
    for &(id, score) in engine {
        if !seen.insert(id) {
            return Err(format!("document {id} appears twice"));
        }
        let Some(hit) = by_id.get(&id) else {
            return Err(format!(
                "document {id} holds no query term yet was returned with score {score}"
            ));
        };
        if !score.is_finite() {
            return Err(format!("document {id} scored {score}"));
        }
        // WHY: the engine's own sequence must honour the ranking contract
        // exactly: non-increasing scores, and equal f32 scores in ascending
        // ID order.
        if let Some((earlier, earlier_score)) = previous {
            if score > earlier_score {
                return Err(format!(
                    "document {id} ({score}) follows document {earlier} ({earlier_score}) with a higher score"
                ));
            }
            if score.to_bits() == earlier_score.to_bits() && id < earlier {
                return Err(format!(
                    "documents {earlier} and {id} tie at {score} but are not in ascending ID order"
                ));
            }
        }
        // WHY: two documents with identical formula inputs have
        // mathematically equal scores, so the contract ties them; a
        // deterministic f32 evaluation gives them identical bits, which is
        // what lets the ascending-ID rule above apply to them exactly.
        if let Some(&(twin, twin_score)) = twins.get(&hit.profile) {
            if twin_score.to_bits() != score.to_bits() {
                return Err(format!(
                    "documents {twin} and {id} have identical formula inputs but scores {twin_score} and {score}"
                ));
            }
        } else {
            twins.insert(&hit.profile, (id, score));
        }
        // WHY: the engine must rank an earlier document above a later one
        // wherever the reference separates them by more than their combined
        // tolerance. Pairs closer than that may land in either order, because
        // f32 rounding can legitimately invert them; the ascending-ID rule
        // above still binds wherever the engine's own scores are equal.
        // `ceiling` is the smallest score + tolerance over earlier positions,
        // so one comparison covers every earlier document. Given the
        // non-increasing check above, a breach here implies this document's
        // score drifts beyond its tolerance, so the drift check below would
        // catch it too; checking order first names the mis-ranked pair.
        if let Some((earlier, bound)) = ceiling
            && hit.score - hit.tolerance > bound
        {
            return Err(format!(
                "document {id} (reference {}) is ranked below document {earlier}, which the reference scores lower by more than their tolerance",
                hit.score
            ));
        }
        let drift = (f64::from(score) - hit.score).abs();
        if drift > hit.tolerance {
            return Err(format!(
                "document {id} scored {score}, but the reference scores {} (drift {drift} beyond tolerance {})",
                hit.score, hit.tolerance
            ));
        }
        if ceiling.is_none_or(|(_, bound)| hit.score + hit.tolerance < bound) {
            ceiling = Some((id, hit.score + hit.tolerance));
        }
        previous = Some((id, score));
    }
    Ok(())
}

/// Terms by descending draw weight, each with the surface forms a generated
/// text may spell it with. The skew repeats early terms within and across
/// documents (driving n(t) toward N, where idf nears zero) while late terms
/// stay rare.
///
/// NOTE: `ΟΔΟΣ` lowercases to a final sigma only when lowercased as its own
/// token, and `İ` lowercases to `i` plus U+0307, which is not alphanumeric,
/// so `i̇stanbul` typed in lowercase splits in two. Both pin the documented
/// split-then-lowercase order.
const VOCABULARY: [&[&str]; 14] = [
    &["the", "The", "THE"],
    &["index", "Index", "INDEX"],
    &["search", "Search"],
    &["rank", "RANK"],
    &["x1", "X1"],
    &["caf\u{e9}", "CAF\u{c9}", "Caf\u{e9}"],
    &["stra\u{df}e", "STRA\u{1e9e}E"],
    &[
        "\u{3bf}\u{3b4}\u{3bf}\u{3c2}",
        "\u{39f}\u{394}\u{39f}\u{3a3}",
        "\u{39f}\u{3b4}\u{3bf}\u{3c2}",
    ],
    &["2026"],
    &["\u{130}stanbul", "\u{130}STANBUL"],
    &["vector", "Vector"],
    &["x\u{663}"],
    &["i\u{307}stanbul"],
    &["rare"],
];

/// Query words no vocabulary entry tokenizes to.
const ABSENT: [&str; 3] = ["zzz", "Absent", "q9"];

/// Separators, space-weighted. Every one is non-alphanumeric; `.` and `'`
/// are case-ignorable, which is what exposes a lowercase-before-split
/// tokenizer through the final sigma.
const SEPARATORS: [&str; 12] = [
    " ", " ", " ", "  ", ", ", ". ", ".", "'", "-", "_", "\t", "\n",
];

const MAX_DOCUMENT_TOKENS: usize = 12;
const MAX_QUERY_TOKENS: usize = 5;
const PROBES_PER_CHECKPOINT: usize = 16;

/// Where `k` sits relative to the reference's match count.
#[derive(Clone, Copy, Debug)]
enum Cap {
    Zero,
    One,
    Half,
    OneBelow,
    Exact,
    OneAbove,
    Beyond,
}

const CAPS: [Cap; 7] = [
    Cap::Zero,
    Cap::One,
    Cap::Half,
    Cap::OneBelow,
    Cap::Exact,
    Cap::OneAbove,
    Cap::Beyond,
];

impl Cap {
    fn resolve(self, matches: usize, live: usize) -> usize {
        match self {
            Self::Zero => 0,
            Self::One => 1,
            Self::Half => matches / 2,
            Self::OneBelow => matches.saturating_sub(1),
            Self::Exact => matches,
            Self::OneAbove => matches + 1,
            Self::Beyond => live + 3,
        }
    }
}

struct Probe {
    text: String,
    cap: Cap,
}

enum Step {
    Insert { id: u64, text: String },
    Remove { id: u64 },
    Check { probes: Vec<Probe> },
}

/// A generated operation sequence, fixed before any reference sees it so
/// every reference replays the same workload.
struct Workload {
    seed: u64,
    steps: Vec<Step>,
}

/// Shape of one generated workload.
#[derive(Clone, Copy)]
struct Spec {
    seed: u64,
    operations: usize,
    /// Distinct IDs drawn from; a small space forces replacement churn.
    id_space: usize,
    checkpoint_every: usize,
}

const SPECS: [Spec; 4] = [
    Spec {
        seed: 0xB25F_0001,
        operations: 60,
        id_space: 6,
        checkpoint_every: 6,
    },
    Spec {
        seed: 0xB25F_0002,
        operations: 120,
        id_space: 40,
        checkpoint_every: 20,
    },
    Spec {
        seed: 0xB25F_0003,
        operations: 400,
        id_space: 160,
        checkpoint_every: 50,
    },
    Spec {
        seed: 0xB25F_0004,
        operations: 1000,
        id_space: 600,
        checkpoint_every: 125,
    },
];

/// Uniform draw in `0..bound`; `bound` is non-zero.
fn draw(rng: &mut XorShift64, bound: usize) -> usize {
    // WHY: `next_f32` lies in [0, 1); the `min` keeps a product that rounds
    // up to `bound` in range.
    ((rng.next_f32() * bound as f32) as usize).min(bound - 1)
}

/// Draw in `0..bound` skewed toward zero: P(result <= j) = sqrt((j + 1) / bound).
fn skewed(rng: &mut XorShift64, bound: usize) -> usize {
    let unit = rng.next_f32();
    ((unit * unit * bound as f32) as usize).min(bound - 1)
}

fn pick<'a>(rng: &mut XorShift64, options: &[&'a str]) -> &'a str {
    options[draw(rng, options.len())]
}

fn surface(rng: &mut XorShift64) -> &'static str {
    let forms = VOCABULARY[skewed(rng, VOCABULARY.len())];
    pick(rng, forms)
}

/// Joins `words` with random separators, sometimes leading or trailing.
fn join(rng: &mut XorShift64, words: &[&str]) -> String {
    let mut text = String::new();
    if rng.next_f32() < 0.2 {
        text.push_str(pick(rng, &SEPARATORS));
    }
    for (position, word) in words.iter().enumerate() {
        if position > 0 {
            text.push_str(pick(rng, &SEPARATORS));
        }
        text.push_str(word);
    }
    if rng.next_f32() < 0.2 {
        text.push_str(pick(rng, &SEPARATORS));
    }
    text
}

/// A document of up to `MAX_DOCUMENT_TOKENS` words; zero words yields a
/// document with no tokens.
fn document(rng: &mut XorShift64) -> String {
    let count = draw(rng, MAX_DOCUMENT_TOKENS + 1);
    let words: Vec<&str> = (0..count).map(|_| surface(rng)).collect();
    join(rng, &words)
}

/// A query of up to `MAX_QUERY_TOKENS` words mixing vocabulary terms, words
/// absent from every document, and repeats of earlier words.
fn probe(rng: &mut XorShift64) -> Probe {
    let count = draw(rng, MAX_QUERY_TOKENS + 1);
    let mut words: Vec<&str> = Vec::with_capacity(count);
    for _ in 0..count {
        let roll = rng.next_f32();
        let word = if roll < 0.15 {
            pick(rng, &ABSENT)
        } else if roll < 0.4 && !words.is_empty() {
            pick(rng, &words)
        } else {
            surface(rng)
        };
        words.push(word);
    }
    Probe {
        text: join(rng, &words),
        cap: CAPS[draw(rng, CAPS.len())],
    }
}

fn generate(spec: Spec) -> Workload {
    let mut rng = XorShift64::new(spec.seed);
    let mut live: BTreeMap<u64, String> = BTreeMap::new();
    let mut steps = Vec::new();
    for operation in 1..=spec.operations {
        let raw = draw(&mut rng, spec.id_space) as u64;
        // WHY: mirroring every fifth raw ID to the top of the u64 range puts
        // ascending-ID tie-breaks beyond small integers.
        let id = if raw.is_multiple_of(5) {
            u64::MAX - raw
        } else {
            raw
        };
        let roll = rng.next_f32();
        if roll < 0.2 {
            live.remove(&id);
            steps.push(Step::Remove { id });
        } else {
            // WHY: re-inserting a live document's text under another ID
            // creates exact score ties.
            let copy = if roll < 0.32 && !live.is_empty() {
                live.values().nth(draw(&mut rng, live.len())).cloned()
            } else {
                None
            };
            let text = copy.unwrap_or_else(|| document(&mut rng));
            live.insert(id, text.clone());
            steps.push(Step::Insert { id, text });
        }
        if operation % spec.checkpoint_every == 0 || operation == spec.operations {
            let probes = (0..PROBES_PER_CHECKPOINT)
                .map(|_| probe(&mut rng))
                .collect();
            steps.push(Step::Check { probes });
        }
    }
    Workload {
        seed: spec.seed,
        steps,
    }
}

/// Counts of the behaviours a replay exercised.
#[derive(Default)]
struct Coverage {
    fresh_inserts: usize,
    replacements: usize,
    present_removals: usize,
    absent_removals: usize,
    tokenless_documents: usize,
    single_term_queries: usize,
    multi_term_queries: usize,
    repeated_term_queries: usize,
    unknown_term_queries: usize,
    no_match_queries: usize,
    exact_tie_queries: usize,
    k_below_matches: usize,
    k_above_matches: usize,
}

impl Coverage {
    fn note_query(
        &mut self,
        reference: &Reference,
        documents: &[Document],
        query: &str,
        hits: &[Hit],
        k: usize,
    ) {
        let tokens = reference.tokenize(query);
        let distinct: BTreeSet<&String> = tokens.iter().collect();
        self.single_term_queries += usize::from(distinct.len() == 1);
        self.multi_term_queries += usize::from(distinct.len() > 1);
        self.repeated_term_queries += usize::from(distinct.len() < tokens.len());
        self.unknown_term_queries += usize::from(distinct.iter().any(|term| {
            documents
                .iter()
                .all(|document| !document.counts.contains_key(*term))
        }));
        self.no_match_queries += usize::from(hits.is_empty());
        let mut profiles = BTreeSet::new();
        self.exact_tie_queries +=
            usize::from(!hits.iter().all(|hit| profiles.insert(&hit.profile)));
        self.k_below_matches += usize::from(k < hits.len());
        self.k_above_matches += usize::from(k > hits.len());
    }

    /// Names of the behaviours no replay reached.
    fn missing(&self) -> Vec<&'static str> {
        [
            ("inserts of fresh IDs", self.fresh_inserts),
            ("replacements", self.replacements),
            ("removals of present IDs", self.present_removals),
            ("removals of absent IDs", self.absent_removals),
            ("documents yielding no tokens", self.tokenless_documents),
            ("single-term queries", self.single_term_queries),
            ("multi-term queries", self.multi_term_queries),
            ("queries repeating a term", self.repeated_term_queries),
            ("queries with an unknown term", self.unknown_term_queries),
            ("queries matching nothing", self.no_match_queries),
            (
                "queries with exactly tied documents",
                self.exact_tie_queries,
            ),
            ("k below the match count", self.k_below_matches),
            ("k above the match count", self.k_above_matches),
        ]
        .into_iter()
        .filter(|(_, count)| *count == 0)
        .map(|(name, _)| name)
        .collect()
    }
}

#[derive(Default)]
struct Outcome {
    disagreements: Vec<String>,
    coverage: Coverage,
}

/// Runs every probe of one checkpoint against the engine and a reference
/// recomputed from the live corpus texts.
fn check(
    index: &Bm25Index<u64>,
    corpus: &BTreeMap<u64, String>,
    reference: &Reference,
    probes: &[Probe],
    context: &str,
    outcome: &mut Outcome,
) -> Result<(), HeuremaError> {
    let documents = reference.snapshot(corpus);
    for probe in probes {
        let hits = reference.rank(&documents, &probe.text);
        let k = probe.cap.resolve(hits.len(), corpus.len());
        let engine_all = index.query(&probe.text, index.len())?;
        let engine_top = index.query(&probe.text, k)?;
        outcome
            .coverage
            .note_query(reference, &documents, &probe.text, &hits, k);
        if let Err(detail) = compare(&hits, &engine_all, &engine_top, k) {
            outcome.disagreements.push(format!(
                "{context}, {} live documents, query {:?}, k = {k}: {detail}",
                corpus.len(),
                probe.text
            ));
        }
    }
    Ok(())
}

/// Replays one workload through a fresh engine and the reference's corpus
/// model, recording every disagreement.
fn replay(
    workload: &Workload,
    reference: &Reference,
    outcome: &mut Outcome,
) -> Result<(), HeuremaError> {
    let mut index = Bm25Index::<u64>::new(FtsConfig::simple());
    let mut corpus: BTreeMap<u64, String> = BTreeMap::new();
    for (position, step) in workload.steps.iter().enumerate() {
        let context = format!("workload {:#x} step {position}", workload.seed);
        match step {
            Step::Insert { id, text } => {
                index.insert(*id, text)?;
                let live = corpus.contains_key(id);
                outcome.coverage.replacements += usize::from(live);
                outcome.coverage.fresh_inserts += usize::from(!live);
                outcome.coverage.tokenless_documents +=
                    usize::from(reference.tokenize(text).is_empty());
                if !live || !matches!(reference.lifecycle, Lifecycle::KeepFirst) {
                    corpus.insert(*id, text.clone());
                }
            }
            Step::Remove { id } => {
                index.remove(id)?;
                let live = corpus.contains_key(id);
                outcome.coverage.present_removals += usize::from(live);
                outcome.coverage.absent_removals += usize::from(!live);
                if !matches!(reference.lifecycle, Lifecycle::IgnoreRemoval) {
                    corpus.remove(id);
                }
            }
            Step::Check { probes } => check(&index, &corpus, reference, probes, &context, outcome)?,
        }
        if index.len() != corpus.len() {
            outcome.disagreements.push(format!(
                "{context}: len() is {}, but {} documents are live",
                index.len(),
                corpus.len()
            ));
        }
    }
    Ok(())
}

fn replay_all(workloads: &[Workload], reference: &Reference) -> Result<Outcome, HeuremaError> {
    let mut outcome = Outcome::default();
    for workload in workloads {
        replay(workload, reference, &mut outcome)?;
    }
    Ok(outcome)
}

fn ids(results: &[(u64, f32)]) -> Vec<u64> {
    results.iter().map(|(id, _)| *id).collect()
}

#[test]
fn engine_matches_the_formula_reference_on_generated_workloads() -> Result<(), HeuremaError> {
    let workloads = SPECS.map(generate);

    let outcome = replay_all(&workloads, &CONTRACT)?;

    assert!(
        outcome.disagreements.is_empty(),
        "the engine disagrees with the formula reference {} times; first:\n{}",
        outcome.disagreements.len(),
        outcome.disagreements[..outcome.disagreements.len().min(5)].join("\n")
    );
    let missing = outcome.coverage.missing();
    assert!(
        missing.is_empty(),
        "the generated workloads never exercised: {missing:?}"
    );
    Ok(())
}

#[test]
fn comparator_detects_every_perturbed_reference() -> Result<(), HeuremaError> {
    let workloads = SPECS.map(generate);

    for (change, perturbed) in PERTURBATIONS {
        let outcome = replay_all(&workloads, &perturbed)?;

        assert!(
            !outcome.disagreements.is_empty(),
            "a reference with {change} must disagree with the engine on the generated workloads, or the comparison cannot see that clause"
        );
    }
    Ok(())
}

#[test]
fn comparator_rejects_results_that_break_the_reference_ranking() -> Result<(), HeuremaError> {
    let corpus: BTreeMap<u64, String> = [
        (1, "alpha beta"),
        (2, "alpha beta"),
        (3, "alpha alpha alpha gamma"),
        (4, "beta delta"),
        (5, "gamma"),
    ]
    .into_iter()
    .map(|(id, text)| (id, text.to_owned()))
    .collect();
    let mut index = Bm25Index::<u64>::new(FtsConfig::simple());
    for (id, text) in &corpus {
        index.insert(*id, text)?;
    }
    let reference = CONTRACT.rank(&CONTRACT.snapshot(&corpus), "alpha");
    let all = index.query("alpha", index.len())?;
    let top = index.query("alpha", 2)?;
    assert_eq!(
        ids(&all),
        [3, 1, 2],
        "fixture: document 3 leads, then the tied documents 1 and 2"
    );
    let agreement = compare(&reference, &all, &top, 2);
    assert!(
        agreement.is_ok(),
        "the engine's own answer agrees: {agreement:?}"
    );

    let mut tie_reversed = all.clone();
    tie_reversed.swap(1, 2);
    let mut misranked = all.clone();
    misranked[0].0 = all[1].0;
    misranked[1].0 = all[0].0;
    let mut drifted = all.clone();
    drifted[0].1 = (f64::from(all[0].1) + 2.0 * reference[0].tolerance) as f32;
    let mut foreign = all.clone();
    foreign[2].0 = 4;
    let mut duplicated = all.clone();
    duplicated[2].0 = 1;
    let full_rankings = [
        ("tied documents in descending ID order", tie_reversed),
        ("IDs swapped across a clear score gap", misranked),
        ("a score twice the tolerance off", drifted),
        ("a document holding no query term", foreign),
        ("a document returned twice", duplicated),
        ("a matching document dropped", all[..2].to_vec()),
    ];
    for (corruption, ranking) in full_rankings {
        assert!(
            compare(&reference, &ranking, &ranking, ranking.len()).is_err(),
            "the comparator rejects {corruption}"
        );
    }
    assert!(
        compare(&reference, &all, &all, 2).is_err(),
        "the comparator rejects a top-k answer longer than k"
    );
    assert!(
        compare(&reference, &all, &[all[0], all[2]], 2).is_err(),
        "the comparator rejects a top-k answer that is not the head of the full ranking"
    );
    Ok(())
}

#[test]
fn reference_reproduces_hand_computed_scores() {
    let corpus = BTreeMap::from([(1, String::from("a b")), (2, String::from("b c c"))]);
    let documents = CONTRACT.snapshot(&corpus);

    // WHY: worked by hand from the documented formula: N = 2, avgdl = 2.5.
    // "c": n = 1, idf = ln(1 + 1.5 / 1.5) = ln 2. Document 2 has tf = 2 and
    // |d| = 3: 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2.5)) = 4.4 / 3.38.
    // "B b": one distinct term, n = 2, idf = ln(1 + 0.5 / 2.5) = ln 1.2.
    // Document 1 (|d| = 2): 2.2 / (1 + 1.2 * 0.85) = 2.2 / 2.02.
    // Document 2 (|d| = 3): 2.2 / (1 + 1.2 * 1.15) = 2.2 / 2.38.
    let cases = [
        ("c", vec![(2, 2.0_f64.ln() * 4.4 / 3.38)]),
        (
            "B b",
            vec![
                (1, 1.2_f64.ln() * 2.2 / 2.02),
                (2, 1.2_f64.ln() * 2.2 / 2.38),
            ],
        ),
    ];
    for (query, expected) in cases {
        let hits = CONTRACT.rank(&documents, query);
        assert_eq!(
            hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            expected.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            "query {query:?} ranks the hand-computed documents"
        );
        for (hit, (_, score)) in hits.iter().zip(&expected) {
            assert!(
                (hit.score - score).abs() <= 1e-12 * score,
                "query {query:?}, document {}: reference {} against hand-computed {score}",
                hit.id,
                hit.score
            );
        }
    }
}

#[test]
fn reference_tokenizer_splits_on_non_alphanumerics_before_lowercasing() {
    let cases: [(&str, &[&str]); 7] = [
        ("Hello, WORLD-42 x_y", &["hello", "world", "42", "x", "y"]),
        ("", &[]),
        (" .,-_' \t\n", &[]),
        (
            "\u{39f}\u{394}\u{39f}\u{3a3}.ALPHA",
            &["\u{3bf}\u{3b4}\u{3bf}\u{3c2}", "alpha"],
        ),
        ("STRA\u{1e9e}E Caf\u{e9}", &["stra\u{df}e", "caf\u{e9}"]),
        ("\u{130}STANBUL x\u{663}", &["i\u{307}stanbul", "x\u{663}"]),
        ("i\u{307}stanbul", &["i", "stanbul"]),
    ];
    for (text, expected) in cases {
        assert_eq!(
            CONTRACT.tokenize(text),
            expected,
            "Simple tokenization of {text:?}"
        );
    }
}
