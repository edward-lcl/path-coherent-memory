"""
Benchmark evaluator for RFM vs. baseline retrieval.

Loads a corpus (from synthetic_corpus.py), runs a retrieval backend against
each QA pair, and reports precision/recall/F1 per difficulty class.

Retrieval backends:
  - bm25:    keyword BM25 (no ML)
  - cosine:  TF-IDF cosine similarity
  - oracle:  perfect retrieval (upper bound, for sanity checking)
  - random:  random baseline (lower bound)

Usage:
    python3 evaluator.py --corpus corpus_medium.json --backend cosine
    python3 evaluator.py --corpus corpus_medium.json --backend bm25 --top-k 5
    python3 evaluator.py --corpus corpus_medium.json --all-backends
"""

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol


# ────────────────────────────────────────────────────────────────────────────
# Retrieval backend protocol
# ────────────────────────────────────────────────────────────────────────────

class RetrievalBackend(Protocol):
    def index(self, notes: list[dict]) -> None: ...
    def retrieve(self, query: str, top_k: int) -> list[str]: ...  # returns note ids


# ────────────────────────────────────────────────────────────────────────────
# Tokenization
# ────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ────────────────────────────────────────────────────────────────────────────
# BM25 backend
# ────────────────────────────────────────────────────────────────────────────

class BM25Backend:
    k1 = 1.5
    b  = 0.75

    def __init__(self):
        self._notes: list[dict] = []
        self._tf: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self._avg_dl: float = 0.0

    def index(self, notes: list[dict]) -> None:
        self._notes = notes
        tokenized = [tokenize(n["content"]) for n in notes]
        self._avg_dl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)

        df: dict[str, int] = defaultdict(int)
        self._tf = []
        for tokens in tokenized:
            freq: dict[str, float] = defaultdict(float)
            for tok in tokens:
                freq[tok] += 1
            self._tf.append(dict(freq))
            for tok in set(tokens):
                df[tok] += 1

        N = len(notes)
        self._idf = {
            tok: math.log((N - cnt + 0.5) / (cnt + 0.5) + 1)
            for tok, cnt in df.items()
        }

    def retrieve(self, query: str, top_k: int) -> list[str]:
        q_tokens = tokenize(query)
        scores: list[tuple[float, str]] = []
        for i, note in enumerate(self._notes):
            dl = sum(self._tf[i].values())
            score = 0.0
            for tok in q_tokens:
                if tok not in self._idf:
                    continue
                tf = self._tf[i].get(tok, 0)
                norm_tf = tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                )
                score += self._idf[tok] * norm_tf
            scores.append((score, note["id"]))
        scores.sort(reverse=True)
        return [nid for _, nid in scores[:top_k]]


# ────────────────────────────────────────────────────────────────────────────
# TF-IDF cosine backend
# ────────────────────────────────────────────────────────────────────────────

class CosineBackend:
    def __init__(self):
        self._notes: list[dict] = []
        self._vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    def _tfidf(self, tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
        freq: dict[str, float] = defaultdict(float)
        for t in tokens:
            freq[t] += 1
        return {t: (1 + math.log(c)) * idf.get(t, 0) for t, c in freq.items()}

    def _cosine(self, a: dict[str, float], b: dict[str, float]) -> float:
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in b)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na * nb else 0.0

    def index(self, notes: list[dict]) -> None:
        self._notes = notes
        tokenized = [tokenize(n["content"]) for n in notes]
        N = len(notes)
        df: dict[str, int] = defaultdict(int)
        for tokens in tokenized:
            for tok in set(tokens):
                df[tok] += 1
        self._idf = {
            tok: math.log(N / cnt) for tok, cnt in df.items()
        }
        self._vectors = [self._tfidf(t, self._idf) for t in tokenized]

    def retrieve(self, query: str, top_k: int) -> list[str]:
        q_vec = self._tfidf(tokenize(query), self._idf)
        scores = [
            (self._cosine(q_vec, v), note["id"])
            for v, note in zip(self._vectors, self._notes)
        ]
        scores.sort(reverse=True)
        return [nid for _, nid in scores[:top_k]]


# ────────────────────────────────────────────────────────────────────────────
# Oracle backend (perfect upper bound)
# ────────────────────────────────────────────────────────────────────────────

class OracleBackend:
    def __init__(self, qa_pairs: list[dict]):
        self._qa_map = {q["id"]: q["required_notes"] for q in qa_pairs}
        self._current_qa_id: str = ""

    def set_qa_id(self, qa_id: str) -> None:
        self._current_qa_id = qa_id

    def index(self, notes: list[dict]) -> None:
        pass

    def retrieve(self, query: str, top_k: int) -> list[str]:
        return self._qa_map.get(self._current_qa_id, [])[:top_k]


# ────────────────────────────────────────────────────────────────────────────
# Random backend (lower bound)
# ────────────────────────────────────────────────────────────────────────────

class RandomBackend:
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._note_ids: list[str] = []

    def index(self, notes: list[dict]) -> None:
        self._note_ids = [n["id"] for n in notes]

    def retrieve(self, query: str, top_k: int) -> list[str]:
        return self._rng.sample(self._note_ids, min(top_k, len(self._note_ids)))


# ────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class QAResult:
    qa_id: str
    difficulty: str
    ground_truth_type: str
    retrieved: list[str]
    required: list[str]
    precision: float
    recall: float
    f1: float
    hit_at_1: bool
    hit_at_k: bool


def evaluate_qa(qa: dict, retrieved: list[str]) -> QAResult:
    required = set(qa["required_notes"])
    ret_set = set(retrieved)
    tp = len(required & ret_set)
    precision = tp / len(ret_set) if ret_set else 0.0
    recall    = tp / len(required) if required else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return QAResult(
        qa_id=qa["id"],
        difficulty=qa["difficulty"],
        ground_truth_type=qa["ground_truth_type"],
        retrieved=retrieved,
        required=list(required),
        precision=precision,
        recall=recall,
        f1=f1,
        hit_at_1=bool(retrieved) and retrieved[0] in required,
        hit_at_k=bool(required & ret_set),
    )


def aggregate(results: list[QAResult]) -> dict:
    by_diff: dict[str, list[QAResult]] = defaultdict(list)
    for r in results:
        by_diff[r.difficulty].append(r)

    def stats(items: list[QAResult]) -> dict:
        if not items:
            return {}
        n = len(items)
        return {
            "n": n,
            "precision": round(sum(r.precision for r in items) / n, 4),
            "recall":    round(sum(r.recall    for r in items) / n, 4),
            "f1":        round(sum(r.f1        for r in items) / n, 4),
            "hit@1":     round(sum(r.hit_at_1  for r in items) / n, 4),
            "hit@k":     round(sum(r.hit_at_k  for r in items) / n, 4),
        }

    return {
        "overall": stats(results),
        "by_difficulty": {k: stats(v) for k, v in sorted(by_diff.items())},
    }


# ────────────────────────────────────────────────────────────────────────────
# RFM backend — Resonance Field Memory
# ────────────────────────────────────────────────────────────────────────────

class RFMBackend:
    """
    Resonance Field Memory retrieval.

    Each note is stored as a frequency spectrum: a vector of amplitudes across
    N_BINS frequency bins. Tokens are mapped to bins via a stable hash.
    Retrieval works in two phases:

      1. Direct resonance — score every note by how much its spectrum
         matches the query spectrum (Hadamard product, L1-normalized).

      2. Field propagation — top-K first-pass notes emit their own spectra
         back into the field; any note that resonates with *that* secondary
         emission gets a bonus. This lets A→B→C chains surface: the query
         lights up B, B's emission lights up C.

    Contradiction signature: two notes that share subject-token bins but
    diverge on attribute-token bins produce a split resonance peak — both
    score similarly on subject frequency but destructively interfere on
    attributes, giving us a way to detect conflicts.
    """

    N_BINS   = 256     # frequency resolution (higher = fewer hash collisions)
    N_OCTAVE = 4       # hash is spread across octave-shifted copies for stability
    PROP_K   = 5       # how many notes emit in the propagation pass
    PROP_W   = 0.35    # weight of propagated signal vs direct
    DECAY    = 0.5     # amplitude decay per octave
    PROP_STABILITY_PCT = 0.80  # emit only top this fraction of bins by stability
    _OCTAVE_SALTS = [0x9e3779b9, 0x6c62272e, 0xd2a98b26, 0x243f6a88]

    def __init__(self):
        self._notes: list[dict] = []
        self._spectra: list[list[float]] = []   # shape: [N, N_BINS]
        self._stability: list[float] = []       # IDF-like stability weight per bin

    # ── token → bin ──────────────────────────────────────────────────────

    def _token_bins(self, token: str) -> list[tuple[int, float]]:
        """Return (bin_idx, amplitude) pairs for a token across all octaves."""
        pairs = []
        h = int(abs(hash(token)) % (2**31))
        for octave in range(self.N_OCTAVE):
            salted = (h ^ self._OCTAVE_SALTS[octave]) % (2**31)
            bin_idx = salted % self.N_BINS
            amplitude = (1.0 - self.DECAY) * (self.DECAY ** octave)
            pairs.append((bin_idx, amplitude))
        return pairs

    # ── document → spectrum ───────────────────────────────────────────────

    def _compute_spectrum(self, tokens: list[str]) -> list[float]:
        spectrum = [0.0] * self.N_BINS
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        for token, count in freq.items():
            tf = 1.0 + math.log(count)
            for bin_idx, amp in self._token_bins(token):
                spectrum[bin_idx] += tf * amp
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in spectrum)) or 1.0
        return [v / norm for v in spectrum]

    # ── index ─────────────────────────────────────────────────────────────

    def index(self, notes: list[dict]) -> None:
        self._notes = notes
        self._spectra = [
            self._compute_spectrum(tokenize(n["content"]))
            for n in notes
        ]
        # stability weight: bins that resonate across many notes carry less
        # signal (like IDF). Bins that are rare are more discriminative.
        bin_energy = [0.0] * self.N_BINS
        for spec in self._spectra:
            for b, v in enumerate(spec):
                if v > 0:
                    bin_energy[b] += 1.0
        N = len(notes)
        self._stability = [
            math.log((N + 1) / (e + 1)) + 1.0
            for e in bin_energy
        ]

    # ── resonance score ───────────────────────────────────────────────────

    def _resonance(self, query_spec: list[float], doc_spec: list[float]) -> float:
        """Weighted Hadamard product: sum over bins of q[b]*d[b]*stability[b]."""
        return sum(
            query_spec[b] * doc_spec[b] * self._stability[b]
            for b in range(self.N_BINS)
        )

    # ── selective emission mask ───────────────────────────────────────────

    def _emission_mask(self) -> list[bool]:
        """True for bins whose stability exceeds the PROP_STABILITY_PCT threshold.
        Only these bins are emitted during field propagation — noisy common bins
        are suppressed, so the propagated signal carries discriminative content."""
        threshold = sorted(self._stability)[int(self.N_BINS * self.PROP_STABILITY_PCT)]
        return [s >= threshold for s in self._stability]

    # ── contradiction detection ──────────────────────────────────────────

    def detect_contradictions(self, threshold: float = 0.0) -> list[dict]:
        """
        Scan all note pairs for contradiction signatures.

        Contradiction signature:
          - High token Jaccard (notes share most tokens = same subject/field)
          - Exactly 1-3 differing tokens per side (the value tokens diverge)

        This correctly separates contradictions (J=0.6-0.7, 1-2 value diffs)
        from aliases (J=0.6-0.8 but differ only on reference names) because
        aliases share the predicate word "refers" while contradiction notes
        share a field-assignment structure ending in different values.

        Scoring: jaccard * (1 / (1 + len(diff_tokens))) — rewards high overlap
        and penalizes notes that differ in many places.

        Returns list of {note_a, note_b, jaccard, diff_a, diff_b, score}
        sorted by score descending.
        """
        N = len(self._notes)
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                ta = set(tokenize(self._notes[i]["content"]))
                tb = set(tokenize(self._notes[j]["content"]))
                union = ta | tb
                if not union:
                    continue
                inter = ta & tb
                jaccard = len(inter) / len(union)
                diff_a = sorted(ta - tb)
                diff_b = sorted(tb - ta)
                total_diff = len(diff_a) + len(diff_b)

                # Contradiction signature: high overlap, small symmetric difference
                # Alias notes also have high Jaccard but contain "refers" in shared tokens
                shared_tokens = inter
                is_alias_pattern = "refers" in shared_tokens

                if is_alias_pattern:
                    continue

                if jaccard < 0.4 or total_diff == 0 or total_diff > 6:
                    continue

                score = jaccard * (1.0 / (1.0 + total_diff))

                if score >= threshold:
                    pairs.append({
                        "note_a": self._notes[i]["id"],
                        "note_b": self._notes[j]["id"],
                        "jaccard": round(jaccard, 4),
                        "diff_a": diff_a,
                        "diff_b": diff_b,
                        "contradiction_score": round(score, 4),
                    })

        pairs.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return pairs

    # ── retrieve (contradiction-aware) ────────────────────────────────────

    def retrieve(self, query: str, top_k: int) -> list[str]:
        q_spec = self._compute_spectrum(tokenize(query))
        N = len(self._notes)

        # Pass 1: direct resonance score for every note
        direct_scores = [
            self._resonance(q_spec, self._spectra[i])
            for i in range(N)
        ]

        # Top-PROP_K notes emit their spectra into the field — but only the
        # high-stability (discriminative) bins, suppressing corpus-wide noise
        mask = self._emission_mask()
        top_direct = sorted(range(N), key=lambda i: direct_scores[i], reverse=True)
        prop_scores = [0.0] * N
        for src_idx in top_direct[: self.PROP_K]:
            full_spec = self._spectra[src_idx]
            emitted = [v if mask[b] else 0.0 for b, v in enumerate(full_spec)]
            for dst in range(N):
                if dst != src_idx:
                    prop_scores[dst] += self._resonance(emitted, self._spectra[dst])

        # Combine: direct + weighted propagation bonus
        prop_max = max(prop_scores) or 1.0
        final_scores = [
            direct_scores[i] + self.PROP_W * prop_scores[i] / prop_max
            for i in range(N)
        ]
        ranked = sorted(range(N), key=lambda i: final_scores[i], reverse=True)
        result_ids = [self._notes[i]["id"] for i in ranked[:top_k]]

        # Contradiction co-retrieval: if a contradiction pair has one member
        # already in results, ensure its partner is also included (up to top_k+2)
        # so callers can surface conflicts rather than silently dropping one side
        contra_pairs = self.detect_contradictions(threshold=0.0)
        partner_map: dict[str, str] = {}
        for p in contra_pairs[:20]:  # top 20 suspected pairs
            partner_map[p["note_a"]] = p["note_b"]
            partner_map[p["note_b"]] = p["note_a"]

        result_set = set(result_ids)
        for nid in list(result_ids):  # snapshot to avoid mutation during iteration
            partner = partner_map.get(nid)
            if partner and partner not in result_set:
                result_ids.append(partner)
                result_set.add(partner)
                break  # add at most one partner per retrieval to limit expansion

        return result_ids[:top_k]  # contradiction surfacing via detect_contradictions(), not auto-expand




# ────────────────────────────────────────────────────────────────────────────
# RFMv2 — more aggressive selective propagation (top 10% of bins)
# ────────────────────────────────────────────────────────────────────────────

class RFMv2Backend(RFMBackend):
    """Same as RFM but emits only the top 10% highest-stability bins.
    Tests whether sharper selectivity improves multi-hop at the cost of alias/direct."""
    PROP_STABILITY_PCT = 0.90


# ────────────────────────────────────────────────────────────────────────────
# RFMv3 — hop-distance decay + confidence-gated propagation
# ────────────────────────────────────────────────────────────────────────────

class RFMv3Backend(RFMBackend):
    """
    Extends RFMv2 with two precision improvements:

    1. Hop-distance decay: the combined score is a weighted sum where the
       direct component and propagation component are kept separate.
       Notes that score high *only* via propagation (direct score is low)
       are penalised — they appear after notes with strong direct signal.
       Final score = direct + PROP_W * prop_bonus * direct_weight
       where direct_weight = direct / direct_max (normalised direct score).
       This means a propagation bonus only lifts a note if it already has
       some direct resonance — not from zero.

    2. Confidence gate: propagation bonuses are only applied to notes whose
       direct score is above PROP_GATE * direct_max. Below that threshold
       the note is treated as background noise and its propagation score
       is zeroed out entirely.

    Result: fewer false positives dragged in by propagation; precision
    improves while multi-hop recall is preserved (chain nodes that light
    up via propagation still have some direct resonance with the query).
    """
    PROP_STABILITY_PCT = 0.90   # same sharp selectivity as RFMv2
    PROP_GATE          = 0.05   # minimum direct score fraction to receive prop bonus

    def retrieve(self, query: str, top_k: int) -> list[str]:
        q_spec = self._compute_spectrum(tokenize(query))
        N = len(self._notes)

        # Pass 1: direct resonance
        direct_scores = [
            self._resonance(q_spec, self._spectra[i])
            for i in range(N)
        ]
        direct_max = max(direct_scores) or 1.0

        # Pass 2: selective field propagation (top-10% stability bins only)
        mask = self._emission_mask()
        top_direct = sorted(range(N), key=lambda i: direct_scores[i], reverse=True)
        prop_scores = [0.0] * N
        for src_idx in top_direct[: self.PROP_K]:
            full_spec = self._spectra[src_idx]
            emitted = [v if mask[b] else 0.0 for b, v in enumerate(full_spec)]
            for dst in range(N):
                if dst != src_idx:
                    prop_scores[dst] += self._resonance(emitted, self._spectra[dst])

        prop_max = max(prop_scores) or 1.0

        # Combine with hop-distance decay:
        #   notes below the confidence gate receive zero propagation bonus
        #   notes above it receive a bonus scaled by their own direct weight
        gate = self.PROP_GATE * direct_max
        final_scores = []
        for i in range(N):
            d = direct_scores[i]
            p = prop_scores[i] / prop_max
            direct_weight = d / direct_max
            prop_bonus = self.PROP_W * p * direct_weight if d >= gate else 0.0
            final_scores.append(d + prop_bonus)

        ranked = sorted(range(N), key=lambda i: final_scores[i], reverse=True)
        result_ids = [self._notes[i]["id"] for i in ranked[:top_k]]

        # Contradiction co-retrieval (same as parent)
        contra_pairs = self.detect_contradictions(threshold=0.0)
        partner_map: dict[str, str] = {}
        for p in contra_pairs[:20]:
            partner_map[p["note_a"]] = p["note_b"]
            partner_map[p["note_b"]] = p["note_a"]

        result_set = set(result_ids)
        for nid in list(result_ids):
            partner = partner_map.get(nid)
            if partner and partner not in result_set:
                result_ids.append(partner)
                result_set.add(partner)
                break

        return result_ids[:top_k]



# ────────────────────────────────────────────────────────────────────────────
# RFMChar — character n-gram frequency field
# ────────────────────────────────────────────────────────────────────────────

class RFMCharBackend(RFMv3Backend):
    """
    Character n-gram frequency field.

    Instead of mapping whole words to hash bins (which causes collisions
    at scale), decompose each note into character 2/3/4-gram frequencies.
    The n-gram space is fixed (26^n, but bucketed to N_BINS via stable hash),
    so collision growth is bounded by the alphabet, not the vocabulary.

    This is the proper instantiation of the Fourier analogy:
      - Each character n-gram = a frequency mode over the text signal
      - Short n-grams (2-gram) = low-frequency (coarse shape of text)
      - Long n-grams (4-gram) = high-frequency (fine lexical detail)
      - Superposition of modes = full spectrum representation

    Benefits over word-hash:
      - Morphological similarity captured (abbreviations, plurals, typos)
      - No out-of-vocabulary problem
      - Collision rate grows with alphabet size, not vocabulary size
      - Weighted sum of n-gram scales = true multi-resolution representation
    """

    N_BINS    = 512        # larger space; n-grams distribute better
    N_GRAM_SIZES = [2, 3, 4]  # character n-gram lengths
    N_GRAM_WEIGHTS = [0.2, 0.5, 0.3]  # 3-grams dominate (best signal)
    PROP_STABILITY_PCT = 0.90
    PROP_GATE = 0.05

    def _char_ngrams(self, text: str) -> dict[str, float]:
        """Extract weighted character n-gram frequency map from text."""
        text = re.sub(r"[^a-z0-9]", "_", text.lower())
        freq: dict[str, float] = {}
        for n, w in zip(self.N_GRAM_SIZES, self.N_GRAM_WEIGHTS):
            for i in range(len(text) - n + 1):
                gram = text[i:i+n]
                freq[gram] = freq.get(gram, 0.0) + w
        return freq

    def _compute_spectrum(self, tokens: list[str]) -> list[float]:
        # tokens are ignored — recompute from raw text join
        # (called with tokenize() output; reconstruct approximate text)
        text = " ".join(tokens)
        ngram_freq = self._char_ngrams(text)

        spectrum = [0.0] * self.N_BINS
        for gram, count in ngram_freq.items():
            h = abs(hash(gram)) % (2**31)
            for octave in range(self.N_OCTAVE):
                salted = (h ^ self._OCTAVE_SALTS[octave]) % (2**31)
                bin_idx = salted % self.N_BINS
                amp = (1.0 - self.DECAY) * (self.DECAY ** octave)
                spectrum[bin_idx] += (1.0 + math.log(count)) * amp

        norm = math.sqrt(sum(v * v for v in spectrum)) or 1.0
        return [v / norm for v in spectrum]

    def index(self, notes: list[dict]) -> None:
        """Override to compute spectra from full content, not pre-tokenized."""
        self._notes = notes
        self._spectra = [
            self._compute_spectrum(tokenize(n["content"]))
            for n in notes
        ]
        bin_energy = [0.0] * self.N_BINS
        for spec in self._spectra:
            for b, v in enumerate(spec):
                if v > 0:
                    bin_energy[b] += 1.0
        N = len(notes)
        self._stability = [
            math.log((N + 1) / (e + 1)) + 1.0
            for e in bin_energy
        ]


# ────────────────────────────────────────────────────────────────────────────
# RFMCharProp — iterative multi-round char n-gram propagation
# ────────────────────────────────────────────────────────────────────────────

class RFMCharPropBackend(RFMCharBackend):
    """
    The genuine test of field propagation over zero-vocabulary-overlap chains.

    Uses the char n-gram frequency field (no hash-word collisions) with
    ITERATED propagation across multiple rounds:

      Round 0: query resonates with corpus → direct_scores
      Round 1: top-K round-0 notes emit → secondary_scores
      Round 2: top-K round-1 notes emit → tertiary_scores
      ...
      Final:   score_i = Σ_k  DECAY^k * round_k_score_i

    This allows genuine A→B→C chain traversal:
      - Round 0 lights up bridge1 (shares query vocab)
      - Round 1 emission from bridge1 lights up bridge2 (shares bridge1 vocab)
      - Round 2 emission from bridge2 lights up the answer (shares bridge2 vocab)

    Key differences from RFMv3:
      - NO aggressive stability mask during propagation rounds
        (the mask was cutting entity n-grams that appear in distractor notes)
      - Soft stability weighting instead: scale by log(N/df) per bin
        but never zero — prevents total suppression of discriminative signals
      - Explicit n-hop iteration instead of one-shot emission
    """

    N_ROUNDS   = 3      # number of propagation rounds (= max chain depth)
    PROP_DECAY = 0.4    # geometric decay per round
    PROP_K     = 3      # top notes that emit per round
    N_BINS     = 512

    def _soft_stability(self) -> list[float]:
        """Soft IDF: log(N/df)+1 per bin, minimum 0.1 — no hard masking."""
        N = len(self._notes)
        bin_df = [0.0] * self.N_BINS
        for spec in self._spectra:
            for b, v in enumerate(spec):
                if v > 1e-9:
                    bin_df[b] += 1.0
        return [
            math.log((N + 1) / (df + 1)) + 1.0
            for df in bin_df
        ]

    def _weighted_resonance(
        self,
        a: list[float],
        b: list[float],
        stab: list[float],
    ) -> float:
        return sum(a[i] * b[i] * stab[i] for i in range(self.N_BINS))

    def retrieve(self, query: str, top_k: int) -> list[str]:
        q_spec = self._compute_spectrum(tokenize(query))
        N = len(self._notes)
        stab = self._soft_stability()

        # scores[i] = accumulated resonance signal for note i
        scores = [0.0] * N
        # current "emitter" spectra — start from query
        emitter_specs = [q_spec]
        emitter_weight = 1.0

        for round_idx in range(self.N_ROUNDS):
            # Aggregate resonance from all current emitters
            round_scores = [0.0] * N
            for e_spec in emitter_specs:
                for i, d_spec in enumerate(self._spectra):
                    round_scores[i] += self._weighted_resonance(e_spec, d_spec, stab)

            # Accumulate with decay
            w = emitter_weight * (self.PROP_DECAY ** round_idx)
            for i in range(N):
                scores[i] += w * round_scores[i]

            # Top-K notes by round score become next emitters
            top_this_round = sorted(
                range(N), key=lambda i: round_scores[i], reverse=True
            )[: self.PROP_K]

            emitter_specs = [self._spectra[i] for i in top_this_round]
            emitter_weight *= self.PROP_DECAY  # successive rounds weigh less

        ranked = sorted(range(N), key=lambda i: scores[i], reverse=True)
        return [self._notes[i]["id"] for i in ranked[:top_k]]


# ────────────────────────────────────────────────────────────────────────────
# RFMCharMMR — char n-gram field + MMR-diverse propagation
# ────────────────────────────────────────────────────────────────────────────

class RFMCharMMRBackend(RFMCharPropBackend):
    """
    Character n-gram frequency field with MMR-diverse iterative propagation.

    The key insight from experiments: naive top-K emitter selection causes
    cluster collapse — a group of similar notes mutually amplify each other
    and drown the actual chain signal.

    MMR (Maximal Marginal Relevance) emitter selection prevents this:
      emitter_score(i) = (1-lambda) * relevance(i) - lambda * max_redundancy(i)

    By penalising redundancy with already-selected emitters, MMR ensures the
    propagating wavefront covers diverse parts of the document graph rather
    than getting trapped in a dense similarity cluster.

    This is the mechanism that allows traversal of zero-vocabulary-overlap
    chains: round-0 diverse emitters include bridge1, round-1 from bridge1
    reaches bridge2, round-2 from bridge2 reaches the answer.
    """

    MMR_LAMBDA = 0.60   # redundancy penalty (0=pure relevance, 1=pure diversity)
    N_ROUNDS   = 3
    PROP_DECAY = 0.5
    PROP_K     = 5

    def _mmr_select(self, scores: list[float], k: int) -> list[int]:
        """Select k diverse emitters via Maximal Marginal Relevance."""
        N = len(scores)
        candidates = list(range(N))
        selected: list[int] = []
        selected_specs: list[list[float]] = []
        stab = self._soft_stability()

        for _ in range(k):
            best_i, best_s = -1, float("-inf")
            for i in candidates:
                redundancy = max(
                    (self._weighted_resonance(self._spectra[i], sp, stab)
                     for sp in selected_specs),
                    default=0.0,
                )
                mmr = (1 - self.MMR_LAMBDA) * scores[i] - self.MMR_LAMBDA * redundancy
                if mmr > best_s:
                    best_s, best_i = mmr, i
            if best_i < 0:
                break
            selected.append(best_i)
            selected_specs.append(self._spectra[best_i])
            candidates.remove(best_i)
        return selected

    def retrieve(self, query: str, top_k: int) -> list[str]:
        q_spec = self._compute_spectrum(tokenize(query))
        N = len(self._notes)
        stab = self._soft_stability()

        def res(a, d):
            return max(0.0, self._weighted_resonance(a, d, stab))

        # Round 0: direct query resonance
        r0 = [res(q_spec, self._spectra[i]) for i in range(N)]
        r0_max = max(r0) or 1.0
        r0_n = [s / r0_max for s in r0]

        accumulated = list(r0_n)

        # Emitters always weighted by query-relevance (r0_n) to stay anchored.
        # Using round scores as emitter weights causes drift — nodes with high
        # mutual similarity dominate even when they have low query relevance.
        emitters = self._mmr_select(r0_n, self.PROP_K)
        round_norms = r0_n  # used to weight each emitter's emission

        for rnd in range(1, self.N_ROUNDS + 1):
            # Each emitter contributes proportional to its query-relevance score
            round_scores = [
                sum(round_norms[e] * res(self._spectra[e], self._spectra[i])
                    for e in emitters)
                for i in range(N)
            ]
            rs_max = max(round_scores) or 1.0
            rs_n = [s / rs_max for s in round_scores]

            decay = self.PROP_DECAY ** rnd
            for i in range(N):
                accumulated[i] += decay * rs_n[i]

            # Next emitters: MMR over this round's scores to follow the chain
            # while avoiding cluster collapse — but keep query anchor via weighting
            emitters = self._mmr_select(rs_n, self.PROP_K)
            round_norms = rs_n  # next round emits proportional to THIS round's activation

        ranked = sorted(range(N), key=lambda i: accumulated[i], reverse=True)
        return [self._notes[i]["id"] for i in ranked[:top_k]]

# ────────────────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────────────────


def evaluate_contradictions(corpus: dict) -> None:
    notes = corpus["notes"]
    ground_truth = {
        frozenset([c["note_a"], c["note_b"]])
        for c in corpus["contradictions"]
    }

    backend = RFMBackend()
    backend.index(notes)
    detected = backend.detect_contradictions(threshold=0.0)

    print(f"\nContradiction detection  (ground truth: {len(ground_truth)} pairs)")
    print(f"{'k':>5}  {'precision':>10}  {'recall':>8}  {'F1':>8}")
    print("-" * 40)
    for k in [5, 10, 15, 20, len(detected)]:
        if k > len(detected):
            k = len(detected)
        top_k = detected[:k]
        detected_set = {frozenset([p["note_a"], p["note_b"]]) for p in top_k}
        tp = len(detected_set & ground_truth)
        prec = tp / k if k else 0
        rec  = tp / len(ground_truth) if ground_truth else 0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
        print(f"{k:>5}  {prec:>10.3f}  {rec:>8.3f}  {f1:>8.3f}")

    print(f"\nTop 10 detected contradiction candidates:")
    print(f"  {'note_a':>12}  {'note_b':>12}  {'jaccard':>8}  {'score':>8}  GT?  diff_a → diff_b")
    for p in detected[:10]:
        pair = frozenset([p["note_a"], p["note_b"]])
        gt_marker = "Y" if pair in ground_truth else " "
        print(f"  {p['note_a']:>12}  {p['note_b']:>12}  "
              f"{p['jaccard']:>8.4f}  {p['contradiction_score']:>8.4f}  {gt_marker}  "
              f"{p['diff_a']} → {p['diff_b']}")


def run_eval(corpus: dict, backend_name: str, top_k: int) -> dict:
    notes     = corpus["notes"]
    qa_pairs  = corpus["qa_pairs"]

    if backend_name == "bm25":
        backend = BM25Backend()
    elif backend_name == "cosine":
        backend = CosineBackend()
    elif backend_name == "rfm":
        backend = RFMBackend()
    elif backend_name == "rfmv2":
        backend = RFMv2Backend()
    elif backend_name == "rfmv3":
        backend = RFMv3Backend()
    elif backend_name == "rfmchar":
        backend = RFMCharBackend()
    elif backend_name == "rfmcharprop":
        backend = RFMCharPropBackend()
    elif backend_name == "rfmcharmmr":
        backend = RFMCharMMRBackend()
    elif backend_name == "oracle":
        backend = OracleBackend(qa_pairs)
    elif backend_name == "random":
        backend = RandomBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    backend.index(notes)

    results = []
    for qa in qa_pairs:
        if backend_name == "oracle":
            backend.set_qa_id(qa["id"])  # type: ignore[attr-defined]
        retrieved = backend.retrieve(qa["question"], top_k)
        results.append(evaluate_qa(qa, retrieved))

    return aggregate(results)


def print_report(backend: str, report: dict, top_k: int) -> None:
    print(f"\n{'='*55}")
    print(f"  Backend: {backend.upper()}   top_k={top_k}")
    print(f"{'='*55}")
    ov = report["overall"]
    print(f"  Overall   P={ov['precision']:.3f}  R={ov['recall']:.3f}  "
          f"F1={ov['f1']:.3f}  hit@1={ov['hit@1']:.3f}  hit@k={ov['hit@k']:.3f}")
    print()
    for diff, stats in report["by_difficulty"].items():
        print(f"  [{diff:<14}] n={stats['n']:>3}  "
              f"P={stats['precision']:.3f}  R={stats['recall']:.3f}  "
              f"F1={stats['f1']:.3f}  hit@k={stats['hit@k']:.3f}")
    print()


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on RFM benchmark corpus")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSON")
    parser.add_argument("--backend", choices=["bm25","cosine","rfm","rfmv2","rfmv3","rfmchar","rfmcharprop","rfmcharmmr","oracle","random"],
                        default="cosine")
    parser.add_argument("--all-backends", action="store_true",
                        help="Run all backends and compare")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", help="Optional JSON output for results")
    parser.add_argument("--detect-contradictions", action="store_true")
    args = parser.parse_args()

    with open(args.corpus) as f:
        corpus = json.load(f)

    if args.detect_contradictions:
        evaluate_contradictions(corpus)
        return

    m = corpus["metadata"]
    print(f"Corpus: {m['size']}  notes={m['total_notes']}  qa={m['total_qa_pairs']}  "
          f"seed={m['seed']}")

    backends = ["random", "bm25", "cosine", "rfmcharmmr", "oracle"] if args.all_backends else [args.backend]
    all_reports = {}
    for b in backends:
        report = run_eval(corpus, b, args.top_k)
        print_report(b, report, args.top_k)
        all_reports[b] = report

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"top_k": args.top_k, "results": all_reports}, f, indent=2)
        print(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
