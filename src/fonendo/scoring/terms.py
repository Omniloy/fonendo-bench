"""Gold-term metrics: term recall and term / other word error rates (B-WER / U-WER).

A clip's gold terms (the medical terms spoken in it) are tokenized with the same normalizer as
reference and hypothesis, then located in the token sequences. Attribution rules:

* **Term recall is all-or-nothing.** A gold-term occurrence in the reference counts as
  recognized only if *every* one of its tokens is aligned as ``equal``. A clinician gives no
  partial credit for half of "amoxicilina/ácido clavulánico".
* **Term words** (B-WER): reference tokens covered by a gold-term occurrence, excluding
  function words: "fractura de fémur" contributes 2 term words, not 3, so the most frequent
  word of the language does not dilute the rate.
* **Insertions** follow Le et al. (Interspeech 2021): an inserted word counts towards the term
  word error rate iff it belongs to the term vocabulary (the union of the tokens of the clip's
  gold terms, minus function words), and towards the other word error rate otherwise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from fonendo.scoring.alignment import Alignment, align
from fonendo.text import tokenize

#: Function words: never term words, even inside a multi-word term ("fractura DE fémur",
#: "prótesis DE LA cadera").
FUNCTION_WORDS: frozenset[str] = frozenset(
    """
    a al ante bajo con contra de del desde durante en entre hacia hasta para por según sin
    so sobre tras el la los las un una unos unas lo y e o u ni que se su sus le les me te
    no es está están tipo grado grados
    """.split()
)


@dataclass(frozen=True)
class Span:
    """One occurrence of a term in a token sequence: tokens ``[start, end)``."""

    start: int
    end: int
    term: str

    @property
    def indices(self) -> range:
        return range(self.start, self.end)


class TermMatcher:
    """A set of terms, normalized like reference and hypothesis, with multi-word matching."""

    def __init__(self, terms: Iterable[str], function_words: frozenset[str] = FUNCTION_WORDS):
        self.function_words = function_words
        self._by_first: dict[str, list[tuple[str, ...]]] = {}
        self._terms: set[tuple[str, ...]] = set()
        for term in terms:
            toks = tuple(tokenize(term))
            if not toks or toks in self._terms:
                continue
            self._terms.add(toks)
            self._by_first.setdefault(toks[0], []).append(toks)
        for candidates in self._by_first.values():
            candidates.sort(key=len, reverse=True)
        #: tokens of all terms, minus function words (insertion attribution)
        self.vocabulary: frozenset[str] = frozenset(
            tok for toks in self._terms for tok in toks if tok not in function_words
        )

    def __len__(self) -> int:
        return len(self._terms)

    def find(self, tokens: Sequence[str]) -> list[Span]:
        """Term occurrences, left to right, longest match first, without overlaps."""
        spans: list[Span] = []
        i, n = 0, len(tokens)
        while i < n:
            for toks in self._by_first.get(tokens[i], ()):
                end = i + len(toks)
                if end <= n and tuple(tokens[i:end]) == toks:
                    spans.append(Span(i, end, " ".join(toks)))
                    i = end
                    break
            else:
                i += 1
        return spans

    def term_word_indices(self, tokens: Sequence[str]) -> set[int]:
        """Indices of the tokens that are term words (covered by a term, not function words)."""
        return {
            i
            for span in self.find(tokens)
            for i in span.indices
            if tokens[i] not in self.function_words
        }


@dataclass(frozen=True)
class TermRecall:
    n_ref: int  # gold-term occurrences found in the normalized reference
    hits: int  # of those, recognized with every token aligned as equal


def term_recall(
    ref: Sequence[str], hyp: Sequence[str], matcher: TermMatcher, alignment: Alignment | None = None
) -> TermRecall:
    """All-or-nothing recall of the term occurrences of the reference."""
    al = alignment or align(ref, hyp)
    kind = al.kind_by_ref()
    spans = matcher.find(ref)
    hits = sum(1 for s in spans if all(kind.get(i) == "equal" for i in s.indices))
    return TermRecall(n_ref=len(spans), hits=hits)


@dataclass(frozen=True)
class TermWordErrors:
    n_term_words: int
    n_other_words: int
    term_errors: int  # S + D on term words + insertions of term-vocabulary words
    other_errors: int  # S + D on other words + every other insertion


def term_word_errors(
    ref: Sequence[str], hyp: Sequence[str], matcher: TermMatcher, alignment: Alignment | None = None
) -> TermWordErrors:
    """B-WER / U-WER counts with the insertion attribution of Le et al. (2021)."""
    al = alignment or align(ref, hyp)
    term_idx = matcher.term_word_indices(ref)
    term_err = other_err = 0
    for op in al.ops:
        if op.kind == "equal":
            continue
        if op.kind == "ins":
            in_term = op.hyp in matcher.vocabulary
        else:  # sub / del
            in_term = op.i in term_idx
        if in_term:
            term_err += 1
        else:
            other_err += 1
    return TermWordErrors(
        n_term_words=len(term_idx),
        n_other_words=len(ref) - len(term_idx),
        term_errors=term_err,
        other_errors=other_err,
    )


def contains_span(tokens: Sequence[str], needle: Sequence[str]) -> bool:
    """True if ``needle`` occurs as a contiguous token span of ``tokens``."""
    if not needle or len(needle) > len(tokens):
        return False
    return f" {' '.join(needle)} " in f" {' '.join(tokens)} "
