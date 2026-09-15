"""Spanish text normalizer used for every metric.

Contract (changing any rule changes ``NORMALIZER_VERSION``, which is written into every score
file; numbers computed with different versions are not comparable):

* spelled-out letter names collapsed to the initialism first ("eme ge" -> "mg", see
  ``fonendo.text.letters``);
* lowercase + Unicode NFC, whitespace collapsed, punctuation removed;
* **accents are kept**: "fémur" and "femur" are different words, and dropping accents would
  hide real spelling errors;
* number words rewritten as digits ("ochocientos setenta y cinco" -> "875") and units that
  follow a number abbreviated ("miligramos" -> "mg", "horas" -> "h").

Separator policy (declared because it changes token counts):

* ``/``, ``-``, ``_`` and the dash family always become a space, so "875/125" -> "875 125" and
  "amoxicilina/ácido clavulánico" -> three tokens, as if dictated.
* The reading connectors {barra, a, y, sobre, partido, guion, guión, coma} are dropped **only
  when they sit between two numeric tokens**, so "875/125" and "ochocientos setenta y cinco
  barra ciento veinticinco" become the same string, and "de 24 a 48 horas" equals
  "24-48 horas". Elsewhere no word is ever dropped.
* A decimal comma or point between digits is kept as a comma ("1.5" -> "1,5"); every other comma
  is removed.
* Digits and letters glued together are split ("10mg" -> "10 mg", "b12" -> "b 12").

Example::

    >>> tokenize("Amoxicilina 875/125 mg cada 8 horas")
    ['amoxicilina', '875', '125', 'mg', 'cada', '8', 'h']
"""

from __future__ import annotations

import re
import unicodedata

from fonendo.text import numbers, units
from fonendo.text.letters import LETTER_COLLAPSE_VERSION, collapse_letter_names

#: Normalizer version: "es<rules>+<letter collapse>". Bump it whenever a rule changes.
NORMALIZER_VERSION = f"es1+{LETTER_COLLAPSE_VERSION}"

#: reading connectors dropped only between two numeric tokens
NUMERIC_CONNECTORS = frozenset({"barra", "a", "y", "sobre", "partido", "guion", "guión", "coma"})

_DECIMAL_MARK = "\x01"
_DECIMAL_RE = re.compile(r"(?<=\d)[.,](?=\d)")
_SEPARATORS_RE = re.compile(r"[/\-‐‑‒–—_]")
_PUNCT_RE = re.compile(r"[^\w\s%º\x01]", re.UNICODE)
_DIGIT_LETTER_RE = re.compile(r"(\d)([^\W\d_])", re.UNICODE)
_LETTER_DIGIT_RE = re.compile(r"([^\W\d_])(\d)", re.UNICODE)
_DIGIT_SYMBOL_RE = re.compile(r"(\d)(%)")
_SPACES_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Lowercase, separators, punctuation and whitespace."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _DECIMAL_RE.sub(_DECIMAL_MARK, text)
    text = _SEPARATORS_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _DIGIT_LETTER_RE.sub(r"\1 \2", text)
    text = _LETTER_DIGIT_RE.sub(r"\1 \2", text)
    text = _DIGIT_SYMBOL_RE.sub(r"\1 \2", text)
    text = text.replace(_DECIMAL_MARK, ",")
    return _SPACES_RE.sub(" ", text).strip()


def _abbreviate_units(tokens: list[str]) -> list[str]:
    """Abbreviate units **only** when the previous output token is numeric."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if out and numbers.is_numeric_token(out[-1]):
            matched = False
            for n in range(units.MAX_UNIT_TOKENS, 0, -1):
                abbrev = units.abbreviation(" ".join(tokens[i : i + n]))
                if abbrev is not None:
                    out.append(abbrev)
                    i += n
                    matched = True
                    break
            if matched:
                continue
            canon = units.ABBREV_ALIASES.get(tokens[i])
            if canon is not None:
                out.append(canon)
                i += 1
                continue
        out.append(tokens[i])
        i += 1
    return out


def _drop_numeric_connectors(tokens: list[str]) -> list[str]:
    """Drop reading connectors that sit between two numeric tokens."""
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if (
            tok in NUMERIC_CONNECTORS
            and out
            and numbers.is_numeric_token(out[-1])
            and i + 1 < len(tokens)
            and numbers.is_numeric_token(tokens[i + 1])
        ):
            continue
        out.append(tok)
    return out


def tokenize(text: str | None) -> list[str]:
    """Normalize ``text`` and return its tokens (the unit of every metric)."""
    tokens = _clean(collapse_letter_names(text or "")).split()
    tokens = numbers.words_to_digits(tokens, units.UNIT_START_WORDS)
    tokens = _abbreviate_units(tokens)
    return _drop_numeric_connectors(tokens)


def normalize(text: str | None) -> str:
    """Normalize ``text`` and return it as one space-separated string."""
    return " ".join(tokenize(text))
