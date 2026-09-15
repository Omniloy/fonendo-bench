"""Spanish number words -> digits ("ochocientos setenta y cinco" -> "875").

This is the direction the scorer needs: it collapses the two ways a system may write a number
(digits or words) to a single string. The parser is deliberately conservative:

* "un", "una", "uno" alone are articles, not numbers. They become 1 only when they carry a
  magnitude ("un millón") or are followed by a unit word ("un gramo").
* The conjunction "y" is absorbed into a number only between a tens word (30..90) and a unit
  word (1..9): "setenta y cinco" is 75; "ocho horas y treinta minutos" is left alone.
* Ordinals and fractions ("medio") are never touched.
* A decimal part is read after "coma" / "punto" as a run of single digits
  ("uno coma cinco" -> "1,5").
"""

from __future__ import annotations

import re

_UNITS: dict[str, int] = {
    "cero": 0,
    "uno": 1,
    "un": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciséis": 16,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiuno": 21,
    "veintiún": 21,
    "veintiuna": 21,
    "veintidós": 22,
    "veintidos": 22,
    "veintitrés": 23,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiséis": 26,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
}

_TENS: dict[str, int] = {
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}

_HUNDREDS: dict[str, int] = {
    "cien": 100,
    "ciento": 100,
    "doscientos": 200,
    "doscientas": 200,
    "trescientos": 300,
    "trescientas": 300,
    "cuatrocientos": 400,
    "cuatrocientas": 400,
    "quinientos": 500,
    "quinientas": 500,
    "seiscientos": 600,
    "seiscientas": 600,
    "setecientos": 700,
    "setecientas": 700,
    "ochocientos": 800,
    "ochocientas": 800,
    "novecientos": 900,
    "novecientas": 900,
}

_THOUSAND = {"mil"}
_MILLION = {"millón", "millon", "millones"}

#: ambiguous tokens: alone they are articles, not numbers
_ARTICLES = {"un", "una", "uno"}

NUMBER_WORDS: frozenset[str] = frozenset(
    set(_UNITS) | set(_TENS) | set(_HUNDREDS) | _THOUSAND | _MILLION
)

_DIGITS_RE = re.compile(r"^\d+(?:,\d+)?$")


def is_numeric_token(tok: str) -> bool:
    """True if the token is already a number written in digits ("8", "1,5")."""
    return bool(_DIGITS_RE.match(tok))


def _parse_integer(tokens: list[str], i: int) -> tuple[int | None, int]:
    """Read an integer starting at ``tokens[i]``. Returns ``(value, n_tokens_consumed)``."""
    total = 0  # accumulated value already multiplied by thousand / million
    group = 0  # current group (< 1000)
    kind: str | None = None  # last kind read inside the group
    consumed = 0
    seen = False
    j = i

    while j < len(tokens):
        tok = tokens[j]
        if tok in _HUNDREDS:
            if kind is not None:
                break
            group += _HUNDREDS[tok]
            kind = "hundreds"
        elif tok in _TENS:
            if kind in ("tens", "units"):
                break
            group += _TENS[tok]
            kind = "tens"
        elif tok in _UNITS:
            if kind in ("tens", "units"):
                break
            group += _UNITS[tok]
            kind = "units"
        elif tok in _THOUSAND:
            total += (group or 1) * 1000
            group = 0
            kind = None
        elif tok in _MILLION:
            total += (group or 1) * 1_000_000
            group = 0
            kind = None
        elif tok == "y":
            # only valid between a tens word (30..90) and a unit word (1..9)
            if (
                kind == "tens"
                and j + 1 < len(tokens)
                and tokens[j + 1] in _UNITS
                and 1 <= _UNITS[tokens[j + 1]] <= 9
            ):
                group += _UNITS[tokens[j + 1]]
                kind = "units"
                j += 2
                consumed += 2
                seen = True
                continue
            break
        else:
            break
        seen = True
        j += 1
        consumed += 1

    if not seen:
        return None, 0
    return total + group, consumed


def _parse_number(tokens: list[str], i: int) -> tuple[str | None, int]:
    """Read a number with an optional decimal part ("uno coma cinco" -> "1,5")."""
    integer, n = _parse_integer(tokens, i)
    if integer is None:
        return None, 0
    j = i + n
    if j + 1 < len(tokens) and tokens[j] in ("coma", "punto"):
        digits: list[str] = []
        k = j + 1
        while k < len(tokens) and tokens[k] in _UNITS and _UNITS[tokens[k]] <= 9:
            digits.append(str(_UNITS[tokens[k]]))
            k += 1
        if digits:
            return f"{integer},{''.join(digits)}", k - i
    return str(integer), n


def words_to_digits(
    tokens: list[str], unit_words: frozenset[str] | set[str] = frozenset()
) -> list[str]:
    """Replace runs of Spanish number words with digit tokens.

    ``unit_words`` disambiguates "un" / "una" / "uno": alone they become 1 only when the next
    token is a unit word ("un gramo" -> "1 gramo"); otherwise they are kept as articles.
    """
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in NUMBER_WORDS:
            value, n = _parse_number(tokens, i)
            if value is not None and n > 0:
                if n == 1 and tok in _ARTICLES:
                    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
                    if nxt not in unit_words:
                        out.append(tok)
                        i += 1
                        continue
                out.append(value)
                i += n
                continue
        out.append(tok)
        i += 1
    return out
