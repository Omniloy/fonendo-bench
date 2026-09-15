"""Collapse spelled-out Spanish letter names to the initialism ("eme ge" -> "mg").

Units and acronyms are often read letter by letter ("eme ge" for mg, "pe ce erre" for PCR), and
a system without inverse text normalization writes what it hears. The collapse is applied to
reference and hypothesis alike, before tokenization, so "pe ce erre" and "PCR" score the same.

Version ``lc1``: the rules below are part of the normalizer; changing them changes the version.
"""

from __future__ import annotations

import re
from collections import Counter

LETTER_COLLAPSE_VERSION = "lc1"

#: Spanish letter names -> letter
LETTER_NAMES: dict[str, str] = {
    "be": "b",
    "ce": "c",
    "de": "d",
    "efe": "f",
    "ge": "g",
    "gue": "g",
    "hache": "h",
    "ache": "h",
    "jota": "j",
    "ka": "k",
    "ele": "l",
    "eme": "m",
    "ene": "n",
    "pe": "p",
    "cu": "q",
    "erre": "r",
    "ese": "s",
    "te": "t",
    "uve": "v",
    "equis": "x",
    "zeta": "z",
    "ye": "y",
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "u",
}
_VOWEL_NAMES = frozenset("aeiou")

#: digit words; part of a run only right after a letter name ("be doce" = B12, "ele cuatro" = L4)
DIGIT_WORDS: dict[str, str] = {
    "uno": "1",
    "dos": "2",
    "tres": "3",
    "cuatro": "4",
    "cinco": "5",
    "seis": "6",
    "siete": "7",
    "ocho": "8",
    "nueve": "9",
    "diez": "10",
    "doce": "12",
}

#: trimmed from both ends of a run before deciding (they are ordinary words far more often)
_RUN_EDGE_TRIM = frozenset({"de", "a", "e", "o", "y", "u"})
_WORD_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
#: a real initialism is short ("ache be a uno ce" = 5 tokens)
_MAX_RUN = 6


def collapse_letter_names(text: str) -> str:
    """'eme ge' -> 'mg', 'pe ce erre' -> 'pcr', 'ache be a uno ce' -> 'hba1c', 'be doce' -> 'b12'.

    A run is >= 2 consecutive letter-name / digit words separated only by spaces or hyphens
    (punctuation breaks it; a digit word joins only right after a letter name other than
    "de", because "de" + number is the preposition). Leading and trailing de/a/e/o/y/u are
    trimmed off the run; what is left is collapsed iff it still has >= 2 tokens, at least one
    consonant letter name, at most 6 tokens and no word repeated 3 or more times (longer or
    repetitive runs are decoding loops: they are left alone so they still count as errors).
    Everything else is returned byte for byte.
    """
    if not text:
        return text or ""
    words = list(_WORD_RE.finditer(text))
    out: list[str] = []
    pos = 0
    i = 0
    n = len(words)
    while i < n:
        w = words[i].group().lower()
        if w not in LETTER_NAMES:
            i += 1
            continue
        # grow the run
        j = i + 1
        kinds = ["L"]
        while j < n:
            gap = text[words[j - 1].end() : words[j].start()]
            if gap.strip(" \t-") or not gap:
                break
            wj = words[j].group().lower()
            if wj in LETTER_NAMES:
                kinds.append("L")
            elif wj in DIGIT_WORDS and kinds[-1] == "L" and words[j - 1].group().lower() != "de":
                kinds.append("D")
            else:
                break
            j += 1
        a, b = i, j  # [a, b) after trimming
        while a < b and kinds[a - i] == "L" and words[a].group().lower() in _RUN_EDGE_TRIM:
            a += 1
        while b > a and kinds[b - 1 - i] == "L" and words[b - 1].group().lower() in _RUN_EDGE_TRIM:
            b -= 1
        # a digit word left at the head after trimming ("de uno") no longer follows a letter
        while a < b and kinds[a - i] == "D":
            a += 1
        seg = [words[k].group().lower() for k in range(a, b)]
        has_consonant = any(s in LETTER_NAMES and s not in _VOWEL_NAMES for s in seg)
        is_loop = len(seg) > _MAX_RUN or max(Counter(seg).values(), default=0) >= 3
        if len(seg) >= 2 and has_consonant and not is_loop:
            out.append(text[pos : words[a].start()])
            out.append("".join(LETTER_NAMES.get(s) or DIGIT_WORDS[s] for s in seg))
            pos = words[b - 1].end()
        i = j
    out.append(text[pos:])
    return "".join(out)
