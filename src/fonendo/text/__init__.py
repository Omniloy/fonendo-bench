"""Spanish text normalization for scoring.

``tokenize(text)`` is the only entry point the metrics use; reference, hypothesis and gold terms
all go through it. See ``fonendo.text.spanish`` for the full rule set.
"""

from fonendo.text.letters import LETTER_COLLAPSE_VERSION, collapse_letter_names
from fonendo.text.spanish import NORMALIZER_VERSION, normalize, tokenize

__all__ = [
    "LETTER_COLLAPSE_VERSION",
    "NORMALIZER_VERSION",
    "collapse_letter_names",
    "normalize",
    "tokenize",
]
