"""Spanish normalizer: every rule on small synthetic sentences."""

import pytest

from fonendo.text import NORMALIZER_VERSION, collapse_letter_names, normalize, tokenize


def test_version_is_pinned():
    # Changing any normalization rule must bump the version (scores are not comparable across).
    assert NORMALIZER_VERSION == "es1+lc1"


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        # case, NFC, punctuation; accents are kept
        ("¡Hola, Mundo!", ["hola", "mundo"]),
        ("El fémur", ["el", "fémur"]),
        ("él", ["él"]),  # decomposed accent -> NFC
        # number words -> digits
        ("ochocientos setenta y cinco", ["875"]),
        ("dos mil veinticuatro", ["2024"]),
        ("un millón", ["1000000"]),
        ("uno coma cinco", ["1,5"]),
        # "y" joins only tens + unit
        ("ocho horas y treinta minutos", ["8", "h", "y", "30", "min"]),
        # articles stay articles unless a unit follows
        ("una casa", ["una", "casa"]),
        ("un gramo", ["1", "g"]),
        # units are abbreviated only after a number
        ("cada 8 horas", ["cada", "8", "h"]),
        ("pasadas unas horas", ["pasadas", "unas", "horas"]),
        ("diez miligramos", ["10", "mg"]),
        ("120 milímetros de mercurio", ["120", "mmhg"]),
        ("50 %", ["50", "%"]),
        ("50%", ["50", "%"]),
        # pinned quirk: "ciento" is read as a number before units are matched
        ("50 por ciento", ["50", "por", "100"]),
        ("5 mgs", ["5", "mg"]),
        # separators and numeric connectors
        ("875/125", ["875", "125"]),
        ("ochocientos setenta y cinco barra ciento veinticinco", ["875", "125"]),
        ("de 24 a 48 horas", ["de", "24", "48", "h"]),
        ("rojo/verde", ["rojo", "verde"]),
        ("la casa y el perro", ["la", "casa", "y", "el", "perro"]),
        # decimal separators
        ("1.5 litros", ["1,5", "l"]),
        ("tomé 1,5", ["tomé", "1,5"]),
        # glued digits and letters
        ("10mg", ["10", "mg"]),
        ("", []),
        (None, []),
    ],
)
def test_tokenize(text, tokens):
    assert tokenize(text) == tokens


def test_normalize_joins_tokens():
    assert normalize("Tres  Gatos.") == "3 gatos"


@pytest.mark.parametrize(
    ("text", "collapsed"),
    [
        ("dosis de cinco eme ge", "dosis de cinco mg"),
        ("pe ce erre negativa", "pcr negativa"),
        ("ache be a uno ce", "hba1c"),
        ("vitamina be doce", "vitamina b12"),
        ("ele cuatro", "l4"),
        ("te-ce", "tc"),
        # a run needs a consonant letter name and >= 2 tokens after trimming
        ("a e i", "a e i"),
        ("de a", "de a"),
        # "de" + number is the preposition
        ("erre de dos coma cinco", "erre de dos coma cinco"),
        # punctuation breaks a run
        ("eme, ge", "eme, ge"),
        # loops are left alone so they still count as errors
        ("efe efe efe", "efe efe efe"),
        ("be ce de efe ge ache jota", "be ce de efe ge ache jota"),
        # everything else is untouched, byte for byte
        ("Nada que  cambiar.", "Nada que  cambiar."),
        ("", ""),
    ],
)
def test_collapse_letter_names(text, collapsed):
    assert collapse_letter_names(text) == collapsed


def test_letter_names_and_written_form_score_the_same():
    assert tokenize("cinco eme ge") == tokenize("5 mg") == ["5", "mg"]
    assert tokenize("una pe ce erre") == tokenize("una PCR")
