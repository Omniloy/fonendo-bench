"""Alignment, gold-term metrics and degenerate-output detection on synthetic token lists."""

from fonendo.scoring.alignment import align
from fonendo.scoring.degenerate import degeneration, repetition_rate_5, tandem_repeats
from fonendo.scoring.terms import (
    FUNCTION_WORDS,
    TermMatcher,
    contains_span,
    term_recall,
    term_word_errors,
)
from fonendo.text import tokenize


def kinds(al):
    return [(o.kind, o.ref, o.hyp) for o in al.ops]


# --------------------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------------------


def test_align_counts():
    al = align("a b c d".split(), "a x c d e".split())
    assert (al.substitutions, al.deletions, al.insertions, al.equal) == (1, 0, 1, 3)
    assert al.errors == 2


def test_align_tie_break_pairs_substitution_leftmost():
    # equal > insertion > deletion > substitution in the backtrace: the substitution is paired
    # with the leftmost word and the insertions go to the end of the ambiguous stretch
    al = align("no tiene perros".split(), "no tiene perro a la vecina".split())
    assert kinds(al) == [
        ("equal", "no", "no"),
        ("equal", "tiene", "tiene"),
        ("sub", "perros", "perro"),
        ("ins", None, "a"),
        ("ins", None, "la"),
        ("ins", None, "vecina"),
    ]


def test_align_empty_sides():
    assert align([], []).ops == ()
    assert align(["a", "b"], []).deletions == 2
    assert align([], ["a"]).insertions == 1


# --------------------------------------------------------------------------------------
# gold terms
# --------------------------------------------------------------------------------------


def test_matcher_longest_match_and_vocabulary():
    m = TermMatcher(["dolor", "dolor de cabeza", "Dolor de cabeza"])
    assert len(m) == 2  # duplicates after normalization are merged
    spans = m.find(tokenize("tiene dolor de cabeza y dolor"))
    assert [(s.start, s.end, s.term) for s in spans] == [
        (1, 4, "dolor de cabeza"),
        (5, 6, "dolor"),
    ]
    assert m.vocabulary == {"dolor", "cabeza"}  # "de" is a function word
    assert "de" in FUNCTION_WORDS


def test_matcher_normalizes_terms_like_text():
    m = TermMatcher(["vitamina B12"])
    assert m.find(tokenize("falta de vitamina be doce"))  # letter names collapse to b12


def test_term_recall_is_all_or_nothing():
    m = TermMatcher(["dolor de cabeza", "fiebre"])
    ref = tokenize("refiere dolor de cabeza y fiebre")
    full = term_recall(ref, tokenize("refiere dolor de cabeza y fiebre"), m)
    partial = term_recall(ref, tokenize("refiere dolor de caveza y fiebre"), m)
    assert (full.n_ref, full.hits) == (2, 2)
    assert (partial.n_ref, partial.hits) == (2, 1)  # one wrong token loses the whole term


def test_term_word_errors_and_insertion_attribution():
    m = TermMatcher(["dolor de cabeza"])
    ref = tokenize("refiere dolor de cabeza")
    # term words: dolor, cabeza (function word "de" excluded); other words: refiere, de
    r = term_word_errors(ref, tokenize("refiere dolor de cabeza"), m)
    assert (r.n_term_words, r.n_other_words, r.term_errors, r.other_errors) == (2, 2, 0, 0)
    # substitution on a term word -> term error
    r = term_word_errors(ref, tokenize("refiere calor de cabeza"), m)
    assert (r.term_errors, r.other_errors) == (1, 0)
    # substitution on the function word inside the term -> other error
    r = term_word_errors(ref, tokenize("refiere dolor en cabeza"), m)
    assert (r.term_errors, r.other_errors) == (0, 1)
    # insertions: a term-vocabulary word counts as a term error, any other word as other
    r = term_word_errors(ref, tokenize("refiere dolor de cabeza cabeza"), m)
    assert (r.term_errors, r.other_errors) == (1, 0)
    r = term_word_errors(ref, tokenize("refiere dolor de cabeza hoy"), m)
    assert (r.term_errors, r.other_errors) == (0, 1)


def test_contains_span():
    assert contains_span(["a", "bb", "c"], ["bb", "c"])
    assert not contains_span(["a", "bbc"], ["bb"])
    assert not contains_span(["a"], [])


# --------------------------------------------------------------------------------------
# degenerate outputs
# --------------------------------------------------------------------------------------


def test_empty_output():
    flags, _ = degeneration(["hola", "mundo"], [])
    assert flags == ["empty"]
    assert degeneration([], [])[0] == []


def test_loop_tandem_repeat():
    ref = "el perro come".split()
    assert degeneration(ref, "el perro come come come come".split())[0] == ["loop"]  # 1-gram x4
    assert degeneration(ref, "el perro come come come".split())[0] == []  # 1-gram x3 is fine
    hyp = "el perro come el perro come el perro come".split()  # 3-gram x3
    flags, detail = degeneration(ref, hyp)
    assert flags == ["loop"] and detail["loop"]["tandem"]


def test_repetition_in_reference_is_not_a_loop():
    ref = "no no no no quiero".split()
    assert degeneration(ref, ref)[0] == []


def test_runaway_length():
    ref = ["uno"]
    hyp = [f"w{i}" for i in range(13)]  # 13 > 2 * 1 + 10
    flags, detail = degeneration(ref, hyp)
    assert flags == ["loop"] and detail["loop"]["runaway"] is True


def test_repetition_helpers():
    assert tandem_repeats("a a a a b".split()) == {("a",)}
    assert repetition_rate_5("a b c d e".split()) == 0.0
    assert repetition_rate_5("a b".split()) == 0.0
