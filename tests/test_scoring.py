"""score(), compare(), compare_macro() and the bootstrap on a small synthetic subset."""

import math

import numpy as np
import pytest

from fonendo.scoring import (
    ALL_METRICS,
    NORMALIZER_VERSION,
    as_hyp_map,
    compare,
    compare_macro,
    resolve_block,
    score,
    score_clip,
)
from fonendo.scoring.bootstrap import MIN_BLOCKS, paired_ratio_diff, ratio_ci


def row(cid, text, terms=(), text_id=None):
    return {
        "clip_id": cid,
        "audio": None,
        "text": text,
        "terms": list(terms),
        "meta": {"text_id": text_id or cid, "duration_s": 2.0},
    }


#: 4 clips, 2 sentences read twice each (text_id s1, s2)
ROWS = [
    row("c1", "El paciente toma ibuprofeno cada ocho horas.", ["ibuprofeno"], "s1"),
    row("c2", "El paciente toma ibuprofeno cada ocho horas.", ["ibuprofeno"], "s1"),
    row("c3", "Presenta dolor de cabeza desde ayer.", ["dolor de cabeza"], "s2"),
    row("c4", "Presenta dolor de cabeza desde ayer.", ["dolor de cabeza"], "s2"),
]
PERFECT = {r["clip_id"]: r["text"] for r in ROWS}
HYPS = [
    {"clip_id": "c1", "hyp": "el paciente toma ibuprofeno cada 8 h", "secs": 0.2},
    {"clip_id": "c2", "hyp": "el paciente toma ibuprofeno", "secs": 0.4},  # 3 deletions
    {"clip_id": "c3", "hyp": "presenta dolor de caveza desde ayer ayer", "secs": 0.6},  # S + I
    {"clip_id": "c4", "hyp": "presenta dolor de cabeza desde ayer", "secs": 0.8},
]


def test_perfect_system():
    res = score(ROWS, PERFECT, n_boot=200)
    m = res["metrics"]
    assert m["wer"] == {"value": 0.0, "ci95": [0.0, 0.0], "n": 26}
    assert m["term_recall"]["value"] == 1.0 and m["term_recall"]["n"] == 4
    assert m["term_word_error_rate"]["value"] == 0.0 and m["term_word_error_rate"]["n"] == 6
    assert m["other_word_error_rate"]["value"] == 0.0
    assert m["insertions_per_1k"]["value"] == 0.0
    assert m["degenerate_rate"]["value"] == 0.0
    assert res["complete"] is True and res["n_scored"] == 4
    assert res["normalizer"] == NORMALIZER_VERSION
    assert res["rtf_median"] is None  # a mapping carries no timings


def test_metric_values():
    res = score(ROWS, HYPS, n_boot=200)
    m, c = res["metrics"], res["counts"]
    # N = 7 + 7 + 6 + 6; errors: c2 3 deletions, c3 1 substitution + 1 insertion
    assert (c["S"], c["D"], c["I"], c["N"]) == (1, 3, 1, 26)
    assert m["wer"]["value"] == round(5 / 26, 6)
    assert m["insertions_per_1k"]["value"] == round(1000 * 1 / 26, 6)
    # term occurrences: 4, one lost (caveza)
    assert m["term_recall"]["value"] == 0.75
    # term words: ibuprofeno x2, dolor x2, cabeza x2 = 6; one substitution on cabeza
    assert m["term_word_error_rate"]["value"] == round(1 / 6, 6)
    # other words 20: 3 deletions + 1 insertion (ayer, not a term word)
    assert m["other_word_error_rate"]["value"] == round(4 / 20, 6)
    assert res["rtf_median"] == 0.25  # median of secs / 2 s
    lo, hi = m["wer"]["ci95"]
    assert lo <= m["wer"]["value"] <= hi
    assert list(m) == list(ALL_METRICS)


def test_public_subset_has_null_term_metrics():
    rows = [row("p1", "hola a todos"), row("p2", "buenas tardes")]
    res = score(rows, {"p1": "hola a todos", "p2": "buenas noches"}, n_boot=100)
    for k in ("term_recall", "term_word_error_rate", "other_word_error_rate"):
        assert res["metrics"][k] is None
    assert res["metrics"]["wer"]["value"] == 0.2


def test_missing_and_error_clips_score_as_empty():
    hyps = HYPS[:2] + [{"clip_id": "c3", "hyp": "", "error": "timeout"}]
    res = score(ROWS, hyps, n_boot=100)
    assert (res["n_missing"], res["n_errors"], res["n_scored"]) == (1, 1, 2)
    assert res["complete"] is False
    assert res["degenerate_clips"] == {"c3": ["empty"], "c4": ["empty"]}
    assert res["counts"]["D"] == 3 + 6 + 6


def test_duplicate_lines_last_success_wins():
    hyps = [
        {"clip_id": "c1", "hyp": "", "error": "boom"},
        {"clip_id": "c1", "hyp": "primera"},
        {"clip_id": "c1", "hyp": "segunda"},
        {"clip_id": "c1", "hyp": "", "error": "later failure"},
        {"clip_id": "c2", "hyp": "", "error": "boom"},
    ]
    ok, errors = as_hyp_map(hyps)
    assert ok == {"c1": "segunda"} and errors == {"c2"}


def test_text_block_groups_renditions():
    res_clip = score(ROWS, HYPS, n_boot=300, block="clip")
    res_text = score(ROWS, HYPS, n_boot=300, block="text")
    assert res_clip["n_blocks"] == 4 and res_text["n_blocks"] == 2
    assert res_text["metrics"]["wer"]["value"] == res_clip["metrics"]["wer"]["value"]
    assert score(ROWS, HYPS, n_boot=300, block="text_id")["block"] == "text"  # alias
    with pytest.raises(ValueError):
        score(ROWS, HYPS, block="speaker")


def test_default_block_is_auto_like_the_cli():
    # rows with meta.text_id (clinical subsets): sentences, the published intervals
    assert resolve_block(ROWS) == "text"
    assert score(ROWS, HYPS, n_boot=300) == score(ROWS, HYPS, n_boot=300, block="text")
    assert compare(ROWS, HYPS, PERFECT, n_boot=300)["block"] == "text"
    # rows without text_id (public subsets): clips
    public = [{**r, "meta": {"duration_s": 2.0}} for r in ROWS]
    assert resolve_block(public) == "clip"
    res = score(public, HYPS, n_boot=300)
    assert res["block"] == "clip" and res["n_blocks"] == 4
    assert res == score(public, HYPS, n_boot=300, block="clip")
    # an explicit block always wins
    assert resolve_block(ROWS, "clip") == "clip" and resolve_block(public, "text") == "text"


def test_default_block_matches_the_cli_rule():
    from fonendo.report import default_block

    public = [{"clip_id": "p", "text": "hola", "terms": [], "meta": {"source_id": "1"}}]
    assert default_block("clinical_test") == resolve_block(ROWS) == "text"
    assert default_block("fleurs_es") == resolve_block(public) == "clip"


def test_bootstrap_is_deterministic():
    a = score(ROWS, HYPS, n_boot=500)
    b = score(ROWS, HYPS, n_boot=500)
    assert a == b
    assert (
        score(ROWS, HYPS, n_boot=500, seed=1)["metrics"]["wer"]["value"]
        == a["metrics"]["wer"]["value"]
    )


def test_score_clip_loop_detection():
    c = score_clip("buenos días", "hola hola hola hola hola")
    assert c.degenerate == ["loop"]
    assert c.insertions + c.substitutions == 5


# --------------------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------------------


def many_rows(n=40):
    return [row(f"r{i}", f"frase número {i} con fiebre alta", ["fiebre"]) for i in range(n)]


def test_compare_identical_systems():
    rows = many_rows()
    hyps = {r["clip_id"]: r["text"] for r in rows}
    res = compare(rows, hyps, hyps, n_boot=300)
    wer = res["metrics"]["wer"]
    assert wer["delta"] == 0.0 and wer["ci95"] == [0.0, 0.0]
    assert wer["p_two_sided"] == 1.0 and wer["significant"] is False


def test_compare_detects_a_real_difference():
    rows = many_rows()
    good = {r["clip_id"]: r["text"] for r in rows}
    bad = {r["clip_id"]: r["text"].replace("fiebre", "liebre") for r in rows}
    res = compare(rows, good, bad, n_boot=500)
    tr = res["metrics"]["term_recall"]
    assert (tr["a"], tr["b"], tr["delta"]) == (1.0, 0.0, 1.0)
    wer = res["metrics"]["wer"]
    assert wer["delta"] < 0 and wer["ci95"][1] < 0 and wer["significant"] is True
    assert wer["p_two_sided"] == 0.0
    assert res["complete_a"] and res["complete_b"]


def test_compare_needs_enough_blocks_to_be_significant():
    d = paired_ratio_diff([0, 0], [1, 1], [1, 1], [1, 1], ["x", "y"], n_boot=100)
    assert d.n_blocks < MIN_BLOCKS and d.significant is False
    assert d.lo == d.hi == -1.0


def test_compare_macro():
    parts = []
    for k in range(3):
        rows = [row(f"m{k}_{i}", f"uno dos tres cuatro {i}") for i in range(10)]
        a = {r["clip_id"]: r["text"] for r in rows}
        b = {r["clip_id"]: r["text"].replace("dos", "doce") for r in rows[: 2 + 3 * k]}
        b = {**a, **b}
        parts.append((rows, a, b))
    res = compare_macro(parts, n_boot=300)
    # system b has 2, 5, 8 substituted clips out of 10 (5 words each) in the three subsets
    expected_b = np.mean([2 / 50, 5 / 50, 8 / 50])
    assert res["a"] == 0.0 and math.isclose(res["b"], expected_b, abs_tol=1e-6)
    assert res["delta"] < 0 and res["n_subsets"] == 3 and res["n_blocks"] == 30


def test_ratio_ci_constant_ratio():
    ci = ratio_ci([1, 2, 3, 4], [10, 20, 30, 40], ["a", "b", "c", "d"], n_boot=200)
    assert math.isclose(ci.value, 0.1) and math.isclose(ci.lo, 0.1) and math.isclose(ci.hi, 0.1)
    empty = ratio_ci([0, 0], [0, 0], ["a", "b"])
    assert empty.value is None
