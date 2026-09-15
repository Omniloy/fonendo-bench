"""Degenerate outputs: empty transcripts, decoding loops and runaway generation.

A clip is flagged when ANY of:

* ``empty``: the hypothesis has no tokens and the reference has some;
* ``loop``: an n-gram repeated back to back (>= 4x for n = 1, >= 3x for n = 2..8) that does not
  repeat that way in the reference; or a 5-gram repetition rate above 0.05 and above the
  reference's + 0.05; or runaway length (hypothesis tokens > 2 x reference tokens + 10).

Degenerate clips stay in every metric (a loop is a real failure); the flag makes them visible.
"""

from __future__ import annotations

from typing import Any


def tandem_repeats(tokens: list[str]) -> set[tuple[str, ...]]:
    """n-grams repeated back to back: >= 4x for n = 1, >= 3x for n = 2..8."""
    found: set[tuple[str, ...]] = set()
    n_tok = len(tokens)
    for n in range(1, 9):
        need = 4 if n == 1 else 3
        if n * need > n_tok:
            break
        i = 0
        while i + n * need <= n_tok:
            gram = tuple(tokens[i : i + n])
            k = 1
            while tokens[i + k * n : i + (k + 1) * n] == list(gram):
                k += 1
            if k >= need:
                found.add(gram)
                i += k * n
            else:
                i += 1
    return found


def repetition_rate_5(tokens: list[str]) -> float:
    """Share of repeated 5-grams: 1 - distinct / total."""
    grams = [tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def degeneration(ref: list[str], hyp: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Return ``(flags, detail)`` for one clip; ``flags`` is a subset of ["empty", "loop"]."""
    flags: list[str] = []
    detail: dict[str, Any] = {}
    if not hyp and ref:
        flags.append("empty")
    tandem = tandem_repeats(hyp) - tandem_repeats(ref)
    rep_hyp, rep_ref = repetition_rate_5(hyp), repetition_rate_5(ref)
    runaway = len(hyp) > 2 * len(ref) + 10
    if tandem or (rep_hyp > 0.05 and rep_hyp > rep_ref + 0.05) or runaway:
        flags.append("loop")
        detail["loop"] = {
            "tandem": [" ".join(g) for g in sorted(tandem)][:3],
            "rep5": round(rep_hyp, 3),
            "runaway": runaway,
        }
    return flags, detail
