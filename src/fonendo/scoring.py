"""Scoring: metrics, 95% bootstrap confidence intervals and paired comparisons.

All metrics are computed on normalized text: lowercase, Unicode NFC, punctuation removed,
accents KEPT, numbers and units rewritten in Spanish words, and runs of spelled-out Spanish
letter names collapsed to the initialism ("eme ge" -> "mg", "pe ce erre" -> "pcr"), identically
for reference and hypothesis. Word alignment is Levenshtein (S, D, I).

Metrics (corpus-level: ratio of sums over clips, not mean of per-clip ratios):

wer                     (S + D + I) / N_ref.
term_recall             clinical only: share of the clip's gold terms (``terms``) found as an
                        exact normalized span in the hypothesis (all-or-nothing per term).
                        Denominator: gold terms that occur as a span in the normalized reference.
term_word_error_rate    clinical only, B-WER of Le et al. (Interspeech 2021): errors on words
                        belonging to the clip's gold terms / number of such reference words
                        (an insertion counts here if the inserted word is a gold-term word).
other_word_error_rate   clinical only, U-WER: the same on every other word.
insertions_per_1k       inserted words per 1,000 reference words (a hallucination proxy).
degenerate_rate         share of clips whose output is empty (for a non-empty reference), loops
                        (a repeated n-gram absent from the reference: >= 4x for n = 1, >= 3x for
                        n = 2..8, or a 5-gram repetition rate above the reference's + 0.05) or
                        runs away (n_hyp > 2 * n_ref + 10).

Confidence intervals: percentile bootstrap, 2,000 resamples, seed 0, the ratio of sums
recomputed on each resample. ``block="clip"`` (default) resamples clips; ``block="text"``
resamples sentences (``meta["text_id"]``: all clips of a sentence move together). A missing clip
or a line with ``error`` is scored as an empty hypothesis and makes the result incomplete.

Status: interface stub; the implementation lands in this module (plus vendored helpers under
``fonendo/``) without changing these signatures or the output format (see CONTRACT.md).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from fonendo.data import read_jsonl

N_BOOT = 2000
SEED = 0
BLOCKS = ("clip", "text")

#: Metrics reported for every subset.
METRICS = ("wer", "insertions_per_1k", "degenerate_rate")
#: Extra metrics for subsets with gold terms (clinical_*); null elsewhere.
TERM_METRICS = ("term_recall", "term_word_error_rate", "other_word_error_rate")

Hyps = Mapping[str, str] | Iterable[dict[str, Any]]


def load_hyps(path: str | Path) -> list[dict[str, Any]]:
    """Read a hypotheses JSONL file (``{"clip_id", "hyp", "secs"[, "error"]}`` per line)."""
    return read_jsonl(path)


def as_hyp_map(hyps: Hyps) -> tuple[dict[str, str], set[str]]:
    """Normalize ``hyps`` to ``({clip_id: hyp}, {clip_ids with error})``.

    Accepts a ``{clip_id: hyp}`` mapping or the rows of a hypotheses file. For duplicated
    clip_ids the last successful line wins.
    """
    if isinstance(hyps, Mapping):
        return {str(k): str(v) for k, v in hyps.items()}, set()
    out: dict[str, str] = {}
    errors: set[str] = set()
    for rec in hyps:
        cid = rec["clip_id"]
        if rec.get("error"):
            if cid not in out:
                errors.add(cid)
            continue
        out[cid] = rec.get("hyp") or ""
        errors.discard(cid)
    return out, errors


def score(
    subset_rows: list[dict[str, Any]],
    hyps: Hyps,
    *,
    block: str = "clip",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Score one system on one subset.

    ``subset_rows`` come from ``load_subset(..., with_audio=False)``. Returns::

        {
          "n_clips": 300, "n_scored": 300, "n_missing": 0, "n_errors": 0, "complete": true,
          "block": "clip", "n_boot": 2000, "seed": 0,
          "metrics": {
            "wer": {"value": 0.081, "ci95": [0.072, 0.091]},
            "term_recall": {"value": 0.93, "ci95": [0.90, 0.95], "n": 412},
            ...                                   # null when not applicable (no gold terms)
          },
          "degenerate_clips": {"<clip_id>": ["loop"]},
          "rtf_median": 0.05                     # median secs / audio duration, when known
        }
    """
    if block not in BLOCKS:
        raise ValueError(f"block must be one of {BLOCKS}")
    raise NotImplementedError("fonendo.scoring.score is not implemented yet")


def compare(
    subset_rows: list[dict[str, Any]],
    hyps_a: Hyps,
    hyps_b: Hyps,
    *,
    block: str = "clip",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Paired comparison of systems A and B on the same clips.

    Both systems are resampled with the SAME bootstrap blocks. For each metric returns
    ``{"a": x, "b": y, "delta": x - y, "ci95": [lo, hi], "p_two_sided": p}``, where ``p`` is
    twice the smaller share of resamples with ``delta <= 0`` or ``delta >= 0``. A difference
    whose CI excludes 0 is reported as significant.
    """
    if block not in BLOCKS:
        raise ValueError(f"block must be one of {BLOCKS}")
    raise NotImplementedError("fonendo.scoring.compare is not implemented yet")
