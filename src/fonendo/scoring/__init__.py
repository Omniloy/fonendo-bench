"""Scoring: metrics, 95% bootstrap confidence intervals and paired comparisons.

All metrics are computed on normalized text (``fonendo.text.tokenize``): spelled-out Spanish
letter names collapsed to the initialism ("eme ge" -> "mg", "pe ce erre" -> "pcr"), lowercase,
Unicode NFC, punctuation removed, accents KEPT, numbers and units rewritten in a canonical
written form, identically for reference, hypothesis and gold terms. Word alignment is
Levenshtein (S, D, I) with a fixed tie-breaking rule (``fonendo.scoring.alignment``).

Metrics (corpus-level: ratio of sums over clips, not mean of per-clip ratios):

wer                     (S + D + I) / N_ref.
term_recall             clinical only: share of the gold-term occurrences of the normalized
                        references recognized in full (every token aligned as equal;
                        all-or-nothing per occurrence). Gold terms that do not occur verbatim in
                        the normalized reference cannot be matched; they are left out of the
                        denominator and counted in ``gold_terms_not_in_ref``.
term_word_error_rate    clinical only, B-WER of Le et al. (Interspeech 2021): S + D on reference
                        words that belong to the clip's gold terms (function words excluded),
                        plus insertions of words of the clip's gold-term vocabulary, divided by
                        the number of such reference words.
other_word_error_rate   clinical only, U-WER: the same on every other word.
insertions_per_1k       inserted words per 1,000 reference words (a hallucination proxy).
degenerate_rate         share of clips whose output is empty (for a non-empty reference), loops
                        or runs away (``fonendo.scoring.degenerate``).

Confidence intervals: percentile bootstrap, 2,000 resamples, seed 0, the ratio of sums
recomputed on each resample (``fonendo.scoring.bootstrap``). ``block="clip"`` (default)
resamples clips; ``block="text"`` resamples sentences (``meta["text_id"]``: all clips of a
sentence move together). A missing clip or a line with ``error`` is scored as an empty
hypothesis and makes the result incomplete.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from fonendo import SAMPLE_RATE
from fonendo.data import read_jsonl
from fonendo.scoring.alignment import Alignment, align
from fonendo.scoring.bootstrap import (
    macro_ratio_ci,
    paired_macro_diff,
    paired_ratio_diff,
    ratio_ci,
)
from fonendo.scoring.degenerate import degeneration
from fonendo.scoring.terms import TermMatcher, contains_span, term_recall, term_word_errors
from fonendo.text import NORMALIZER_VERSION, tokenize

N_BOOT = 2000
SEED = 0
BLOCKS = ("clip", "text")

#: Metrics reported for every subset.
METRICS = ("wer", "insertions_per_1k", "degenerate_rate")
#: Extra metrics for subsets with gold terms (clinical_*); null elsewhere.
TERM_METRICS = ("term_recall", "term_word_error_rate", "other_word_error_rate")
#: Report order.
ALL_METRICS = ("wer", *TERM_METRICS, "insertions_per_1k", "degenerate_rate")

#: Decimals of every reported value.
DECIMALS = 6

Hyps = Mapping[str, str] | Iterable[dict[str, Any]]

__all__ = [
    "ALL_METRICS",
    "BLOCKS",
    "METRICS",
    "N_BOOT",
    "NORMALIZER_VERSION",
    "SEED",
    "TERM_METRICS",
    "ClipScore",
    "as_hyp_map",
    "compare",
    "compare_macro",
    "load_hyps",
    "score",
    "score_clip",
    "score_macro",
]


# --------------------------------------------------------------------------------------
# hypotheses
# --------------------------------------------------------------------------------------


def load_hyps(path: str | Path) -> list[dict[str, Any]]:
    """Read a hypotheses JSONL file (``{"clip_id", "hyp", "secs"[, "error"]}`` per line)."""
    return read_jsonl(path)


def _hyp_records(hyps: Hyps) -> dict[str, dict[str, Any]]:
    """``{clip_id: {"hyp", "secs", "error"}}``; for duplicated clip_ids the last successful
    line wins, and an error line is kept only when the clip has no successful line."""
    if isinstance(hyps, Mapping):
        return {str(k): {"hyp": str(v), "secs": None, "error": None} for k, v in hyps.items()}
    out: dict[str, dict[str, Any]] = {}
    for rec in hyps:
        cid = str(rec["clip_id"])
        if rec.get("error"):
            if cid not in out or out[cid]["error"]:
                out[cid] = {"hyp": "", "secs": None, "error": str(rec["error"])}
            continue
        out[cid] = {"hyp": rec.get("hyp") or "", "secs": rec.get("secs"), "error": None}
    return out


def as_hyp_map(hyps: Hyps) -> tuple[dict[str, str], set[str]]:
    """Normalize ``hyps`` to ``({clip_id: hyp}, {clip_ids with error})``.

    Accepts a ``{clip_id: hyp}`` mapping or the rows of a hypotheses file. For duplicated
    clip_ids the last successful line wins.
    """
    recs = _hyp_records(hyps)
    return (
        {cid: r["hyp"] for cid, r in recs.items() if not r["error"]},
        {cid for cid, r in recs.items() if r["error"]},
    )


# --------------------------------------------------------------------------------------
# per-clip scoring
# --------------------------------------------------------------------------------------


@dataclass
class ClipScore:
    """Counts of one clip; every metric is a ratio of sums of these over clips."""

    clip_id: str
    block: str
    n_ref: int = 0
    n_hyp: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    term_n: int = 0  # gold-term occurrences in the normalized reference
    term_hits: int = 0
    terms_not_in_ref: int = 0
    term_words: int = 0
    term_word_errors: int = 0
    other_words: int = 0
    other_word_errors: int = 0
    degenerate: list[str] = field(default_factory=list)
    degenerate_detail: dict[str, Any] = field(default_factory=dict)
    missing: bool = False
    error: str | None = None
    secs: float | None = None
    duration_s: float | None = None

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


_MATCHERS: dict[tuple[str, ...], TermMatcher] = {}


def _matcher(terms: tuple[str, ...]) -> TermMatcher:
    m = _MATCHERS.get(terms)
    if m is None:
        m = _MATCHERS[terms] = TermMatcher(terms)
    return m


def score_clip(ref_text: str, hyp_text: str, terms: Iterable[str] = (), **fields: Any) -> ClipScore:
    """Score one clip. ``fields`` fill the bookkeeping attributes of ``ClipScore``
    (``clip_id``, ``block``, ``missing``, ``error``, ``secs``, ``duration_s``)."""
    fields.setdefault("clip_id", "")
    fields.setdefault("block", fields["clip_id"])
    c = ClipScore(**fields)
    ref, hyp = tokenize(ref_text), tokenize(hyp_text)
    al: Alignment = align(ref, hyp)
    c.n_ref, c.n_hyp = len(ref), len(hyp)
    c.substitutions, c.deletions, c.insertions = al.substitutions, al.deletions, al.insertions

    gold = tuple(dict.fromkeys(terms))
    if gold:
        m = _matcher(gold)
        rec = term_recall(ref, hyp, m, al)
        c.term_n, c.term_hits = rec.n_ref, rec.hits
        c.terms_not_in_ref = sum(1 for t in gold if not contains_span(ref, tokenize(t)))
        tw = term_word_errors(ref, hyp, m, al)
        c.term_words, c.term_word_errors = tw.n_term_words, tw.term_errors
        c.other_words, c.other_word_errors = tw.n_other_words, tw.other_errors

    c.degenerate, c.degenerate_detail = degeneration(ref, hyp)
    return c


def _block_key(row: dict[str, Any], block: str) -> str:
    if block == "clip":
        return str(row["clip_id"])
    meta = row.get("meta") or {}
    text_id = meta.get("text_id", row.get("text_id"))
    return str(text_id) if text_id is not None else str(row["clip_id"])


def _duration(row: dict[str, Any]) -> float | None:
    audio = row.get("audio")
    if audio is not None:
        return len(audio) / SAMPLE_RATE
    meta = row.get("meta") or {}
    for key in ("duration_s", "duration", "audio_s"):
        if meta.get(key):
            return float(meta[key])
    return None


def _check_block(block: str) -> str:
    if block == "text_id":  # accepted alias
        block = "text"
    if block not in BLOCKS:
        raise ValueError(f"block must be one of {BLOCKS}")
    return block


def score_clips(
    subset_rows: list[dict[str, Any]], hyps: Hyps, *, block: str = "clip"
) -> list[ClipScore]:
    """Per-clip counts for every row of the subset, in subset order."""
    block = _check_block(block)
    recs = _hyp_records(hyps)
    clips = []
    for row in subset_rows:
        cid = str(row["clip_id"])
        rec = recs.get(cid)
        clips.append(
            score_clip(
                row["text"],
                rec["hyp"] if rec else "",
                row.get("terms") or (),
                clip_id=cid,
                block=_block_key(row, block),
                missing=rec is None,
                error=rec["error"] if rec else None,
                secs=rec["secs"] if rec else None,
                duration_s=_duration(row),
            )
        )
    return clips


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------

_Get = Callable[[ClipScore], float]

#: metric -> (numerator, denominator, scale)
_DEFS: dict[str, tuple[_Get, _Get, float]] = {
    "wer": (lambda c: c.errors, lambda c: c.n_ref, 1.0),
    "term_recall": (lambda c: c.term_hits, lambda c: c.term_n, 1.0),
    "term_word_error_rate": (lambda c: c.term_word_errors, lambda c: c.term_words, 1.0),
    "other_word_error_rate": (lambda c: c.other_word_errors, lambda c: c.other_words, 1.0),
    "insertions_per_1k": (lambda c: c.insertions, lambda c: c.n_ref, 1000.0),
    "degenerate_rate": (lambda c: float(bool(c.degenerate)), lambda c: 1.0, 1.0),
}


def _columns(clips: list[ClipScore], metric: str) -> tuple[list[float], list[float], float]:
    num, den, scale = _DEFS[metric]
    return [float(num(c)) for c in clips], [float(den(c)) for c in clips], scale


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), DECIMALS)


def _has_terms(subset_rows: list[dict[str, Any]]) -> bool:
    return any(row.get("terms") for row in subset_rows)


def aggregate(
    clips: list[ClipScore], *, has_terms: bool = True, n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, dict[str, Any] | None]:
    """Point value, 95% CI and denominator of every metric over ``clips``."""
    blocks = [c.block for c in clips]
    out: dict[str, dict[str, Any] | None] = {}
    for metric in ALL_METRICS:
        if metric in TERM_METRICS and not has_terms:
            out[metric] = None
            continue
        num, den, scale = _columns(clips, metric)
        ci = ratio_ci(num, den, blocks, scale=scale, n_boot=n_boot, seed=seed)
        if ci.value is None:
            out[metric] = None
            continue
        out[metric] = {"value": _r(ci.value), "ci95": [_r(ci.lo), _r(ci.hi)], "n": int(ci.den)}
    return out


def score(
    subset_rows: list[dict[str, Any]],
    hyps: Hyps,
    *,
    block: str = "clip",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Score one system on one subset.

    ``subset_rows`` come from ``load_subset(..., with_audio=False)`` (any list of rows with
    ``clip_id``, ``text``, ``terms`` and ``meta`` works, e.g. a filtered subset). ``hyps`` is a
    ``{clip_id: hyp}`` mapping or the rows of a hypotheses file. Returns::

        {
          "n_clips": 300, "n_scored": 300, "n_missing": 0, "n_errors": 0, "complete": true,
          "normalizer": "es1+lc1", "block": "clip", "n_boot": 2000, "seed": 0, "n_blocks": 300,
          "metrics": {
            "wer": {"value": 0.081, "ci95": [0.072, 0.091], "n": 5130},
            "term_recall": {"value": 0.93, "ci95": [0.90, 0.95], "n": 412},
            ...                                   # null when not applicable (no gold terms)
          },
          "counts": {"S": .., "D": .., "I": .., "N": .., "gold_terms_not_in_ref": ..},
          "degenerate_clips": {"<clip_id>": ["loop"]},
          "rtf_median": 0.05                     # median secs / audio duration, when known
        }
    """
    block = _check_block(block)
    clips = score_clips(subset_rows, hyps, block=block)
    n_missing = sum(c.missing for c in clips)
    n_errors = sum(1 for c in clips if c.error)
    rtfs = [
        c.secs / c.duration_s
        for c in clips
        if c.secs is not None and c.duration_s and not c.error and not c.missing
    ]
    return {
        "n_clips": len(clips),
        "n_scored": len(clips) - n_missing - n_errors,
        "n_missing": n_missing,
        "n_errors": n_errors,
        "complete": n_missing == 0 and n_errors == 0,
        "normalizer": NORMALIZER_VERSION,
        "block": block,
        "n_boot": n_boot,
        "seed": seed,
        "n_blocks": len({c.block for c in clips}),
        "metrics": aggregate(clips, has_terms=_has_terms(subset_rows), n_boot=n_boot, seed=seed),
        "counts": {
            "S": sum(c.substitutions for c in clips),
            "D": sum(c.deletions for c in clips),
            "I": sum(c.insertions for c in clips),
            "N": sum(c.n_ref for c in clips),
            "gold_terms_not_in_ref": sum(c.terms_not_in_ref for c in clips),
        },
        "degenerate_clips": {c.clip_id: c.degenerate for c in clips if c.degenerate},
        "rtf_median": round(median(rtfs), 5) if rtfs else None,
    }


def compare_clips(
    clips_a: list[ClipScore],
    clips_b: list[ClipScore],
    *,
    has_terms: bool = True,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, dict[str, Any] | None]:
    """Paired A - B difference of every metric over clips scored on the same rows."""
    if [c.clip_id for c in clips_a] != [c.clip_id for c in clips_b]:
        raise ValueError("clips_a and clips_b must cover the same clips in the same order")
    blocks = [c.block for c in clips_a]
    out: dict[str, dict[str, Any] | None] = {}
    for metric in ALL_METRICS:
        if metric in TERM_METRICS and not has_terms:
            out[metric] = None
            continue
        na, da, scale = _columns(clips_a, metric)
        nb, db, _ = _columns(clips_b, metric)
        d = paired_ratio_diff(na, da, nb, db, blocks, scale=scale, n_boot=n_boot, seed=seed)
        out[metric] = None if d is None else _paired_dict(d)
    return out


def _paired_dict(d: Any) -> dict[str, Any]:
    return {
        "a": _r(d.a),
        "b": _r(d.b),
        "delta": _r(d.delta),
        "ci95": [_r(d.lo), _r(d.hi)],
        "p_two_sided": _r(d.p_two_sided),
        "significant": d.significant,
        "n_blocks": d.n_blocks,
    }


def compare_macro(
    parts: Iterable[tuple[list[dict[str, Any]], Hyps, Hyps]],
    *,
    metric: str = "wer",
    block: str = "clip",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Paired A - B comparison of a metric macro-averaged over several subsets.

    ``parts`` is a sequence of ``(subset_rows, hyps_a, hyps_b)``, one per subset. The macro value
    is the unweighted mean of the per-subset values (e.g. the mean WER over the three
    real-speech subsets). Each subset is resampled on its own blocks; one generator seeded with
    ``seed`` draws the subsets in the order given, so keep a fixed order (the leaderboard uses
    fleurs_es, voxpopuli_es, mediaspeech_health).
    """
    block = _check_block(block)
    if metric not in _DEFS:
        raise ValueError(f"unknown metric {metric!r}; choose from {ALL_METRICS}")
    raw = []
    for rows, hyps_a, hyps_b in parts:
        ca = score_clips(rows, hyps_a, block=block)
        cb = score_clips(rows, hyps_b, block=block)
        na, da, _ = _columns(ca, metric)
        nb, db, _ = _columns(cb, metric)
        raw.append((na, da, nb, db, [c.block for c in ca]))
    d = paired_macro_diff(raw, scale=_DEFS[metric][2], n_boot=n_boot, seed=seed)
    return {
        "metric": metric,
        "n_subsets": len(raw),
        "n_clips": [len(p[4]) for p in raw],
        "block": block,
        "n_boot": n_boot,
        "seed": seed,
        **_paired_dict(d),
    }


def score_macro(
    parts: Iterable[tuple[list[dict[str, Any]], Hyps]],
    *,
    metric: str = "wer",
    block: str = "clip",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """A metric macro-averaged over several subsets, with a stratified bootstrap 95% CI.

    ``parts`` is a sequence of ``(subset_rows, hyps)``, one per subset. The value is the
    unweighted mean of the per-subset values (e.g. the mean WER over the three real-speech
    subsets). Each subset is resampled on its own blocks; one generator seeded with ``seed``
    draws the subsets in the order given, so keep a fixed order (the leaderboard uses
    fleurs_es, voxpopuli_es, mediaspeech_health).
    """
    block = _check_block(block)
    if metric not in _DEFS:
        raise ValueError(f"unknown metric {metric!r}; choose from {ALL_METRICS}")
    raw, n_clips = [], []
    for rows, hyps in parts:
        clips = score_clips(rows, hyps, block=block)
        num, den, _ = _columns(clips, metric)
        raw.append((num, den, [c.block for c in clips]))
        n_clips.append(len(clips))
    ci = macro_ratio_ci(raw, scale=_DEFS[metric][2], n_boot=n_boot, seed=seed)
    return {
        "metric": metric,
        "value": _r(ci.value),
        "ci95": [_r(ci.lo), _r(ci.hi)],
        "n_subsets": len(raw),
        "n_clips": n_clips,
        "block": block,
        "n_boot": n_boot,
        "seed": seed,
    }


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
    ``{"a": x, "b": y, "delta": x - y, "ci95": [lo, hi], "p_two_sided": p, "significant": bool,
    "n_blocks": k}``, where ``p`` is twice the smaller share of resamples with ``delta <= 0`` or
    ``delta >= 0``. A difference is significant when its CI excludes 0 (and there are at least
    5 blocks). Metrics that are undefined for the subset are null.
    """
    block = _check_block(block)
    clips_a = score_clips(subset_rows, hyps_a, block=block)
    clips_b = score_clips(subset_rows, hyps_b, block=block)

    def complete(clips: list[ClipScore]) -> bool:
        return not any(c.missing or c.error for c in clips)

    return {
        "n_clips": len(clips_a),
        "complete_a": complete(clips_a),
        "complete_b": complete(clips_b),
        "normalizer": NORMALIZER_VERSION,
        "block": block,
        "n_boot": n_boot,
        "seed": seed,
        "definition": "delta = a - b; paired block bootstrap, same resampled blocks for a and b",
        "metrics": compare_clips(
            clips_a, clips_b, has_terms=_has_terms(subset_rows), n_boot=n_boot, seed=seed
        ),
    }
