"""Block bootstrap for ratio metrics: confidence intervals and paired differences.

Two rules this module enforces:

1. **Blocks are resampled, not observations.** The same sentence read by several voices is not
   several independent observations; the caller declares the block (a clip, or a sentence).
2. **Compare systems through the CI of the paired difference**, never through two separate
   CIs: overlapping intervals are not a test, and neither are disjoint ones.

Every metric is a ratio of sums (errors / words, hits / terms, ...), so each observation carries
a numerator and a denominator and the ratio is recomputed on the sums of every resample, never
by averaging per-clip ratios.

Procedure (fixed, so results are reproducible bit for bit): observations are summed per block;
blocks are sorted by key; ``numpy.random.default_rng(seed).integers(0, K, size=(n_boot, K))``
draws the resamples; the 95% interval is the 2.5% and 97.5% ``numpy.quantile`` of the resampled
statistic (resamples whose denominator is 0 are skipped).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Below this many blocks a block bootstrap does not produce a meaningful interval (with k
#: blocks there are only C(2k-1, k) distinct resamples; with 2 blocks, three), so no difference
#: is called significant.
MIN_BLOCKS = 5


def _block_sums(blocks: Sequence[str], *columns: Sequence[float]) -> tuple[list[str], np.ndarray]:
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(columns))
    for k, block in enumerate(blocks):
        row = acc[block]
        for c, col in enumerate(columns):
            row[c] += col[k]
    keys = sorted(acc)
    return keys, np.array([acc[k] for k in keys], dtype=float).reshape(len(keys), len(columns))


@dataclass(frozen=True)
class RatioCI:
    value: float | None
    lo: float | None
    hi: float | None
    num: float
    den: float
    n_blocks: int


def ratio_ci(
    num: Sequence[float],
    den: Sequence[float],
    blocks: Sequence[str],
    *,
    scale: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> RatioCI:
    """Point value ``scale * sum(num) / sum(den)`` with a percentile block-bootstrap 95% CI."""
    keys, m = _block_sums(blocks, num, den)
    tot_num, tot_den = float(m[:, 0].sum()), float(m[:, 1].sum())
    if tot_den == 0:
        return RatioCI(None, None, None, tot_num, 0.0, len(keys))
    value = scale * tot_num / tot_den
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    s_num, s_den = m[idx, 0].sum(1), m[idx, 1].sum(1)
    ok = s_den > 0
    vals = scale * s_num[ok] / s_den[ok]
    lo, hi = np.quantile(vals, [0.025, 0.975]) if vals.size else (np.nan, np.nan)
    return RatioCI(float(value), float(lo), float(hi), tot_num, tot_den, len(keys))


@dataclass(frozen=True)
class PairedDiff:
    a: float
    b: float
    delta: float  # a - b
    lo: float
    hi: float
    p_two_sided: float  # achieved significance level of the bootstrap, not a formal test
    n_blocks: int
    n_resamples: int

    @property
    def significant(self) -> bool:
        """The CI of the difference excludes 0 (never with fewer than ``MIN_BLOCKS`` blocks)."""
        return self.n_blocks >= MIN_BLOCKS and not (self.lo <= 0.0 <= self.hi)


def paired_ratio_diff(
    num_a: Sequence[float],
    den_a: Sequence[float],
    num_b: Sequence[float],
    den_b: Sequence[float],
    blocks: Sequence[str],
    *,
    scale: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> PairedDiff | None:
    """Paired difference A - B of a ratio metric, resampling the same blocks for both systems.

    Returns None when the metric is undefined for either system (total denominator 0).
    """
    keys, m = _block_sums(blocks, num_a, den_a, num_b, den_b)
    k = len(keys)
    if m[:, 1].sum() == 0 or m[:, 3].sum() == 0:
        return None
    if k < 2:
        raise ValueError(f"need >= 2 bootstrap blocks, got {k}")

    def stat(n: float, d: float) -> float:
        return scale * n / d if d else float("nan")

    a = stat(m[:, 0].sum(), m[:, 1].sum())
    b = stat(m[:, 2].sum(), m[:, 3].sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    sums = m[idx].sum(axis=1)  # (n_boot, 4)
    diffs = np.array([stat(s[0], s[1]) - stat(s[2], s[3]) for s in sums], dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError("every bootstrap resample gave a non-finite statistic")
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p = min(1.0, 2.0 * min(float(np.mean(diffs <= 0.0)), float(np.mean(diffs >= 0.0))))
    return PairedDiff(
        a=float(a),
        b=float(b),
        delta=float(a - b),
        lo=float(lo),
        hi=float(hi),
        p_two_sided=p,
        n_blocks=k,
        n_resamples=int(diffs.size),
    )


#: One part of a macro average: (num_a, den_a, num_b, den_b, blocks).
Part = tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float], Sequence[str]]


def paired_macro_diff(
    parts: Sequence[Part], *, scale: float = 1.0, n_boot: int = 2000, seed: int = 0
) -> PairedDiff:
    """Paired difference A - B of the unweighted mean of a ratio metric over several subsets.

    Each subset is resampled on its own blocks (a stratified bootstrap); one generator seeded
    with ``seed`` draws the resamples of the subsets in the order given. ``n_blocks`` is the
    total number of blocks.
    """
    rng = np.random.default_rng(seed)
    a_vals, b_vals, diffs, n_blocks = [], [], [], 0
    for num_a, den_a, num_b, den_b, blocks in parts:
        keys, m = _block_sums(blocks, num_a, den_a, num_b, den_b)
        if m[:, 1].sum() == 0 or m[:, 3].sum() == 0:
            raise ValueError("the metric is undefined on one of the subsets")
        k = len(keys)
        n_blocks += k
        a_vals.append(scale * m[:, 0].sum() / m[:, 1].sum())
        b_vals.append(scale * m[:, 2].sum() / m[:, 3].sum())
        idx = rng.integers(0, k, size=(n_boot, k))
        s = m[idx].sum(axis=1)  # (n_boot, 4)
        with np.errstate(divide="ignore", invalid="ignore"):
            diffs.append(scale * (s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]))
    d = np.mean(diffs, axis=0)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("every bootstrap resample gave a non-finite statistic")
    lo, hi = np.quantile(d, [0.025, 0.975])
    a, b = float(np.mean(a_vals)), float(np.mean(b_vals))
    p = min(1.0, 2.0 * min(float(np.mean(d <= 0.0)), float(np.mean(d >= 0.0))))
    return PairedDiff(a, b, a - b, float(lo), float(hi), p, n_blocks, int(d.size))


def macro_ratio_ci(
    parts: Sequence[tuple[Sequence[float], Sequence[float], Sequence[str]]],
    *,
    scale: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> RatioCI:
    """Unweighted mean of a ratio metric over several subsets, with a stratified 95% CI.

    ``parts`` holds one ``(num, den, blocks)`` per subset. Each subset is resampled on its own
    blocks; one generator seeded with ``seed`` draws the subsets in the order given (the same
    procedure as :func:`paired_macro_diff`). ``num`` and ``den`` of the result are the totals
    over every subset; ``n_blocks`` is the total number of blocks.
    """
    rng = np.random.default_rng(seed)
    values, draws, n_blocks, tot_num, tot_den = [], [], 0, 0.0, 0.0
    for num, den, blocks in parts:
        keys, m = _block_sums(blocks, num, den)
        if m[:, 1].sum() == 0:
            raise ValueError("the metric is undefined on one of the subsets")
        k = len(keys)
        n_blocks += k
        tot_num += float(m[:, 0].sum())
        tot_den += float(m[:, 1].sum())
        values.append(scale * m[:, 0].sum() / m[:, 1].sum())
        idx = rng.integers(0, k, size=(n_boot, k))
        s = m[idx].sum(axis=1)  # (n_boot, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            draws.append(scale * s[:, 0] / s[:, 1])
    d = np.mean(draws, axis=0)
    d = d[np.isfinite(d)]
    lo, hi = np.quantile(d, [0.025, 0.975]) if d.size else (np.nan, np.nan)
    return RatioCI(float(np.mean(values)), float(lo), float(hi), tot_num, tot_den, n_blocks)
