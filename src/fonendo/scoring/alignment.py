"""Word alignment of reference and hypothesis (Levenshtein with backtrace).

Costs: substitution 1, deletion 1, insertion 1 (classic WER). ``difflib`` is not used: its
opcodes come from a longest-common-subsequence match and do not minimize edit distance.

Deterministic tie-breaking. The backtrace walks the matrix from the end and tries, in order,
**equal > insertion > deletion > substitution**. Insertions and deletions are therefore pushed
towards the end of an ambiguous stretch and a substitution is paired with the leftmost word:
"no refiere alergias" vs "no refiere alergia a la amoxicilina" gives sub(alergias->alergia) +
ins(a, la, amoxicilina), not sub(alergias->amoxicilina) + ins(alergia, a, la), which has the
same cost but hides the inserted word inside a substitution. With tied costs several optimal
alignments exist and the per-category counts (not the total) depend on the one chosen, so the
rule is part of the metric definition.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

OpKind = Literal["equal", "sub", "del", "ins"]


@dataclass(frozen=True)
class Op:
    """One edit operation. ``i`` indexes the reference (None for insertions), ``j`` the
    hypothesis (None for deletions)."""

    kind: OpKind
    i: int | None
    j: int | None
    ref: str | None
    hyp: str | None


@dataclass(frozen=True)
class Alignment:
    ops: tuple[Op, ...]
    n_ref: int
    n_hyp: int

    def count(self, kind: OpKind) -> int:
        return sum(1 for o in self.ops if o.kind == kind)

    @property
    def equal(self) -> int:
        return self.count("equal")

    @property
    def substitutions(self) -> int:
        return self.count("sub")

    @property
    def deletions(self) -> int:
        return self.count("del")

    @property
    def insertions(self) -> int:
        return self.count("ins")

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def kind_by_ref(self) -> dict[int, OpKind]:
        """Reference index -> kind of the operation that consumes it."""
        return {o.i: o.kind for o in self.ops if o.i is not None}


def align(ref: Iterable[str], hyp: Iterable[str]) -> Alignment:
    """Minimum edit distance alignment of two token lists."""
    r = list(ref)
    h = list(hyp)
    n, m = len(r), len(h)

    # d[i][j] = distance between r[:i] and h[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = r[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            if ri == h[j - 1]:
                row[j] = prev[j - 1]
            else:
                row[j] = 1 + min(prev[j - 1], prev[j], row[j - 1])

    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and r[i - 1] == h[j - 1] and d[i][j] == d[i - 1][j - 1]:
            ops.append(Op("equal", i - 1, j - 1, r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ops.append(Op("ins", None, j - 1, None, h[j - 1]))
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(Op("del", i - 1, None, r[i - 1], None))
            i -= 1
        else:
            ops.append(Op("sub", i - 1, j - 1, r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1

    ops.reverse()
    return Alignment(tuple(ops), n, m)
