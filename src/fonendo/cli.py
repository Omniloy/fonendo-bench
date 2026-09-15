"""The ``fonendo`` command.

fonendo fetch [SUBSET ...]                     build the public subsets under --data-dir
fonendo run --model M --subset S [--limit N]   transcribe -> results/raw/M/S.jsonl
fonendo score --subset S --hyps F [--out J]    metrics with 95% CIs
fonendo compare A B --subset S                 paired comparison of two hypotheses files
fonendo report [--results-dir results]         leaderboard tables from scored results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fonendo import __version__
from fonendo.data import DEFAULT_DATA_DIR, SUBSETS, load_subset

PUBLIC_SUBSETS = [n for n, s in SUBSETS.items() if s.kind == "public"]


def _not_ready(what: str) -> int:
    print(f"fonendo {what}: not implemented yet", file=sys.stderr)
    return 2


def _dump(obj: dict, out: str | None) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
    print(text)


def cmd_fetch(args: argparse.Namespace) -> int:
    unknown = sorted(set(args.subsets) - set(PUBLIC_SUBSETS))
    if unknown:
        print(f"fonendo fetch: not a public subset: {', '.join(unknown)}", file=sys.stderr)
        return 2
    try:
        from fonendo.fetch import fetch  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        if exc.name != "fonendo.fetch":
            raise
        return _not_ready("fetch")
    fetch(args.subsets or PUBLIC_SUBSETS, Path(args.data_dir))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from fonendo.runners import get_runner, run_subset

    overrides = {"device": args.device} if args.device else {}
    runner = get_runner(args.model, **overrides)
    rows = load_subset(args.subset, args.data_dir, hf_dir=args.hf_dir, limit=args.limit)
    out = Path(args.out or Path("results/raw") / args.model / f"{args.subset}.jsonl")
    run_subset(runner, rows, out, subset=args.subset)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from fonendo.scoring import load_hyps, score

    rows = load_subset(args.subset, args.data_dir, hf_dir=args.hf_dir, with_audio=False)
    try:
        result = score(rows, load_hyps(args.hyps), block=args.block, n_boot=args.n_boot)
    except NotImplementedError:
        return _not_ready("score")
    _dump({"subset": args.subset, "hyps": str(args.hyps), **result}, args.out)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from fonendo.scoring import compare, load_hyps

    rows = load_subset(args.subset, args.data_dir, hf_dir=args.hf_dir, with_audio=False)
    try:
        result = compare(
            rows, load_hyps(args.a), load_hyps(args.b), block=args.block, n_boot=args.n_boot
        )
    except NotImplementedError:
        return _not_ready("compare")
    _dump({"subset": args.subset, "a": str(args.a), "b": str(args.b), **result}, args.out)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    try:
        from fonendo.report import build_report  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        if exc.name != "fonendo.report":
            raise
        return _not_ready("report")
    build_report(Path(args.results_dir), Path(args.out) if args.out else None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fonendo", description="Spanish clinical speech-to-text benchmark"
    )
    p.add_argument("--version", action="version", version=f"fonendo {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def data_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--data-dir",
            default=str(DEFAULT_DATA_DIR),
            help="public subsets built by `fonendo fetch` (default: %(default)s)",
        )
        sp.add_argument(
            "--hf-dir",
            default=None,
            help="local clone of the HF dataset (default: download from the Hub)",
        )

    def stat_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--block",
            choices=("clip", "text"),
            default="clip",
            help="bootstrap unit: clips, or sentences via meta.text_id",
        )
        sp.add_argument("--n-boot", type=int, default=2000)
        sp.add_argument("--out", default=None, help="also write the JSON result here")

    sp = sub.add_parser("fetch", help="download and build the public subsets")
    sp.add_argument(
        "subsets",
        nargs="*",
        metavar="SUBSET",
        help=f"any of {', '.join(PUBLIC_SUBSETS)} (default: all)",
    )
    sp.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("run", help="transcribe a subset with one model")
    sp.add_argument("--model", required=True)
    sp.add_argument("--subset", required=True, choices=sorted(SUBSETS))
    sp.add_argument("--out", default=None, help="default: results/raw/<model>/<subset>.jsonl")
    sp.add_argument("--limit", type=int, default=None, help="first N clips only (smoke test)")
    sp.add_argument("--device", default=None, help="local runners: cuda, cuda:1, mps, cpu")
    data_args(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("score", help="score one hypotheses file")
    sp.add_argument("--subset", required=True, choices=sorted(SUBSETS))
    sp.add_argument("--hyps", required=True, type=Path)
    data_args(sp)
    stat_args(sp)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("compare", help="paired comparison of two hypotheses files")
    sp.add_argument("a", type=Path)
    sp.add_argument("b", type=Path)
    sp.add_argument("--subset", required=True, choices=sorted(SUBSETS))
    data_args(sp)
    stat_args(sp)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("report", help="build leaderboard tables from scored results")
    sp.add_argument("--results-dir", default="results")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
