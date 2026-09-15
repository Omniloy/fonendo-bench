"""The ``fonendo`` command.

fonendo fetch [SUBSET ...]                     build the public subsets under --data-dir
fonendo models                                 list the registered models
fonendo run --model M --subset S [--limit N]   transcribe -> results/raw/M/S.jsonl
fonendo score --subset S --hyps F [--out J]    metrics with 95% CIs
fonendo compare A B --subset S                 paired comparison of two hypotheses files
fonendo report [--results-dir results/raw]     summary.json + leaderboard.md from your runs

Expected failures (unknown model, missing extra, missing API key, data not fetched or not
accessible) are reported as one line on stderr with exit code 2; ``fonendo --traceback ...``
shows the full traceback instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fonendo import __version__
from fonendo.data import DEFAULT_DATA_DIR, SUBSETS, DataUnavailableError, load_subset

PUBLIC_SUBSETS = [n for n, s in SUBSETS.items() if s.kind == "public"]

#: Errors with an actionable message: printed as one line instead of a traceback.
EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    DataUnavailableError,
    FileNotFoundError,
    ImportError,
    KeyError,
    RuntimeError,
)


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
    from fonendo.fetch import fetch

    fetch(
        args.subsets or PUBLIC_SUBSETS,
        Path(args.data_dir),
        force=args.force,
        keep_downloads=args.keep_downloads,
    )
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from fonendo.runners import EXPERIMENTAL, LOCAL_REGISTRY, REGISTRY

    for name, factory in REGISTRY.items():
        kind = "local " if name in LOCAL_REGISTRY else "remote"
        extra = getattr(factory, "extra", "?")
        install = f"pip install 'fonendo[{extra}]'"
        flag = "  (experimental)" if name in EXPERIMENTAL else ""
        print(f"{name:32s} {kind}  {install:40s}{flag}".rstrip())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from fonendo.runners import get_runner, run_subset

    overrides = {"device": args.device} if args.device else {}
    runner = get_runner(args.model, **overrides)
    runner.check_env()  # a missing API key fails before the subset is loaded
    rows = load_subset(args.subset, args.data_dir, hf_dir=args.hf_dir, limit=args.limit)
    out = Path(args.out or Path("results/raw") / args.model / f"{args.subset}.jsonl")
    run_subset(runner, rows, out, subset=args.subset)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from fonendo.report import score_cell
    from fonendo.scoring import load_hyps

    rows = load_subset(args.subset, args.data_dir, hf_dir=args.hf_dir, with_audio=False)
    result = score_cell(
        args.subset, rows, load_hyps(args.hyps), block=args.block, n_boot=args.n_boot
    )
    _dump({"subset": args.subset, "hyps": str(args.hyps), **result}, args.out)
    if not result["complete"]:
        print(
            f"fonendo score: incomplete ({result['n_missing']} missing, "
            f"{result['n_errors']} error lines; scored as empty hypotheses)",
            file=sys.stderr,
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from fonendo.report import resolve_block
    from fonendo.scoring import compare, load_hyps

    rows = load_subset(args.subset, args.data_dir, hf_dir=args.hf_dir, with_audio=False)
    result = compare(
        rows,
        load_hyps(args.a),
        load_hyps(args.b),
        block=resolve_block(args.subset, args.block),
        n_boot=args.n_boot,
    )
    _dump({"subset": args.subset, "a": str(args.a), "b": str(args.b), **result}, args.out)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from fonendo.report import build_report

    build_report(
        Path(args.results_dir),
        Path(args.out) if args.out else None,
        data_dir=args.data_dir,
        hf_dir=args.hf_dir,
        published=args.published,
        n_boot=args.n_boot,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fonendo", description="Spanish clinical speech-to-text benchmark"
    )
    p.add_argument("--version", action="version", version=f"fonendo {__version__}")
    p.add_argument(
        "--traceback",
        action="store_true",
        help="show the full Python traceback of an error instead of a one-line message",
    )
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
            choices=("auto", "clip", "text"),
            default="auto",
            help="bootstrap unit: clips, or sentences via meta.text_id "
            "(default auto: sentences for clinical subsets, clips otherwise)",
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
    sp.add_argument("--force", action="store_true", help="rebuild subsets already built")
    sp.add_argument(
        "--keep-downloads",
        action="store_true",
        help="keep the downloaded source files in <data-dir>/.downloads",
    )
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("models", help="list the registered models and their pip extras")
    sp.set_defaults(func=cmd_models)

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

    sp = sub.add_parser("report", help="summary.json + leaderboard.md from your runs")
    sp.add_argument(
        "--results-dir",
        default="results/raw",
        help="holds <model>/<subset>.jsonl (default: %(default)s)",
    )
    sp.add_argument("--out", default=None, help="output directory (default: --results-dir)")
    sp.add_argument(
        "--published",
        default=None,
        help="also list the systems of a published summary.json, e.g. results/summary.json",
    )
    sp.add_argument("--n-boot", type=int, default=2000)
    data_args(sp)
    sp.set_defaults(func=cmd_report)
    return p


def _message(exc: BaseException) -> str:
    # KeyError wraps its message in quotes; show the message itself
    if isinstance(exc, KeyError) and len(exc.args) == 1 and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc) or type(exc).__name__


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except EXPECTED_ERRORS as exc:
        if args.traceback:
            raise
        print(f"fonendo {args.command}: error: {_message(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
