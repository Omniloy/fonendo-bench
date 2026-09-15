"""Runner base class, lazy factories and the resumable run loop.

A runner wraps ONE system in its default configuration: it receives a float32 mono waveform
and returns the raw transcript. It never sees the reference, the gold terms or any other
per-clip information. See CONTRACT.md, "Adding a runner".
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fonendo import SAMPLE_RATE, __version__
from fonendo.data import read_jsonl

KINDS = ("local", "remote")
#: Hard cap on parallel requests to a remote API, whatever the runner asks for.
MAX_REMOTE_CONCURRENCY = 3


class Runner:
    """Base class for every system in the benchmark.

    Subclasses implement :meth:`transcribe` (and optionally :meth:`load`,
    :meth:`transcribe_batch`, :meth:`close`). Constructor arguments describe the system; they
    are written next to the hypotheses so a run can be reproduced.

    Args:
        name: registry key and results folder name, ``[a-z0-9_]+`` (e.g. ``whisper_large_v3``).
        kind: ``"local"`` (weights run on this machine) or ``"remote"`` (a hosted API).
        extra: the pip extra that installs this runner's dependencies (``fonendo[<extra>]``).
        label: display name in reports (default: ``name``).
        model_id: upstream identifier (HF repo id, or API model name).
        revision: pinned model revision (HF commit hash) or API version, if any.
        env_vars: environment variables the runner needs (API keys). Values are read at
            runtime from the environment and are never logged or written anywhere.
        batch_size: clips per :meth:`transcribe_batch` call (local runners only).
        max_concurrency: parallel requests (remote runners only; capped at 3).
    """

    def __init__(
        self,
        name: str,
        kind: str,
        *,
        extra: str,
        label: str | None = None,
        model_id: str = "",
        revision: str | None = None,
        env_vars: Sequence[str] = (),
        batch_size: int = 1,
        max_concurrency: int = 1,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.name = name
        self.kind = kind
        self.extra = extra
        self.label = label or name
        self.model_id = model_id
        self.revision = revision
        self.env_vars = tuple(env_vars)
        self.batch_size = max(1, int(batch_size))
        self.max_concurrency = max(1, min(int(max_concurrency), MAX_REMOTE_CONCURRENCY))
        self._loaded = False

    # -- lifecycle -------------------------------------------------------------------

    def check_env(self) -> None:
        """Fail early, naming (never printing) any missing environment variable."""
        missing = [v for v in self.env_vars if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"{self.name}: set the environment variable(s) {', '.join(missing)}")

    def load(self) -> None:
        """Load weights or open the API client. Called once, before the first clip."""

    def warmup(self) -> None:
        """Run once before timing. Local default: transcribe 1 s of silence."""
        if self.kind == "local":
            self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    def close(self) -> None:
        """Release resources (GPU memory, sockets)."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.check_env()
            self.load()
            self._loaded = True

    # -- inference -------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        """Return the raw transcript of one clip (float32 mono; ``sr`` is always 16000).

        Default configuration only: Spanish forced when the system allows it, greedy or the
        decoding the model card recommends, temperature 0, no prompt or context of any kind.
        Return the text as produced; scoring normalizes it.
        """
        raise NotImplementedError

    def transcribe_batch(self, audios: list[np.ndarray], sr: int) -> list[str]:
        """Transcribe several clips; must equal ``[transcribe(a, sr) for a in audios]``."""
        return [self.transcribe(a, sr) for a in audios]

    def info(self) -> dict[str, Any]:
        """Run metadata written to ``<out>.run.json``. Subclasses may add fields."""
        return {
            "model": self.name,
            "label": self.label,
            "kind": self.kind,
            "extra": self.extra,
            "model_id": self.model_id,
            "revision": self.revision,
        }


# --------------------------------------------------------------------------------------
# registry helpers
# --------------------------------------------------------------------------------------

Factory = Callable[..., Runner]


def lazy(target: str, *, extra: str, **defaults: Any) -> Factory:
    """A registry factory that imports ``"package.module:Class"`` only when called.

    Keeps ``import fonendo.runners`` free of heavy dependencies; a missing extra becomes an
    actionable error. Call-time kwargs override ``defaults``.
    """
    module_name, _, attr = target.partition(":")

    def factory(**kwargs: Any) -> Runner:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"{target}: missing dependencies, run: pip install 'fonendo[{extra}]' ({exc})"
            ) from exc
        return getattr(module, attr)(**{**defaults, **kwargs})

    factory.target = target  # type: ignore[attr-defined]
    factory.extra = extra  # type: ignore[attr-defined]
    return factory


# --------------------------------------------------------------------------------------
# run loop
# --------------------------------------------------------------------------------------


def _done_ids(out_path: Path) -> set[str]:
    """Keep successful lines of an existing output file; drop error lines so they are retried."""
    if not out_path.exists():
        return set()
    good = [r for r in read_jsonl(out_path) if not r.get("error")]
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in good:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {r["clip_id"] for r in good}


def run_subset(
    runner: Runner,
    rows: list[dict[str, Any]],
    out_path: str | Path,
    *,
    subset: str,
    log: Callable[[str], None] = print,
) -> Path:
    """Transcribe ``rows`` (from ``load_subset``) into ``out_path`` (one JSON line per clip).

    Resumable: clip_ids already in ``out_path`` are skipped, lines with ``error`` are retried.
    Each line is ``{"clip_id", "hyp", "secs"}`` (+ ``"error"`` on failure) and is flushed at
    once. ``secs`` is the wall time of the transcription call only (a batch's time is split
    evenly over its clips). Run metadata goes to ``<out_path stem>.run.json``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    todo = [r for r in rows if r["clip_id"] not in done]
    log(f"{runner.name} / {subset}: {len(done)} done, {len(todo)} to go -> {out_path}")
    if not todo:
        return out_path

    runner.ensure_loaded()
    runner.warmup()
    lock = threading.Lock()
    n_err = 0

    with open(out_path, "a", encoding="utf-8") as fh:

        def emit(rec: dict[str, Any]) -> None:
            nonlocal n_err
            n_err += bool(rec.get("error"))
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

        def one(row: dict[str, Any]) -> None:
            t0 = time.perf_counter()
            try:
                hyp = runner.transcribe(row["audio"], SAMPLE_RATE)
                emit(
                    {
                        "clip_id": row["clip_id"],
                        "hyp": hyp,
                        "secs": round(time.perf_counter() - t0, 4),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - recorded and retried on the next run
                emit(
                    {
                        "clip_id": row["clip_id"],
                        "hyp": "",
                        "secs": round(time.perf_counter() - t0, 4),
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                    }
                )

        if runner.kind == "remote" and runner.max_concurrency > 1:
            with ThreadPoolExecutor(runner.max_concurrency) as pool:
                list(pool.map(one, todo))
        elif runner.batch_size > 1:
            for i in range(0, len(todo), runner.batch_size):
                chunk = todo[i : i + runner.batch_size]
                t0 = time.perf_counter()
                try:
                    hyps = list(runner.transcribe_batch([r["audio"] for r in chunk], SAMPLE_RATE))
                    if len(hyps) != len(chunk):
                        raise ValueError(f"batch returned {len(hyps)} texts for {len(chunk)} clips")
                except Exception:  # noqa: BLE001 - fall back to one clip at a time
                    for r in chunk:
                        one(r)
                    continue
                secs = round((time.perf_counter() - t0) / len(chunk), 4)
                for r, h in zip(chunk, hyps, strict=True):
                    emit({"clip_id": r["clip_id"], "hyp": h, "secs": secs})
        else:
            for row in todo:
                one(row)

    meta = {
        **runner.info(),
        "subset": subset,
        "n_clips": len(rows),
        "n_errors_last_run": n_err,
        "fonendo_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.with_suffix(".run.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    runner.close()
    log(f"{runner.name} / {subset}: finished, {n_err} error(s) this run")
    return out_path
