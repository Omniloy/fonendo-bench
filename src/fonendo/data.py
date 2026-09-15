"""Subsets and loading.

``load_subset(name, data_dir)`` returns one dict per clip, in the subset's canonical order::

    {
        "clip_id": str,          # stable id, identical across releases
        "audio":   np.ndarray,   # float32, mono, 16 kHz (None when with_audio=False)
        "text":    str,          # reference transcript (not normalized)
        "terms":   list[str],    # gold medical terms spoken in the clip (clinical only, else [])
        "meta":    dict,         # everything else; clinical rows carry meta["text_id"]
    }

Clinical subsets (``clinical_test``, ``clinical_dev``) come from the gated Hugging Face dataset
``Omniloy/fonendo-bench`` (request access, then ``HF_TOKEN`` or ``huggingface-cli login``), or
from a local clone of that dataset repo (``hf_dir`` / ``FONENDO_HF_DIR``). Public subsets
(``fleurs_es``, ``voxpopuli_es``, ``mediaspeech_health``) are rebuilt from their original
sources by ``fonendo fetch`` into ``<data_dir>/<subset>/`` (see CONTRACT.md).
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fonendo import HF_DATASET, SAMPLE_RATE


@dataclass(frozen=True)
class SubsetInfo:
    name: str
    kind: str  # "clinical" (gated HF dataset) | "public" (rebuilt by `fonendo fetch`)
    n_clips: int
    role: str  # "test" (reported) | "dev" (sanity checks and development only, never reported)
    source: str
    description: str


SUBSETS: dict[str, SubsetInfo] = {
    s.name: s
    for s in (
        SubsetInfo(
            "clinical_test",
            "clinical",
            300,
            "test",
            HF_DATASET,
            "Synthetic Spanish clinical dictation with gold medical terms.",
        ),
        SubsetInfo(
            "clinical_dev",
            "clinical",
            60,
            "dev",
            HF_DATASET,
            "Held-out clinical sentences, text-disjoint from clinical_test.",
        ),
        SubsetInfo(
            "fleurs_es",
            "public",
            300,
            "test",
            "google/fleurs (es_419, test)",
            "Read Wikipedia sentences, human speakers.",
        ),
        SubsetInfo(
            "voxpopuli_es",
            "public",
            200,
            "test",
            "facebook/voxpopuli (es, test)",
            "European Parliament speeches, human speakers.",
        ),
        SubsetInfo(
            "mediaspeech_health",
            "public",
            200,
            "test",
            "MediaSpeech (es)",
            "Broadcast media speech, health-related segments.",
        ),
    )
}

#: Default location of the public subsets built by `fonendo fetch`.
DEFAULT_DATA_DIR = Path(os.environ.get("FONENDO_DATA_DIR", "data"))

HF_DATASET_URL = f"https://huggingface.co/datasets/{HF_DATASET}"


class DataUnavailableError(RuntimeError):
    """A subset cannot be loaded: no access to the gated dataset, or a bad local copy."""


class MalformedFileError(ValueError):
    """A JSONL file (hypotheses, manifest) has a line that is not a valid record."""


# --------------------------------------------------------------------------------------
# JSONL helpers (shared by runners, scoring and the CLI)
# --------------------------------------------------------------------------------------


def parse_jsonl_line(line: bytes | str) -> dict[str, Any]:
    """Parse one JSONL line into a dict; raise ``ValueError`` if it is not a JSON object."""
    if isinstance(line, bytes):
        line = line.decode("utf-8")  # UnicodeDecodeError is a ValueError
    try:
        rec = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at column {exc.colno}: {exc.msg}") from None
    if not isinstance(rec, dict):
        raise ValueError(f"expected a JSON object, got {type(rec).__name__}")
    return rec


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the records of a JSONL file, skipping blank lines.

    Raises :class:`MalformedFileError` naming the file and line when a line is not a JSON
    object (or the file is not UTF-8).
    """
    with open(path, "rb") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield parse_jsonl_line(line)
            except ValueError as exc:
                raise MalformedFileError(f"{path}, line {lineno}: {exc}") from None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines (see :func:`iter_jsonl`)."""
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------------------


def load_audio_16k(src: str | Path | bytes) -> np.ndarray:
    """Decode a file path or encoded bytes to float32 mono 16 kHz (soxr HQ if resampling)."""
    import soundfile as sf

    data, sr = sf.read(
        io.BytesIO(src) if isinstance(src, bytes) else str(src), dtype="float32", always_2d=True
    )
    audio = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != SAMPLE_RATE:
        import soxr

        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ")
    return np.ascontiguousarray(audio, dtype=np.float32)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def subset_info(name: str) -> SubsetInfo:
    try:
        return SUBSETS[name]
    except KeyError:
        raise KeyError(f"unknown subset {name!r}; choose from {sorted(SUBSETS)}") from None


def load_subset(
    name: str,
    data_dir: str | Path | None = None,
    *,
    hf_dir: str | Path | None = None,
    token: str | None = None,
    with_audio: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load a subset as a list of clip dicts (see the module docstring for the row format).

    ``data_dir`` holds the public subsets built by ``fonendo fetch`` (default ``./data`` or
    ``FONENDO_DATA_DIR``). ``hf_dir`` (or ``FONENDO_HF_DIR``) points at a local clone of the HF
    dataset repo; otherwise clinical subsets are downloaded from the Hub with ``token`` (default:
    ``HF_TOKEN`` or the cached login). ``with_audio=False`` skips decoding (``audio`` is None),
    which is all scoring needs.
    """
    info = subset_info(name)
    if info.kind == "clinical":
        rows = _load_clinical(name, hf_dir or os.environ.get("FONENDO_HF_DIR"), token, with_audio)
    else:
        rows = _load_public(name, Path(data_dir) if data_dir else DEFAULT_DATA_DIR, with_audio)
    if len(rows) != info.n_clips:
        raise ValueError(f"{name}: expected {info.n_clips} clips, found {len(rows)}")
    return rows[:limit] if limit else rows


#: Columns of the HF dataset that map to top-level row keys; every other column goes to meta.
_CLINICAL_COLUMNS = ("clip_id", "audio", "text", "terms")


def _load_clinical(
    name: str, hf_dir: str | Path | None, token: str | None, with_audio: bool
) -> list[dict[str, Any]]:
    import datasets

    source = str(hf_dir) if hf_dir else HF_DATASET
    # token=None lets huggingface_hub resolve HF_TOKEN or the cached login itself
    kwargs: dict[str, Any] = {"token": token} if token and not hf_dir else {}
    try:
        dsd = datasets.load_dataset(source, name, **kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised with what to do about it
        first_line = (str(exc).strip().splitlines() or [""])[0]
        cause = f"{type(exc).__name__}: {first_line}" if first_line else type(exc).__name__
        if hf_dir:
            hint = (
                f"check that {hf_dir} is a copy of the {HF_DATASET} dataset repository "
                f"(README.md and data/{name}/*.parquet)"
            )
        else:
            hint = (
                f"the clinical subsets are gated: request access on {HF_DATASET_URL} and email "
                "info@omniloy.com, then log in with an account that has access (`hf auth "
                "login`) or set HF_TOKEN; or point --hf-dir (FONENDO_HF_DIR) at a local copy "
                "of the dataset"
            )
        raise DataUnavailableError(f"cannot load {name} from {source} ({cause}); {hint}") from exc
    if len(dsd) != 1:
        raise ValueError(f"{source}/{name}: expected a single split, found {list(dsd)}")
    ds = next(iter(dsd.values()))
    # decode=False: decode with soundfile ourselves (no torchcodec/ffmpeg dependency)
    ds = ds.cast_column("audio", datasets.Audio(decode=False))

    rows = []
    for rec in ds:
        audio = None
        if with_audio:
            a = rec["audio"]
            audio = load_audio_16k(a["bytes"] if a.get("bytes") else a["path"])
        rows.append(
            {
                "clip_id": rec["clip_id"],
                "audio": audio,
                "text": rec["text"],
                "terms": list(rec.get("terms") or []),
                "meta": {k: v for k, v in rec.items() if k not in _CLINICAL_COLUMNS},
            }
        )
    return rows


def _load_public(name: str, data_dir: Path, with_audio: bool) -> list[dict[str, Any]]:
    subset_dir = data_dir / name
    manifest = subset_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} not found; run `fonendo fetch {name}` first")
    rows = []
    for rec in iter_jsonl(manifest):
        rows.append(
            {
                "clip_id": rec["clip_id"],
                "audio": load_audio_16k(subset_dir / rec["audio_path"]) if with_audio else None,
                "text": rec["text"],
                "terms": [],
                "meta": dict(rec.get("meta") or {}),
            }
        )
    return rows
