"""``fonendo fetch``: rebuild the public subsets from their original sources.

The clip selection ships with the package as one manifest per subset (``manifests/<subset>.jsonl``):
clip id, reference text, source dataset + pinned revision + original id, and the number of
samples and sha256 of the expected 16 kHz PCM16 audio. ``fetch`` downloads the original files at
the pinned revision, keeps only the selected clips, converts them to 16 kHz mono PCM16 WAV and
checks every clip against the manifest::

    <data_dir>/<subset>/manifest.jsonl          {clip_id, audio_path, text, meta}, subset order
    <data_dir>/<subset>/audio/<clip_id>.wav     16 kHz, mono, PCM16
    <data_dir>/<subset>/fetch_report.json       how many clips match the expected audio

Sources (all public, no token needed):

* ``fleurs_es``: ``google/fleurs``, config ``es_419``, test split. The ``test.tsv`` index is
  downloaded and ``audio/test.tar.gz`` (582 MB) is streamed: only the selected members are
  kept, the archive never touches the disk. Reference: ``raw_transcription``.
* ``voxpopuli_es``: ``facebook/voxpopuli``, ``es`` test parquet (994 MB, downloaded to
  ``<data_dir>/.downloads/`` and deleted afterwards). Reference: ``raw_text``, stripped.
* ``mediaspeech_health``: ``ymoslem/MediaSpeech``, ``es`` (two parquet files, 586 MB; the corpus
  ships a single ``train`` split built as an ASR test set). Reference: ``sentence`` with
  whitespace collapsed.

Audio conversion (identical for every source): decode with soundfile to float32, average the
channels, resample to 16 kHz with soxr (HQ) when needed, clip to [-1, 1], write PCM16. With the
pinned revisions this reproduces the benchmark audio sample for sample; ``fetch_report.json``
lists any clip whose audio differs (for example after a different soxr release resampled it).
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import shutil
import sys
import tarfile
import time
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fonendo import SAMPLE_RATE, __version__
from fonendo.data import SUBSETS, read_jsonl, write_jsonl


@dataclass(frozen=True)
class Source:
    dataset: str
    revision: str
    config: str
    split: str
    files: tuple[str, ...]


SOURCES: dict[str, Source] = {
    "fleurs_es": Source(
        "google/fleurs",
        "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
        "es_419",
        "test",
        ("data/es_419/test.tsv", "data/es_419/audio/test.tar.gz"),
    ),
    "voxpopuli_es": Source(
        "facebook/voxpopuli",
        "42f01879c780b4a2e90ec0b4f616c2ece526e4f1",
        "es",
        "test",
        ("es/test-00000-of-00001.parquet",),
    ),
    "mediaspeech_health": Source(
        "ymoslem/MediaSpeech",
        "4008a968760f2187b0c5b2b2db965f1283433059",
        "es",
        "train",
        ("es/train-00000-of-00002.parquet", "es/train-00001-of-00002.parquet"),
    ),
}

#: A source record: original id -> (encoded audio bytes, reference text derived from the source)
Found = dict[str, tuple[bytes, str]]


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------------------


def manifests_dir() -> Path:
    """Directory holding the packaged clip selections (``manifests/<subset>.jsonl``)."""
    from importlib.resources import files

    packaged = Path(str(files("fonendo"))) / "manifests"
    if packaged.is_dir():  # installed wheel
        return packaged
    return Path(__file__).resolve().parents[2] / "manifests"  # source checkout / editable


def read_manifest(subset: str) -> list[dict[str, Any]]:
    if subset not in SOURCES:
        raise KeyError(f"{subset!r} is not a public subset; choose from {sorted(SOURCES)}")
    path = manifests_dir() / f"{subset}.jsonl"
    rows = read_jsonl(path)
    src = SOURCES[subset]
    n_expected = SUBSETS[subset].n_clips
    if len(rows) != n_expected:
        raise ValueError(f"{path}: expected {n_expected} clips, found {len(rows)}")
    for r in rows:
        s = r["source"]
        if (s["dataset"], s["revision"], s["config"], s["split"]) != (
            src.dataset,
            src.revision,
            src.config,
            src.split,
        ):
            raise ValueError(f"{path}: {r['clip_id']} has an unexpected source {s}")
    if len({r["source"]["id"] for r in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate source ids")
    return rows


# --------------------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------------------


def pcm16_sha256(pcm: np.ndarray) -> str:
    """sha256 of the little-endian int16 samples (independent of the WAV header)."""
    return hashlib.sha256(np.ascontiguousarray(pcm, dtype="<i2").tobytes()).hexdigest()


def write_wav_16k(encoded: bytes, dest: Path) -> np.ndarray:
    """Decode ``encoded`` to 16 kHz mono, write ``dest`` as PCM16 and return its int16 samples."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if sr != SAMPLE_RATE:
        import soxr

        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ")
    audio = np.clip(audio, -1.0, 1.0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".wav.part")
    sf.write(str(tmp), audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    tmp.replace(dest)
    pcm, _ = sf.read(str(dest), dtype="int16")
    return pcm


# --------------------------------------------------------------------------------------
# downloads
# --------------------------------------------------------------------------------------


def _open_stream(src: Source, filename: str):
    """Open a streaming HTTP response for a file of the source repo at the pinned revision.

    Plain HTTPS (the Hub resolves to a CDN URL): no Hub client, no chunk cache on disk.
    """
    from huggingface_hub import get_token, hf_hub_url

    req = urllib.request.Request(
        hf_hub_url(src.dataset, filename, repo_type="dataset", revision=src.revision),
        headers={"User-Agent": f"fonendo/{__version__}"},
    )
    token = get_token()
    if token:  # not needed for these public datasets; never forwarded on redirects
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=120)


def _download(src: Source, filename: str, dest_dir: Path, retries: int = 3) -> Path:
    """Download one file of the source repo at the pinned revision into ``dest_dir``.

    A complete earlier download (``keep_downloads``) is reused.
    """
    dest = dest_dir / src.dataset.replace("/", "__") / src.revision / filename
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with _open_stream(src, filename) as resp, open(part, "wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0)
                done, next_report = 0, 0.25
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total and done / total >= next_report:
                        _log(f"    {filename}: {done / 1e6:.0f}/{total / 1e6:.0f} MB")
                        next_report += 0.25
            if total and part.stat().st_size != total:
                raise OSError(f"incomplete download ({part.stat().st_size} of {total} bytes)")
            part.replace(dest)
            return dest
        except OSError as exc:  # URLError, IncompleteRead, timeouts
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
            print(f"    {filename}: {exc}; retrying ({attempt}/{retries})", file=sys.stderr)
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def _collect_fleurs(src: Source, wanted: set[str], dl_dir: Path) -> Found:
    tsv = _download(src, src.files[0], dl_dir)
    # columns: id, file_name, raw_transcription, transcription, chars, num_samples, gender
    texts: dict[str, str] = {}
    with open(tsv, encoding="utf-8", newline="") as fh:
        for rec in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(rec) > 2 and rec[1] in wanted:
                texts.setdefault(rec[1], rec[2])
    found: Found = {}
    _log(f"  streaming {src.files[1]} (about 582 MB, nothing but the selected clips is kept)")
    with _open_stream(src, src.files[1]) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if member.isfile() and name in wanted and name not in found:
                    fh = tar.extractfile(member)
                    assert fh is not None
                    found[name] = (fh.read(), texts.get(name, ""))
                    if len(found) == len(wanted):
                        break
    return found


def _iter_parquet(path: Path, columns: list[str]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64, columns=columns):
        yield from batch.to_pylist()


def _collect_parquet(
    src: Source,
    wanted: set[str],
    dl_dir: Path,
    keep: bool,
    columns: list[str],
    id_of: Callable[[dict[str, Any]], str],
    text_of: Callable[[dict[str, Any]], str],
) -> Found:
    found: Found = {}
    for filename in src.files:
        _log(f"  downloading {src.dataset}/{filename}")
        path = _download(src, filename, dl_dir)
        for rec in _iter_parquet(path, columns):
            key = id_of(rec)
            if key in wanted and key not in found:
                found[key] = (rec["audio"]["bytes"], text_of(rec))
        if not keep:  # free the disk before the next file
            path.unlink(missing_ok=True)
    return found


def _collect(subset: str, wanted: set[str], dl_dir: Path, keep: bool) -> Found:
    src = SOURCES[subset]
    if subset == "fleurs_es":
        return _collect_fleurs(src, wanted, dl_dir)
    if subset == "voxpopuli_es":
        return _collect_parquet(
            src,
            wanted,
            dl_dir,
            keep,
            ["audio_id", "raw_text", "audio"],
            lambda r: r["audio_id"],
            lambda r: (r["raw_text"] or "").strip(),
        )
    if subset == "mediaspeech_health":
        return _collect_parquet(
            src,
            wanted,
            dl_dir,
            keep,
            ["audio", "sentence"],
            lambda r: Path(r["audio"]["path"]).stem,
            lambda r: " ".join((r["sentence"] or "").split()),
        )
    raise KeyError(subset)


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------


def _is_built(subset_dir: Path, n_clips: int) -> bool:
    manifest = subset_dir / "manifest.jsonl"
    if not manifest.is_file():
        return False
    rows = read_jsonl(manifest)
    return len(rows) == n_clips and all((subset_dir / r["audio_path"]).is_file() for r in rows)


def _library_versions() -> dict[str, str]:
    import soundfile as sf
    import soxr

    return {
        "fonendo": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "soundfile": sf.__version__,
        "libsndfile": sf.__libsndfile_version__,
        "soxr": soxr.__version__,
    }


def fetch_subset(
    subset: str, data_dir: str | Path, *, force: bool = False, keep_downloads: bool = False
) -> dict[str, Any]:
    """Build one public subset under ``data_dir/<subset>/`` and return its fetch report."""
    rows = read_manifest(subset)
    src = SOURCES[subset]
    subset_dir = Path(data_dir) / subset
    report_path = subset_dir / "fetch_report.json"
    if not force and _is_built(subset_dir, len(rows)):
        _log(f"{subset}: already built in {subset_dir} (use force=True to rebuild)")
        return json.loads(report_path.read_text()) if report_path.is_file() else {}

    _log(f"{subset}: {len(rows)} clips from {src.dataset}@{src.revision[:12]}")
    subset_dir.mkdir(parents=True, exist_ok=True)
    (subset_dir / "manifest.jsonl").unlink(missing_ok=True)  # written last, marks completion
    wanted = {r["source"]["id"] for r in rows}
    dl_root = Path(data_dir) / ".downloads"
    try:
        found = _collect(subset, wanted, dl_root, keep_downloads)
    finally:
        if not keep_downloads:
            shutil.rmtree(dl_root / src.dataset.replace("/", "__"), ignore_errors=True)
            with contextlib.suppress(OSError):
                dl_root.rmdir()  # only when empty

    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"{subset}: {len(missing)} clips not found in the source: {missing[:5]}")
    text_mismatch = [r["clip_id"] for r in rows if found[r["source"]["id"]][1] != r["text"]]
    if text_mismatch:
        raise RuntimeError(
            f"{subset}: the source reference differs from the manifest for {len(text_mismatch)} "
            f"clips (first: {text_mismatch[:3]}); the source revision may have been rewritten"
        )

    out_rows, audio_mismatch = [], []
    for r in rows:
        rel = f"audio/{r['clip_id']}.wav"
        pcm = write_wav_16k(found[r["source"]["id"]][0], subset_dir / rel)
        if len(pcm) != r["num_samples"] or pcm16_sha256(pcm) != r["sha256"]:
            audio_mismatch.append(
                {"clip_id": r["clip_id"], "num_samples": len(pcm), "expected": r["num_samples"]}
            )
        out_rows.append(
            {
                "clip_id": r["clip_id"],
                "audio_path": rel,
                "text": r["text"],
                "meta": {
                    **r.get("meta", {}),
                    "source": f"{src.dataset}@{src.revision}",
                    "source_id": r["source"]["id"],
                    "duration_s": round(len(pcm) / SAMPLE_RATE, 3),
                },
            }
        )

    report = {
        "subset": subset,
        "source": {
            "dataset": src.dataset,
            "revision": src.revision,
            "config": src.config,
            "split": src.split,
            "files": list(src.files),
        },
        "n_clips": len(rows),
        "n_audio_identical": len(rows) - len(audio_mismatch),
        "audio_mismatches": audio_mismatch,
        "hours": round(sum(r["meta"]["duration_s"] for r in out_rows) / 3600, 4),
        "versions": _library_versions(),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_jsonl(subset_dir / "manifest.jsonl", out_rows)
    status = "all identical" if not audio_mismatch else f"{len(audio_mismatch)} DIFFER"
    _log(
        f"{subset}: {len(rows)} clips written to {subset_dir}; audio vs expected sha256: "
        f"{report['n_audio_identical']}/{len(rows)} identical ({status})"
    )
    if audio_mismatch:
        print(
            f"WARNING {subset}: {len(audio_mismatch)} clips differ from the benchmark audio; "
            f"see {report_path}. Scores on these clips may not be comparable.",
            file=sys.stderr,
        )
    return report


def fetch(
    subsets: Iterable[str],
    data_dir: str | Path,
    *,
    force: bool = False,
    keep_downloads: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build the given public subsets under ``data_dir`` (see the module docstring).

    ``force`` rebuilds subsets that are already complete. Source files are downloaded to
    ``<data_dir>/.downloads/`` and deleted once their clips are extracted (peak extra disk: the
    largest source file, about 1 GB); ``keep_downloads`` keeps them for later rebuilds. The FLEURS
    archive is always streamed, never stored.
    """
    return {
        s: fetch_subset(s, data_dir, force=force, keep_downloads=keep_downloads) for s in subsets
    }
