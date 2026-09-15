"""Offline checks of the packaged public-subset manifests and of the fetch audio conversion."""

from __future__ import annotations

import io
import json
import re

import numpy as np
import pytest
import soundfile as sf

from fonendo.data import SUBSETS, load_subset
from fonendo.fetch import SOURCES, pcm16_sha256, read_manifest, write_wav_16k

PUBLIC = [name for name, info in SUBSETS.items() if info.kind == "public"]


def test_every_public_subset_has_a_source():
    assert sorted(SOURCES) == sorted(PUBLIC)
    for src in SOURCES.values():
        assert re.fullmatch(r"[0-9a-f]{40}", src.revision), "revisions are pinned commit hashes"


@pytest.mark.parametrize("subset", PUBLIC)
def test_manifest(subset):
    rows = read_manifest(subset)  # checks count, source and unique source ids
    assert len(rows) == SUBSETS[subset].n_clips
    ids = [r["clip_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    for r in rows:
        assert set(r) == {"clip_id", "text", "source", "num_samples", "sha256", "meta"}
        assert r["text"] and r["text"] == r["text"].strip()
        assert re.fullmatch(r"[0-9a-f]{64}", r["sha256"])
        assert 16_000 <= r["num_samples"] <= 30 * 16_000  # 1 to 30 s
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", r["clip_id"]), "clip ids are file-name safe"


def _encode(audio: np.ndarray, sr: int, fmt: str = "WAV", subtype: str = "PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


def test_pcm16_16k_input_is_copied_exactly(tmp_path):
    rng = np.random.default_rng(0)
    pcm = rng.integers(-32768, 32767, size=16_000, dtype=np.int16)
    for fmt in ("WAV", "FLAC"):
        got = write_wav_16k(_encode(pcm, 16_000, fmt), tmp_path / f"x_{fmt}.wav")
        assert np.array_equal(got, pcm)
        assert pcm16_sha256(got) == pcm16_sha256(pcm)
    info = sf.info(str(tmp_path / "x_WAV.wav"))
    assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")


def test_stereo_48k_is_downmixed_and_resampled(tmp_path):
    t = np.arange(48_000) / 48_000
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    got = write_wav_16k(_encode(np.stack([tone, tone], axis=1), 48_000), tmp_path / "s.wav")
    assert len(got) == 16_000
    assert 0.45 < np.abs(got).max() / 32768 < 0.55


def test_load_public_subset_from_fetched_layout(tmp_path, monkeypatch):
    subset = "voxpopuli_es"
    n = SUBSETS[subset].n_clips
    d = tmp_path / subset
    rows = []
    for i in range(n):
        rel = f"audio/c{i}.wav"
        (d / "audio").mkdir(parents=True, exist_ok=True)
        sf.write(str(d / rel), np.zeros(1600, dtype=np.int16), 16_000, subtype="PCM_16")
        rows.append({"clip_id": f"c{i}", "audio_path": rel, "text": "hola", "meta": {"k": i}})
    (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = load_subset(subset, tmp_path, limit=3)
    assert [r["clip_id"] for r in got] == ["c0", "c1", "c2"]
    assert got[0]["audio"].dtype == np.float32 and got[0]["audio"].shape == (1600,)
    assert got[0]["terms"] == [] and got[2]["meta"] == {"k": 2}
