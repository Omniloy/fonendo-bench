"""Tests of the resumable run loop (``run_subset``) with a fake local runner.

No network, no model downloads.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fonendo import SAMPLE_RATE
from fonendo.data import MalformedFileError, read_jsonl
from fonendo.runners.base import Runner, run_subset


class EchoRunner(Runner):
    """Returns ``"texto <n>"`` for a clip of n samples and remembers what it transcribed."""

    def __init__(self) -> None:
        super().__init__("echo", "local", extra="dev")
        self.seen: list[int] = []

    def warmup(self) -> None:  # keep the call log to real clips
        pass

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        assert sr == SAMPLE_RATE
        self.seen.append(len(audio))
        return f"texto {len(audio)}"


def _rows(n: int = 4) -> list[dict]:
    return [
        {"clip_id": f"c{i}", "audio": np.zeros(100 + i, dtype=np.float32), "text": "", "meta": {}}
        for i in range(n)
    ]


def _line(i: int) -> str:
    return json.dumps({"clip_id": f"c{i}", "hyp": f"texto {100 + i}", "secs": 0.1}) + "\n"


def test_fresh_run_and_rerun_writes_nothing(tmp_path):
    out = tmp_path / "echo" / "clinical_dev.jsonl"
    runner = EchoRunner()
    run_subset(runner, _rows(), out, subset="clinical_dev", log=lambda m: None)
    assert [r["clip_id"] for r in read_jsonl(out)] == ["c0", "c1", "c2", "c3"]
    assert (tmp_path / "echo" / "clinical_dev.run.json").is_file()
    again = EchoRunner()
    run_subset(again, _rows(), out, subset="clinical_dev", log=lambda m: None)
    assert again.seen == [] and len(read_jsonl(out)) == 4


@pytest.mark.parametrize(
    "tail",
    [
        '{"clip_id": "c2", "hyp": "tex',  # cut in the middle of a value
        '{"clip_id": "c2"',  # cut after the id
        '{"clip_id": "c2", "hyp": "á',  # cut in the middle of a UTF-8 character
    ],
)
def test_resume_drops_a_truncated_last_line_and_redoes_its_clip(tmp_path, tail):
    out = tmp_path / "run.jsonl"
    raw = (_line(0) + _line(1)).encode() + tail.encode()
    if "á" in tail:
        raw = raw[:-1]  # leave half of the two-byte character
    out.write_bytes(raw)
    logs: list[str] = []
    runner = EchoRunner()
    run_subset(runner, _rows(), out, subset="clinical_dev", log=logs.append)
    recs = read_jsonl(out)  # every line parses again
    assert [r["clip_id"] for r in recs] == ["c0", "c1", "c2", "c3"]
    assert runner.seen == [102, 103]  # c2 redone, c0 and c1 kept
    warning = [m for m in logs if m.startswith("warning:")]
    assert len(warning) == 1 and "line 3" in warning[0] and "transcribed again" in warning[0]


def test_resume_retries_error_lines(tmp_path):
    out = tmp_path / "run.jsonl"
    err = json.dumps({"clip_id": "c1", "hyp": "", "secs": 0.0, "error": "Timeout: x"}) + "\n"
    out.write_text(_line(0) + err)
    runner = EchoRunner()
    run_subset(runner, _rows(2), out, subset="clinical_dev", log=lambda m: None)
    assert runner.seen == [101]
    assert [r.get("error") for r in read_jsonl(out)] == [None, None]


def test_resume_refuses_a_broken_line_before_the_end(tmp_path):
    out = tmp_path / "run.jsonl"
    before = _line(0) + '{"clip_id": "c1", "hy\n' + _line(2)
    out.write_text(before)
    with pytest.raises(MalformedFileError, match="line 2"):
        run_subset(EchoRunner(), _rows(), out, subset="clinical_dev", log=lambda m: None)
    assert out.read_text() == before  # left untouched
