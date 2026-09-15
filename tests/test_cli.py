"""Tests of the CLI's error handling: expected failures are one line on stderr, exit code 2.

No network, no model downloads.
"""

from __future__ import annotations

import json

import pytest

from fonendo import cli, data
from fonendo.report import build_summary, render_leaderboard, system_entry


def _fail_if_called(*args, **kwargs):
    raise AssertionError("the subset must not be loaded")


def test_unknown_model_is_one_line(capsys):
    rc = cli.main(["run", "--model", "no_such_model", "--subset", "clinical_dev"])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith("fonendo run: error: unknown model 'no_such_model'")
    assert "Traceback" not in err and err.count("\n") == 1


def test_missing_api_key_fails_before_loading_the_subset(monkeypatch, capsys):
    monkeypatch.delenv("FONENDO_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("FONENDO_OPENAI_MODEL", raising=False)
    monkeypatch.setattr(cli, "load_subset", _fail_if_called)
    rc = cli.main(["run", "--model", "openai_compatible", "--subset", "clinical_test"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "FONENDO_OPENAI_BASE_URL" in err and "Traceback" not in err


def test_traceback_flag_reraises(monkeypatch):
    with pytest.raises(KeyError):
        cli.main(["--traceback", "run", "--model", "no_such_model", "--subset", "clinical_dev"])


def test_gated_dataset_error_says_what_to_do(monkeypatch, tmp_path, capsys):
    import datasets

    def not_found(*args, **kwargs):
        raise FileNotFoundError("Dataset 'Omniloy/fonendo-bench' doesn't exist on the Hub")

    monkeypatch.delenv("FONENDO_HF_DIR", raising=False)
    monkeypatch.setattr(datasets, "load_dataset", not_found)
    hyps = tmp_path / "h.jsonl"
    hyps.write_text(json.dumps({"clip_id": "x", "hyp": ""}) + "\n")
    rc = cli.main(["score", "--subset", "clinical_test", "--hyps", str(hyps)])
    err = capsys.readouterr().err
    assert rc == 2 and "Traceback" not in err
    assert "gated" in err and "info@omniloy.com" in err and "--hf-dir" in err
    with pytest.raises(data.DataUnavailableError):
        data.load_subset("clinical_test")


def test_missing_public_subset_is_one_line(tmp_path, capsys):
    hyps = tmp_path / "h.jsonl"
    hyps.write_text("")
    rc = cli.main(
        ["score", "--subset", "fleurs_es", "--hyps", str(hyps), "--data-dir", str(tmp_path)]
    )
    err = capsys.readouterr().err
    assert rc == 2 and "fonendo fetch fleurs_es" in err and "Traceback" not in err


def test_models_marks_experimental_runners(capsys):
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out.splitlines()
    flagged = {line.split()[0] for line in out if line.endswith("(experimental)")}
    assert {"voxtral_small_24b", "mlx_whisper_large_v3", "mlx_whisper_large_v3_turbo"} <= flagged
    assert "whisper_large_v3" not in flagged


def test_leaderboard_lists_licenses_and_experimental_runners():
    big = {"label": "Big", "type": "open", "license": "Apache-2.0", "runner": "voxtral_small_24b"}
    entries = [
        system_entry("voxtral_small_24b", big, {}),
        system_entry("other", {"label": "Other", "type": "open", "license": "MIT"}, {}),
    ]
    md = render_leaderboard(build_summary(entries, generated="2026-01-01"))
    assert "| Big | open weights | Apache-2.0 | `voxtral_small_24b` (experimental) |" in md
    assert "| Other | open weights | MIT | – |" in md
    assert "pinned to the top" in md
