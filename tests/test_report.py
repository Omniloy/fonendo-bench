"""Tests of the leaderboard builder, the macro-average CI and the merged model registry.

Synthetic sentences only; no network, no model downloads.
"""

from __future__ import annotations

import json
import sys

import pytest

from fonendo import cli
from fonendo.report import (
    build_report,
    build_summary,
    default_block,
    render_leaderboard,
    score_cell,
    system_entry,
)
from fonendo.scoring import score, score_macro
from fonendo.scoring.bootstrap import macro_ratio_ci

SENTENCES = [
    "el gato duerme en la alfombra roja",
    "mañana lloverá por la tarde en la costa",
    "la reunión empieza a las diez en punto",
    "compramos pan fresco y queso curado",
    "el tren llega con veinte minutos de retraso",
]


def _clinical_rows() -> list[dict]:
    rows = []
    for i in range(20):
        cond = "clean" if i % 2 == 0 else "degraded"
        rows.append(
            {
                "clip_id": f"c{i:02d}",
                "audio": None,
                "text": f"se pauta ibuprofeno y paracetamol al paciente {SENTENCES[i % 5]}",
                "terms": ["ibuprofeno", "paracetamol"],
                "meta": {"text_id": f"t{i // 2}", "condition": cond, "duration_s": 4.0},
            }
        )
    return rows


def _public_rows(n: int, offset: int = 0) -> list[dict]:
    return [
        {
            "clip_id": f"p{offset + i:03d}",
            "audio": None,
            "text": SENTENCES[i % 5],
            "terms": [],
            "meta": {"duration_s": 3.0},
        }
        for i in range(n)
    ]


def test_default_block():
    assert default_block("clinical_test") == "text"
    assert default_block("clinical_dev") == "text"
    assert default_block("fleurs_es") == "clip"


def test_score_cell_splits_clinical_by_condition():
    rows = _clinical_rows()
    hyps = {r["clip_id"]: r["text"] for r in rows}
    hyps["c01"] = "se pauta ibuprofeno al paciente"  # degraded clip loses a term
    cell = score_cell("clinical_test", rows, hyps)
    assert cell["block"] == "text"
    assert cell["n_blocks"] == 10
    split = cell["by_condition"]
    assert split["clean"]["n_clips"] == split["degraded"]["n_clips"] == 10
    assert split["clean"]["metrics"]["wer"]["value"] == 0.0
    assert split["clean"]["metrics"]["term_recall"]["value"] == 1.0
    assert split["degraded"]["metrics"]["wer"]["value"] > 0.0
    assert split["degraded"]["metrics"]["term_recall"]["value"] < 1.0
    # the split equals scoring the filtered rows on their own
    alone = score([r for r in rows if r["meta"]["condition"] == "degraded"], hyps, block="text")
    assert split["degraded"]["metrics"] == alone["metrics"]


def test_score_cell_public_has_no_split_and_null_terms():
    rows = _public_rows(10)
    cell = score_cell("fleurs_es", rows, {r["clip_id"]: r["text"] for r in rows})
    assert "by_condition" not in cell
    assert cell["block"] == "clip"
    assert cell["metrics"]["term_recall"] is None


def test_score_macro_value_is_mean_of_subset_values():
    parts = []
    for k, off in enumerate((0, 100, 200)):
        rows = _public_rows(12, off)
        hyps = {r["clip_id"]: r["text"] for r in rows}
        for r in rows[: k + 1]:
            hyps[r["clip_id"]] = "otra cosa"
        parts.append((rows, hyps))
    wers = [score(rows, hyps)["metrics"]["wer"]["value"] for rows, hyps in parts]
    macro = score_macro(parts)
    assert macro["value"] == pytest.approx(sum(wers) / 3, abs=1e-6)
    lo, hi = macro["ci95"]
    assert lo <= macro["value"] <= hi
    assert macro["n_clips"] == [12, 12, 12]


def test_macro_ratio_ci_is_deterministic():
    part = ([1.0, 0.0, 2.0, 1.0], [5.0, 4.0, 6.0, 5.0], ["a", "b", "c", "d"])
    a = macro_ratio_ci([part, part], n_boot=500, seed=0)
    b = macro_ratio_ci([part, part], n_boot=500, seed=0)
    assert a == b
    assert a.value == pytest.approx(4 / 20)
    assert a.n_blocks == 8


def _entry(system_id: str, label: str, type_: str, wer_hyp_suffix: str = "", note=None) -> dict:
    rows = _clinical_rows()
    hyps = {r["clip_id"]: r["text"] + wer_hyp_suffix for r in rows}
    cells = {"clinical_test": score_cell("clinical_test", rows, hyps, n_boot=200)}
    info = {"label": label, "type": type_, "runner": None, "note": note}
    return system_entry(system_id, info, cells)


def test_summary_and_leaderboard_order_and_notes():
    systems = [
        _entry("b_open", "B open", "open", " extra"),
        _entry("a_api", "A api", "api"),
        _entry("owner", "Owner system", "results-only", " uno dos", note="Evaluated by its owner."),
    ]
    summary = build_summary(systems, generated="2026-01-01")
    assert summary["subsets"]["clinical_test"]["by_condition"] == {"clean": 10, "degraded": 10}
    assert summary["bootstrap"]["block"]["clinical_test"] == "text"
    res = summary["systems"][0]["results"]["clinical_test"]
    assert set(res) >= {"wer", "term_recall", "bwer", "uwer", "insertions_per_1k", "by_condition"}
    assert "degenerate_clips" not in json.dumps(summary)  # no per-clip data

    md = render_leaderboard(summary)
    table = [line for line in md.splitlines() if line.startswith("| ")]
    labels = [line.split("|")[1].strip() for line in table[1:4]]
    assert labels == ["**Owner system** ¹", "A api", "B open"]  # results-only first, then WER
    assert "¹ **Owner system**: Evaluated by its owner." in md
    assert "## Real Spanish speech" not in md  # no real-speech cells


def _write_public_subset(data_dir, name: str, n: int) -> None:
    d = data_dir / name
    d.mkdir(parents=True)
    with open(d / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in _public_rows(n):
            rec = {"clip_id": r["clip_id"], "audio_path": "audio/x.wav", "text": r["text"]}
            fh.write(json.dumps({**rec, "meta": r["meta"]}) + "\n")


def test_build_report_scores_runs_and_merges_published(tmp_path):
    data_dir = tmp_path / "data"
    _write_public_subset(data_dir, "fleurs_es", 300)
    raw = tmp_path / "raw"
    for model, bad in (("model_a", 0), ("model_b", 30)):
        (raw / model).mkdir(parents=True)
        with open(raw / model / "fleurs_es.jsonl", "w", encoding="utf-8") as fh:
            for i, r in enumerate(_public_rows(300)):
                hyp = "algo distinto" if i < bad else r["text"]
                fh.write(json.dumps({"clip_id": r["clip_id"], "hyp": hyp, "secs": 0.3}) + "\n")
    (raw / "model_a" / "fleurs_es.run.json").write_text(json.dumps({"label": "Model A"}))
    published = tmp_path / "published.json"
    published.write_text(json.dumps(build_summary([_entry("owner", "Owner", "results-only")])))

    summary = build_report(
        raw,
        tmp_path / "out",
        data_dir=data_dir,
        published=published,
        n_boot=200,
        log=lambda m: None,
    )
    ids = [s["id"] for s in summary["systems"]]
    assert ids == ["owner", "model_a", "model_b"]
    a, b = summary["systems"][1], summary["systems"][2]
    assert a["label"] == "Model A (your run)"
    assert a["results"]["fleurs_es"]["wer"]["value"] == 0.0
    assert b["results"]["fleurs_es"]["wer"]["value"] > 0.0
    assert "real_speech_mean_wer" not in a  # needs all three real-speech subsets
    assert (tmp_path / "out" / "summary.json").is_file()
    assert "Model A (your run)" in (tmp_path / "out" / "leaderboard.md").read_text()


def test_cli_score_uses_auto_block(tmp_path, capsys):
    data_dir = tmp_path / "data"
    _write_public_subset(data_dir, "fleurs_es", 300)
    hyps = tmp_path / "h.jsonl"
    hyps.write_text(
        "".join(
            json.dumps({"clip_id": r["clip_id"], "hyp": r["text"]}) + "\n"
            for r in _public_rows(300)
        )
    )
    out = tmp_path / "s.json"
    rc = cli.main(
        ["score", "--subset", "fleurs_es", "--hyps", str(hyps), "--data-dir", str(data_dir),
         "--n-boot", "100", "--out", str(out)]
    )  # fmt: skip
    capsys.readouterr()
    result = json.loads(out.read_text())
    assert rc == 0 and result["block"] == "clip" and result["complete"]
    assert result["metrics"]["wer"]["value"] == 0.0


def test_registry_merges_local_and_remote_without_heavy_imports():
    from fonendo.runners import LOCAL_REGISTRY, REGISTRY, REMOTE_REGISTRY

    assert set(REGISTRY) == set(LOCAL_REGISTRY) | set(REMOTE_REGISTRY)
    assert not set(LOCAL_REGISTRY) & set(REMOTE_REGISTRY)
    assert {"whisper_large_v3", "soniox_stt_rt_v5", "deepgram_nova3_es"} <= set(REGISTRY)
    assert all(name.replace("_", "").isalnum() and name.islower() for name in REGISTRY)
    # importing the registry must not import any model dependency
    for heavy in ("torch", "transformers", "nemo", "vllm", "mlx_whisper"):
        assert heavy not in sys.modules
