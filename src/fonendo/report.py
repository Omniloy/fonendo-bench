"""Leaderboard: ``summary.json`` and ``leaderboard.md`` from hypotheses files.

``fonendo report`` scans ``<results_dir>/<model>/<subset>.jsonl`` (the layout ``fonendo run``
writes), scores every ``test`` subset it finds, and writes two files:

* ``summary.json``: per system and subset, every metric with its 95% CI; for the clinical
  subset also the clean / degraded split; for the three real-speech subsets their mean WER.
* ``leaderboard.md``: the same numbers as Markdown tables.

The published leaderboard (``results/summary.json`` and ``results/leaderboard.md`` in the
repository) has the same format, so a local run can be merged into it (``--published``).

Bootstrap blocks follow the data: the clinical subsets hold several renditions of each
sentence, so their intervals resample sentences (``block="text"``); the real-speech subsets
resample clips.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from fonendo import __version__
from fonendo.data import SUBSETS, load_subset
from fonendo.scoring import N_BOOT, NORMALIZER_VERSION, SEED, load_hyps, score, score_macro

#: Subsets shown in the leaderboard, in display order.
REPORT_SUBSETS: tuple[str, ...] = tuple(n for n, s in SUBSETS.items() if s.role == "test")
#: Real-speech subsets averaged into the mean WER (this order fixes the bootstrap draws).
REAL_SPEECH: tuple[str, ...] = ("fleurs_es", "voxpopuli_es", "mediaspeech_health")
#: Acoustic conditions of the clinical clips (``meta["condition"]``).
CONDITIONS: tuple[str, ...] = ("clean", "degraded")

#: summary.json key -> scorer metric key
SUMMARY_METRICS: dict[str, str] = {
    "wer": "wer",
    "term_recall": "term_recall",
    "bwer": "term_word_error_rate",
    "uwer": "other_word_error_rate",
    "insertions_per_1k": "insertions_per_1k",
    "degenerate_rate": "degenerate_rate",
}

METRIC_DEFINITIONS: dict[str, str] = {
    "wer": "word error rate, (S + D + I) / reference words, on normalized text",
    "term_recall": (
        "clinical only: share of the gold medical terms of the reference found in full in the "
        "hypothesis (every word of the term correct)"
    ),
    "bwer": (
        "clinical only: B-WER (Le et al., 2021), the error rate on the reference words that "
        "belong to gold medical terms"
    ),
    "uwer": "clinical only: U-WER, the error rate on every other reference word",
    "insertions_per_1k": "inserted words per 1,000 reference words",
    "degenerate_rate": "share of clips whose output is empty, loops, or runs away",
}

#: What "default configuration" means, as stated in summary.json and leaderboard.md.
DEFAULT_CONFIGURATION = (
    "no custom vocabulary, keyterms or context prompt; Spanish selected where the system "
    "allows it; instruction-following models get only the fixed transcription instruction "
    "they need"
)

#: The ``configuration`` statement of summary.json.
CONFIGURATION = (
    f"default configuration ({DEFAULT_CONFIGURATION}) for every system except the "
    "results-only ones. This covers the systems run with this package and the open-weights "
    "rows without a runner (runner null), which Omniloy ran outside the package on the same "
    "audio, with the same scoring and the settings given in each row. Results-only systems "
    "were evaluated by their owner under the conditions given in their note."
)

SUBSET_TITLES: dict[str, str] = {
    "clinical_test": "Clinical dictation",
    "fleurs_es": "FLEURS",
    "voxpopuli_es": "VoxPopuli",
    "mediaspeech_health": "MediaSpeech",
}

#: ``type`` values of a system entry, in leaderboard group order.
TYPES: dict[str, str] = {
    "results-only": "results only",
    "api": "commercial API",
    "open": "open weights",
    "local-run": "your run",
}


def default_block(subset: str) -> str:
    """Bootstrap unit of a subset: sentences for clinical subsets, clips otherwise."""
    return "text" if SUBSETS[subset].kind == "clinical" else "clip"


def resolve_block(subset: str, block: str | None) -> str:
    return default_block(subset) if block in (None, "auto") else block


# --------------------------------------------------------------------------------------
# scoring one cell
# --------------------------------------------------------------------------------------


def score_cell(
    subset: str,
    rows: list[dict[str, Any]],
    hyps: Any,
    *,
    block: str | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """:func:`fonendo.scoring.score` plus, for clinical subsets, the clean / degraded split.

    ``block=None`` (or ``"auto"``) picks :func:`default_block`. The split scores the rows
    whose ``meta["condition"]`` is ``clean`` or ``degraded`` on their own, with the same
    block rule, and is added as ``result["by_condition"]``.
    """
    block = resolve_block(subset, block)
    if not isinstance(hyps, Mapping):
        hyps = list(hyps)
    result = score(rows, hyps, block=block, n_boot=n_boot, seed=seed)
    if SUBSETS[subset].kind == "clinical":
        split: dict[str, Any] = {}
        for cond in CONDITIONS:
            part = [r for r in rows if (r.get("meta") or {}).get("condition") == cond]
            if part:
                s = score(part, hyps, block=block, n_boot=n_boot, seed=seed)
                split[cond] = {"n_clips": s["n_clips"], "metrics": s["metrics"]}
        if split:
            result["by_condition"] = split
    return result


def _compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, metric in SUMMARY_METRICS.items():
        m = metrics.get(metric)
        out[key] = None if m is None else {"value": m["value"], "ci95": m["ci95"]}
    return out


def cell_summary(cell: Mapping[str, Any]) -> dict[str, Any]:
    """The summary.json form of a :func:`score_cell` result (no per-clip data)."""
    out: dict[str, Any] = {
        "n_clips": cell["n_clips"],
        "complete": cell["complete"],
        "block": cell["block"],
        **_compact(cell["metrics"]),
    }
    if "by_condition" in cell:
        out["by_condition"] = {
            cond: {"n_clips": part["n_clips"], **_compact(part["metrics"])}
            for cond, part in cell["by_condition"].items()
        }
    return out


# --------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------


def system_entry(
    system_id: str,
    info: Mapping[str, Any],
    cells: Mapping[str, Mapping[str, Any]],
    macro_wer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One system of summary.json. ``info`` carries the descriptive fields (label, type,
    provider, model_id, license, runner, settings, note); ``cells`` maps subset -> a
    :func:`score_cell` result."""
    entry: dict[str, Any] = {
        "id": system_id,
        "label": info.get("label") or system_id,
        "type": info.get("type", "local-run"),
        "provider": info.get("provider"),
        "model_id": info.get("model_id"),
        "license": info.get("license"),
        "runner": info.get("runner"),
        "settings": info.get("settings"),
        "note": info.get("note"),
        "results": {s: cell_summary(cells[s]) for s in REPORT_SUBSETS if s in cells},
    }
    if macro_wer is not None:
        entry["real_speech_mean_wer"] = {
            "value": macro_wer["value"],
            "ci95": macro_wer["ci95"],
            "subsets": list(REAL_SPEECH),
        }
    return entry


def build_summary(
    systems: Iterable[Mapping[str, Any]], *, generated: str | None = None
) -> dict[str, Any]:
    """Wrap system entries (:func:`system_entry`) into the summary.json document."""
    systems = list(systems)
    clinical_splits = {}
    for s in systems:
        split = s.get("results", {}).get("clinical_test", {}).get("by_condition")
        if split:
            clinical_splits = {c: v["n_clips"] for c, v in split.items()}
            break
    return {
        "benchmark": "fonendo-bench",
        "fonendo_version": __version__,
        "generated": generated or date.today().isoformat(),
        "configuration": CONFIGURATION,
        "normalizer": NORMALIZER_VERSION,
        "bootstrap": {
            "ci": 0.95,
            "n_boot": N_BOOT,
            "seed": SEED,
            "block": {s: default_block(s) for s in REPORT_SUBSETS},
        },
        "subsets": {
            s: {
                "n_clips": SUBSETS[s].n_clips,
                "source": SUBSETS[s].source,
                **({"by_condition": clinical_splits} if s == "clinical_test" else {}),
            }
            for s in REPORT_SUBSETS
        },
        "metrics": METRIC_DEFINITIONS,
        "systems": systems,
    }


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------


def _pct(m: Mapping[str, Any] | None, *, ci: bool = True) -> str:
    if not m or m.get("value") is None:
        return "–"
    v = f"{100 * m['value']:.1f}"
    if ci and m.get("ci95") and None not in m["ci95"]:
        lo, hi = m["ci95"]
        return f"{v} <sub>{100 * lo:.1f}–{100 * hi:.1f}</sub>"
    return v


def _num(m: Mapping[str, Any] | None, *, ci: bool = True) -> str:
    if not m or m.get("value") is None:
        return "–"
    v = f"{m['value']:.1f}"
    if ci and m.get("ci95") and None not in m["ci95"]:
        lo, hi = m["ci95"]
        return f"{v} <sub>{lo:.1f}–{hi:.1f}</sub>"
    return v


def _name(s: Mapping[str, Any]) -> str:
    label = s["label"]
    if s["type"] == "results-only":
        return f"**{label}** ¹"
    return label


def _run(s: Mapping[str, Any]) -> str:
    if not s.get("runner"):
        return "–"
    from fonendo.runners import EXPERIMENTAL

    return f"`{s['runner']}`" + (" (experimental)" if s["runner"] in EXPERIMENTAL else "")


def _cell(text: Any) -> str:
    return "–" if text in (None, "") else str(text).replace("|", "\\|").replace("\n", " ")


def _group_sorted(
    systems: list[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], float | None]
) -> list[Mapping[str, Any]]:
    """Results-only rows first, then every other row by ``key`` (rows without it last)."""
    first = [s for s in systems if s["type"] == "results-only"]
    rest = [s for s in systems if s["type"] != "results-only"]
    rest.sort(key=lambda s: (key(s) is None, key(s) or 0.0, s["label"].lower()))
    return first + rest


def _val(s: Mapping[str, Any], subset: str, metric: str) -> float | None:
    m = s.get("results", {}).get(subset, {}).get(metric)
    return None if not m else m.get("value")


def _clinical_wer(s: Mapping[str, Any]) -> float | None:
    return _val(s, "clinical_test", "wer")


def render_leaderboard(summary: Mapping[str, Any]) -> str:
    """The leaderboard.md text of a summary.json document."""
    systems = list(summary["systems"])
    no_runner = [s for s in systems if s["type"] == "open" and not s.get("runner")]
    lines = [
        "# fonendo-bench leaderboard",
        "",
        f"Generated {summary['generated']} by `fonendo report` (fonendo "
        f"{summary['fonendo_version']}, normalizer `{summary['normalizer']}`). Every system "
        "except the results-only rows was evaluated in its **default configuration**: "
        f"{DEFAULT_CONFIGURATION}."
        + (
            " That covers the systems run with this package and the open-weights rows without "
            "a runner (`--model` –), which Omniloy ran outside the package on the same audio, "
            "with the same scoring and the settings listed under *Systems*."
            if no_runner
            else ""
        )
        + " Results-only rows (¹) were evaluated by their owner under the conditions given in "
        "their note.",
        "",
        "Values are percentages (insertions: per 1,000 reference words), with the 95% bootstrap "
        f"interval in small type ({summary['bootstrap']['n_boot']:,} resamples, seed "
        f"{summary['bootstrap']['seed']}; clinical: whole sentences resampled, real speech: "
        "clips resampled). Lower is better except for term recall. Results-only rows are "
        "pinned to the top of each table; the other rows are sorted by WER (real speech: by "
        "mean WER). Neighbouring rows whose intervals overlap may not differ, so use "
        "`fonendo compare` for a paired test before calling one system better than another.",
        "",
    ]

    clinical = [s for s in systems if "clinical_test" in s.get("results", {})]
    if clinical:
        cl = summary["subsets"]["clinical_test"]
        lines += [
            f"## Clinical dictation (`clinical_test`, {cl['n_clips']} clips)",
            "",
            "| System | Type | WER | Term recall | B-WER (term words) | U-WER (other words) "
            "| Insertions / 1k | Degenerate |",
            "|---|---|--:|--:|--:|--:|--:|--:|",
        ]
        for s in _group_sorted(clinical, lambda s: _val(s, "clinical_test", "wer")):
            r = s["results"]["clinical_test"]
            lines.append(
                f"| {_name(s)} | {TYPES.get(s['type'], s['type'])} | {_pct(r['wer'])} "
                f"| {_pct(r['term_recall'])} | {_pct(r['bwer'])} | {_pct(r['uwer'])} "
                f"| {_num(r['insertions_per_1k'])} | {_pct(r['degenerate_rate'], ci=False)} |"
            )
        lines.append("")

        split = cl.get("by_condition") or {}
        if split and any("by_condition" in s["results"]["clinical_test"] for s in clinical):
            n_clean, n_deg = split.get("clean", "?"), split.get("degraded", "?")
            lines += [
                "### Clean and degraded audio",
                "",
                f"Clean: {n_clean} clips. Degraded: {n_deg} clips with added noise (20 to 5 dB "
                "SNR) and, on many clips, a phone or low-bitrate codec, room reverberation or a "
                "speed or pitch change. The two groups are mostly different sentences.",
                "",
                "| System | WER clean | WER degraded | Term recall clean | Term recall degraded |",
                "|---|--:|--:|--:|--:|",
            ]
            for s in _group_sorted(clinical, lambda s: _val(s, "clinical_test", "wer")):
                bc = s["results"]["clinical_test"].get("by_condition") or {}
                c, d = bc.get("clean", {}), bc.get("degraded", {})
                lines.append(
                    f"| {_name(s)} | {_pct(c.get('wer'))} | {_pct(d.get('wer'))} "
                    f"| {_pct(c.get('term_recall'))} | {_pct(d.get('term_recall'))} |"
                )
            lines.append("")

    real = [s for s in systems if any(x in s.get("results", {}) for x in REAL_SPEECH)]
    if real:

        def mean_wer(s: Mapping[str, Any]) -> float | None:
            m = s.get("real_speech_mean_wer")
            return None if not m else m["value"]

        heads = " | ".join(
            f"{SUBSET_TITLES[x]} ({summary['subsets'][x]['n_clips']})" for x in REAL_SPEECH
        )
        lines += [
            "## Real Spanish speech (word error rate)",
            "",
            "Human speakers: read Wikipedia sentences (FLEURS), European Parliament speeches "
            "(VoxPopuli) and health-related broadcast media (MediaSpeech). The mean is the "
            "unweighted average of the three WERs.",
            "",
            f"| System | Type | Mean WER | {heads} |",
            "|---|---|--:|" + "--:|" * len(REAL_SPEECH),
        ]
        for s in _group_sorted(real, mean_wer):
            cells = " | ".join(_pct((s["results"].get(x) or {}).get("wer")) for x in REAL_SPEECH)
            lines.append(
                f"| {_name(s)} | {TYPES.get(s['type'], s['type'])} "
                f"| {_pct(s.get('real_speech_mean_wer'))} | {cells} |"
            )
        lines.append("")

    lines += [
        "## Systems",
        "",
        "License of each model, the `fonendo run --model` name of its runner and the settings "
        "of the published run.",
        "",
        "| System | Type | License | `--model` | Settings |",
        "|---|---|---|---|---|",
    ]
    for s in _group_sorted(systems, _clinical_wer):
        lines.append(
            f"| {_name(s)} | {TYPES.get(s['type'], s['type'])} | {_cell(s.get('license'))} "
            f"| {_run(s)} | {_cell(s.get('settings'))} |"
        )
    lines.append("")

    notes = [s for s in systems if s.get("note")]
    lines += ["## Notes", ""]
    for s in notes:
        mark = "¹ " if s["type"] == "results-only" else ""
        lines.append(f"* {mark}**{s['label']}**: {s['note']}")
    if no_runner:
        names = ", ".join(s["label"] for s in _group_sorted(no_runner, _clinical_wer))
        lines.append(
            f"* **Open-weights rows without a runner** ({names}): not runnable with this "
            "package yet. "
            "Omniloy ran them outside the package in their default configuration (as defined "
            "at the top), on the same audio and with the same scoring; the *Systems* table "
            "gives the settings of each."
        )
    lines += [
        "* **Type**: *commercial API* = hosted service called through its public streaming "
        "API; *open weights* = model run locally; *results only* = evaluated by its owner, not "
        "runnable with this package.",
        "* **Options not used**: the commercial APIs offer custom vocabulary, keyterm or "
        "context features (Soniox, Deepgram) and Whisper accepts a text prompt. None of them "
        "was used; they could raise those systems' clinical scores.",
        "* **`--model`**: the `fonendo run --model` name of the runner for the row's model; "
        "`–` means the row has no runner in this package yet (see *Open-weights rows without a "
        "runner* above; results-only rows are not runnable). *(experimental)*: the runner "
        "was not re-run with this package against the published row, so the reproduction is "
        "not verified (`fonendo models` lists these runners).",
        "* **Degenerate**: share of clips whose output is empty, loops or runs away; such "
        "outputs are scored as they are (an empty output counts every reference word as "
        "deleted). Metrics are corpus-level, so one runaway output of hundreds of words can "
        "dominate a system's WER and insertion rate; a very wide interval is the sign of it.",
        "* Metric definitions and the bootstrap procedure: see the README, section *Methodology*.",
        "",
    ]
    return "\n".join(lines)


def write_report(summary: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``leaderboard.md`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js, md = out_dir / "summary.json", out_dir / "leaderboard.md"
    js.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md.write_text(render_leaderboard(summary), encoding="utf-8")
    return js, md


# --------------------------------------------------------------------------------------
# `fonendo report`
# --------------------------------------------------------------------------------------


def _run_info(model_dir: Path, model: str) -> dict[str, Any]:
    """Descriptive fields of a local run, from its ``*.run.json`` files and the registry."""
    info: dict[str, Any] = {"type": "local-run", "runner": None}
    for rj in sorted(model_dir.glob("*.run.json")):
        try:
            meta = json.loads(rj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        info.setdefault("label", meta.get("label"))
        info.setdefault("model_id", meta.get("model_id"))
        if meta.get("model"):
            info["runner"] = meta["model"]
    try:
        from fonendo.runners import REGISTRY

        if model in REGISTRY:
            info["runner"] = model
    except ImportError:  # pragma: no cover
        pass
    info.setdefault("label", model)
    return info


#: Appended to the id of a local run when it is merged with a published summary.json.
LOCAL_ID_SUFFIX = "-local"


def _unique_id(candidate: str, taken: set[str]) -> str:
    n, out = 2, candidate
    while out in taken:
        out, n = f"{candidate}-{n}", n + 1
    return out


def build_report(
    results_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
    hf_dir: str | Path | None = None,
    published: str | Path | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> dict[str, Any]:
    """Score every ``<results_dir>/<model>/<subset>.jsonl`` of a ``test`` subset and write
    ``summary.json`` + ``leaderboard.md`` to ``out_dir`` (default: ``results_dir``).

    Subsets whose data cannot be loaded (not fetched, no access to the gated dataset) are
    skipped with a message. ``published`` merges the systems of a published summary.json
    (e.g. the repository's ``results/summary.json``) so a local run can be read next to them;
    the local entries then get the id ``<model>-local`` and the label ``<label> (your run)``,
    so every id in the merged summary.json stays unique.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir
    files: dict[str, dict[str, Path]] = {}
    for f in sorted(results_dir.glob("*/*.jsonl")):
        if f.stem in REPORT_SUBSETS:
            files.setdefault(f.parent.name, {})[f.stem] = f
    if not files and not published:
        raise FileNotFoundError(f"no <model>/<subset>.jsonl files under {results_dir}")

    rows: dict[str, list[dict[str, Any]]] = {}
    for subset in REPORT_SUBSETS:
        if not any(subset in v for v in files.values()):
            continue
        try:
            rows[subset] = load_subset(subset, data_dir, hf_dir=hf_dir, with_audio=False)
        except Exception as exc:  # noqa: BLE001 - reported and skipped
            log(f"report: skipping {subset} ({type(exc).__name__}: {exc})")

    entries = []
    for model, by_subset in sorted(files.items()):
        cells = {}
        hyps_of = {}
        for subset, path in by_subset.items():
            if subset not in rows:
                continue
            hyps_of[subset] = load_hyps(path)
            cells[subset] = score_cell(
                subset, rows[subset], hyps_of[subset], n_boot=n_boot, seed=seed
            )
            if not cells[subset]["complete"]:
                log(f"report: {model}/{subset} is incomplete ({cells[subset]['n_scored']} scored)")
        if not cells:
            continue
        macro = None
        if all(s in cells for s in REAL_SPEECH):
            macro = score_macro(
                [(rows[s], hyps_of[s]) for s in REAL_SPEECH], n_boot=n_boot, seed=seed
            )
        entries.append(system_entry(model, _run_info(results_dir / model, model), cells, macro))
        log(f"report: {model}: {', '.join(cells)}")

    if published:
        pub = json.loads(Path(published).read_text(encoding="utf-8"))
        taken = {s["id"] for s in pub["systems"]}
        for e in entries:
            # a local run of a published model must not reuse the published row's id
            e["id"] = _unique_id(f"{e['id']}{LOCAL_ID_SUFFIX}", taken)
            taken.add(e["id"])
            e["label"] = f"{e['label']} (your run)"
        log(f"report: merged {len(pub['systems'])} published systems ({len(entries)} local)")
        entries = [*pub["systems"], *entries]

    summary = build_summary(entries)
    js, md = write_report(summary, out_dir)
    log(f"report: wrote {js} and {md}")
    return summary


__all__ = [
    "CONDITIONS",
    "LOCAL_ID_SUFFIX",
    "REAL_SPEECH",
    "REPORT_SUBSETS",
    "SUMMARY_METRICS",
    "build_report",
    "build_summary",
    "cell_summary",
    "default_block",
    "render_leaderboard",
    "score_cell",
    "system_entry",
    "write_report",
]
