# fonendo-bench: package contract

This file is the interface contract of the `fonendo` package. Every module, runner and script
in this repository follows it; change it first, in the same commit, when an interface has to
change.

fonendo-bench measures Spanish speech-to-text on clinical dictation and on real Spanish speech.
Every system that the package runs is evaluated in its **default configuration**: no custom
vocabulary, keyterms or context prompt; Spanish selected where the system allows it;
instruction-following models get only the fixed transcription instruction they need; no
per-clip information of any kind. Those numbers describe what a user gets out of the box. The
open-weights leaderboard rows without a runner (`runner: null`, section 6) were run by Omniloy
outside the package under the same rules. Results-only rows are evaluated by their owner and
state their conditions in their `note`; the published OmniScribe 2 row used patient-record
context that included the spoken medical terms.

## 1. Package

| item | value |
|---|---|
| distribution / import name | `fonendo` (src layout: `src/fonendo/`) |
| Python | >= 3.10 |
| license | Apache-2.0 for the code (`LICENSE`); each dataset keeps its own license |
| entry point | `fonendo` -> `fonendo.cli:main` |
| core dependencies | `numpy`, `soundfile`, `soxr`, `pyarrow`, `datasets`, `huggingface_hub` |
| per-runner dependencies | one pip extra per runner family (section 4.3) |

```
src/fonendo/
  __init__.py        __version__, HF_DATASET = "Omniloy/fonendo-bench", SAMPLE_RATE = 16000
  data.py            SUBSETS, load_subset(), load_audio_16k(), JSONL helpers
  fetch.py           builds the public subsets (`fonendo fetch`)
  text/              Spanish normalizer (tokenize, NORMALIZER_VERSION)
  scoring/           score(), compare(), score_macro(), compare_macro(); alignment, gold-term
                     metrics, degenerate detection, bootstrap
  report.py          score_cell(), summary.json and leaderboard.md (`fonendo report`)
  cli.py             the `fonendo` command
  runners/
    __init__.py      REGISTRY (= LOCAL_REGISTRY + REMOTE_REGISTRY), get_runner()
    base.py          Runner, lazy(), run_subset()
    local/           open-weights runners, one module per family (whisper.py, nemo.py, ...)
    remote/          hosted-API runners, one module per provider (soniox.py, deepgram.py, ...)
manifests/           clip selections of the public subsets (shipped in the wheel as
                     fonendo/manifests)
```

## 2. Subsets

| name | clips | role | kind | source |
|---|---:|---|---|---|
| `clinical_test` | 300 | test | clinical | gated HF dataset `Omniloy/fonendo-bench` |
| `clinical_dev` | 60 | dev | clinical | gated HF dataset, sentences disjoint from `clinical_test` |
| `fleurs_es` | 300 | test | public | `google/fleurs`, `es_419`, test split |
| `voxpopuli_es` | 200 | test | public | `facebook/voxpopuli`, `es`, test split |
| `mediaspeech_health` | 200 | test | public | MediaSpeech (es), health-related segments |

* Clip IDs are stable identifiers, identical across releases and between the dataset and the
  results files. Order = the order of the dataset / manifest.
* `test` subsets are reported. `clinical_dev` is for smoke tests and runner development only;
  it is never used to pick settings per system and never appears in the leaderboard.
* Clinical subsets contain several renditions of the same sentence (different voices and
  acoustic conditions); `meta["text_id"]` identifies the sentence.

## 3. Data (`fonendo.data`)

### 3.1 `load_subset`

```python
load_subset(name, data_dir=None, *, hf_dir=None, token=None, with_audio=True, limit=None)
    -> list[dict]
```

Each row:

| key | type | meaning |
|---|---|---|
| `clip_id` | `str` | stable clip id |
| `audio` | `np.ndarray` float32, mono, 16 kHz | `None` when `with_audio=False` |
| `text` | `str` | reference transcript, not normalized |
| `terms` | `list[str]` | gold medical terms spoken in the clip (clinical subsets), `[]` otherwise |
| `meta` | `dict` | all other fields; clinical rows carry `text_id` |

`load_subset` raises if the number of clips differs from the table above. `limit` keeps the first
N clips after that check. Audio is always decoded by `load_audio_16k` (soundfile, channel
average, soxr HQ resampling when the file is not 16 kHz); runners receive the array and must not
re-read files.

### 3.2 Clinical subsets: HF dataset

* Hub: `datasets.load_dataset("Omniloy/fonendo-bench", "<clinical_test|clinical_dev>")` with a
  token of an account that was granted access (`HF_TOKEN` or `huggingface-cli login`). Access
  is by request (info@omniloy.com).
* Local: `load_subset(..., hf_dir=PATH)` or `FONENDO_HF_DIR=PATH` loads a local clone of the
  dataset repo with the same call (`load_dataset(PATH, name)`), so the repo's `README.md` must
  declare both configs in its YAML `configs:` block.
* Each config has exactly one split, named `test`.
* Columns: `clip_id` (string), `audio` (Audio, 16 kHz mono), `text` (string), `terms`
  (sequence of string), `text_id` (string), plus descriptive metadata columns (for example
  specialty, synthetic voice, acoustic condition, duration). Every column except `clip_id`,
  `audio`, `text` and `terms` ends up in `meta`.
* The loader casts `audio` to `Audio(decode=False)` and decodes the bytes with soundfile, so no
  torchcodec / ffmpeg dependency is needed. (Building the dataset: recent `datasets` releases
  need torchcodec to *encode* an `Audio` column; build `audio` as a `{"bytes", "path"}` struct
  and `cast_column("audio", Audio(sampling_rate=16000, decode=False))` instead.)

### 3.3 Public subsets: `fonendo fetch`

`fonendo fetch [SUBSET ...] [--data-dir data]` rebuilds the public subsets from their original
sources. The clip selection ships with the package (clip id, source dataset + revision, source
row / audio id, reference text, and the sha256 of the expected 16 kHz PCM16 audio so users can
verify they scored the same audio). Output:

```
<data_dir>/<subset>/manifest.jsonl     one line per clip, in subset order
<data_dir>/<subset>/audio/<clip_id>.wav  16 kHz, mono, PCM16
```

`manifest.jsonl` line: `{"clip_id": str, "audio_path": "audio/<clip_id>.wav", "text": str,
"meta": {...}}` (`audio_path` relative to the subset directory). `data/` is git-ignored; the
default location is `./data` or `FONENDO_DATA_DIR`.

* `meta` carries `source`, `source_id` and `duration_s`, plus per subset: FLEURS
  `sentence_id`, `gender`; VoxPopuli `speaker_id`, `gender`; MediaSpeech `health_keywords`.
* `<data_dir>/<subset>/fetch_report.json` records how many clips are sample-identical to the
  packaged sha256, any mismatch, and the library versions used.
* Source files are downloaded to `<data_dir>/.downloads/` and deleted after use (`--keep-downloads`
  keeps them). The FLEURS archive is streamed, never stored. A complete subset is skipped on
  the next run (`--force` rebuilds it).

## 4. Runners (`fonendo.runners`)

### 4.1 `Runner`

```python
class Runner:
    def __init__(self, name, kind, *, extra, label=None, model_id="", revision=None,
                 env_vars=(), batch_size=1, max_concurrency=1): ...
    def load(self) -> None                                  # weights / API client, once
    def warmup(self) -> None                                # local: 1 s of silence
    def transcribe(self, audio: np.ndarray, sr: int) -> str # REQUIRED
    def transcribe_batch(self, audios, sr) -> list[str]     # optional, same results
    def close(self) -> None
    def info(self) -> dict                                  # written to <out>.run.json
```

| attribute | meaning |
|---|---|
| `name` | registry key and results folder, `[a-z0-9_]+`, e.g. `whisper_large_v3` |
| `kind` | `"local"` (weights run on your hardware) or `"remote"` (hosted API) |
| `extra` | pip extra that installs the runner's dependencies |
| `label` | display name in reports |
| `model_id`, `revision` | upstream id (HF repo or API model) and pinned revision / API version |
| `env_vars` | environment variables the runner needs (API keys); checked before `load()` |
| `batch_size` | local runners: clips per `transcribe_batch` call |
| `max_concurrency` | remote runners: parallel requests, hard-capped at 3 |

### 4.2 `REGISTRY`

`REGISTRY: dict[str, Factory]` in `fonendo/runners/__init__.py` maps a model name to a
factory returning a `Runner`. Register with `lazy()` so importing the registry never imports a
model's dependencies; a missing extra becomes `pip install 'fonendo[<extra>] @ git+https://github.com/Omniloy/fonendo-bench'` (the package is not on PyPI):

```python
LOCAL_REGISTRY["whisper_large_v3"] = lazy(
    "fonendo.runners.local.whisper:WhisperRunner",
    extra="whisper",
)
```

`LOCAL_REGISTRY` lives in `fonendo/runners/local/__init__.py`, `REMOTE_REGISTRY` in
`fonendo/runners/remote/__init__.py`; `REGISTRY` merges both and refuses duplicate names.

Factories accept keyword overrides; the CLI passes only `device` (`--device`).

### 4.3 Extras

| extra | family |
|---|---|
| `whisper` | Whisper checkpoints via `transformers` |
| `nemo` | NVIDIA NeMo models (Parakeet, Canary); own virtual environment |
| `voxtral` | Mistral Voxtral Mini 3B and Mini 4B Realtime via `transformers` |
| `voxtral-vllm` | Mistral Voxtral Small 24B via vLLM; own virtual environment |
| `cohere` | Cohere Transcribe |
| `granite` | IBM Granite Speech |
| `mlx-whisper` | Whisper on Apple Silicon (MLX); macOS only |
| `soniox` | Soniox API (remote) |
| `deepgram` | Deepgram API (remote) |
| `openai-compatible` | any OpenAI-compatible `/audio/transcriptions` endpoint (remote or self-hosted) |
| `dev` | tests and lint |

The published Parakeet and Canary hypotheses were produced with NeMo from GitHub at commit
`ca3f93a5`; the `nemo` extra installs the PyPI 3.x release, which runs the same code but
decodes a few clips differently (see `fonendo/runners/local/nemo.py`).

Pin lower and upper bounds that were actually tested. When two families cannot share one
environment (conflicting `transformers` or `torch` pins), say so in the runner's docstring and
in the README; use one virtual environment per extra.

### 4.4 Hypotheses file

`fonendo run --model M --subset S` calls `run_subset` and writes
`results/raw/<M>/<S>.jsonl` (override with `--out`), one JSON object per line, flushed per clip:

| field | type | meaning |
|---|---|---|
| `clip_id` | str | from the subset |
| `hyp` | str | raw system output; no post-processing (scoring normalizes) |
| `secs` | float | wall time of the transcription call only (no audio I/O, no model load); a batch's time is split evenly over its clips |
| `error` | str | only on failure (`hyp` is `""`); such lines are retried on the next run |

```json
{"clip_id": "fleurs_1661_6253518291639218682", "hyp": "No mencionó una cifra ...", "secs": 0.412}
```

* **Resumable**: clip ids already present without `error` are skipped; rerunning a finished
  command writes nothing. A last line that does not parse (a run killed mid-write) is dropped
  with a warning and its clip is redone; an unparsable line before the last one stops the run
  with a `MalformedFileError` and leaves the file untouched.
* Next to it, `<S>.run.json` records `Runner.info()` (model, label, kind, extra, model id,
  revision, plus any runner-specific fields such as device, dtype, decoding flags, API region),
  fonendo and Python versions, platform and finish time. It must never contain secrets.

### 4.5 Rules for every runner

1. **Default configuration.** No custom vocabulary, keyterms or context prompt; Spanish
   selected when the system has a language option (record it in `info()`); an
   instruction-following model gets only the fixed transcription instruction it needs (record
   it in `info()`), the same for every clip; decoding as the model card recommends (greedy or
   its default beam), temperature 0; the same settings for every subset. Settings are never
   tuned per subset or per clip.
2. **Input.** The 16 kHz float32 array from `load_subset`. Clips are at most ~30 s: no chunking,
   no VAD, unless the system does it internally by default.
3. **Bounded output.** Cap generation (e.g. `max_new_tokens=1024`) so a runaway loop cannot
   stall a run; scoring flags it as degenerate.
4. **Secrets.** API keys come only from the environment variables in `env_vars`
   (`SONIOX_API_KEY`, `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, ...; `HF_TOKEN` for gated weights).
   Never log, print, or write them; `.env` is git-ignored.
5. **Remote APIs.** At most 3 concurrent requests, retries with backoff inside `transcribe`,
   the region used recorded in `info()`.
6. **Weights.** Load from the HF cache (`HF_HUB_CACHE`); pin `revision` to a commit hash;
   document size on disk and tested device in the docstring.

### 4.6 Adding a runner

1. Write `src/fonendo/runners/local/<family>.py` (or `remote/<provider>.py`) with a `Runner`
   subclass; its docstring lists the model id + revision, tested versions, device, decoding
   flags and size on disk.
2. Add its dependencies to the family's extra in `pyproject.toml` (create the extra if new).
3. Register it in `LOCAL_REGISTRY` (or `REMOTE_REGISTRY`) with `lazy(...)`.
4. Smoke test: `fonendo run --model M --subset clinical_dev --limit 5` writes 5 lines; the same
   command again writes nothing; `fonendo score --subset clinical_dev --hyps ...` looks sane.
5. Full run of every `test` subset; each file has exactly the subset's clip count and no
   `error` line.
6. `fonendo report --published results/summary.json` scores every file under `results/raw/`
   and lists the new model next to the published ones.

## 5. Scoring (`fonendo.scoring`)

```python
score(subset_rows, hyps, *, block="auto", n_boot=2000, seed=0) -> dict
compare(subset_rows, hyps_a, hyps_b, *, block="auto", n_boot=2000, seed=0) -> dict
load_hyps(path) -> list[dict]
```

`subset_rows` come from `load_subset(..., with_audio=False)`; `hyps` is either a
`{clip_id: hyp}` mapping or the rows of a hypotheses file (`load_hyps`, which raises
`fonendo.data.MalformedFileError` naming the file and line when a line is not a JSON object,
has no `clip_id` or has a non-string `hyp`). A clip that is missing or has an `error` line is
scored as an empty hypothesis and makes the result `complete: false`.

### 5.1 Normalization and alignment

Reference and hypothesis go through the same normalizer: lowercase, Unicode NFC, punctuation
removed, accents kept, number words rewritten as digits and units after a number abbreviated
("quinientos miligramos" -> "500 mg"), and runs of spelled-out
Spanish letter names collapsed to the initialism ("eme ge" -> "mg"). Word alignment is
Levenshtein with substitutions (S), deletions (D) and insertions (I). The normalizer is
versioned and its version is written into every score file.

### 5.2 Metrics

All metrics are corpus-level (ratio of sums over clips).

| key | subsets | definition |
|---|---|---|
| `wer` | all | (S + D + I) / reference words |
| `term_recall` | clinical | share of the gold-term occurrences of the normalized reference whose every word is aligned as correct in the hypothesis (all-or-nothing per term); denominator: gold terms that occur as a span in the normalized reference |
| `term_word_error_rate` | clinical | B-WER (Le et al., Interspeech 2021): errors on reference words that belong to the clip's gold terms / number of such words; an insertion counts here when the inserted word is a gold-term word. Function words (`FUNCTION_WORDS` in `scoring/terms.py`: "de", "la", "con", ...) are never term words, even inside a multi-word term |
| `other_word_error_rate` | clinical | U-WER: the same on all other words, function words included |
| `insertions_per_1k` | all | inserted words per 1,000 reference words |
| `degenerate_rate` | all | share of clips whose output is empty (non-empty reference), loops (an n-gram absent from the reference repeated >= 4x for n = 1 or >= 3x for n = 2..8, or a 5-gram repetition rate above the reference's + 0.05) or runs away (hypothesis words > 2 x reference words + 10) |

Term metrics are `null` on subsets without gold terms.

### 5.3 Confidence intervals and paired comparison

* 95% percentile bootstrap, 2,000 resamples, seed 0; the ratio of sums is recomputed on each
  resample.
* `block="clip"` resamples clips; `block="text"` resamples sentences through
  `meta["text_id"]` (all renditions of a sentence move together; the conservative choice for
  the clinical subsets). The default everywhere is `auto`, the rule of the leaderboard: the
  CLI (`--block auto`) picks sentences for clinical subsets and clips otherwise, and the Python
  API (`block="auto"`, `fonendo.scoring.resolve_block`) picks sentences when the rows carry
  `meta["text_id"]` (the clinical subsets) and clips otherwise. The result's `block` field is
  the unit actually used.
* `compare` resamples A and B with the same blocks and reports, per metric, `a`, `b`,
  `delta = a - b`, its `ci95`, `p_two_sided` (twice the smaller share of resamples on either
  side of 0), `significant` and `n_blocks`. A difference is called significant only when its
  CI excludes 0 and there are at least 5 blocks.
* `score_macro(parts, metric="wer")` averages a metric over several subsets (unweighted) with
  a stratified bootstrap CI; `compare_macro` is its paired version. One generator draws the
  subsets in the order given; the leaderboard uses fleurs_es, voxpopuli_es,
  mediaspeech_health.

### 5.4 Score file

`fonendo score --subset S --hyps F --out results/<M>/<S>.score.json`:

```json
{
  "subset": "clinical_test",
  "hyps": "results/<M>/clinical_test.jsonl",
  "n_clips": 300, "n_scored": 300, "n_missing": 0, "n_errors": 0, "complete": true,
  "normalizer": "<version>", "block": "text", "n_boot": 2000, "seed": 0,
  "metrics": {
    "wer":                   {"value": 0.081, "ci95": [0.072, 0.091]},
    "term_recall":           {"value": 0.930, "ci95": [0.905, 0.953], "n": 412},
    "term_word_error_rate":  {"value": 0.052, "ci95": [0.041, 0.064], "n": 530},
    "other_word_error_rate": {"value": 0.084, "ci95": [0.075, 0.094], "n": 4410},
    "insertions_per_1k":     {"value": 6.1,   "ci95": [4.8, 7.5]},
    "degenerate_rate":       {"value": 0.0,   "ci95": [0.0, 0.0]}
  },
  "degenerate_clips": {},
  "rtf_median": 0.05
}
```

(Values are illustrative.) The normalizer version is the constant `es1+lc1`. The file also
carries `n_blocks`, a `counts` object `{S, D, I, N, gold_terms_not_in_ref}`, and, for clinical
subsets, `by_condition: {clean: {n_clips, metrics}, degraded: {n_clips, metrics}}` (rows split
by `meta["condition"]`, same block rule). Every metric also carries `n`, its denominator.
`rtf_median` is the median of `secs / audio duration` when the hypotheses carry `secs`; it
depends on hardware and is informative only.

## 6. Results and report

```
results/
  summary.json       published leaderboard data: per system and test subset, every metric
                     with its 95% CI (no hypotheses, no per-clip data)
  leaderboard.md     the same numbers as tables (rendered from summary.json)
  raw/               your own runs, <model>/<subset>.jsonl + .run.json (git-ignored)
```

* A published cell is complete: exactly the subset's clips, no `error` line.
* `summary.json`: `{benchmark, fonendo_version, generated, configuration, normalizer,
  bootstrap {ci, n_boot, seed, block per subset}, subsets, metrics (definitions), systems}`.
  Each system: `{id, label, type, provider, model_id, license, runner, settings, note,
  results: {<subset>: {n_clips, complete, block, wer, term_recall, bwer, uwer,
  insertions_per_1k, degenerate_rate, by_condition?}}, real_speech_mean_wer}`; each metric is
  `{value, ci95}` or null (`ci95` may be null on results-only rows). `type` is `api`, `open`,
  `results-only` or `local-run`; `runner` is the `--model` name of the row's runner, or null.
  Runners listed in `fonendo.runners.EXPERIMENTAL` were not re-run against their published
  row; the leaderboard marks them *experimental*.
* **Results-only systems** (evaluated by their owner, no runner in `REGISTRY`) appear with
  `type: "results-only"`, `runner: null` and a `note` that the leaderboard shows verbatim.
  They are pinned to the top of every leaderboard table; the other rows are sorted by WER.
* **Open-weights rows without a runner** (`type: "open"`, `runner: null`) were run by Omniloy
  outside the package in the default configuration, on the same audio and with the same
  scoring; `settings` describes the run. `configuration` and the leaderboard say so.
* `fonendo report` scores every `<results-dir>/<model>/<subset>.jsonl` of a `test` subset
  (default `results/raw`) and writes `summary.json` + `leaderboard.md` to `--out`;
  `--published results/summary.json` adds the published systems to the tables. The local
  entries then get the id `<model>-local` (with `-2`, `-3`, ... if that is taken too) and the
  label `<label> (your run)`, so ids stay unique.

## 7. CLI

```
fonendo fetch [SUBSET ...] [--data-dir DIR] [--force] [--keep-downloads]
fonendo models
fonendo run --model M --subset S [--limit N] [--device D] [--out F] [--data-dir DIR] [--hf-dir DIR]
fonendo score --subset S --hyps F [--block auto|clip|text] [--n-boot N] [--out F]
fonendo compare A.jsonl B.jsonl --subset S [--block auto|clip|text] [--out F]
fonendo report [--results-dir results/raw] [--out DIR] [--published results/summary.json]
```

Environment: `HF_TOKEN` (gated dataset and weights), `FONENDO_DATA_DIR`, `FONENDO_HF_DIR`,
`HF_HUB_CACHE`, and the API keys of remote runners.

Expected failures (unknown model, missing extra, missing API key, public subset not fetched,
no access to the gated dataset, a malformed hypotheses file) print one line,
`fonendo <command>: error: <message>`, on stderr and exit with code 2;
`fonendo --traceback <command> ...` shows the full traceback. `fonendo run` checks the
runner's environment variables before it loads the subset; `fonendo score` and
`fonendo compare` read the hypotheses files before they load the subset.

## 8. Repository hygiene

* No secrets, local absolute paths, or machine names in code, docs, results or commit messages.
* Audio never enters git (`*.wav`, `data/` are ignored); clinical audio is distributed only
  through the gated dataset. The one exception is the showcase: `docs/audio/` holds 13 short
  MP3 demo clips, `clinical_test` clips chosen by Omniloy for illustration (© Omniloy, not
  under the code license; credits and terms in the README, section License).
* `results/` publishes aggregate numbers only (`summary.json`, `leaderboard.md`): no
  hypotheses and no per-clip data.
* Code comments and docs are in English; references and hypotheses stay as produced (Spanish).
