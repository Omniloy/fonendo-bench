# fonendo-bench

**A reproducible Spanish clinical speech-to-text benchmark, published by
[Omniloy](https://omniloy.com).**

**▶ [Listen to the demos](https://omniloy.github.io/fonendo-bench/)** · [Leaderboard](results/leaderboard.md) · [Showcase (text)](SHOWCASE.md) · [Dataset (gated, request access)](https://huggingface.co/datasets/Omniloy/fonendo-bench)

fonendo-bench measures how well speech-to-text systems transcribe Spanish clinical dictation
(drug names with doses, diagnoses, imaging and laboratory tests, abbreviations, numbers and
units) and, as a control, real Spanish speech. It ships:

* a Python package and CLI, `fonendo`, that runs open-weights models and hosted APIs on the
  same audio, scores them with one normalizer, and reports every number with a 95% bootstrap
  interval and paired comparisons;
* the clinical test set, as a gated dataset on Hugging Face
  ([Omniloy/fonendo-bench](https://huggingface.co/datasets/Omniloy/fonendo-bench));
* recipes that rebuild three public real-speech test sets from their original sources, sample
  for sample;
* the [leaderboard](results/leaderboard.md) of 26 systems, three commercial APIs and 22 open
  models among them.

Every system on the leaderboard except OmniScribe 2 is evaluated in its **default
configuration**: no custom vocabulary, keyterms or context prompt; Spanish selected where the
system allows it; instruction-following models get only the fixed transcription instruction
they need. None of them sees the reference, the terms or any other per-clip information. That
holds for the systems the package runs and for the open-weights rows that have no runner in the
package yet (note ²). Those numbers describe what a user gets out of the box. The one exception
is OmniScribe 2, a results-only row: it uses context from the patient's record, and in this
test that context included the medical terms spoken in each clinical clip (a best case; see
note ¹).

## Leaderboard

Clinical: `clinical_test`, 300 clips (147 clean, 153 degraded). Real speech: the mean WER over
FLEURS, VoxPopuli and MediaSpeech (700 clips). Values in %, with the 95% interval in small
type. OmniScribe 2 (results only, given patient-record context) is pinned to the top; the
other rows are sorted by clinical WER. Lower is better except for term recall. Full tables
(B-WER, U-WER, insertions, degenerate outputs, per-subset WER, intervals for every cell, and
the license and settings of every row): [results/leaderboard.md](results/leaderboard.md); the
same data as JSON: [results/summary.json](results/summary.json).

| System | Type | Clinical WER | Term recall | WER clean | WER degraded | Real speech WER |
|---|---|--:|--:|--:|--:|--:|
| **OmniScribe 2 (Omniloy, self-hosted, not publicly available)** ¹ | results only | 8.5 <sub>6.9–10.3</sub> | 85.7 <sub>81.9–89.4</sub> | 3.7 | 13.3 | 6.6 |
| Soniox stt-rt-v5 | API | 7.2 <sub>6.0–8.5</sub> | 70.8 <sub>65.5–76.2</sub> | 4.8 | 9.5 | 7.4 <sub>6.7–8.2</sub> |
| Voxtral Small 24B (FP8 weights) | open | 8.5 <sub>6.7–10.3</sub> | 70.3 <sub>64.6–75.7</sub> | 4.0 | 12.8 | 6.6 <sub>6.0–7.2</sub> |
| Cohere Transcribe | open | 9.3 <sub>7.8–10.9</sub> | 60.4 <sub>54.6–66.4</sub> | 5.2 | 13.4 | 6.8 <sub>6.2–7.4</sub> |
| Whisper large-v3-turbo | open | 10.3 <sub>8.4–12.3</sub> | 63.8 <sub>58.1–69.4</sub> | 4.8 | 15.6 | 9.4 <sub>8.1–10.9</sub> |
| Whisper large-v3 | open | 10.6 <sub>8.6–12.7</sub> | 64.1 <sub>58.1–69.9</sub> | 5.2 | 15.9 | 10.3 <sub>8.1–13.3</sub> |
| Deepgram Nova-3 (es) | API | 11.5 <sub>9.5–13.6</sub> | 62.8 <sub>56.7–68.7</sub> | 5.6 | 17.2 | 8.0 <sub>7.3–8.7</sub> |
| Canary-1B-v2 | open | 11.8 <sub>9.8–13.9</sub> | 56.0 <sub>49.7–62.0</sub> | 5.5 | 18.0 | 7.1 <sub>6.5–7.7</sub> |
| Gemma 4 E4B ² | open | 11.9 <sub>9.8–14.0</sub> | 63.3 <sub>57.4–69.0</sub> | 5.3 | 18.4 | 7.7 <sub>7.1–8.5</sub> |
| Parakeet-TDT v3 | open | 12.3 <sub>10.1–14.6</sub> | 58.6 <sub>52.6–64.4</sub> | 6.2 | 18.4 | 10.8 <sub>9.6–12.2</sub> |
| Voxtral Mini 4B Realtime | open | 13.9 <sub>11.2–16.6</sub> | 59.9 <sub>54.1–65.7</sub> | 5.8 | 21.8 | 8.1 <sub>7.1–9.2</sub> |
| Hojo-ASR-Multi-V1 ² | open | 14.0 <sub>11.8–16.3</sub> | 56.5 <sub>50.7–62.4</sub> | 7.8 | 20.1 | 7.3 <sub>6.7–8.0</sub> |
| Whisper large-v3 LoS ² | open | 15.0 <sub>12.9–17.4</sub> | 52.1 <sub>45.7–57.9</sub> | 8.0 | 21.9 | 12.8 <sub>10.6–15.6</sub> |
| omniASR-LLM-7B ² | open | 15.2 <sub>12.8–18.2</sub> | 49.7 <sub>44.3–55.3</sub> | 7.8 | 22.5 | 6.5 <sub>5.8–7.3</sub> |
| Granite Speech 4.1 2B NAR ² | open | 15.4 <sub>13.3–17.4</sub> | 43.8 <sub>37.5–50.0</sub> | 8.8 | 21.9 | 7.6 <sub>7.0–8.2</sub> |
| MOSS-Transcribe-Diarize ² | open | 15.7 <sub>13.3–18.1</sub> | 48.7 <sub>42.3–54.5</sub> | 9.0 | 22.2 | 12.0 <sub>10.5–13.7</sub> |
| Nemotron 3.5 ASR Streaming ² | open | 17.0 <sub>14.8–19.2</sub> | 44.0 <sub>38.0–49.6</sub> | 9.8 | 24.0 | 7.9 <sub>7.2–8.7</sub> |
| Deepgram Flux Multilingual | API | 17.7 <sub>15.0–20.4</sub> | 49.7 <sub>43.5–55.8</sub> | 7.3 | 27.9 | 8.0 <sub>7.3–8.7</sub> |
| Granite Speech 4.1 2B | open | 19.2 <sub>16.5–22.0</sub> | 46.9 <sub>41.1–52.5</sub> | 10.0 | 28.2 | 11.5 <sub>10.2–12.9</sub> |
| Whisper large-v3 clinical-assistance ² | open | 20.2 <sub>14.3–30.1</sub> | 55.5 <sub>49.7–61.1</sub> | 16.7 | 23.6 | 10.6 <sub>8.4–14.0</sub> |
| Canary-1B-flash ² | open | 20.5 <sub>17.6–23.6</sub> | 47.1 <sub>41.3–53.1</sub> | 9.8 | 31.0 | 17.1 <sub>15.1–19.3</sub> |
| Phi-4-multimodal-instruct ² | open | 24.0 <sub>10.4–49.8</sub> | 59.1 <sub>52.7–65.2</sub> | 6.7 | 41.0 | 10.0 <sub>7.0–15.2</sub> |
| Granite Speech 4.1 2B Plus ² | open | 26.3 <sub>21.1–34.2</sub> | 38.8 <sub>32.7–44.8</sub> | 12.7 | 39.5 | 14.6 <sub>9.4–24.6</sub> |
| Parakeet-RNNT 1.1B es (projecte-aina) ² | open | 27.8 <sub>24.0–31.6</sub> | 40.6 <sub>35.0–46.2</sub> | 11.3 | 43.9 | 17.2 <sub>15.8–18.8</sub> |
| Voxtral Mini 3B | open | 29.0 <sub>9.7–67.8</sub> | 57.8 <sub>51.5–63.8</sub> | 6.6 | 50.8 | 7.0 <sub>6.5–7.7</sub> |
| VibeVoice-ASR ² | open | 67.7 <sub>21.9–130.7</sub> | 50.0 <sub>43.5–55.9</sub> | 7.2 | 126.7 | 12.6 <sub>7.7–21.9</sub> |

¹ **OmniScribe 2** is Omniloy's self-hosted clinical transcription system; it is not publicly
available and not runnable with this package (results only, evaluated by Omniloy). OmniScribe 2
uses context from the patient's record. In this test that context included the medical terms
spoken in each clinical clip, so its clinical numbers are a best case. It transcribed whole
clips offline. Its real-speech numbers used no context and are given rounded, without
intervals.
² No runner for this model in this package yet; its row comes from Omniloy's own run outside
the package, in the same default configuration, on the same audio and with the same scoring.
Its settings are listed in [results/leaderboard.md](results/leaderboard.md) and
`summary.json`.

Reading the table:

* Neighbouring rows often do not differ: two intervals that overlap are not a test. Use
  `fonendo compare` (a paired bootstrap on the same clips) before calling one system better.
* Metrics are corpus-level, so a single runaway output (a loop of hundreds of words) can
  dominate a system's WER; a very wide interval is the sign of it (Voxtral Mini 3B, Phi-4,
  VibeVoice-ASR on degraded audio).
* Voxtral Small 24B was run with FP8 weight-only quantization (see
  [Supported models](#supported-models)).

## Listen to the demos

The [showcase](https://omniloy.github.io/fonendo-bench/) ([source](docs/index.html)) plays
clinical clips next to what each system wrote, with the errors marked;
[SHOWCASE.md](SHOWCASE.md) is the text version with links to the audio files. It compares
OmniScribe 2 with three commercial APIs and three widely used open models; the leaderboard
above has all 22 open models.

## Datasets

| subset | clips | hours | speech | source | license |
|---|--:|--:|---|---|---|
| `clinical_test` | 300 | 0.64 | synthetic voices, clinical dictation | [Omniloy/fonendo-bench](https://huggingface.co/datasets/Omniloy/fonendo-bench) (gated) | Omniloy fonendo-bench evaluation license |
| `clinical_dev` | 60 | 0.13 | same, sentences disjoint from test; never reported | same | same |
| `fleurs_es` | 300 | 1.04 | read Wikipedia sentences | [google/fleurs](https://huggingface.co/datasets/google/fleurs), `es_419` test | CC BY 4.0 |
| `voxpopuli_es` | 200 | 0.64 | European Parliament speeches | [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli), `es` test | CC0 (see the dataset card for the European Parliament's terms) |
| `mediaspeech_health` | 200 | 0.80 | broadcast media, health-related segments | [MediaSpeech](https://huggingface.co/datasets/ymoslem/MediaSpeech) (es) | CC BY 4.0 |

### Clinical subsets (gated)

Fictitious Spanish clinical sentences, written for the benchmark in the register of clinical
notes, discharge summaries and test orders, synthesised with text-to-speech (11 voices from
Spain and Latin America). About half of the clips are clean; the other half are degraded with
added noise (20 to 5 dB SNR) and, on many clips, a phone or low-bitrate codec, room
reverberation or a speed or pitch change. Each clip lists the medical terms it contains
(`terms`), which drive the term metrics. The sentences describe no real patient. Details are in
the [dataset card](https://huggingface.co/datasets/Omniloy/fonendo-bench).

**Access.** Request access on the
[dataset page](https://huggingface.co/datasets/Omniloy/fonendo-bench) and email
**info@omniloy.com** with your name, your organisation and your intended use. The license
allows evaluation and research, including commercial evaluation; it forbids redistributing the
audio and training or adapting models on it. Once access is granted, `fonendo` downloads the
subsets itself with your Hugging Face token (`hf auth login`, `huggingface-cli login` or
`HF_TOKEN`).

### Public subsets

The clip selection ships with the package (`manifests/`): the clip ids, the source dataset and
its pinned revision, the row of each clip, the reference text and the sha256 of the expected
16 kHz audio. `fonendo fetch` downloads the sources, extracts the selected clips, converts them
to 16 kHz mono PCM16 and checks every clip against its sha256 (`fetch_report.json`). About
290 MB of audio is kept; source files are downloaded to `data/.downloads/` and deleted as soon
as their clips are extracted (the FLEURS archive is streamed, never stored).

Attribution: FLEURS (Conneau et al., 2022; CC BY 4.0), VoxPopuli (Wang et al., 2021; CC0),
MediaSpeech (Kolobov et al., 2021; CC BY 4.0). The rebuilt audio keeps the license of its
source.

## Quick start

```bash
git clone https://github.com/Omniloy/fonendo-bench
cd fonendo-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[whisper]"      # core + the extra of the model family you want to run
fonendo models                   # registered models and their extras

# 1. data
fonendo fetch                    # public subsets -> ./data (or --data-dir, FONENDO_DATA_DIR)
hf auth login                    # clinical subsets: an account that was granted access
                                 # (or huggingface-cli login, or export HF_TOKEN=...)

# 2. run a local model (writes results/raw/<model>/<subset>.jsonl, resumable)
fonendo run --model whisper_large_v3_turbo --subset clinical_dev --limit 5    # smoke test
fonendo run --model whisper_large_v3_turbo --subset clinical_test
fonendo run --model whisper_large_v3_turbo --subset fleurs_es --device cuda

# 3. run a hosted API (keys are read from the environment, never logged or written)
pip install -e ".[soniox]"
export SONIOX_API_KEY=...        # your key
export SONIOX_REGION=eu          # the region of the project that owns the key
fonendo run --model soniox_stt_rt_v5 --subset clinical_test

# 4. score and compare
fonendo score --subset clinical_test \
    --hyps results/raw/whisper_large_v3_turbo/clinical_test.jsonl --out turbo.score.json
fonendo compare --subset clinical_test \
    results/raw/soniox_stt_rt_v5/clinical_test.jsonl \
    results/raw/whisper_large_v3_turbo/clinical_test.jsonl

# 5. your runs next to the published leaderboard
fonendo report --published results/summary.json --out my_report
```

* `fonendo run` writes one JSON line per clip, `{"clip_id", "hyp", "secs"}`, flushed as it
  goes; a rerun skips finished clips and retries failed ones, and a last line cut off by an
  interrupted run is dropped with a warning and its clip transcribed again. A `.run.json` next
  to it records the model, revision, backend versions, device and decoding settings.
* `fonendo score` prints every metric with its 95% interval (for the clinical subset also the
  clean / degraded split); `fonendo compare` prints the paired difference A - B, its interval
  and whether it is significant.
* `fonendo report --published results/summary.json` lists your runs next to the published
  rows; a local run gets the id `<model>-local` and the label "(your run)", so it never
  collides with the published row of the same model.
* To score without downloading the dataset again, point `--hf-dir` (or `FONENDO_HF_DIR`) at a
  local clone of the dataset repository.

From Python:

```python
from fonendo.data import load_subset
from fonendo.scoring import load_hyps, score

rows = load_subset("clinical_test", with_audio=False)  # clip_id, text, terms, meta
result = score(rows, load_hyps("results/raw/my_model/clinical_test.jsonl"))
print(result["metrics"]["wer"])  # {"value": ..., "ci95": [..., ...], "n": ...}
print(result["block"])           # "text": sentences resampled, as on the leaderboard
```

`score()` and `compare()` use the same bootstrap rule as the CLI by default (`block="auto"`:
whole sentences when the rows carry `meta["text_id"]`, as the clinical subsets do, clips
otherwise), so the intervals match the published ones; pass `block="clip"` or `block="text"`
to choose. The public subsets load the same way (`load_subset("fleurs_es", "data")`).

`load_subset(name)` returns one dict per clip with `audio` as a float32 mono 16 kHz array,
`text` (the reference), `terms` (gold medical terms, clinical subsets only) and `meta`.

## Supported models

| `--model` | system | kind | extra | hardware | model license |
|---|---|---|---|---|---|
| `whisper_large_v3` | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | local | `whisper` | CUDA GPU, 3.1 GiB peak; 3.1 GB download | Apache-2.0 |
| `whisper_large_v3_turbo` | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | local | `whisper` | CUDA GPU, 1.6 GiB; 1.6 GB | MIT |
| `voxtral_mini_3b` | [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | local | `voxtral` | CUDA GPU, 8.8 GiB; 9.4 GB | Apache-2.0 |
| `voxtral_mini_4b_realtime` | [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | local | `voxtral` | CUDA GPU, 8.5 GiB; 8.9 GB | Apache-2.0 |
| `voxtral_small_24b` | [mistralai/Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507), FP8 weights (**experimental**: not re-run with this package) | local | `voxtral-vllm` | CUDA GPU, about 28 GB (FP8); 48.5 GB download; own venv | Apache-2.0 |
| `cohere_transcribe` | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | local | `cohere` | CUDA GPU, 3.9 GiB; 4.1 GB; gated: accept the terms on the model page | Apache-2.0 |
| `granite_speech_4p1_2b` | [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | local | `granite` | CUDA GPU, 4.4 GiB; 4.4 GB | Apache-2.0 |
| `parakeet_tdt_0p6b_v3` | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | local | `nemo` | CUDA GPU, 4.7 GiB; 2.5 GB; own venv | CC-BY-4.0 |
| `canary_1b_v2` | [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | local | `nemo` | CUDA GPU, 7.2 GiB; 6.4 GB; own venv | CC-BY-4.0 |
| `mlx_whisper_large_v3` | Whisper large-v3, MLX conversion (**experimental**: not scored) | local | `mlx-whisper` | Apple Silicon Mac; 3.1 GB | Apache-2.0 |
| `mlx_whisper_large_v3_turbo` | Whisper large-v3-turbo, MLX conversion (**experimental**: not scored) | local | `mlx-whisper` | Apple Silicon Mac; 1.6 GB | MIT |
| `soniox_stt_rt_v5` | Soniox `stt-rt-v5`, real-time API | remote | `soniox` | `SONIOX_API_KEY` (+ `SONIOX_REGION`) | commercial API |
| `deepgram_nova3_es` | Deepgram Nova-3, Spanish, streaming | remote | `deepgram` | `DEEPGRAM_API_KEY` (+ `DEEPGRAM_REGION`) | commercial API |
| `deepgram_flux_multi` | Deepgram Flux Multilingual, streaming | remote | `deepgram` | `DEEPGRAM_API_KEY` (+ `DEEPGRAM_REGION`) | commercial API |
| `openai_compatible` | any server with an OpenAI-style `/v1/audio/transcriptions` endpoint (**experimental**) | remote | `openai-compatible` | `FONENDO_OPENAI_BASE_URL`, `FONENDO_OPENAI_MODEL` | depends on the server |

Notes:

* **Tested hardware.** Local runners were tested on an NVIDIA A100 80 GB with torch 2.14 and
  transformers 5.17; any CUDA GPU with the memory listed should work. Each module's docstring
  (`src/fonendo/runners/local/`) gives the pinned revision, download size, decoding settings and
  tested versions.
* **Reproduction.** Re-run with this package on the first 20 `clinical_test` clips, the
  Whisper (large-v3 and large-v3-turbo), Voxtral Mini 3B, Voxtral Mini 4B Realtime, Cohere,
  Granite, Parakeet and Canary runners returned the published transcripts byte for byte.
  Hosted APIs change and streaming is not bit-exact: in a 5-clip check per service, Nova-3 and
  Flux matched 5 of 5 and Soniox 4 of 5 (one word differed on a noisy clip).
* **Experimental runners.** `voxtral_small_24b` (48.5 GB download, about 28 GB of GPU memory)
  and the two MLX backends were not re-run with this package: their code paths are tested, but
  their output has not been checked against a published row. `openai_compatible` has no
  published row. `fonendo models` and the leaderboard mark these runners *experimental*.
* **Virtual environments.** `whisper`, `voxtral`, `cohere` and `granite` share one
  environment. `nemo` and `voxtral-vllm` pin their own torch builds: give each its own
  environment. The published Parakeet and Canary rows used NeMo from GitHub at commit
  `ca3f93a5`; to reproduce them exactly, install
  `"nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@ca3f93a516ff172e32951928aea6f96a977a7600"`.
  The PyPI release that the `nemo` extra installs decodes a few clips differently.
* **Voxtral Small 24B** runs with vLLM's FP8 weight-only quantization of the bf16 checkpoint
  (the published row used it); `precision="bf16"` needs about 55 GB of GPU memory and can score
  slightly differently.
* **Language.** Parakeet-TDT v3, Voxtral Mini 4B Realtime and Granite Speech have no language
  option; they follow the audio.
* **Hosted APIs** stream the audio at real time in 100 ms frames, as a microphone would, with
  at most 3 connections per provider. A full run of the four test subsets costs about $0.50
  (Soniox) to $1.60 (Deepgram Flux) at September 2026 prices. See
  [docs/remote-runners.md](docs/remote-runners.md) for what is sent, retries and costs.
* **MLX** is a convenience backend for Macs, not the reference: its transcripts can differ from
  the `transformers` Whisper runner on a few clips. Report its results under its own name.

## Methodology

**Default configuration.** No custom vocabulary, keyterms or context prompt; Spanish selected
where the system allows it; instruction-following models (for example Granite Speech, Gemma and
Phi-4) get only the fixed transcription instruction they need, the same for every clip. A
runner receives the 16 kHz float32 waveform and nothing else: no reference, no terms, no clip
metadata. Decoding follows the model card (greedy or its default beam, temperature 0); output
is capped so a loop cannot stall a run; the same settings are used for every subset. The
open-weights rows without a runner (note ²) were run by Omniloy under the same rules. The
settings of every published row are in [results/leaderboard.md](results/leaderboard.md) and
`results/summary.json`. Some systems have options that this setting leaves out: Soniox and
Deepgram offer custom vocabulary, keyterm or context features, and Whisper accepts a text
prompt. None was used; they could raise those systems' clinical scores. OmniScribe 2, a
results-only row, is the exception: it used patient-record context that included the spoken
medical terms (see note ¹ under the leaderboard).

**Normalization.** Reference, hypothesis and gold terms go through the same Spanish
normalizer (version `es1+lc1`): spelled-out letter names collapsed to the initialism ("eme
erre ene" -> "mrn"), lowercase, Unicode NFC, punctuation removed, **accents kept**, number
words written as digits and units after a number abbreviated ("quinientos miligramos" ->
"500 mg"). Alignment is word-level Levenshtein (substitutions S, deletions D, insertions I).

**Metrics** (corpus-level: ratios of sums over clips, never means of per-clip ratios):

| metric | definition |
|---|---|
| WER | (S + D + I) / reference words |
| term recall | clinical only: share of the gold medical terms of the reference that appear in full in the hypothesis (every word of the term correct, all or nothing) |
| B-WER | clinical only: error rate on the reference words that belong to gold terms, insertions of term words included (Le et al., 2021); function words ("de", "la", "con", ...) are never term words, even inside a multi-word term |
| U-WER | clinical only: the same on every other word, function words included |
| insertions / 1k | inserted words per 1,000 reference words (a proxy for hallucinated text) |
| degenerate | share of clips whose output is empty, loops, or runs away (more than twice the reference length plus 10 words) |

Missing clips and failed calls count as empty outputs and mark the result incomplete. Empty
and looping outputs are scored as they are.

**Confidence intervals and comparisons.** 95% percentile bootstrap, 2,000 resamples, seed 0,
the ratio of sums recomputed on each resample. The clinical subset contains several
renditions of some sentences, so its intervals resample whole sentences (`meta.text_id`); the
real-speech subsets resample clips. The CLI and the Python API choose this automatically
(`--block auto`, `block="auto"`); `--block clip` resamples clips on the clinical subset too,
which gives narrower intervals that ignore the shared sentences (point values do not change).
`fonendo compare` resamples both systems with the same blocks and reports the interval of the
difference; a difference is called significant only when that interval excludes zero. The
real-speech mean is the unweighted mean of the three WERs, with a stratified bootstrap.

**Clean vs degraded.** Clinical clips are split by `condition`: 147 clean (the text-to-speech
output) and 153 degraded (noise, codecs, reverberation, speed or pitch changes). The two
groups are mostly different sentences, so the split compares two groups of clips, not the same
clips getting worse.

**Caveats.**

* *Synthetic voices.* The clinical audio is text-to-speech. It lacks the hesitations,
  disfluencies, speaking rate and microphones of real clinicians, so absolute clinical error
  rates are likely optimistic; treat the clinical subset as a controlled test of medical
  vocabulary. The real-speech subsets are the counterpart with human speakers.
* *Streaming vs offline.* The three commercial APIs are real-time streaming services. The
  open models, and OmniScribe 2, transcribe the whole clip at once, with one exception:
  Nemotron 3.5 ASR Streaming ran in 1.12 s streaming chunks. Voxtral Mini 4B Realtime is a
  streaming model but was run on the whole clip offline (480 ms delay setting). Streaming
  recognizers decide with less right context, which usually costs accuracy, so compare the
  groups with that in mind.
* *Size.* 300 clinical clips (242 sentences) and 700 real-speech clips. Differences of a point
  or two of WER are often inside the interval; `clinical_dev` is for smoke tests only and is
  never used to choose settings.
* *Timing.* `secs` and `rtf_median` depend on hardware and, for the APIs, on the real-time
  pacing; they are informative only and not part of the leaderboard.

## Adding a model

1. Write a `Runner` subclass in `src/fonendo/runners/local/<family>.py` (open weights) or
   `src/fonendo/runners/remote/<provider>.py` (hosted API). Implement `load()` and
   `transcribe(audio, sr) -> str` (optionally `transcribe_batch`), use the default
   configuration, pin the model revision, and read any key from an environment variable.
2. Add its dependencies to an extra in `pyproject.toml` and register it with `lazy(...)` in
   `LOCAL_REGISTRY` or `REMOTE_REGISTRY`, so importing `fonendo` never imports the model's
   dependencies.
3. Smoke test: `fonendo run --model <name> --subset clinical_dev --limit 5` writes 5 lines, the
   same command again writes nothing, and `fonendo score` on the file looks sane.
4. Run every test subset (`clinical_test`, `fleurs_es`, `voxpopuli_es`,
   `mediaspeech_health`), then `fonendo report --published results/summary.json` to see the
   model next to the published ones.
5. Open a pull request with the runner and its docstring (model id, revision, tested versions
   and hardware, decoding settings). We re-run new models before adding them to the published
   leaderboard.

[CONTRACT.md](CONTRACT.md) is the full interface contract: runner rules, file formats, score
fields and the CLI.

## Repository layout

```
src/fonendo/        the package: data, fetch, text (normalizer), scoring, report, runners, cli
manifests/          clip selections of the public subsets (shipped with the package)
results/            the published leaderboard: summary.json, leaderboard.md
docs/               the showcase page (GitHub Pages) and docs/remote-runners.md
tests/              unit tests (pytest), no network and no model downloads
```

Development: `pip install -e ".[dev]"`, then `pytest` and `ruff check src tests`.

## Citation

```bibtex
@misc{omniloy2026fonendobench,
  title        = {fonendo-bench: a reproducible {S}panish clinical speech-to-text benchmark},
  author       = {{Omniloy}},
  year         = {2026},
  howpublished = {\url{https://github.com/Omniloy/fonendo-bench}},
  note         = {Clinical subsets: \url{https://huggingface.co/datasets/Omniloy/fonendo-bench}}
}
```

Please also cite the public datasets you use: FLEURS (Conneau et al., 2022), VoxPopuli (Wang
et al., 2021) and MediaSpeech (Kolobov et al., 2021).

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE). The clinical
dataset is distributed separately under the Omniloy fonendo-bench evaluation license (see the
[dataset card](https://huggingface.co/datasets/Omniloy/fonendo-bench)). Audio rebuilt by
`fonendo fetch` keeps the license of its source (FLEURS CC BY 4.0, VoxPopuli CC0, MediaSpeech
CC BY 4.0). Each model keeps its own license: the Supported models table above lists the
runnable ones, and the Systems table of [results/leaderboard.md](results/leaderboard.md) (and
`summary.json`) lists every leaderboard row. Commercial APIs are subject to their providers'
terms.

The 13 demo clips in `docs/audio/` are MP3 renditions of `clinical_test` clips, © Omniloy.
They are published for listening on the showcase and are not covered by the Apache-2.0 code
license; like the rest of the clinical set, they may not be redistributed or used to train
models (see the dataset license). Voices: ElevenLabs and Kokoro-82M (Apache-2.0). The
degraded clips mix in noise from MUSAN (Snyder, Chen and Povey, 2015; CC BY 4.0) and room
impulse responses from OpenSLR SLR28 (Ko et al., 2017; Apache-2.0).

## Contact

info@omniloy.com
