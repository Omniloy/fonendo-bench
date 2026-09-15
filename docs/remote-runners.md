# Remote runners (hosted APIs)

fonendo-bench evaluates three hosted speech-to-text services through their public APIs. There is
also an experimental runner for any server that speaks the OpenAI audio API. They all
live in `src/fonendo/runners/remote/`.

| `--model` | service | extra | environment |
|---|---|---|---|
| `soniox_stt_rt_v5` | Soniox `stt-rt-v5`, real-time WebSocket | `soniox` | `SONIOX_API_KEY`, optional `SONIOX_REGION` (`us` default, `eu`) |
| `deepgram_nova3_es` | Deepgram Nova-3, Spanish, live streaming | `deepgram` | `DEEPGRAM_API_KEY`, optional `DEEPGRAM_REGION` (`us` default, `eu`) |
| `deepgram_flux_multi` | Deepgram Flux Multilingual (`flux-general-multi`), live streaming | `deepgram` | `DEEPGRAM_API_KEY`, optional `DEEPGRAM_REGION` |
| `openai_compatible` | **experimental**: any `/v1/audio/transcriptions` server | `openai-compatible` | `FONENDO_OPENAI_BASE_URL`, `FONENDO_OPENAI_MODEL`, optional `FONENDO_OPENAI_API_KEY` |

```bash
pip install "fonendo[soniox]"            # or fonendo[deepgram], fonendo[openai-compatible]
export SONIOX_API_KEY=...                # read at run time; never logged or written
export SONIOX_REGION=eu                  # the region of the project that owns the key
fonendo run --model soniox_stt_rt_v5 --subset clinical_dev --limit 5   # smoke test, ~1 min
fonendo run --model soniox_stt_rt_v5 --subset clinical_test
```

## What is sent

Every runner uses the service's **default configuration**: no custom vocabulary, keyterms or
context prompt, and Spanish selected (Nova-3: `language=es`; Soniox: a strict Spanish
language hint; Flux: a Spanish language hint). No per-clip information is sent. The exact request settings are also written
to `<subset>.run.json`.

* **Soniox**: the first frame is `{"model": "stt-rt-v5", "audio_format": "pcm_s16le",
  "sample_rate": 16000, "num_channels": 1, "language_hints": ["es"], "language_hints_strict":
  true, "enable_endpoint_detection": false}` plus the key. Then the audio, then 0.5 s of
  silence. An **empty text frame** ends the stream (an empty binary frame is ignored by the
  server, which then times out with 408). The transcript is the text of the final tokens only,
  without the `<end>`/`<fin>` control tokens.
* **Deepgram Nova-3**: `wss://…/v1/listen?model=nova-3&language=es&encoding=linear16&
  sample_rate=16000&channels=1&smart_format=true&punctuate=true&interim_results=true&
  endpointing=300`. Then the audio and 0.5 s of silence, then `Finalize`, then `CloseStream`
  once the `from_finalize` result has arrived. The transcript is the `is_final` results only.
* **Deepgram Flux**: `wss://…/v2/listen?model=flux-general-multi&language_hint=es&
  encoding=linear16&sample_rate=16000`, with Flux's default end-of-turn settings. Then the audio
  and 1.0 s of silence, then `ForceEndTurn` if a turn is still open, then `CloseStream`
  (`CloseStream` alone never emits the last `EndOfTurn`). The transcript is the `EndOfTurn`
  transcripts in turn order. A turn still open at the close contributes its last update. When
  Flux never opens a turn, the transcript is empty: that is a valid output, and scoring counts
  it as degenerate.

Audio is PCM16 16 kHz mono (the loader's float32 samples × 32768, bit-exact for PCM16
sources). It is sent in 100 ms frames on an absolute wall-clock schedule, which is **exactly
real time**, the way a live microphone would send it. As a result `secs` (and so
`rtf_median`) is about 1.0 for these runners by construction. It measures the pacing, not the
service's speed.

## Concurrency, retries, errors

* **At most 3 open connections per provider per process.** The run loop uses at most 3
  worker threads for a remote runner, and each connection also takes one of 3 slots of a
  process-wide limiter shared by every runner of that provider. Nova-3 and Flux share one
  limit because they share the key. A slot is held only while a connection is open, never
  during a retry backoff. The limiter does not coordinate separate processes, so run a single
  `fonendo run` per API key at a time (run the subsets one after another).
* **Retries**: connection errors, timeouts, HTTP 408/429/5xx, server-side errors and streams
  that end without their closing message are retried 3 times inside `transcribe` (backoff 2,
  4 and 8 s plus jitter). Authentication, payment and bad-request errors (400/401/402/403,
  Soniox 413) fail immediately, and the error names the environment variable to check.
* A clip that still fails is written as `{"hyp": "", "error": ...}`, and the next
  `fonendo run` with the same arguments retries only those clips. Finished clips are never
  sent twice.
* Keys come only from the environment. They are placed in the one request field that needs
  them (Soniox: the first WebSocket frame; Deepgram: the `Authorization` header; OpenAI-style:
  the bearer token) and are scrubbed from every error message.

## Cost and time of a full run

The four `test` subsets hold 1,000 clips and **3.12 h of audio** (clinical_test 0.64 h,
fleurs_es 1.04 h, voxpopuli_es 0.64 h, mediaspeech_health 0.80 h). The trailing silence is
streamed and billed too, which gives about 3.26 h billed for Soniox and Nova-3 and about 3.4 h
of connection time for Flux. `clinical_dev` (60 clips, 0.13 h) costs a few cents.

| service | price used for the estimate (September 2026) | 4 test subsets | wall time at 3 streams |
|---|---|---:|---:|
| Soniox `stt-rt-v5` | about $0.14 per hour of streamed audio (audio plus output text, as billed in our runs) | about $0.50 | about 1 h 10 min |
| Deepgram Nova-3 (streaming, pay as you go) | $0.0077 per minute | about $1.50 | about 1 h 10 min |
| Deepgram Flux (streaming, pay as you go) | about $0.0078 per minute | about $1.60 | about 1 h 15 min |

These are estimates. Prices, free credits and promotions change, and failed attempts are
billed for the time they were connected. Check the vendor's pricing page before a large run.
A 5-clip smoke test costs well under one cent per service.

## Reproducibility

Hosted models are updated by their vendors, and streaming recognizers are not bit-for-bit
deterministic. Their output can depend on network timing. Nova-3 runs record the model
version the server reports (`server_model_info` in `run.json`); Soniox pins the model by name
(`stt-rt-v5`). In September 2026 we re-ran 5 `clinical_test` clips per service with these
runners against the transcripts of the published runs. Nova-3 and Flux matched on 5 of 5 clips. Soniox
matched on 4 of 5; the fifth, a noisy clip, differed in its last word. The region (`us` or
`eu`) selects where audio is processed, not the model. The published runs used the EU
endpoints, and the region is written to `run.json`.

## Experimental: OpenAI-compatible servers

`openai_compatible` posts each clip as a WAV file to `{FONENDO_OPENAI_BASE_URL}/v1/audio/
transcriptions` with `model=$FONENDO_OPENAI_MODEL`, `language=es`, `temperature=0` and
`response_format=json`, and no prompt. It works with self-hosted servers that implement this
endpoint (for example vLLM serving a speech model it supports) and with hosted OpenAI-style
APIs:

```bash
pip install "fonendo[openai-compatible]"      # standard library only
export FONENDO_OPENAI_BASE_URL=http://localhost:8000
export FONENDO_OPENAI_MODEL=<model name as the server knows it>
fonendo run --model openai_compatible --subset clinical_dev --limit 5 \
    --out results/raw/<your_model>/clinical_dev.jsonl
```

It is experimental because servers differ in which form fields they honour, in their default
decoding and in how they chunk long audio. Its results describe that server's configuration
as much as the model, and no published fonendo-bench number comes from it. `run.json` records
the model name but never the server address or the key; record the server and its version
yourself. Local servers are not billed, but they use the same 3-request concurrency cap.
