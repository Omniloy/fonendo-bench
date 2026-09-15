"""Soniox ``stt-rt-v5``: real-time streaming over a WebSocket (hosted API, no weights).

Registry name: ``soniox_stt_rt_v5`` (extra ``soniox``; key ``SONIOX_API_KEY``).

Endpoint, chosen with ``SONIOX_REGION`` (or the ``region=`` argument):

=======  ================================================
``us``   ``wss://stt-rt.soniox.com/transcribe-websocket`` (default)
``eu``   ``wss://stt-rt.eu.soniox.com/transcribe-websocket``
=======  ================================================

An API key belongs to the region of the Soniox project that created it; use the matching
region. The published fonendo-bench numbers were produced on the EU endpoint (the region used is
written to ``<subset>.run.json``). The region selects where the audio is processed, not the
model.

Protocol, one WebSocket per clip:

1. First (text) frame: the JSON configuration. Default configuration only::

       {"api_key": ..., "model": "stt-rt-v5", "audio_format": "pcm_s16le",
        "sample_rate": 16000, "num_channels": 1, "language_hints": ["es"],
        "language_hints_strict": true, "enable_endpoint_detection": false}

   Spanish is forced (``language_hints`` + ``language_hints_strict``). No ``context`` key is
   ever sent, no diarization, no language identification, no translation.
2. Binary frames: PCM16 16 kHz mono in 100 ms frames (3,200 bytes) sent on an absolute
   wall-clock schedule, frame ``i`` at ``t0 + i * 0.1 s``: exactly real time, no drift. Then
   0.5 s of digital silence at the same pace.
3. An EMPTY TEXT frame ends the audio. (An empty *binary* frame is ignored by the server, which
   then closes with ``408 request_timeout`` after about 20 s.) Messages are read until
   ``{"finished": true}``.

The transcript is the concatenation of the ``text`` of the tokens with ``is_final: true``, outer
whitespace stripped; the control tokens ``<end>`` and ``<fin>`` are dropped. Interim tokens never
reach the transcript. Endpoint detection is off and no manual finalize is sent, so Soniox
finalizes everything when the stream ends.

Reliability: connection errors, timeouts, HTTP 408/429/5xx on the upgrade, server errors with
those codes and a close without ``finished`` are retried 3 times (backoff 2, 4, 8 s + jitter).
Error codes 400, 401, 402, 403 and 413 fail at once. At most 3 connections are open at a time
in this process (:func:`~fonendo.runners.remote._common.provider_slot`).

Because audio is streamed at 1x, ``secs`` is about the clip duration + 0.6 s; the real-time
factor of this runner is not a speed measure. Billing is per second of streamed audio (clip +
0.5 s silence); see ``docs/remote-runners.md`` for the cost of a full run.

Tested with websockets 15.0.1, Python 3.12, September 2026.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import numpy as np
from websockets.asyncio.client import connect  # websockets>=13; missing -> install hint

from fonendo import SAMPLE_RATE
from fonendo.runners.base import Runner
from fonendo.runners.remote._common import (
    DEFAULT_BACKOFF_S,
    FatalError,
    RetryableError,
    chunks,
    env_value,
    provider_slot,
    run_coro,
    scrub,
    ssl_context,
    status_is_retryable,
    to_pcm16,
    websocket_body,
    websocket_status,
    with_retries,
)

ENDPOINTS = {
    "us": "wss://stt-rt.soniox.com/transcribe-websocket",
    "eu": "wss://stt-rt.eu.soniox.com/transcribe-websocket",
}
ENV_KEY = "SONIOX_API_KEY"
ENV_REGION = "SONIOX_REGION"

CHUNK_MS = 100
TRAIL_SILENCE_S = 0.5
FINISH_TIMEOUT_S = 60.0
FATAL_CODES = frozenset({400, 401, 402, 403, 413})
CONTROL_TOKENS = frozenset({"<end>", "<fin>"})
_REGION_HINT = f" (check {ENV_KEY} and that {ENV_REGION} matches the region of the key's project)"


class SonioxRealtimeRunner(Runner):
    """Soniox real-time WebSocket API, default configuration, Spanish forced."""

    provider = "soniox"

    def __init__(
        self,
        name: str = "soniox_stt_rt_v5",
        *,
        model_id: str = "stt-rt-v5",
        label: str | None = "Soniox stt-rt-v5",
        region: str | None = None,
        url: str | None = None,
        realtime: bool = True,
        max_concurrency: int = 3,
        backoff_s: tuple[float, ...] = DEFAULT_BACKOFF_S,
        device: str | None = None,  # accepted for CLI symmetry; unused (hosted API)
    ) -> None:
        super().__init__(
            name,
            "remote",
            extra="soniox",
            label=label,
            model_id=model_id,
            revision=None,
            env_vars=(ENV_KEY,),
            max_concurrency=max_concurrency,
        )
        if url:
            self.region, self.url = "custom", url
        else:
            self.region = (region or os.environ.get(ENV_REGION) or "us").strip().lower()
            if self.region not in ENDPOINTS:
                raise ValueError(f"{ENV_REGION} must be one of {sorted(ENDPOINTS)}")
            self.url = ENDPOINTS[self.region]
        self.realtime = realtime
        self.backoff_s = tuple(backoff_s)
        self._ssl = None

    # -- lifecycle -------------------------------------------------------------------

    def load(self) -> None:
        self._ssl = ssl_context() if self.url.startswith("wss://") else None

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "provider": "Soniox",
            "region": self.region,
            "endpoint": self.url if self.region != "custom" else "custom",
            "language": "es",
            "config": {
                "language_hints": ["es"],
                "language_hints_strict": True,
                "enable_endpoint_detection": False,
                "context": None,
            },
            "streaming": {
                "chunk_ms": CHUNK_MS,
                "pace": "1x real time" if self.realtime else "as fast as possible",
                "trailing_silence_s": TRAIL_SILENCE_S,
                "end_of_audio": "empty text frame",
                "transcript": "final tokens only",
            },
        }

    # -- inference -------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr}")
        key = env_value(ENV_KEY)
        pcm = to_pcm16(audio)

        def attempt() -> str:
            with provider_slot(self.provider):
                return run_coro(lambda: self._stream(pcm, key))

        return with_retries(attempt, backoff_s=self.backoff_s)

    async def _stream(self, pcm: bytes, key: str) -> str:
        config = {
            "api_key": key,
            "model": self.model_id,
            "audio_format": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
            "num_channels": 1,
            "language_hints": ["es"],
            "language_hints_strict": True,
            "enable_endpoint_detection": False,
        }
        frame_bytes = SAMPLE_RATE * CHUNK_MS // 1000 * 2
        frames = chunks(pcm, frame_bytes)
        frames += [b"\x00" * frame_bytes] * round(TRAIL_SILENCE_S * 1000 / CHUNK_MS)
        final: list[str] = []
        state = {"finished": False}

        try:
            async with connect(
                self.url,
                ssl=self._ssl,
                max_size=None,
                open_timeout=20,
                close_timeout=10,
                ping_interval=20,
                ping_timeout=20,
                compression=None,
            ) as ws:
                await ws.send(json.dumps(config, ensure_ascii=False))

                async def reader() -> None:
                    async for msg in ws:
                        if isinstance(msg, bytes):
                            continue
                        data = json.loads(msg)
                        if data.get("error_code") is not None:
                            code = int(data.get("error_code") or 0)
                            text = scrub(
                                f"{code} {data.get('error_type', '')}: "
                                f"{data.get('error_message', '')}",
                                key,
                            )
                            if code in FATAL_CODES:
                                hint = _REGION_HINT if code == 401 else ""
                                raise FatalError(f"Soniox error {text}{hint}")
                            raise RetryableError(f"Soniox error {text}")
                        for tok in data.get("tokens") or []:
                            txt = tok.get("text", "")
                            if tok.get("is_final") and txt not in CONTROL_TOKENS:
                                final.append(txt)
                        if data.get("finished"):
                            state["finished"] = True
                            return

                rtask = asyncio.create_task(reader())
                t0 = time.perf_counter()
                try:
                    for i, frame in enumerate(frames):
                        if self.realtime:
                            delay = t0 + i * CHUNK_MS / 1000.0 - time.perf_counter()
                            if delay > 0:
                                await asyncio.sleep(delay)
                        if rtask.done():  # the server answered with an error mid-stream
                            break
                        await ws.send(frame)
                    if not rtask.done():
                        await ws.send("")  # end of audio: an EMPTY TEXT frame
                    await asyncio.wait_for(rtask, timeout=FINISH_TIMEOUT_S)
                except Exception:
                    # an error message read by the reader explains a failed send: prefer it
                    if rtask.done() and not rtask.cancelled() and rtask.exception() is not None:
                        raise rtask.exception() from None  # type: ignore[misc]
                    raise
                finally:
                    if not rtask.done():
                        rtask.cancel()
        except (FatalError, RetryableError):
            raise
        except asyncio.TimeoutError as exc:
            raise RetryableError("Soniox: timeout waiting for `finished`") from exc
        except Exception as exc:  # handshake status, OSError, abnormal close
            code = websocket_status(exc)
            if code is not None:
                msg = scrub(f"Soniox: HTTP {code} on upgrade {websocket_body(exc)}", key)
                if code in FATAL_CODES or not status_is_retryable(code):
                    if code == 401:
                        msg += _REGION_HINT
                    raise FatalError(msg) from None
                raise RetryableError(msg) from None
            raise RetryableError(scrub(f"Soniox: {type(exc).__name__}: {exc}", key)) from None
        if not state["finished"]:
            raise RetryableError("Soniox: connection closed without `finished`")
        return "".join(final).strip()
