"""Deepgram streaming speech-to-text (hosted API, no weights): Nova-3 and Flux.

Registry names (extra ``deepgram``; key ``DEEPGRAM_API_KEY``, sent as ``Authorization: Token``):

* ``deepgram_nova3_es``   Nova-3, monolingual Spanish, ``/v1/listen`` streaming
* ``deepgram_flux_multi`` Flux Multilingual (``flux-general-multi``), ``/v2/listen`` streaming

Endpoint host, chosen with ``DEEPGRAM_REGION`` (or the ``region=`` argument):

=======  ========================================
``us``   ``wss://api.deepgram.com`` (default)
``eu``   ``wss://api.eu.deepgram.com`` (EU data residency)
=======  ========================================

The published fonendo-bench numbers were produced on the EU endpoint (the region used is
written to ``<subset>.run.json``). Both runners send the audio as PCM16 16 kHz mono in 100 ms
frames on an absolute wall-clock schedule (exactly real time), open one WebSocket per clip and
hold at most 3 connections at a time in this process; Nova-3 and Flux share the limit because
they share the key. No keyterm, keyword, search or replacement option is ever sent.

Retries: connection errors, timeouts, HTTP 408/429/5xx on the handshake, server errors and
streams that end without their closing messages are retried 3 times (backoff 2, 4, 8 s +
jitter). Other HTTP errors (400, 401, 402, 403) fail at once.

``secs`` is about the clip duration plus the trailing silence and the flush, because audio is
streamed at 1x; it is not a speed measure. Deepgram bills streamed audio per second; see
``docs/remote-runners.md`` for the cost of a full run.

Tested with websockets 15.0.1, Python 3.12, September 2026.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any
from urllib.parse import urlencode

import numpy as np
from websockets.asyncio.client import connect  # websockets>=13; missing -> install hint
from websockets.exceptions import ConnectionClosed

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

HOSTS = {"us": "wss://api.deepgram.com", "eu": "wss://api.eu.deepgram.com"}
ENV_KEY = "DEEPGRAM_API_KEY"
ENV_REGION = "DEEPGRAM_REGION"
CHUNK_MS = 100
FRAME_BYTES = SAMPLE_RATE * CHUNK_MS // 1000 * 2  # 3,200 bytes of PCM16


class _DeepgramStreamingRunner(Runner):
    """Shared plumbing: endpoint selection, key, limiter, retries, handshake classification."""

    provider = "deepgram"
    path = ""  # "/v1/listen" or "/v2/listen"

    def __init__(
        self,
        name: str,
        *,
        model_id: str,
        label: str | None,
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
            extra="deepgram",
            label=label,
            model_id=model_id,
            revision=None,
            env_vars=(ENV_KEY,),
            max_concurrency=max_concurrency,
        )
        if url:  # full listen URL without query (tests, proxies)
            self.region, self.url = "custom", url
        else:
            self.region = (region or os.environ.get(ENV_REGION) or "us").strip().lower()
            if self.region not in HOSTS:
                raise ValueError(f"{ENV_REGION} must be one of {sorted(HOSTS)}")
            self.url = HOSTS[self.region] + self.path
        self.realtime = realtime
        self.backoff_s = tuple(backoff_s)
        self._ssl = None

    def load(self) -> None:
        self._ssl = ssl_context() if self.url.startswith("wss://") else None

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "provider": "Deepgram",
            "region": self.region,
            "endpoint": self.url if self.region != "custom" else "custom",
            "language": "es",
        }

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr}")
        key = env_value(ENV_KEY)
        pcm = to_pcm16(audio)

        def attempt() -> str:
            with provider_slot(self.provider):
                try:
                    return run_coro(lambda: self._stream(pcm, key))
                except (FatalError, RetryableError):
                    raise
                except asyncio.TimeoutError as exc:
                    raise RetryableError(f"{self.label}: timeout") from exc
                except Exception as exc:  # handshake status, OSError, abnormal close
                    raise self._classify(exc, key) from None

        return with_retries(attempt, backoff_s=self.backoff_s)

    def _connect_kwargs(self, key: str) -> dict[str, Any]:
        return {
            "additional_headers": {"Authorization": f"Token {key}"},
            "ssl": self._ssl,
            "open_timeout": 20,
            "close_timeout": 5,
            "ping_interval": 20,
            "ping_timeout": 20,
            "max_size": 2**24,
        }

    def _classify(self, exc: BaseException, key: str) -> Exception:
        code = websocket_status(exc)
        if code is None:
            return RetryableError(scrub(f"{self.label}: {type(exc).__name__}: {exc}", key))
        msg = scrub(f"{self.label}: HTTP {code} on handshake {websocket_body(exc)}", key)
        if status_is_retryable(code):
            return RetryableError(msg)
        if code == 401:
            msg += f" (check {ENV_KEY})"
        return FatalError(msg)

    def _frames(self, pcm: bytes, trail_s: float) -> list[bytes]:
        return chunks(pcm, FRAME_BYTES) + [b"\x00" * FRAME_BYTES] * round(trail_s * 1000 / CHUNK_MS)

    async def _pace(self, t0: float, i: int) -> None:
        if self.realtime:
            delay = t0 + i * CHUNK_MS / 1000.0 - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

    async def _stream(self, pcm: bytes, key: str) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Nova-3
# --------------------------------------------------------------------------------------

NOVA3_PARAMS = (
    ("model", "nova-3"),
    ("language", "es"),
    ("encoding", "linear16"),
    ("sample_rate", str(SAMPLE_RATE)),
    ("channels", "1"),
    ("smart_format", "true"),
    ("punctuate", "true"),
    ("interim_results", "true"),
    ("endpointing", "300"),
)
NOVA3_TRAIL_S = 0.5
NOVA3_FINALIZE_WAIT_S = 10.0
NOVA3_CLOSE_WAIT_S = 15.0


class DeepgramNova3Runner(_DeepgramStreamingRunner):
    """Deepgram Nova-3, monolingual Spanish (``language=es``), live streaming.

    Query: ``model=nova-3 language=es encoding=linear16 sample_rate=16000 channels=1
    smart_format=true punctuate=true interim_results=true endpointing=300``: the settings of
    Deepgram's streaming quick start, with Spanish forced. ``smart_format`` only formats the text
    (numbers, dates, punctuation); scoring normalizes it.

    Per clip: the audio at 1x, then 0.5 s of digital silence, then ``{"type": "Finalize"}``;
    after its ``from_finalize`` response (or 10 s) ``{"type": "CloseStream"}``, and messages are
    read until the server's ``Metadata`` and close. The transcript is the ``is_final: true``
    ``Results`` transcripts joined with spaces, in arrival order (``speech_final`` segments
    included); interim results never reach it. The model version the server reports in
    ``Metadata.model_info`` is recorded in ``info()``.
    """

    path = "/v1/listen"

    def __init__(
        self,
        name: str = "deepgram_nova3_es",
        *,
        model_id: str = "nova-3",
        label: str | None = "Deepgram Nova-3 (es)",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, model_id=model_id, label=label, **kwargs)
        self._server_model_info: Any = None

    def info(self) -> dict[str, Any]:
        params = dict(NOVA3_PARAMS)
        params["model"] = self.model_id
        return {
            **super().info(),
            "query": params,
            "server_model_info": self._server_model_info,
            "streaming": {
                "chunk_ms": CHUNK_MS,
                "pace": "1x real time" if self.realtime else "as fast as possible",
                "trailing_silence_s": NOVA3_TRAIL_S,
                "end_of_audio": "Finalize, then CloseStream",
                "transcript": "is_final results only",
            },
        }

    async def _stream(self, pcm: bytes, key: str) -> str:
        params = [(k, self.model_id if k == "model" else v) for k, v in NOVA3_PARAMS]
        url = f"{self.url}?{urlencode(params)}"
        frames = self._frames(pcm, NOVA3_TRAIL_S)
        finals: list[str] = []
        st: dict[str, Any] = {"metadata": None, "finalized": False}
        finalize_evt = asyncio.Event()

        async with connect(url, **self._connect_kwargs(key)) as ws:

            async def receiver() -> None:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        m = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    typ = m.get("type")
                    if typ == "Results":
                        alts = (m.get("channel") or {}).get("alternatives") or [{}]
                        text = (alts[0].get("transcript") or "").strip()
                        if m.get("is_final"):
                            if text:
                                finals.append(text)
                            if m.get("from_finalize"):
                                st["finalized"] = True
                                finalize_evt.set()
                    elif typ == "Metadata":
                        st["metadata"] = m

            rtask = asyncio.create_task(receiver())
            t0 = time.perf_counter()
            try:
                for i, frame in enumerate(frames):
                    await self._pace(t0, i)
                    if rtask.done():
                        break
                    await ws.send(frame)
                if not rtask.done():
                    await ws.send(json.dumps({"type": "Finalize"}))
                    try:
                        await asyncio.wait_for(finalize_evt.wait(), NOVA3_FINALIZE_WAIT_S)
                    except asyncio.TimeoutError:
                        pass  # CloseStream below still flushes the remaining audio
                    await ws.send(json.dumps({"type": "CloseStream"}))
                await asyncio.wait_for(rtask, NOVA3_CLOSE_WAIT_S)
            finally:
                if not rtask.done():
                    rtask.cancel()
            close_code = ws.close_code

        metadata = st["metadata"]
        if close_code not in (1000, None) and not finals and metadata is None:
            raise RetryableError(f"{self.label}: abnormal close {close_code}")
        if metadata is None and not st["finalized"]:
            raise RetryableError(f"{self.label}: stream ended without Metadata or Finalize reply")
        if metadata and metadata.get("model_info"):
            self._server_model_info = metadata["model_info"]
        return " ".join(finals).strip()


# --------------------------------------------------------------------------------------
# Flux
# --------------------------------------------------------------------------------------

FLUX_PARAMS = (
    ("model", "flux-general-multi"),
    ("language_hint", "es"),
    ("encoding", "linear16"),
    ("sample_rate", str(SAMPLE_RATE)),
)
FLUX_PAD_S = 1.0
FLUX_FLUSH_WAIT_S = 6.0
FLUX_CLOSE_WAIT_S = 8.0


class DeepgramFluxRunner(_DeepgramStreamingRunner):
    """Deepgram Flux Multilingual (``flux-general-multi``), ``/v2/listen``, Spanish hint.

    Query: ``model=flux-general-multi language_hint=es encoding=linear16 sample_rate=16000``.
    Flux defaults are kept (``eot_threshold`` 0.7, ``eot_timeout_ms`` 5000, no eager end of
    turn); no ``Configure`` message is sent. Flux has no smart formatting: it writes numbers as
    words and marks a word cut at the end of the audio with a trailing ``--``; the text is
    returned as produced (scoring normalizes it).

    Per clip: the audio at 1x, then 1.0 s of digital silence (Flux's text lags the audio by about
    1 s; with 0.5 s the flush cut the last word), then, only if a turn is still open,
    ``{"type": "ForceEndTurn"}`` and a wait (<= 6 s) for its ``EndOfTurn``; then
    ``{"type": "CloseStream"}`` and messages are read until the server closes. ``CloseStream``
    alone does not emit an ``EndOfTurn`` (the last message would be an interim ``Update``),
    hence the explicit ``ForceEndTurn``.

    The transcript is the ``EndOfTurn`` transcripts in ``turn_index`` order joined with spaces;
    a turn still open when the socket closes contributes its last ``Update`` transcript.
    A clip in which Flux never opens a turn returns an empty transcript (a valid output that
    scoring counts as degenerate, not an error).
    """

    path = "/v2/listen"

    def __init__(
        self,
        name: str = "deepgram_flux_multi",
        *,
        model_id: str = "flux-general-multi",
        label: str | None = "Deepgram Flux Multilingual",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, model_id=model_id, label=label, **kwargs)

    def info(self) -> dict[str, Any]:
        params = dict(FLUX_PARAMS)
        params["model"] = self.model_id
        return {
            **super().info(),
            "query": params,
            "streaming": {
                "chunk_ms": CHUNK_MS,
                "pace": "1x real time" if self.realtime else "as fast as possible",
                "trailing_silence_s": FLUX_PAD_S,
                "end_of_audio": "ForceEndTurn if a turn is open, then CloseStream",
                "transcript": "EndOfTurn transcripts (last Update of a turn left open)",
            },
        }

    async def _stream(self, pcm: bytes, key: str) -> str:
        params = [(k, self.model_id if k == "model" else v) for k, v in FLUX_PARAMS]
        url = f"{self.url}?{urlencode(params)}"
        frames = self._frames(pcm, FLUX_PAD_S)
        st: dict[str, Any] = {
            "finals": {},  # turn_index -> EndOfTurn transcript
            "open": {},  # turn_index -> last transcript of a turn without EndOfTurn yet
            "error": None,
            "close": None,
        }
        changed = asyncio.Event()

        ws = await connect(url, **self._connect_kwargs(key))

        async def receiver() -> None:
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        m = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    typ = m.get("type")
                    if typ == "TurnInfo":
                        event = m.get("event")
                        ti = int(m.get("turn_index", 0))
                        text = (m.get("transcript") or "").strip()
                        if event == "EndOfTurn":
                            st["finals"][ti] = text
                            st["open"].pop(ti, None)
                        elif ti not in st["finals"] and (
                            event != "Update" or text or ti in st["open"]
                        ):
                            # a turn is open once StartOfTurn (or any text) arrived for it; the
                            # empty Updates Flux streams between turns do not open one
                            st["open"][ti] = text
                        changed.set()
                    elif typ == "Error":
                        st["error"] = m
                        changed.set()
            except ConnectionClosed as exc:
                st["close"] = f"{type(exc).__name__}: {exc}"
            changed.set()

        rtask = asyncio.create_task(receiver())
        try:
            t0 = time.perf_counter()
            for i, frame in enumerate(frames):
                await self._pace(t0, i)
                if rtask.done():
                    why = st["close"] or st["error"]
                    raise RetryableError(
                        scrub(f"{self.label}: socket closed mid-stream ({why})", key)
                    )
                await ws.send(frame)
            if st["open"] and not rtask.done():
                await ws.send(json.dumps({"type": "ForceEndTurn"}))
                deadline = time.perf_counter() + FLUX_FLUSH_WAIT_S
                while st["open"] and not rtask.done():
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    changed.clear()
                    try:
                        await asyncio.wait_for(changed.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
            if not rtask.done():
                await ws.send(json.dumps({"type": "CloseStream"}))
            try:
                await asyncio.wait_for(asyncio.shield(rtask), timeout=FLUX_CLOSE_WAIT_S)
            except asyncio.TimeoutError:
                pass  # keep what arrived; a turn left open contributes its last Update
        except ConnectionClosed as exc:
            raise RetryableError(
                scrub(f"{self.label}: socket closed while sending: {exc}", key)
            ) from None
        finally:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - already closed
                pass
            if not rtask.done():
                rtask.cancel()
                try:
                    await rtask
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

        if st["error"]:
            raise RetryableError(
                scrub(f"{self.label}: server Error {json.dumps(st['error'])}", key)
            )
        parts = list(st["finals"].items()) + [(ti, t) for ti, t in st["open"].items() if t]
        return " ".join(text for _, text in sorted(parts) if text).strip()
