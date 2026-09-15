"""Soniox and Deepgram streaming runners against local mock WebSocket servers (no network).

The mocks follow the documented message flow of each API closely enough to check what the
runners send (configuration, query, audio bytes, end-of-stream messages) and which server
messages reach the transcript (final text only).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pytest

pytest.importorskip("websockets")

from websockets.asyncio.server import serve  # noqa: E402

from fonendo.runners.remote._common import FatalError, RetryableError, to_pcm16  # noqa: E402
from fonendo.runners.remote.deepgram import DeepgramFluxRunner, DeepgramNova3Runner  # noqa: E402
from fonendo.runners.remote.soniox import SonioxRealtimeRunner  # noqa: E402

FAKE_KEY = "fake-key-for-tests-abcdef"
FRAME = 3200


class MockWS:
    """A WebSocket server on 127.0.0.1 running ``handler(ws, mock)`` on its own event loop."""

    def __init__(self, handler, process_request=None) -> None:
        self.handler = handler
        self.process_request = process_request
        self.log: list[dict] = []
        self.active = 0
        self.max_active = 0
        self.connections = 0
        self.options: dict = {}
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(5)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)

        async def main() -> None:
            async def wrapped(ws) -> None:
                with self._lock:
                    self.connections += 1
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    await self.handler(ws, self)
                finally:
                    with self._lock:
                        self.active -= 1

            async def count_handshake(connection, request):
                if self.process_request is None:
                    return None
                with self._lock:
                    self.connections += 1
                return self.process_request(connection, request)

            async with serve(wrapped, "127.0.0.1", 0, process_request=count_handshake) as server:
                self.port = server.sockets[0].getsockname()[1]
                self._stop = asyncio.Event()
                self._ready.set()
                await self._stop.wait()

        self._loop.run_until_complete(main())

    def url(self, path: str) -> str:
        return f"ws://127.0.0.1:{self.port}{path}"

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(5)


def _audio(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(1)
    pcm = rng.integers(-20000, 20000, int(16000 * seconds), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


@pytest.fixture()
def keys(monkeypatch):
    monkeypatch.setenv("SONIOX_API_KEY", FAKE_KEY)
    monkeypatch.setenv("DEEPGRAM_API_KEY", FAKE_KEY)


def _loaded(runner):
    runner.ensure_loaded()
    return runner


# --------------------------------------------------------------------------------------
# Soniox
# --------------------------------------------------------------------------------------


async def soniox_handler(ws, mock: MockWS) -> None:
    cfg = json.loads(await ws.recv())
    rec = {"config": cfg, "audio_bytes": 0, "empty_binary": 0, "ended": False}
    mock.log.append(rec)
    if cfg.get("api_key") != FAKE_KEY:
        await ws.send(json.dumps({"error_code": 401, "error_message": f"bad {cfg.get('api_key')}"}))
        return
    if mock.options.get("drop_first") and mock.connections == 1:
        return  # close without `finished`
    async for msg in ws:
        if isinstance(msg, bytes):
            if not msg:
                rec["empty_binary"] += 1
                continue
            first = rec["audio_bytes"] == 0
            rec["audio_bytes"] += len(msg)
            if first:
                await ws.send(json.dumps({"tokens": [{"text": "Ola", "is_final": False}]}))
                await ws.send(json.dumps({"tokens": [{"text": " Hola", "is_final": True}]}))
        elif msg == "":
            rec["ended"] = True
            break
    await asyncio.sleep(mock.options.get("finish_delay", 0.0))
    toks = [
        {"text": ",", "is_final": True},
        {"text": " mundo", "is_final": True},
        {"text": "<end>", "is_final": True},
        {"text": " interino", "is_final": False},
    ]
    await ws.send(json.dumps({"tokens": toks, "total_audio_proc_ms": 1000}))
    await ws.send(json.dumps({"tokens": [], "finished": True}))


def test_soniox_default_config_and_final_text(keys):
    mock = MockWS(soniox_handler)
    try:
        runner = _loaded(SonioxRealtimeRunner(url=mock.url("/ws"), realtime=False))
        audio = _audio(0.75)
        assert runner.transcribe(audio, 16000) == "Hola, mundo"
        (rec,) = mock.log
        assert rec["config"] == {
            "api_key": FAKE_KEY,
            "model": "stt-rt-v5",
            "audio_format": "pcm_s16le",
            "sample_rate": 16000,
            "num_channels": 1,
            "language_hints": ["es"],
            "language_hints_strict": True,
            "enable_endpoint_detection": False,
        }  # no `context`, ever
        assert rec["audio_bytes"] == len(to_pcm16(audio)) + 5 * FRAME  # clip + 0.5 s silence
        assert rec["ended"] and rec["empty_binary"] == 0  # ended by an EMPTY TEXT frame
        info = json.dumps(runner.info())
        assert FAKE_KEY not in info
    finally:
        mock.close()


def test_soniox_bad_key_is_fatal_and_scrubbed(monkeypatch):
    monkeypatch.setenv("SONIOX_API_KEY", "wrong-" + FAKE_KEY)
    mock = MockWS(soniox_handler)
    try:
        runner = _loaded(SonioxRealtimeRunner(url=mock.url("/ws"), realtime=False))
        with pytest.raises(FatalError) as err:
            runner.transcribe(_audio(0.2), 16000)
        assert "401" in str(err.value) and FAKE_KEY not in str(err.value)
        assert len(mock.log) == 1  # not retried
    finally:
        mock.close()


def test_soniox_retries_close_without_finished(keys):
    mock = MockWS(soniox_handler)
    mock.options["drop_first"] = True
    try:
        runner = _loaded(
            SonioxRealtimeRunner(url=mock.url("/ws"), realtime=False, backoff_s=(0.0, 0.0))
        )
        assert runner.transcribe(_audio(0.3), 16000) == "Hola, mundo"
        assert mock.connections == 2
    finally:
        mock.close()


def test_soniox_realtime_pacing(keys):
    mock = MockWS(soniox_handler)
    try:
        runner = _loaded(SonioxRealtimeRunner(url=mock.url("/ws")))
        t0 = time.perf_counter()
        runner.transcribe(_audio(0.3), 16000)
        # 3 clip frames + 5 silence frames, frame i sent at t0 + i * 0.1 s -> >= 0.7 s
        assert time.perf_counter() - t0 >= 0.69
    finally:
        mock.close()


def test_provider_limit_caps_connections(keys):
    mock = MockWS(soniox_handler)
    mock.options["finish_delay"] = 0.2
    try:
        runner = _loaded(SonioxRealtimeRunner(url=mock.url("/ws"), realtime=False))
        threads = [
            threading.Thread(target=runner.transcribe, args=(_audio(0.1), 16000)) for _ in range(7)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        assert mock.connections == 7 and mock.max_active <= 3
    finally:
        mock.close()


# --------------------------------------------------------------------------------------
# Deepgram Nova-3
# --------------------------------------------------------------------------------------


async def nova3_handler(ws, mock: MockWS) -> None:
    req = urlsplit(ws.request.path)
    rec = {
        "path": req.path,
        "query": parse_qsl(req.query),
        "auth": ws.request.headers.get("Authorization"),
        "audio_bytes": 0,
        "control": [],
    }
    mock.log.append(rec)
    sent_first = False
    async for msg in ws:
        if isinstance(msg, bytes):
            rec["audio_bytes"] += len(msg)
            if not sent_first:
                sent_first = True
                alt = {"channel": {"alternatives": [{"transcript": "primer"}]}}
                await ws.send(json.dumps({"type": "Results", "is_final": False, **alt}))
                alt = {"channel": {"alternatives": [{"transcript": "Primer segmento."}]}}
                await ws.send(json.dumps({"type": "Results", "is_final": True, **alt}))
            continue
        m = json.loads(msg)
        rec["control"].append(m["type"])
        if m["type"] == "Finalize":
            alt = {"channel": {"alternatives": [{"transcript": "Segundo."}]}}
            await ws.send(
                json.dumps({"type": "Results", "is_final": True, "from_finalize": True, **alt})
            )
        elif m["type"] == "CloseStream":
            info = {"u1": {"name": "general-nova-3", "version": "2099-01-01.0", "arch": "nova-3"}}
            await ws.send(
                json.dumps(
                    {"type": "Metadata", "request_id": "r1", "duration": 1.0, "model_info": info}
                )
            )
            await ws.close()
            return


def test_nova3_query_protocol_and_text(keys):
    mock = MockWS(nova3_handler)
    try:
        runner = _loaded(DeepgramNova3Runner(url=mock.url("/v1/listen"), realtime=False))
        audio = _audio(0.5)
        assert runner.transcribe(audio, 16000) == "Primer segmento. Segundo."
        (rec,) = mock.log
        assert rec["path"] == "/v1/listen"
        assert rec["auth"] == f"Token {FAKE_KEY}"
        assert rec["query"] == [
            ("model", "nova-3"),
            ("language", "es"),
            ("encoding", "linear16"),
            ("sample_rate", "16000"),
            ("channels", "1"),
            ("smart_format", "true"),
            ("punctuate", "true"),
            ("interim_results", "true"),
            ("endpointing", "300"),
        ]  # no keyterm / keywords / search / replace
        assert rec["audio_bytes"] == len(to_pcm16(audio)) + 5 * FRAME
        assert rec["control"] == ["Finalize", "CloseStream"]
        info = runner.info()
        assert info["server_model_info"]["u1"]["version"] == "2099-01-01.0"
        assert FAKE_KEY not in json.dumps(info)
    finally:
        mock.close()


def _rejecting_server(status: int) -> MockWS:
    """Rejects every handshake with ``status``, echoing the Authorization header in the body."""
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    def process_request(connection, request):
        body = f"echo {request.headers.get('Authorization')}".encode()
        return Response(status, "err", Headers([("Content-Length", str(len(body)))]), body)

    async def never(ws, mock):  # handshakes are all rejected
        await ws.close()

    return MockWS(never, process_request)


@pytest.mark.parametrize(("status", "fatal"), [(401, True), (400, True), (503, False)])
def test_deepgram_handshake_errors(keys, status, fatal):
    mock = _rejecting_server(status)
    try:
        runner = _loaded(
            DeepgramNova3Runner(url=mock.url("/v1/listen"), realtime=False, backoff_s=(0.0, 0.0))
        )
        with pytest.raises(FatalError if fatal else RetryableError) as err:
            runner.transcribe(_audio(0.2), 16000)
        assert f"HTTP {status}" in str(err.value)
        assert FAKE_KEY not in str(err.value)
        assert mock.connections == (1 if fatal else 3)
    finally:
        mock.close()


# --------------------------------------------------------------------------------------
# Deepgram Flux
# --------------------------------------------------------------------------------------
# Paced at 1x (short clips): whether a turn is open when the audio ends depends on server
# messages arriving during the stream, as with the real service.


def _turn(event: str, ti: int, text: str) -> str:
    return json.dumps({"type": "TurnInfo", "event": event, "turn_index": ti, "transcript": text})


async def flux_handler(ws, mock: MockWS) -> None:
    req = urlsplit(ws.request.path)
    rec = {"path": req.path, "query": parse_qsl(req.query), "audio_bytes": 0, "control": []}
    mock.log.append(rec)
    mode = mock.options.get("mode", "normal")
    await ws.send(json.dumps({"type": "Connected", "request_id": "r1"}))
    frames = 0
    async for msg in ws:
        if isinstance(msg, bytes):
            rec["audio_bytes"] += len(msg)
            frames += 1
            if mode == "silent":
                if frames == 1:
                    await ws.send(_turn("Update", 0, ""))  # empty Updates do not open a turn
                continue
            if frames == 1:
                await ws.send(_turn("StartOfTurn", 0, ""))
                await ws.send(_turn("Update", 0, "hola"))
            elif frames == 2:
                await ws.send(_turn("EndOfTurn", 0, "Hola."))
            elif frames == 3:
                await ws.send(_turn("StartOfTurn", 1, ""))
                await ws.send(_turn("Update", 1, "qué"))
            elif frames == 4:
                await ws.send(_turn("Update", 1, "qué tal"))
            continue
        m = json.loads(msg)
        rec["control"].append(m["type"])
        if m["type"] == "ForceEndTurn" and mode == "normal":
            await ws.send(_turn("EndOfTurn", 1, "¿Qué tal?"))
        elif m["type"] == "CloseStream":
            await ws.close()
            return


def test_flux_force_end_turn_and_text(keys):
    mock = MockWS(flux_handler)
    try:
        runner = _loaded(DeepgramFluxRunner(url=mock.url("/v2/listen")))
        audio = _audio(0.3)
        assert runner.transcribe(audio, 16000) == "Hola. ¿Qué tal?"
        (rec,) = mock.log
        assert rec["path"] == "/v2/listen"
        assert rec["query"] == [
            ("model", "flux-general-multi"),
            ("language_hint", "es"),
            ("encoding", "linear16"),
            ("sample_rate", "16000"),
        ]
        assert rec["audio_bytes"] == len(to_pcm16(audio)) + 10 * FRAME  # clip + 1.0 s silence
        assert rec["control"] == ["ForceEndTurn", "CloseStream"]  # no Configure message
    finally:
        mock.close()


def test_flux_open_turn_falls_back_to_last_update(keys, monkeypatch):
    import fonendo.runners.remote.deepgram as dg

    monkeypatch.setattr(dg, "FLUX_FLUSH_WAIT_S", 0.3)
    mock = MockWS(flux_handler)
    mock.options["mode"] = "no_eot"  # ForceEndTurn is never answered
    try:
        runner = _loaded(DeepgramFluxRunner(url=mock.url("/v2/listen")))
        assert runner.transcribe(_audio(0.3), 16000) == "Hola. qué tal"
        assert mock.log[0]["control"] == ["ForceEndTurn", "CloseStream"]
    finally:
        mock.close()


def test_flux_no_turn_returns_empty(keys):
    mock = MockWS(flux_handler)
    mock.options["mode"] = "silent"
    try:
        runner = _loaded(DeepgramFluxRunner(url=mock.url("/v2/listen")))
        assert runner.transcribe(_audio(0.3), 16000) == ""
        assert mock.log[0]["control"] == ["CloseStream"]  # nothing to flush
    finally:
        mock.close()
