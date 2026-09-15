"""The EXPERIMENTAL openai-compatible runner against a tiny local mock server (no network)."""

from __future__ import annotations

import io
import json
import threading
import time
import wave
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from fonendo.data import read_jsonl
from fonendo.runners.base import run_subset
from fonendo.runners.remote import REMOTE_REGISTRY
from fonendo.runners.remote._common import FatalError, RetryableError, to_pcm16
from fonendo.runners.remote.openai_compatible import OpenAICompatibleRunner, endpoint_url

FAKE_KEY = "sk-fake-for-tests-0123456789"


class _Handler(BaseHTTPRequestHandler):
    server: _MockServer

    def log_message(self, *args) -> None:  # keep test output clean
        pass

    def _reply(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        srv = self.server
        body = self.rfile.read(int(self.headers["Content-Length"]))
        with srv.lock:
            srv.in_flight += 1
            srv.max_in_flight = max(srv.max_in_flight, srv.in_flight)
        try:
            time.sleep(srv.delay_s)
            msg = BytesParser(policy=email_policy).parsebytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body
            )
            fields, files = {}, {}
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if part.get_filename():
                    files[name] = part.get_payload(decode=True)
                else:
                    fields[name] = part.get_content().strip()
            record = {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "fields": fields,
                "files": files,
            }
            with srv.lock:
                srv.requests.append(record)
                fail = srv.fail_first > 0
                srv.fail_first -= fail
            if fail:
                self._reply(503, b'{"error": "busy"}')
            elif srv.status != 200:
                # echo the auth header back: the runner must scrub it from its error message
                self._reply(srv.status, json.dumps({"error": f"bad {record['auth']}"}).encode())
            else:
                with wave.open(io.BytesIO(files["file"])) as w:
                    n, rate, ch = w.getnframes(), w.getframerate(), w.getnchannels()
                text = f" hola {n} {rate} {ch} "
                if srv.plain_text:
                    self._reply(200, text.encode(), "text/plain; charset=utf-8")
                else:
                    self._reply(200, json.dumps({"text": text}).encode())
        finally:
            with srv.lock:
                srv.in_flight -= 1


class _MockServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.lock = threading.Lock()
        self.requests: list[dict] = []
        self.fail_first = 0
        self.status = 200
        self.plain_text = False
        self.delay_s = 0.0
        self.in_flight = 0
        self.max_in_flight = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture()
def server():
    srv = _MockServer()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def clean_env(monkeypatch):
    for var in (
        "FONENDO_OPENAI_BASE_URL",
        "FONENDO_OPENAI_MODEL",
        "FONENDO_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _audio(seconds: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pcm = rng.integers(-20000, 20000, int(16000 * seconds), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _runner(server: _MockServer, **kw) -> OpenAICompatibleRunner:
    r = OpenAICompatibleRunner(
        base_url=server.base_url, model="mock-asr", backoff_s=(0.0, 0.0, 0.0), **kw
    )
    r.ensure_loaded()
    return r


def test_endpoint_url():
    want = "http://h:8000/v1/audio/transcriptions"
    for base in ("http://h:8000", "http://h:8000/", "http://h:8000/v1", "http://h:8000/v1/"):
        assert endpoint_url(base) == want


def test_request_fields_and_audio(server, clean_env):
    clean_env.setenv("FONENDO_OPENAI_API_KEY", FAKE_KEY)
    audio = _audio(0.5)
    hyp = _runner(server).transcribe(audio, 16000)
    assert hyp == "hola 8000 16000 1"  # stripped; server saw 8,000 frames of 16 kHz mono
    (req,) = server.requests
    assert req["path"] == "/v1/audio/transcriptions"
    assert req["auth"] == f"Bearer {FAKE_KEY}"
    assert req["fields"] == {
        "model": "mock-asr",
        "language": "es",
        "temperature": "0",
        "response_format": "json",
    }  # default configuration: no prompt field
    with wave.open(io.BytesIO(req["files"]["file"])) as w:
        assert w.readframes(w.getnframes()) == to_pcm16(audio)  # bit-exact PCM16


def test_no_key_means_no_auth_header(server, clean_env):
    _runner(server).transcribe(_audio(0.1), 16000)
    assert server.requests[0]["auth"] is None


def test_fallback_key(server, clean_env):
    clean_env.setenv("OPENAI_API_KEY", FAKE_KEY)
    _runner(server).transcribe(_audio(0.1), 16000)
    assert server.requests[0]["auth"] == f"Bearer {FAKE_KEY}"


def test_retries_transient_errors(server, clean_env):
    server.fail_first = 2
    assert _runner(server).transcribe(_audio(0.1), 16000) == "hola 1600 16000 1"
    assert len(server.requests) == 3


def test_gives_up_after_three_retries(server, clean_env):
    server.fail_first = 10
    with pytest.raises(RetryableError, match="HTTP 503"):
        _runner(server).transcribe(_audio(0.1), 16000)
    assert len(server.requests) == 4


def test_client_error_is_fatal_and_scrubbed(server, clean_env):
    clean_env.setenv("FONENDO_OPENAI_API_KEY", FAKE_KEY)
    server.status = 400
    with pytest.raises(FatalError) as err:
        _runner(server).transcribe(_audio(0.1), 16000)
    assert len(server.requests) == 1  # never retried
    assert "HTTP 400" in str(err.value)
    assert FAKE_KEY not in str(err.value) and "***" in str(err.value)


def test_plain_text_reply(server, clean_env):
    server.plain_text = True
    assert _runner(server).transcribe(_audio(0.1), 16000) == "hola 1600 16000 1"


def test_connection_refused_is_retryable(clean_env):
    srv = _MockServer()
    base = srv.base_url
    srv.server_close()  # nothing listens on that port any more
    r = OpenAICompatibleRunner(base_url=base, model="m", backoff_s=(0.0,))
    r.ensure_loaded()
    with pytest.raises(RetryableError):
        r.transcribe(_audio(0.1), 16000)


def test_registry_env_config(server, clean_env):
    factory = REMOTE_REGISTRY["openai_compatible"]
    r = factory()
    with pytest.raises(RuntimeError, match="FONENDO_OPENAI_BASE_URL, FONENDO_OPENAI_MODEL"):
        r.ensure_loaded()
    clean_env.setenv("FONENDO_OPENAI_BASE_URL", server.base_url + "/v1")
    clean_env.setenv("FONENDO_OPENAI_MODEL", "mock-asr")
    r = factory(backoff_s=(0.0,))
    assert r.kind == "remote" and r.extra == "openai-compatible"
    assert r.label == "mock-asr (experimental)"
    r.ensure_loaded()
    assert r.transcribe(_audio(0.1), 16000) == "hola 1600 16000 1"


def test_run_subset_end_to_end(server, clean_env, tmp_path):
    clean_env.setenv("FONENDO_OPENAI_API_KEY", FAKE_KEY)
    server.delay_s = 0.05
    rows = [
        {"clip_id": f"c{i}", "audio": _audio(0.2, seed=i), "text": "", "terms": [], "meta": {}}
        for i in range(8)
    ]
    runner = OpenAICompatibleRunner(
        base_url=server.base_url, model="mock-asr", max_concurrency=9, backoff_s=(0.0,)
    )
    assert runner.max_concurrency == 3
    out = tmp_path / "raw" / "openai_compatible" / "clinical_dev.jsonl"
    run_subset(runner, rows, out, subset="clinical_dev", log=lambda _: None)
    lines = read_jsonl(out)
    assert sorted(r["clip_id"] for r in lines) == [f"c{i}" for i in range(8)]
    assert all(r["hyp"] == "hola 3200 16000 1" and "error" not in r for r in lines)
    assert server.max_in_flight <= 3
    meta_text = out.with_suffix(".run.json").read_text()
    meta = json.loads(meta_text)
    assert meta["experimental"] is True and meta["model_id"] == "mock-asr"
    assert FAKE_KEY not in meta_text and "127.0.0.1" not in meta_text
    n_requests = len(server.requests)
    run_subset(runner, rows, out, subset="clinical_dev", log=lambda _: None)  # resumable
    assert len(server.requests) == n_requests
