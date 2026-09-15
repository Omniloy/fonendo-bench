"""EXPERIMENTAL: any server that implements the OpenAI ``/v1/audio/transcriptions`` endpoint.

Registry name: ``openai_compatible`` (extra ``openai-compatible``; standard library only).

Works with self-hosted servers that expose the OpenAI audio API (for example vLLM's
``vllm serve <model>`` for speech models it supports) and with hosted OpenAI-style APIs.
Configuration comes from the environment (or the matching constructor arguments):

=============================  ==========================================================
``FONENDO_OPENAI_BASE_URL``    server root, e.g. ``http://localhost:8000``; a trailing
                               ``/v1`` is accepted (required)
``FONENDO_OPENAI_MODEL``       the ``model`` form field, as the server names it (required)
``FONENDO_OPENAI_API_KEY``     bearer token (optional; falls back to ``OPENAI_API_KEY``;
                               omitted when neither is set, e.g. a local server)
=============================  ==========================================================

Request, one per clip: ``POST {base_url}/v1/audio/transcriptions`` as ``multipart/form-data``
with ``file`` (the clip as a 16 kHz mono PCM16 WAV), ``model``, ``language=es``,
``temperature=0`` and ``response_format=json``. No ``prompt`` is ever sent. The transcript is
the ``text`` field of the JSON reply (a plain-text reply is used as is).

Retries: connection errors, timeouts and HTTP 408/425/429/5xx are retried 3 times (backoff 2,
4, 8 s + jitter); other HTTP errors fail at once. At most 3 requests are in flight in this
process.

Why experimental: servers differ in which form fields they honour (some ignore ``language`` or
``temperature``), in their default decoding and in how they chunk long audio, so a result
obtained through this runner describes that server's configuration as much as the model. It is
not used for any published fonendo-bench result. Name your results folder after the model
(``fonendo run --model openai_compatible --out results/raw/<your_model>/<subset>.jsonl``) and
record the server and its version yourself. ``info()`` records the model name, never the
server address or the key.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
import uuid
import wave
from typing import Any

import numpy as np

from fonendo import SAMPLE_RATE
from fonendo.runners.base import Runner
from fonendo.runners.remote._common import (
    DEFAULT_BACKOFF_S,
    FatalError,
    RetryableError,
    provider_slot,
    scrub,
    status_is_retryable,
    to_pcm16,
    with_retries,
)

ENV_BASE_URL = "FONENDO_OPENAI_BASE_URL"
ENV_MODEL = "FONENDO_OPENAI_MODEL"
ENV_KEY = "FONENDO_OPENAI_API_KEY"
ENV_KEY_FALLBACK = "OPENAI_API_KEY"


def endpoint_url(base_url: str) -> str:
    """``{base_url}/v1/audio/transcriptions``, tolerating a trailing slash or ``/v1``."""
    base = base_url.strip().rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/audio/transcriptions"


def wav_bytes(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """A mono PCM16 WAV file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(to_pcm16(audio))
    return buf.getvalue()


def multipart(
    fields: dict[str, str], file_field: str, filename: str, data: bytes
) -> tuple[bytes, str]:
    """Encode ``multipart/form-data``; returns ``(body, content_type)``."""
    boundary = f"fonendo-{uuid.uuid4().hex}"
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{boundary}\r\n".encode())
        out.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        out.write(f"{v}\r\n".encode())
    out.write(f"--{boundary}\r\n".encode())
    out.write(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
    )
    out.write(data)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


class OpenAICompatibleRunner(Runner):
    """EXPERIMENTAL runner for OpenAI-style ``/v1/audio/transcriptions`` servers."""

    provider = "openai_compatible"

    def __init__(
        self,
        name: str = "openai_compatible",
        *,
        base_url: str | None = None,
        model: str | None = None,
        label: str | None = None,
        language: str = "es",
        timeout_s: float = 120.0,
        max_concurrency: int = 3,
        backoff_s: tuple[float, ...] = DEFAULT_BACKOFF_S,
        device: str | None = None,  # accepted for CLI symmetry; unused (remote server)
    ) -> None:
        self._base_url = base_url
        self._model = model
        env_vars = tuple(
            v for v, given in ((ENV_BASE_URL, base_url), (ENV_MODEL, model)) if not given
        )
        super().__init__(
            name,
            "remote",
            extra="openai-compatible",
            label=label,
            model_id=model or os.environ.get(ENV_MODEL, "").strip(),
            revision=None,
            env_vars=env_vars,
            max_concurrency=max_concurrency,
        )
        if label is None:
            self.label = f"{self.model_id or 'openai-compatible'} (experimental)"
        self.language = language
        self.timeout_s = float(timeout_s)
        self.backoff_s = tuple(backoff_s)
        self.url = ""

    # -- lifecycle -------------------------------------------------------------------

    def load(self) -> None:
        base = self._base_url or os.environ.get(ENV_BASE_URL, "").strip()
        model = self._model or os.environ.get(ENV_MODEL, "").strip()
        if not base or not model:
            raise FatalError(f"set {ENV_BASE_URL} and {ENV_MODEL}")
        self.url = endpoint_url(base)
        self.model_id = model

    def _api_key(self) -> str | None:
        return (
            os.environ.get(ENV_KEY, "").strip() or os.environ.get(ENV_KEY_FALLBACK, "").strip()
        ) or None

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "experimental": True,
            "api": "openai-compatible /v1/audio/transcriptions",
            "form": {
                "language": self.language,
                "temperature": 0,
                "response_format": "json",
                "prompt": None,
            },
        }

    # -- inference -------------------------------------------------------------------

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr}")
        if not self.url:
            self.load()
        key = self._api_key()
        body, ctype = multipart(
            {
                "model": self.model_id,
                "language": self.language,
                "temperature": "0",
                "response_format": "json",
            },
            "file",
            "audio.wav",
            wav_bytes(audio, sr),
        )
        headers = {"Content-Type": ctype, "Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        def attempt() -> str:
            with provider_slot(self.provider):
                return self._post(body, headers, key)

        return with_retries(attempt, backoff_s=self.backoff_s)

    def _post(self, body: bytes, headers: dict[str, str], key: str | None) -> str:
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                raw = resp.read()
                ctype = resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace") if exc.fp else ""
            msg = scrub(f"HTTP {exc.code} from the transcription server: {detail}", key)
            if status_is_retryable(exc.code):
                raise RetryableError(msg) from None
            raise FatalError(msg) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise RetryableError(scrub(f"{type(exc).__name__}: {exc}", key)) from None
        text = raw.decode("utf-8", "replace")
        if "json" in ctype.lower() or text.lstrip().startswith("{"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RetryableError(f"invalid JSON reply: {text[:200]!r}") from exc
            if not isinstance(data, dict) or not isinstance(data.get("text"), str):
                raise FatalError(f"reply has no 'text' field: {text[:200]!r}")
            return data["text"].strip()
        return text.strip()
