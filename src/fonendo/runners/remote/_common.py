"""Shared helpers for the hosted-API runners: PCM conversion, a per-provider concurrency limiter,
retries with backoff, secret scrubbing and a safe way to run a coroutine from ``transcribe``.

Nothing here reads a key from disk: keys come from environment variables only, are placed in
the one request field that needs them (a header or the first WebSocket frame) and are removed
from every error message by :func:`scrub`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import random
import ssl
import threading
import time
from collections.abc import Callable, Coroutine, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")

#: Hard cap on simultaneous connections per provider in one process, whatever the runner or the
#: caller asks for (the run loop caps its thread pool at 3 as well).
MAX_STREAMS_PER_PROVIDER = 3
#: Default retry schedule: seconds to wait before retry 1, 2, 3 (+ up to 1 s of jitter).
DEFAULT_BACKOFF_S: tuple[float, ...] = (2.0, 4.0, 8.0)


class RetryableError(RuntimeError):
    """Transient failure (network, timeout, HTTP 408/429/5xx, stream closed early)."""


class FatalError(RuntimeError):
    """Permanent failure (bad request, authentication, payment); never retried."""


# --------------------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------------------


def to_pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] -> little-endian PCM16 bytes.

    ``x * 32768`` rounded and clipped: the exact inverse of the loader's ``int16 / 32768`` for
    PCM16 sources, so a 16 kHz PCM16 file reaches the API bit-exact.
    """
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.clip(np.round(a * 32768.0), -32768, 32767).astype("<i2").tobytes()


def chunks(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]


# --------------------------------------------------------------------------------------
# concurrency limiter
# --------------------------------------------------------------------------------------

_LIMITERS: dict[str, threading.BoundedSemaphore] = {}
_LIMITERS_LOCK = threading.Lock()


@contextmanager
def provider_slot(provider: str) -> Iterator[None]:
    """Hold one of :data:`MAX_STREAMS_PER_PROVIDER` connection slots for ``provider``.

    Shared by every runner of the same provider in this process (Deepgram Nova-3 and Flux use
    the same key, so they share ``"deepgram"``). The slot is held only while a connection is
    open, never during a retry backoff. It does not coordinate separate processes: run one
    ``fonendo run`` per API key at a time.
    """
    with _LIMITERS_LOCK:
        sem = _LIMITERS.setdefault(provider, threading.BoundedSemaphore(MAX_STREAMS_PER_PROVIDER))
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


# --------------------------------------------------------------------------------------
# secrets
# --------------------------------------------------------------------------------------


def env_value(name: str) -> str:
    """Read an environment variable, failing with its NAME (never a value) when unset."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise FatalError(f"set the environment variable {name}")
    return value


def scrub(text: object, *secrets: str | None, limit: int = 500) -> str:
    """``str(text)`` with every secret replaced by ``***``, truncated to ``limit`` chars."""
    s = str(text)
    for secret in secrets:
        if secret:
            s = s.replace(secret, "***")
    return s[:limit]


# --------------------------------------------------------------------------------------
# retries
# --------------------------------------------------------------------------------------


def with_retries(
    attempt: Callable[[], T],
    *,
    backoff_s: Sequence[float] = DEFAULT_BACKOFF_S,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Call ``attempt()``; retry :class:`RetryableError` ``len(backoff_s)`` times.

    :class:`FatalError` (and any other exception) propagates at once. After the last retry the
    last :class:`RetryableError` propagates; the run loop records it as an ``error`` line, which
    the next run retries.
    """
    last: Exception | None = None
    for i in range(len(backoff_s) + 1):
        try:
            return attempt()
        except RetryableError as exc:
            last = exc
        if i < len(backoff_s):
            if on_retry is not None:
                on_retry(i + 1, last)
            delay = backoff_s[i]
            if delay > 0:
                time.sleep(delay + random.uniform(0.0, 1.0))
    assert last is not None
    raise last


def status_is_retryable(code: int) -> bool:
    return code in (408, 425, 429) or code >= 500


# --------------------------------------------------------------------------------------
# asyncio from a synchronous transcribe()
# --------------------------------------------------------------------------------------


def run_coro(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a fresh coroutine to completion from synchronous code.

    ``fonendo run`` calls ``transcribe`` from worker threads, where ``asyncio.run`` is safe. When
    the caller already runs an event loop in this thread (a notebook), the coroutine runs on a
    short-lived helper thread instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


def ssl_context() -> ssl.SSLContext:
    """Default TLS context, with certifi's CA bundle when installed (python.org macOS builds
    ship without system certificates)."""
    ctx = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return ctx
    ctx.load_verify_locations(certifi.where())
    return ctx


def websocket_status(exc: BaseException) -> int | None:
    """HTTP status of a rejected WebSocket handshake (``websockets.InvalidStatus``), else None."""
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return int(code) if isinstance(code, int) else None


def websocket_body(exc: BaseException, limit: int = 300) -> str:
    response = getattr(exc, "response", None)
    body = getattr(response, "body", None) or b""
    try:
        return bytes(body)[:limit].decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - best effort, diagnostic only
        return ""
