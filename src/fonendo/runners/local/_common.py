"""Helpers shared by the local (open-weights) runners.

Dependency-light on purpose: ``torch`` is imported inside the functions, so this module can be
imported by runners whose extra does not install torch (``mlx-whisper``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from fonendo import SAMPLE_RATE

#: torch dtype name -> attribute of the ``torch`` module
_DTYPES = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def resolve_device(device: str | None) -> str:
    """``device`` if given, else ``cuda`` when available, then ``mps``, then ``cpu``."""
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str | None, device: str, reference: str) -> Any:
    """torch dtype for ``device``.

    ``dtype=None`` means the precision the published results were produced with
    (``reference``) on GPUs (CUDA, MPS) and float32 on CPU, where half precision is slow or
    unsupported.
    """
    import torch

    name = dtype or (reference if not device.startswith("cpu") else "float32")
    try:
        return getattr(torch, _DTYPES[name])
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}; use one of {sorted(_DTYPES)}") from None


def dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def check_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """The runners take the 16 kHz float32 mono array from ``load_subset`` as is."""
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio (load_subset output), got {sr} Hz")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio (1-D array), got shape {audio.shape}")
    return np.ascontiguousarray(audio)


def quiet_transformers() -> None:
    """Silence per-call generation notices (they flood the log, one per clip)."""
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


def quiet_nemo() -> None:
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")
    try:
        from nemo.utils import logging as nemo_logging

        nemo_logging.setLevel(logging.ERROR)
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


def snapshot(model_id: str, revision: str | None, allow_patterns: list[str] | None = None) -> Path:
    """Local directory of ``model_id`` at ``revision`` (downloaded into the HF cache if needed).

    ``model_id`` may also be a local directory holding the same files (offline copies); it is
    then used as is and ``revision`` is ignored.
    """
    if Path(model_id).is_dir():
        return Path(model_id)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    try:
        return Path(snapshot_download(model_id, revision=revision, allow_patterns=allow_patterns))
    except GatedRepoError as exc:
        raise gated_repo_hint(model_id, exc) from exc


def gated_repo_hint(model_id: str, exc: Exception) -> RuntimeError:
    """Actionable error for a gated repository whose terms the user has not accepted."""
    return RuntimeError(
        f"{model_id} is a gated repository. Open https://huggingface.co/{model_id} while signed "
        "in, accept its terms of use, then make your token available (environment variable "
        f"HF_TOKEN, or `hf auth login`). Original error: {type(exc).__name__}: {exc}"
    )
