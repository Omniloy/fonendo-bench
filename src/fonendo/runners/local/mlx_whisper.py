"""Whisper large-v3 / large-v3-turbo on Apple Silicon with MLX (extra ``mlx-whisper``).

=====================  ================================================================
model                  ``mlx_whisper_large_v3``: `mlx-community/whisper-large-v3-mlx`_ @
                       ``49e6aa28`` (3.1 GB, fp16)
                       ``mlx_whisper_large_v3_turbo``: `mlx-community/whisper-large-v3-turbo`_
                       @ ``a4aaeec0`` (1.6 GB, fp16)
                       MLX conversions of `openai/whisper-large-v3`_ (Apache-2.0) and
                       `openai/whisper-large-v3-turbo`_ (MIT); the licenses of the original
                       checkpoints apply
backend                ``mlx-whisper`` (``mlx_whisper.transcribe``, a port of OpenAI's
                       reference ``transcribe()``); Apple Silicon Macs only
tested                 mlx-whisper 0.4.3, mlx 0.32.2, Python 3.12, macOS on Apple Silicon:
                       the code path end to end (with the small ``whisper-tiny`` conversion);
                       the large-v3 / turbo conversions have not been scored with it
decoding               greedy, temperature 0 with no temperature fallback, compression-ratio /
                       log-prob / no-speech thresholds off, Spanish forced (``language="es"``),
                       timestamp-token decoding (the reference default), no word timestamps,
                       no initial prompt
=====================  ================================================================

This is a convenience backend for Mac users, not the reference implementation: the published
Whisper rows come from :mod:`fonendo.runners.local.whisper` (``transformers`` on CUDA). The
two implementations share the checkpoint and the decoding settings, but not the numerics (MLX
fp16 kernels, a different mel front end) nor the output cap (the reference ``transcribe()``
stops a window after 224 tokens, the ``transformers`` runner after 444), so hypotheses can
differ on a few clips. Store their results under their own model names.

.. _mlx-community/whisper-large-v3-mlx: https://huggingface.co/mlx-community/whisper-large-v3-mlx
.. _mlx-community/whisper-large-v3-turbo:
   https://huggingface.co/mlx-community/whisper-large-v3-turbo
.. _openai/whisper-large-v3: https://huggingface.co/openai/whisper-large-v3
.. _openai/whisper-large-v3-turbo: https://huggingface.co/openai/whisper-large-v3-turbo
"""

from __future__ import annotations

from typing import Any

import mlx_whisper
import numpy as np

from fonendo.runners.base import Runner
from fonendo.runners.local._common import check_audio, snapshot


class MLXWhisperRunner(Runner):
    """A Whisper checkpoint converted to MLX, default configuration, Spanish forced."""

    def __init__(
        self,
        name: str = "mlx_whisper_large_v3_turbo",
        *,
        model_id: str = "mlx-community/whisper-large-v3-turbo",
        revision: str | None = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
        label: str | None = "Whisper large-v3-turbo (MLX)",
        card: str = "https://huggingface.co/openai/whisper-large-v3-turbo",
        weights_license: str = "MIT",
        device: str | None = None,
        language: str = "es",
    ) -> None:
        if device not in (None, "mps", "gpu"):
            raise ValueError("MLX runners use the Apple GPU; --device is not used")
        super().__init__(
            name, "local", extra="mlx-whisper", label=label, model_id=model_id, revision=revision
        )
        self.card = card
        self.weights_license = weights_license
        self.language = language

    def load(self) -> None:
        self.path = str(snapshot(self.model_id, self.revision))

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        audio = check_audio(audio, sr)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.path,
            language=self.language,
            task="transcribe",
            temperature=0.0,
            compression_ratio_threshold=None,
            logprob_threshold=None,
            no_speech_threshold=None,
            condition_on_previous_text=False,
            word_timestamps=False,
            initial_prompt=None,
            verbose=None,
        )
        return str(result["text"]).strip()

    def info(self) -> dict[str, Any]:
        version = getattr(mlx_whisper, "__version__", None)
        if version is None:
            from importlib.metadata import version as _v

            version = _v("mlx-whisper")
        return {
            **super().info(),
            "card": self.card,
            "weights_license": self.weights_license,
            "backend": f"mlx-whisper {version}",
            "device": "Apple GPU (MLX)",
            "dtype": "float16",
            "language": self.language,
            "decoding": "greedy, temperature 0, no fallback, timestamp tokens",
        }
