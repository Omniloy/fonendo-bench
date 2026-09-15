"""Whisper large-v3 and large-v3-turbo through Hugging Face ``transformers`` (extra ``whisper``).

=====================  ================================================================
model                  ``whisper_large_v3``: `openai/whisper-large-v3`_ @ ``06f233fe``
                       (1.55 B parameters; license Apache-2.0)
                       ``whisper_large_v3_turbo``: `openai/whisper-large-v3-turbo`_ @
                       ``41f01f3f`` (809 M parameters: the large-v3 encoder with a 4-layer
                       decoder; license MIT)
download               only ``model.safetensors`` (fp16) and the small config / tokenizer
                       files: 3.1 GB (large-v3), 1.6 GB (turbo)
tested                 torch 2.14.0, transformers 5.17.0, accelerate 1.15.0, Python 3.12,
                       NVIDIA A100 80 GB, fp16, SDPA attention; peak GPU memory 3.1 GiB
                       (large-v3), 1.6 GiB (turbo)
decoding               greedy (``num_beams=1``, ``do_sample=False``), temperature 0 with no
                       temperature fallback and no compression-ratio / log-prob / no-speech
                       thresholds; Spanish forced (``language="es"``, ``task="transcribe"``);
                       timestamp-token decoding (``return_timestamps=True``)
output cap             ``max_new_tokens = 444`` (the decoder has 448 positions, 4 are the forced
                       prefix); timestamp tokens share this budget
=====================  ================================================================

Short-form path: every clip is at most 30 s, so the whole clip is one Whisper window (no
chunking, no VAD); this is what ``pipeline("automatic-speech-recognition")`` does for such
clips, called here through ``model.generate`` so the decoding flags are explicit. The
hypothesis is ``tokenizer.decode(..., skip_special_tokens=True).strip()`` (timestamp tokens
removed). One clip per call.

Why timestamp decoding: it is the default of OpenAI's reference ``transcribe()``. Decoding with
``<|notimestamps|>`` makes both checkpoints (turbo much more often) stop early on real speech,
returning a short phrase such as "Gracias." for a whole clip.

Apple Silicon: see :mod:`fonendo.runners.local.mlx_whisper` for an MLX backend of the same
checkpoints (a different implementation; its outputs are close to, not identical with, these).

.. _openai/whisper-large-v3: https://huggingface.co/openai/whisper-large-v3
.. _openai/whisper-large-v3-turbo: https://huggingface.co/openai/whisper-large-v3-turbo
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import transformers  # noqa: F401 - fail at import time when the extra is missing

from fonendo import SAMPLE_RATE
from fonendo.runners.base import Runner
from fonendo.runners.local._common import (
    check_audio,
    dtype_name,
    quiet_transformers,
    resolve_device,
    resolve_dtype,
)

MAX_TARGET_POSITIONS = 448
N_PREFIX_TOKENS = 4  # budget reserve for <|startoftranscript|><|es|><|transcribe|>(+1)


class WhisperRunner(Runner):
    """A Whisper checkpoint in its default configuration (see the module docstring)."""

    def __init__(
        self,
        name: str = "whisper_large_v3",
        *,
        model_id: str = "openai/whisper-large-v3",
        revision: str | None = "06f233fe06e710322aca913c1bc4249a0d71fce1",
        label: str | None = "Whisper large-v3",
        card: str = "https://huggingface.co/openai/whisper-large-v3",
        weights_license: str = "Apache-2.0",
        device: str | None = None,
        dtype: str | None = None,
        language: str = "es",
    ) -> None:
        super().__init__(
            name, "local", extra="whisper", label=label, model_id=model_id, revision=revision
        )
        self.card = card
        self.weights_license = weights_license
        self._device_arg = device
        self._dtype_arg = dtype
        self.language = language
        self.max_new_tokens = MAX_TARGET_POSITIONS - N_PREFIX_TOKENS

    def load(self) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        quiet_transformers()
        self.device = resolve_device(self._device_arg)
        self.dtype = resolve_dtype(self._dtype_arg, self.device, "float16")
        kw: dict[str, Any] = {"revision": self.revision} if self.revision else {}
        self.processor = WhisperProcessor.from_pretrained(self.model_id, **kw)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(
                self.model_id, dtype=self.dtype, attn_implementation="sdpa", **kw
            )
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        audio = check_audio(audio, sr)
        feats = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        out = self.model.generate(
            feats.to(self.device, dtype=self.dtype),
            language=self.language,
            task="transcribe",
            return_timestamps=True,
            num_beams=1,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=self.max_new_tokens,
        )
        seq = (out["sequences"] if isinstance(out, dict) else out)[0].tolist()
        return self.processor.tokenizer.decode(seq, skip_special_tokens=True).strip()

    def close(self) -> None:
        self.model = self.processor = None
        self._loaded = False  # a later run_subset() loads again
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "card": self.card,
            "weights_license": self.weights_license,
            "backend": f"transformers {transformers.__version__}, torch {torch.__version__}",
            "device": getattr(self, "device", self._device_arg),
            "dtype": dtype_name(getattr(self, "dtype", self._dtype_arg)),
            "language": self.language,
            "decoding": "greedy, temperature 0, no fallback, timestamp tokens",
            "max_new_tokens": self.max_new_tokens,
        }
