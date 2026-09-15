"""Cohere Transcribe through ``transformers`` (extra ``cohere``).

=====================  ================================================================
model                  `CohereLabs/cohere-transcribe-03-2026`_ @ ``b1eacc26`` (2.07 B
                       parameters, Fast-Conformer encoder-decoder; license Apache-2.0)
access                 **gated**: open the model page while signed in to Hugging Face,
                       accept Cohere's terms of use (access is granted automatically), then
                       expose your token (``HF_TOKEN`` or ``hf auth login``)
download               ``model.safetensors`` (bf16, 4.1 GB), ``tokenizer.model`` and the
                       config files (``*.json``); the repository's remote code is not used
tested                 torch 2.14.0, transformers 5.17.0 (native
                       ``CohereAsrForConditionalGeneration``), Python 3.12, NVIDIA A100 80 GB,
                       bf16; peak GPU memory 3.9 GiB
decoding               greedy (``num_beams=1``, ``do_sample=False``; the checkpoint's
                       configuration is beam search with beam size 1), ``max_new_tokens=448``
language               Spanish forced: the processor's ``language="es"`` builds the decoder
                       prompt with the ``<|es|>`` tokens (the model has no language detection;
                       a language token is mandatory); ``punctuation=True``
=====================  ================================================================

Do not load this model with ``trust_remote_code=True``: with ``transformers`` 5.x the
repository's own modeling code produces wrong output; the native class does not need it. Clips
are at most 30 s, below the processor's chunking threshold; a clip that would be chunked raises
an error rather than being transcribed differently.

.. _CohereLabs/cohere-transcribe-03-2026:
   https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import transformers

from fonendo import SAMPLE_RATE
from fonendo.runners.base import Runner
from fonendo.runners.local._common import (
    check_audio,
    dtype_name,
    quiet_transformers,
    resolve_device,
    resolve_dtype,
    snapshot,
)

ALLOW_PATTERNS = ["*.json", "model.safetensors", "tokenizer.model"]


class CohereTranscribeRunner(Runner):
    """Cohere Transcribe in its default configuration, Spanish forced."""

    def __init__(
        self,
        name: str = "cohere_transcribe",
        *,
        model_id: str = "CohereLabs/cohere-transcribe-03-2026",
        revision: str | None = "b1eacc2686a3d08ceaae5f24a88b1d519620bc09",
        label: str | None = "Cohere Transcribe",
        card: str = "https://huggingface.co/CohereLabs/cohere-transcribe-03-2026",
        weights_license: str = "Apache-2.0 (gated: accept the terms on the model page)",
        device: str | None = None,
        dtype: str | None = None,
        language: str = "es",
        max_new_tokens: int = 448,
    ) -> None:
        super().__init__(
            name, "local", extra="cohere", label=label, model_id=model_id, revision=revision
        )
        self.card = card
        self.weights_license = weights_license
        self._device_arg = device
        self._dtype_arg = dtype
        self.language = language
        self.max_new_tokens = max_new_tokens

    def load(self) -> None:
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        quiet_transformers()
        self.device = resolve_device(self._device_arg)
        self.dtype = resolve_dtype(self._dtype_arg, self.device, "bfloat16")
        src = str(snapshot(self.model_id, self.revision, ALLOW_PATTERNS))
        self.processor = AutoProcessor.from_pretrained(src)
        self.model = (
            CohereAsrForConditionalGeneration.from_pretrained(src, dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        audio = check_audio(audio, sr)
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            language=self.language,
            punctuation=True,
        )
        chunk_index = inputs.pop("audio_chunk_index", None)
        if chunk_index is not None and any(c is not None for _, c in chunk_index):
            raise RuntimeError(f"the processor chunked a {len(audio) / SAMPLE_RATE:.1f} s clip")
        inputs = inputs.to(self.device)
        inputs["input_features"] = inputs["input_features"].to(self.dtype)
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, num_beams=1, do_sample=False
        )
        return self.processor.decode(out, skip_special_tokens=True)[0].strip()

    def close(self) -> None:
        self.model = None
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
            "decoding": "greedy, punctuation on",
            "max_new_tokens": self.max_new_tokens,
        }
