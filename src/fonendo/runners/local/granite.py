"""IBM Granite Speech 4.1 2B through ``transformers`` (extra ``granite``).

=====================  ================================================================
model                  `ibm-granite/granite-speech-4.1-2b`_ @ ``de575db6`` (2.31 B
                       parameters; license Apache-2.0)
download               the three ``model-*.safetensors`` shards (bf16) and the config /
                       tokenizer files: 4.4 GB; not ``out_llm.safetensors`` (a CTC head used
                       only for self-speculative decoding) nor the sample audio
tested                 torch 2.14.0, transformers 5.17.0, accelerate 1.15.0, torchaudio 2.11.0
                       (see below), Python 3.12, NVIDIA A100 80 GB, bf16; peak GPU memory
                       4.4 GiB
prompt                 the model card's ASR prompt through the chat template, audio first:
                       ``USER: <|audio|>transcribe the speech to text.\\n ASSISTANT:``
decoding               the card's recipe: greedy (``do_sample=False``, ``num_beams=1``), no
                       repetition penalty, ``max_new_tokens=1024``
language               **cannot be forced**: the model has no language token or option; the
                       card's non-English ASR recipe is the English instruction above, and the
                       model follows the language of the audio
=====================  ================================================================

``GraniteSpeechFeatureExtractor`` imports ``torchaudio`` (for ``MelSpectrogram``). torchaudio
wheels are built against one torch release each, so installing the extra lets pip pick a
matching torch / torchaudio pair. The published run used torch 2.14.0 with torchaudio 2.11.0
installed without its dependencies (the mel spectrogram is plain torch code); either setup
works.

.. _ibm-granite/granite-speech-4.1-2b: https://huggingface.co/ibm-granite/granite-speech-4.1-2b
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import transformers

from fonendo.runners.base import Runner
from fonendo.runners.local._common import (
    check_audio,
    dtype_name,
    quiet_transformers,
    resolve_device,
    resolve_dtype,
    snapshot,
)

#: The model card's plain ASR instruction (the card's prompt table, "ASR").
INSTRUCTION = "transcribe the speech to text."
ALLOW_PATTERNS = ["*.json", "*.jinja", "model-*.safetensors", "*.txt"]


class GraniteSpeechRunner(Runner):
    """Granite Speech 4.1 2B with the model card's ASR prompt."""

    def __init__(
        self,
        name: str = "granite_speech_4p1_2b",
        *,
        model_id: str = "ibm-granite/granite-speech-4.1-2b",
        revision: str | None = "de575db64086f84fdc79da4932d1076e965bc546",
        label: str | None = "Granite Speech 4.1 2B",
        card: str = "https://huggingface.co/ibm-granite/granite-speech-4.1-2b",
        weights_license: str = "Apache-2.0",
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int = 1024,
    ) -> None:
        super().__init__(
            name, "local", extra="granite", label=label, model_id=model_id, revision=revision
        )
        self.card = card
        self.weights_license = weights_license
        self._device_arg = device
        self._dtype_arg = dtype
        self.max_new_tokens = max_new_tokens

    def load(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        quiet_transformers()
        self.device = resolve_device(self._device_arg)
        self.dtype = resolve_dtype(self._dtype_arg, self.device, "bfloat16")
        src = str(snapshot(self.model_id, self.revision, ALLOW_PATTERNS))
        self.processor = AutoProcessor.from_pretrained(src)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            src, device_map=self.device, dtype=self.dtype
        ).eval()
        self.prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<|audio|>" + INSTRUCTION}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        audio = check_audio(audio, sr)
        wav = torch.from_numpy(audio).unsqueeze(0)
        inputs = self.processor(self.prompt, wav, device=self.device, return_tensors="pt").to(
            self.device
        )
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1
        )
        new = out[0, inputs["input_ids"].shape[-1] :]
        return self.processor.tokenizer.decode(new, skip_special_tokens=True).strip()

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
            "language": "auto (the model has no language option)",
            "prompt": INSTRUCTION,
            "decoding": "greedy",
            "max_new_tokens": self.max_new_tokens,
        }
