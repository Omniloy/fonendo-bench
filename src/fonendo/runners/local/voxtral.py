"""Voxtral Mini 3B and Voxtral Mini 4B Realtime through ``transformers`` (extra ``voxtral``).

Both runners build the model input with ``mistral-common`` (the tokenizer the official
``transformers`` processors call internally) from the checkpoint's ``tekken.json``, then run
``model.generate`` on it. The audio is handed to ``mistral-common`` as an in-memory 32-bit float
WAV, so the model sees exactly the samples returned by ``load_subset``.

Tested with torch 2.14.0, transformers 5.17.0, mistral-common 1.9.1, Python 3.12 on an NVIDIA
A100 80 GB, bf16 (the checkpoints' dtype), SDPA attention, one clip per call.

Voxtral Mini 3B (``voxtral_mini_3b``)
    `mistralai/Voxtral-Mini-3B-2507`_ @ ``3060fe34`` (4.7 B parameters including the audio
    encoder; license Apache-2.0). Downloads the ``transformers`` shards and config files
    (9.4 GB), not the duplicate ``consolidated.safetensors``. Peak GPU memory 8.8 GiB.
    Input: the model's dedicated transcription mode,
    ``<s>[INST][BEGIN_AUDIO][AUDIO]...[/INST]lang:es[TRANSCRIBE]`` (``encode_transcription``
    with ``language="es"``): Spanish is forced through the language slot and no instruction
    text is sent. Log-mel features as the official processor computes them (Whisper feature
    extractor, padded to a multiple of 30 s). Decoding: greedy (the card: temperature 0 for
    transcription), ``max_new_tokens=1024``.

Voxtral Mini 4B Realtime (``voxtral_mini_4b_realtime``)
    `mistralai/Voxtral-Mini-4B-Realtime-2602`_ @ ``2769294d`` (4.4 B parameters; license
    Apache-2.0). Downloads ``model.safetensors`` and the config / tokenizer files (8.9 GB).
    Peak GPU memory 8.5 GiB. Run offline on the whole clip, which is what
    ``VoxtralRealtimeProcessor(audio, is_streaming=False)`` prepares: ``encode_transcription``
    with ``streaming=OFFLINE`` and a transcription delay of 480 ms (the card's recommended
    setting and the ``tekken.json`` default), the audio padded as ``mistral-common`` pads it,
    log-mel from ``VoxtralRealtimeFeatureExtractor``. Decoding: greedy; the model emits one
    token per 80 ms audio frame, so the output length is bounded by the clip length.
    **Language cannot be forced**: the model has no language option; it follows the audio.
    Speed with ``transformers`` at batch size 1 (a real-time factor near 0.75 on an A100) is far
    below what the model reaches with its streaming server; do not read ``secs`` as its
    latency.

.. _mistralai/Voxtral-Mini-3B-2507: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
.. _mistralai/Voxtral-Mini-4B-Realtime-2602:
   https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import mistral_common
import numpy as np
import soundfile as sf
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


def _raw_audio(audio: np.ndarray):
    """``mistral-common`` RawAudio holding a bit-exact float32 WAV of ``audio``."""
    from mistral_common.protocol.instruct.chunk import RawAudio

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return RawAudio(data=buf.getvalue(), format="wav")


class _VoxtralBase(Runner):
    allow_patterns: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        *,
        model_id: str,
        revision: str | None,
        label: str | None,
        card: str,
        weights_license: str,
        device: str | None,
        dtype: str | None,
        max_new_tokens: int,
    ) -> None:
        super().__init__(
            name, "local", extra="voxtral", label=label, model_id=model_id, revision=revision
        )
        self.card = card
        self.weights_license = weights_license
        self._device_arg = device
        self._dtype_arg = dtype
        self.max_new_tokens = max_new_tokens

    def _snapshot(self) -> Path:
        return snapshot(self.model_id, self.revision, list(self.allow_patterns))

    def _setup_device(self) -> None:
        quiet_transformers()
        self.device = resolve_device(self._device_arg)
        self.dtype = resolve_dtype(self._dtype_arg, self.device, "bfloat16")

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
            "backend": (
                f"transformers {transformers.__version__}, torch {torch.__version__}, "
                f"mistral-common {mistral_common.__version__}"
            ),
            "device": getattr(self, "device", self._device_arg),
            "dtype": dtype_name(getattr(self, "dtype", self._dtype_arg)),
            "max_new_tokens": self.max_new_tokens,
        }


class VoxtralMiniRunner(_VoxtralBase):
    """Voxtral Mini 3B in its transcription mode, Spanish forced (``lang:es``)."""

    allow_patterns = ("*.json", "model-*.safetensors")

    def __init__(
        self,
        name: str = "voxtral_mini_3b",
        *,
        model_id: str = "mistralai/Voxtral-Mini-3B-2507",
        revision: str | None = "3060fe34b35ba5d44202ce9ff3c097642914f8f3",
        label: str | None = "Voxtral Mini 3B",
        card: str = "https://huggingface.co/mistralai/Voxtral-Mini-3B-2507",
        weights_license: str = "Apache-2.0",
        device: str | None = None,
        dtype: str | None = None,
        language: str = "es",
        max_new_tokens: int = 1024,
    ) -> None:
        super().__init__(
            name,
            model_id=model_id,
            revision=revision,
            label=label,
            card=card,
            weights_license=weights_license,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
        self.language = language

    def load(self) -> None:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from transformers import AutoFeatureExtractor, VoxtralForConditionalGeneration

        self._setup_device()
        snap = self._snapshot()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(snap))
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            str(snap), dtype=self.dtype, attn_implementation="sdpa", device_map=self.device
        ).eval()
        self.tokenizer = MistralTokenizer.from_file(str(snap / "tekken.json"))

    def _features(self, audios) -> torch.Tensor:
        fe = self.feature_extractor
        chunks = []
        for a in audios:
            f = fe(
                a.audio_array,
                sampling_rate=SAMPLE_RATE,
                padding=True,
                truncation=False,
                pad_to_multiple_of=fe.n_samples,  # 30 s windows, as the official processor
                return_tensors="pt",
            )["input_features"]
            chunks.append(f.reshape(fe.feature_size, -1, fe.nb_max_frames).transpose(0, 1))
        return torch.cat(chunks)

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        from mistral_common.protocol.transcription.request import TranscriptionRequest
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        audio = check_audio(audio, sr)
        tok = self.tokenizer.encode_transcription(
            TranscriptionRequest(
                model=self.model_id, audio=_raw_audio(audio), language=self.language
            )
        )
        input_ids = torch.tensor([list(tok.tokens)], device=self.model.device)
        feats = self._features(tok.audios).to(self.model.device, self.dtype)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            input_features=feats,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        new = out[0, input_ids.shape[-1] :].tolist()
        return self.tokenizer.decode(new, special_token_policy=SpecialTokenPolicy.IGNORE).strip()

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "language": f"{self.language} (transcription mode, lang:{self.language})",
            "decoding": "greedy",
        }


class VoxtralRealtimeRunner(_VoxtralBase):
    """Voxtral Mini 4B Realtime, offline on the whole clip, 480 ms delay, language auto."""

    allow_patterns = (
        "model.safetensors",
        "config.json",
        "processor_config.json",
        "generation_config.json",
        "params.json",
        "tekken.json",
    )

    def __init__(
        self,
        name: str = "voxtral_mini_4b_realtime",
        *,
        model_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602",
        revision: str | None = "2769294da9567371363522aac9bbcfdd19447add",
        label: str | None = "Voxtral Mini 4B Realtime",
        card: str = "https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602",
        weights_license: str = "Apache-2.0",
        device: str | None = None,
        dtype: str | None = None,
        delay_ms: int = 480,
        max_new_tokens: int = 1024,
    ) -> None:
        super().__init__(
            name,
            model_id=model_id,
            revision=revision,
            label=label,
            card=card,
            weights_license=weights_license,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
        self.delay_ms = int(delay_ms)

    def load(self) -> None:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from transformers import VoxtralRealtimeForConditionalGeneration
        from transformers.models.voxtral_realtime.feature_extraction_voxtral_realtime import (
            VoxtralRealtimeFeatureExtractor,
        )

        self._setup_device()
        snap = self._snapshot()
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            str(snap), dtype=self.dtype, attn_implementation="sdpa", device_map=self.device
        ).eval()
        fe_cfg = json.loads((snap / "processor_config.json").read_text())["feature_extractor"]
        fe_cfg.pop("feature_extractor_type", None)
        self.feature_extractor = VoxtralRealtimeFeatureExtractor(**fe_cfg)
        self.tokenizer = MistralTokenizer.from_file(str(snap / "tekken.json"))
        audio_cfg = self.tokenizer.instruct_tokenizer.audio_encoder.audio_config
        enc = audio_cfg.encoding_config
        fe = self.feature_extractor
        # the consistency checks VoxtralRealtimeProcessor runs
        assert fe.win_length == enc.window_size and fe.hop_length == enc.hop_length
        assert fe.feature_size == enc.num_mel_bins and fe.sampling_rate == audio_cfg.sampling_rate
        self.num_delay_tokens = int(audio_cfg.get_num_delay_tokens(self.delay_ms))

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        from mistral_common.protocol.transcription.request import (
            StreamingMode,
            TranscriptionRequest,
        )
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        audio = check_audio(audio, sr)
        tok = self.tokenizer.encode_transcription(
            TranscriptionRequest(
                model=self.model_id,
                audio=_raw_audio(audio),
                streaming=StreamingMode.OFFLINE,
                language=None,
                target_streaming_delay_ms=self.delay_ms,
            )
        )
        feats = self.feature_extractor(
            [a.audio_array for a in tok.audios],
            sampling_rate=SAMPLE_RATE,
            padding=True,
            truncation=False,
            center=True,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_features"]
        input_ids = torch.tensor([list(tok.tokens)], device=self.model.device)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            input_features=feats.to(self.model.device, self.dtype),
            num_delay_tokens=self.num_delay_tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        new = out[0, input_ids.shape[-1] :].tolist()
        return self.tokenizer.decode(new, special_token_policy=SpecialTokenPolicy.IGNORE).strip()

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "language": "auto (the model has no language option)",
            "decoding": f"greedy, offline, transcription delay {self.delay_ms} ms",
        }
