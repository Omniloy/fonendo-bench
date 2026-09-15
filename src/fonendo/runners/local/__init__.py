"""Local runners: open-weights models that run on your own hardware.

Every runner transcribes in the model's **default configuration** (CONTRACT.md, section 4.5):
no custom vocabulary, keyterms or context prompt; Spanish selected when the model has a
language option; an instruction-following model (Granite Speech) gets only the model card's
fixed transcription instruction; greedy decoding (or the model card's default); the same
settings for every subset.
Model revisions are pinned to Hugging Face commit hashes.

The CUDA runners reproduce the published runs exactly: every runner below except
``voxtral_small_24b`` and the MLX backend was re-run with this package on the first 20
``clinical_test`` clips and returned the same text, byte for byte, as the published run (NeMo
runs need the pinned NeMo build, see :mod:`fonendo.runners.local.nemo`).

=============================  ================  =================  ==========================
model name                     extra             weights license    notes
=============================  ================  =================  ==========================
``whisper_large_v3``           ``whisper``       Apache-2.0         fp16, CUDA reference
``whisper_large_v3_turbo``     ``whisper``       MIT                fp16, CUDA reference
``parakeet_tdt_0p6b_v3``       ``nemo``          CC-BY-4.0          language not forceable
``canary_1b_v2``               ``nemo``          CC-BY-4.0
``voxtral_mini_3b``            ``voxtral``       Apache-2.0         bf16
``voxtral_mini_4b_realtime``   ``voxtral``       Apache-2.0         language not forceable
``voxtral_small_24b``          ``voxtral-vllm``  Apache-2.0         FP8 weights (see module)
``cohere_transcribe``          ``cohere``        Apache-2.0         gated: accept the terms
``granite_speech_4p1_2b``      ``granite``       Apache-2.0         language not forceable
``mlx_whisper_large_v3``       ``mlx-whisper``   Apache-2.0         Apple Silicon backend
``mlx_whisper_large_v3_turbo`` ``mlx-whisper``   MIT                Apple Silicon backend
=============================  ================  =================  ==========================

Families that pin different ``torch`` builds need separate virtual environments: one for
``whisper`` + ``voxtral`` + ``cohere`` (+ ``granite``), one for ``nemo``, one for
``voxtral-vllm``; ``mlx-whisper`` runs on macOS only. Each module's docstring gives the model
card link, the download size, the tested versions and hardware, and the decoding settings.
"""

from __future__ import annotations

from fonendo.runners.base import Factory, lazy

_P = "fonendo.runners.local"

LOCAL_REGISTRY: dict[str, Factory] = {
    "canary_1b_v2": lazy(f"{_P}.nemo:CanaryRunner", extra="nemo"),
    "cohere_transcribe": lazy(f"{_P}.cohere:CohereTranscribeRunner", extra="cohere"),
    "granite_speech_4p1_2b": lazy(f"{_P}.granite:GraniteSpeechRunner", extra="granite"),
    "mlx_whisper_large_v3": lazy(
        f"{_P}.mlx_whisper:MLXWhisperRunner",
        extra="mlx-whisper",
        name="mlx_whisper_large_v3",
        model_id="mlx-community/whisper-large-v3-mlx",
        revision="49e6aa286ad60c14352c404340ded53710378a11",
        label="Whisper large-v3 (MLX)",
        card="https://huggingface.co/openai/whisper-large-v3",
        weights_license="Apache-2.0",
    ),
    "mlx_whisper_large_v3_turbo": lazy(
        f"{_P}.mlx_whisper:MLXWhisperRunner",
        extra="mlx-whisper",
        name="mlx_whisper_large_v3_turbo",
    ),
    "parakeet_tdt_0p6b_v3": lazy(f"{_P}.nemo:ParakeetTDTRunner", extra="nemo"),
    "voxtral_mini_3b": lazy(f"{_P}.voxtral:VoxtralMiniRunner", extra="voxtral"),
    "voxtral_mini_4b_realtime": lazy(f"{_P}.voxtral:VoxtralRealtimeRunner", extra="voxtral"),
    "voxtral_small_24b": lazy(f"{_P}.voxtral_vllm:VoxtralVLLMRunner", extra="voxtral-vllm"),
    "whisper_large_v3": lazy(f"{_P}.whisper:WhisperRunner", extra="whisper"),
    "whisper_large_v3_turbo": lazy(
        f"{_P}.whisper:WhisperRunner",
        extra="whisper",
        name="whisper_large_v3_turbo",
        model_id="openai/whisper-large-v3-turbo",
        revision="41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        label="Whisper large-v3-turbo",
        card="https://huggingface.co/openai/whisper-large-v3-turbo",
        weights_license="MIT",
    ),
}

__all__ = ["LOCAL_REGISTRY"]
