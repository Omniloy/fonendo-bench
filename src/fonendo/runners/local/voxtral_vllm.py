"""Mistral Voxtral Small 24B through vLLM (extra ``voxtral-vllm``; own virtual environment).

`mistralai/Voxtral-Small-24B-2507`_ @ ``da5b4240`` (24.3 B parameters; license Apache-2.0).
Downloads the Mistral-format files only: ``consolidated.safetensors`` (bf16, 48.5 GB),
``params.json``, ``tekken.json`` and the small config files; not the duplicate
``transformers`` shards.

PRECISION CAVEAT. The model card's bf16 weights need about 55 GB of GPU memory. The published
fonendo-bench numbers for this model were produced on a shared A100 with about 28 GB available,
so they use vLLM's **online FP8 quantization** of the bf16 checkpoint
(``precision="fp8"``, the default here, so the published numbers can be reproduced):

* ``quantization="fp8_per_block"``: fp8 (e4m3) weights with one scale per 128x128 block, for
  every linear layer of the language model (attention and MLP);
* the audio encoder, the audio-language adapter, the token embeddings and the output head stay
  in bf16;
* on an A100 (compute capability 8.0, no FP8 tensor cores) the layers run weight-only with the
  Marlin kernel (fp8 weights, bf16 activations); the FP8 GEMM kernels vLLM would otherwise try
  first are disabled through ``VLLM_DISABLED_KERNELS`` on such GPUs;
* ~25 GiB of weights, ~28 GB for the process.

Pass ``precision="bf16"`` (``fonendo`` API) to run the checkpoint as published by Mistral; the
results can differ slightly from the FP8 numbers.

Input: the model's transcription mode, ``<s>[INST][BEGIN_AUDIO][AUDIO]...[/INST]lang:es
[TRANSCRIBE]``, built with ``mistral-common`` (``encode_transcription`` with ``language="es"``)
and handed to vLLM as token ids plus the audio array (what vLLM's own Voxtral transcription
endpoint does). Spanish is forced through the language slot; no instruction text is sent.
Decoding: greedy (temperature 0), ``max_tokens=1024``, one clip per request.

Tested configuration of the published run: vLLM 0.29.0, torch 2.13.0, transformers 5.17.0,
mistral-common 1.11.7, Python 3.12, NVIDIA A100 80 GB (``precision="fp8"``,
``gpu_memory_utilization=0.35``). vLLM pins its own torch, so this extra needs its own virtual
environment. This runner was not re-run inside this package (a 48.5 GB download that needs more
GPU memory than the validation budget); its code path is the one of the published run.

.. _mistralai/Voxtral-Small-24B-2507: https://huggingface.co/mistralai/Voxtral-Small-24B-2507
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# The engine runs inside this process (one process to stop, its memory visible per process).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# Greedy decoding needs no top-k/top-p kernel; the FlashInfer sampler would JIT-compile one.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import vllm  # noqa: E402 - fail at import time when the extra is missing

from fonendo.runners.base import Runner  # noqa: E402
from fonendo.runners.local._common import check_audio, snapshot  # noqa: E402
from fonendo.runners.local.voxtral import _raw_audio  # noqa: E402

PRECISIONS = ("fp8", "bf16")
QUANT_IGNORE = ["re:.*whisper_encoder.*", "re:.*audio_language_adapter.*"]
#: FP8 GEMM kernels vLLM tries before Marlin; they cannot run on GPUs below compute
#: capability 8.9 (A100 = 8.0), but some report themselves as supported there.
FP8_KERNELS_BEFORE_MARLIN = (
    "FlashInferFP8ScaledMMLinearKernel,CutlassFP8ScaledMMLinearKernel,"
    "B12xTensorFP8ScaledMMLinearKernel,PerTensorTorchFP8ScaledMMLinearKernel,"
    "ChannelWiseTorchFP8ScaledMMLinearKernel,FlashInferFp8DeepGEMMDynamicBlockScaledKernel,"
    "DeepGemmFp8BlockScaledMMKernel,CutlassFp8BlockScaledMMKernel,B12xFp8BlockScaledMMKernel"
)


class VoxtralVLLMRunner(Runner):
    """Voxtral Small 24B in its transcription mode on vLLM (FP8 weights by default)."""

    allow_patterns = (
        "consolidated.safetensors",
        "params.json",
        "tekken.json",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
    )

    def __init__(
        self,
        name: str = "voxtral_small_24b",
        *,
        model_id: str = "mistralai/Voxtral-Small-24B-2507",
        revision: str | None = "da5b42409f279fdd92febee0511a6c32828569c1",
        label: str | None = "Voxtral Small 24B (FP8 weights)",
        card: str = "https://huggingface.co/mistralai/Voxtral-Small-24B-2507",
        weights_license: str = "Apache-2.0",
        device: str | None = None,
        precision: str = "fp8",
        gpu_memory_utilization: float | None = None,
        max_model_len: int = 3072,
        language: str = "es",
        max_new_tokens: int = 1024,
    ) -> None:
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        if device not in (None, "cuda", "cuda:0"):
            raise ValueError("voxtral_small_24b runs on one CUDA GPU (vLLM); --device is not used")
        super().__init__(
            name, "local", extra="voxtral-vllm", label=label, model_id=model_id, revision=revision
        )
        self.card = card
        self.weights_license = weights_license
        self.precision = precision
        self.gpu_memory_utilization = gpu_memory_utilization or (
            0.35 if precision == "fp8" else 0.9
        )
        self.max_model_len = max_model_len
        self.language = language
        self.max_new_tokens = max_new_tokens

    def load(self) -> None:
        import torch
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        snap = snapshot(self.model_id, self.revision, list(self.allow_patterns))
        kw: dict[str, Any] = {}
        if self.precision == "fp8":
            if torch.cuda.get_device_capability(0) < (8, 9):
                os.environ.setdefault("VLLM_DISABLED_KERNELS", FP8_KERNELS_BEFORE_MARLIN)
            kw = {"quantization": "fp8_per_block", "quantization_config": {"ignore": QUANT_IGNORE}}
        self.llm = vllm.LLM(
            model=str(snap),
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"audio": 1},
            cudagraph_capture_sizes=[1, 2, 4],
            seed=0,
            **kw,
        )
        self.tokenizer = MistralTokenizer.from_file(str(snap / "tekken.json"))
        self.sampling = vllm.SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens, seed=0)

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        from mistral_common.protocol.transcription.request import TranscriptionRequest
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        audio = check_audio(audio, sr)
        tok = self.tokenizer.encode_transcription(
            TranscriptionRequest(
                model=self.model_id, audio=_raw_audio(audio), language=self.language
            )
        )
        prompt = {
            "prompt_token_ids": list(tok.tokens),
            "multi_modal_data": {"audio": [(a.audio_array, a.sampling_rate) for a in tok.audios]},
        }
        out = self.llm.generate([prompt], self.sampling, use_tqdm=False)[0].outputs[0]
        return self.tokenizer.decode(
            list(out.token_ids), special_token_policy=SpecialTokenPolicy.IGNORE
        ).strip()

    def close(self) -> None:
        self.llm = None
        self._loaded = False  # a later run_subset() loads again

    def info(self) -> dict[str, Any]:
        precision = (
            "fp8 per-block (128x128) weight scales on the language-model linears, bf16 audio "
            "encoder / adapter / embeddings / output head"
            if self.precision == "fp8"
            else "bf16"
        )
        return {
            **super().info(),
            "card": self.card,
            "weights_license": self.weights_license,
            "backend": f"vllm {vllm.__version__}",
            "device": "cuda",
            "precision": precision,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "language": f"{self.language} (transcription mode, lang:{self.language})",
            "decoding": "greedy (temperature 0)",
            "max_new_tokens": self.max_new_tokens,
        }
