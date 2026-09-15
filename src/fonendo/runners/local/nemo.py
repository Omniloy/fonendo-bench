"""NVIDIA NeMo ASR models: Parakeet-TDT v3 and Canary-1B-v2 (extra ``nemo``).

Both checkpoints are downloaded as a single ``.nemo`` file with ``huggingface_hub`` and restored
with ``nemo.collections.asr``. They need their own virtual environment: NeMo pins ``lightning``,
``lhotse`` and other packages that the ``transformers``-based extras do not need.

Weights are fp32; the encoder runs under ``torch.autocast(bfloat16)`` on CUDA, the recipe of
NeMo's own evaluation script (``transcribe_speech.py``, ``amp=True amp_dtype=bfloat16``). On CPU
or MPS autocast is off and everything runs in fp32.

NeMo version. The published results were produced with NeMo from GitHub,
``nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@ca3f93a5`` (3.1.0.dev), torch
2.14.0, Python 3.12, NVIDIA A100 80 GB; with that build this code reproduces them exactly. The
PyPI release 3.0.0 also runs this code unchanged, but its decoding differs on a few clips (on
the first 20 clinical_test clips: 1 hypothesis differs for Parakeet, 4 for Canary, each by one
or two words). Install the pinned build to reproduce the published numbers.

Parakeet-TDT v3 (``parakeet_tdt_0p6b_v3``)
    `nvidia/parakeet-tdt-0.6b-v3`_ @ ``541d1f99``, file ``parakeet-tdt-0.6b-v3.nemo`` (2.5 GB,
    627 M parameters; license CC-BY-4.0). FastConformer encoder + TDT transducer, 25 European
    languages. Decoding: TDT ``greedy_batch`` with label looping (the model card's evaluation
    decoding and NeMo's default for this model), no timestamps. **Language cannot be forced**:
    the model has no language token or option and detects the language itself. Peak GPU memory
    4.7 GiB.

Canary-1B-v2 (``canary_1b_v2``)
    `nvidia/canary-1b-v2`_ @ ``d4557063``, file ``canary-1b-v2.nemo`` (6.4 GB: the 3.9 GB
    FastConformer + Transformer encoder-decoder, plus a 2.5 GB CTC model used only for
    timestamps and long-audio chunking, which is not restored because clips are at most 30 s;
    license CC-BY-4.0). Decoding: the checkpoint's own configuration, beam search with
    ``beam_size=1`` (greedy); Spanish forced with ``source_lang="es", target_lang="es"``,
    punctuation and capitalization on (``pnc="yes"``). Peak GPU memory 7.2 GiB.

.. _nvidia/parakeet-tdt-0.6b-v3: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
.. _nvidia/canary-1b-v2: https://huggingface.co/nvidia/canary-1b-v2
"""

from __future__ import annotations

import contextlib
import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import nemo  # noqa: E402 - fail at import time when the extra is missing

from fonendo.runners.base import Runner  # noqa: E402
from fonendo.runners.local._common import check_audio, quiet_nemo, resolve_device  # noqa: E402


class _NemoRunner(Runner):
    nemo_file = ""

    def __init__(
        self,
        name: str,
        *,
        model_id: str,
        revision: str | None,
        label: str | None,
        card: str,
        weights_license: str,
        device: str | None = None,
    ) -> None:
        super().__init__(
            name, "local", extra="nemo", label=label, model_id=model_id, revision=revision
        )
        self.card = card
        self.weights_license = weights_license
        self._device_arg = device

    def _nemo_path(self) -> str:
        """The ``.nemo`` file: downloaded alone (not the whole repository), or a local copy."""
        if Path(self.model_id).is_file():
            return self.model_id
        if Path(self.model_id).is_dir():
            return str(Path(self.model_id) / self.nemo_file)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.model_id, self.nemo_file, revision=self.revision)

    def _autocast(self):
        if self.device.startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

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
            "backend": f"nemo_toolkit {nemo.__version__}, torch {torch.__version__}",
            "device": getattr(self, "device", self._device_arg),
            "dtype": "float32 weights, bfloat16 autocast (CUDA)",
        }


class ParakeetTDTRunner(_NemoRunner):
    """Parakeet-TDT v3: TDT greedy decoding, language detected by the model."""

    nemo_file = "parakeet-tdt-0.6b-v3.nemo"

    def __init__(
        self,
        name: str = "parakeet_tdt_0p6b_v3",
        *,
        model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
        revision: str | None = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        label: str | None = "Parakeet-TDT v3",
        card: str = "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
        weights_license: str = "CC-BY-4.0",
        device: str | None = None,
    ) -> None:
        super().__init__(
            name,
            model_id=model_id,
            revision=revision,
            label=label,
            card=card,
            weights_license=weights_license,
            device=device,
        )

    def load(self) -> None:
        from nemo.collections.asr.models import ASRModel
        from omegaconf import open_dict

        quiet_nemo()
        self.device = resolve_device(self._device_arg)
        model = ASRModel.restore_from(self._nemo_path(), map_location=torch.device(self.device))
        model.eval()
        cfg = copy.deepcopy(model.cfg.decoding)
        with open_dict(cfg):
            cfg.strategy = "greedy_batch"
            cfg.greedy.loop_labels = True
            cfg.compute_timestamps = False
        model.change_decoding_strategy(cfg, verbose=False)
        quiet_nemo()
        self.model = model

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        audio = check_audio(audio, sr)
        sig = torch.from_numpy(audio).to(self.device)[None]
        sig_len = torch.tensor([sig.shape[1]], device=self.device)
        with self._autocast():
            enc, enc_len = self.model.forward(input_signal=sig, input_signal_length=sig_len)
        hyps = self.model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=enc.float(), encoded_lengths=enc_len, return_hypotheses=False
        )
        h = hyps[0]
        return (h.text if hasattr(h, "text") else str(h)).strip()

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "language": "auto (the model has no language option)",
            "decoding": "TDT greedy_batch, label looping, no timestamps",
        }


class CanaryRunner(_NemoRunner):
    """Canary-1B-v2: the checkpoint's beam search with beam_size=1, Spanish forced."""

    nemo_file = "canary-1b-v2.nemo"

    def __init__(
        self,
        name: str = "canary_1b_v2",
        *,
        model_id: str = "nvidia/canary-1b-v2",
        revision: str | None = "d455706339a6b32e1aa40f82c713a482a0c938e2",
        label: str | None = "Canary-1B-v2",
        card: str = "https://huggingface.co/nvidia/canary-1b-v2",
        weights_license: str = "CC-BY-4.0",
        device: str | None = None,
        language: str = "es",
    ) -> None:
        super().__init__(
            name,
            model_id=model_id,
            revision=revision,
            label=label,
            card=card,
            weights_license=weights_license,
            device=device,
        )
        self.language = language

    def load(self) -> None:
        from nemo.collections.asr.models import ASRModel
        from omegaconf import open_dict

        quiet_nemo()
        self.device = resolve_device(self._device_arg)
        path = self._nemo_path()
        cfg = ASRModel.restore_from(path, return_config=True, map_location=torch.device("cpu"))
        with open_dict(cfg):
            cfg.restore_timestamps_model = False  # timestamps / long-audio chunking only
        model = ASRModel.restore_from(
            path, override_config_path=cfg, map_location=torch.device(self.device)
        )
        model.eval()
        dec = copy.deepcopy(model.cfg.decoding)
        with open_dict(dec):
            dec.strategy = "beam"
            dec.beam.beam_size = 1
        model.change_decoding_strategy(dec)
        quiet_nemo()
        self.model = model

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        audio = check_audio(audio, sr)
        with self._autocast():
            out = self.model.transcribe(
                [audio],
                batch_size=1,
                source_lang=self.language,
                target_lang=self.language,
                pnc="yes",
                verbose=False,
            )
        h = out[0]
        return (h.text if hasattr(h, "text") else str(h)).strip()

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "language": self.language,
            "decoding": "beam search, beam_size 1 (greedy), pnc=yes",
        }
