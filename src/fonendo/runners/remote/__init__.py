"""Runners for hosted speech-to-text APIs (``kind="remote"``).

=======================  =====================  ===================  ======================
registry name            system                 extra                environment
=======================  =====================  ===================  ======================
``soniox_stt_rt_v5``     Soniox stt-rt-v5       ``soniox``           ``SONIOX_API_KEY``,
                         (real-time WebSocket)                       ``SONIOX_REGION``
``deepgram_nova3_es``    Deepgram Nova-3,       ``deepgram``         ``DEEPGRAM_API_KEY``,
                         Spanish (streaming)                         ``DEEPGRAM_REGION``
``deepgram_flux_multi``  Deepgram Flux          ``deepgram``         ``DEEPGRAM_API_KEY``,
                         Multilingual                                ``DEEPGRAM_REGION``
``openai_compatible``    any OpenAI-style       ``openai-compatible``  ``FONENDO_OPENAI_*``
                         transcription server
                         (EXPERIMENTAL)
=======================  =====================  ===================  ======================

Every runner sends the audio in the provider's default configuration: no custom vocabulary,
keyterms or context prompt, and Spanish selected (a language hint where the service takes
only a hint). Keys are read from the environment at call
time and never logged or written. At most 3 connections per provider are open at once in a
process (:func:`fonendo.runners.remote._common.provider_slot`); failed calls are retried with
backoff inside ``transcribe``. See ``docs/remote-runners.md`` for regions and costs.

``REMOTE_REGISTRY`` holds lazy factories (see :func:`fonendo.runners.base.lazy`), so importing
this package imports neither ``websockets`` nor any runner module.
"""

from __future__ import annotations

from fonendo.runners.base import Factory, lazy

REMOTE_REGISTRY: dict[str, Factory] = {
    "deepgram_flux_multi": lazy(
        "fonendo.runners.remote.deepgram:DeepgramFluxRunner",
        extra="deepgram",
        name="deepgram_flux_multi",
    ),
    "deepgram_nova3_es": lazy(
        "fonendo.runners.remote.deepgram:DeepgramNova3Runner",
        extra="deepgram",
        name="deepgram_nova3_es",
    ),
    "openai_compatible": lazy(
        "fonendo.runners.remote.openai_compatible:OpenAICompatibleRunner",
        extra="openai-compatible",
        name="openai_compatible",
    ),
    "soniox_stt_rt_v5": lazy(
        "fonendo.runners.remote.soniox:SonioxRealtimeRunner",
        extra="soniox",
        name="soniox_stt_rt_v5",
    ),
}

__all__ = ["REMOTE_REGISTRY"]
