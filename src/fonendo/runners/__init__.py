"""Model registry.

``REGISTRY`` maps a model name (the ``--model`` value and the results folder name) to a factory
returning a :class:`~fonendo.runners.base.Runner`. Register runners with :func:`lazy` so that
importing this package never imports a model's dependencies::

    REGISTRY["whisper_large_v3"] = lazy(
        "fonendo.runners.whisper:WhisperRunner", extra="whisper",
        name="whisper_large_v3", model_id="openai/whisper-large-v3",
    )

See CONTRACT.md, "Adding a runner".
"""

from __future__ import annotations

from fonendo.runners.base import Factory, Runner, lazy, run_subset

REGISTRY: dict[str, Factory] = {
    # one entry per runnable model, alphabetical; added by the runner modules' authors
}


def get_runner(name: str, **kwargs) -> Runner:
    """Instantiate a registered runner (does not load weights; ``run_subset`` does)."""
    try:
        factory = REGISTRY[name]
    except KeyError:
        known = sorted(REGISTRY) or "none"
        raise KeyError(f"unknown model {name!r}; registered: {known}") from None
    return factory(**kwargs)


__all__ = ["REGISTRY", "Factory", "Runner", "get_runner", "lazy", "run_subset"]
