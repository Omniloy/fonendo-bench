"""Model registry.

``REGISTRY`` maps a model name (the ``--model`` value and the results folder name) to a factory
returning a :class:`~fonendo.runners.base.Runner`. Register runners with :func:`lazy` so that
importing this package never imports a model's dependencies::

    LOCAL_REGISTRY["whisper_large_v3"] = lazy(
        "fonendo.runners.local.whisper:WhisperRunner", extra="whisper",
    )

Local runners (open weights on your hardware) live in :mod:`fonendo.runners.local`, hosted APIs
in :mod:`fonendo.runners.remote`; ``REGISTRY`` merges both. See CONTRACT.md, "Adding a runner".
"""

from __future__ import annotations

from fonendo.runners.base import Factory, Runner, lazy, run_subset
from fonendo.runners.local import LOCAL_REGISTRY
from fonendo.runners.remote import REMOTE_REGISTRY

_overlap = set(LOCAL_REGISTRY) & set(REMOTE_REGISTRY)
if _overlap:  # pragma: no cover - a packaging error
    raise RuntimeError(f"model names registered twice: {sorted(_overlap)}")

REGISTRY: dict[str, Factory] = dict(sorted({**LOCAL_REGISTRY, **REMOTE_REGISTRY}.items()))


def get_runner(name: str, **kwargs) -> Runner:
    """Instantiate a registered runner (does not load weights; ``run_subset`` does)."""
    try:
        factory = REGISTRY[name]
    except KeyError:
        known = sorted(REGISTRY) or "none"
        raise KeyError(f"unknown model {name!r}; registered: {known}") from None
    return factory(**kwargs)


__all__ = [
    "LOCAL_REGISTRY",
    "REGISTRY",
    "REMOTE_REGISTRY",
    "Factory",
    "Runner",
    "get_runner",
    "lazy",
    "run_subset",
]
