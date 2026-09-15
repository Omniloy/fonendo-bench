"""fonendo-bench: a reproducible Spanish clinical speech-to-text benchmark.

Modules (see CONTRACT.md for the full interface contract):

* ``fonendo.data``     subset registry and ``load_subset(name, data_dir)``
* ``fonendo.runners``  ``Runner`` base class, ``REGISTRY`` of model factories, the run loop
* ``fonendo.scoring``  ``score(rows, hyps)`` and ``compare(...)`` with bootstrap CIs
* ``fonendo.cli``      the ``fonendo`` command (fetch, run, score, compare, report)

Every system the package runs is evaluated in its default configuration: no custom
vocabulary, keyterms or context prompt; Spanish selected where the system allows it;
instruction-following models get only the fixed transcription instruction they need; no
per-clip information of any kind. Results-only leaderboard rows are evaluated by their owner
and state their conditions.
"""

__version__ = "0.1.0.dev0"

HF_DATASET = "Omniloy/fonendo-bench"
SAMPLE_RATE = 16_000

__all__ = ["HF_DATASET", "SAMPLE_RATE", "__version__"]
