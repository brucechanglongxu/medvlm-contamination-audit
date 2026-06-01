"""medvlm_contam — contamination audit toolkit for medical VLMs.

Public surface area:
    - benchmarks: dataset loaders returning :class:`BenchmarkExample` iterables
    - models: VLM scorer interface (token-level conditional log-probs)
    - detectors: exchangeability test, Min-K%++ scoring, MM-Detect-style probes, image-NN
    - audit:    end-to-end driver tying loaders + scorers + detectors together
"""

from .version import __version__

__all__ = ["__version__"]
