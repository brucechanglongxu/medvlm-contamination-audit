# Medical VLM Contamination Audit

A controlled audit of **pretraining-data contamination** in public medical
vision–language (VLM) benchmarks. This repository contains the detector
toolkit, unit tests, and reproducibility scripts accompanying the paper:

> **A Controlled Audit of Pretraining Contamination in Public Medical
> Vision–Language Benchmarks.** Bruce Changlong Xu, Lan Wu.

Public medical VQA benchmarks (SLAKE, PathVQA, VQA-RAD, OmniMedVQA) have been
freely downloadable for years, yet reported accuracy implicitly assumes their
examples were absent from pretraining. This project tests that assumption with
a family of complementary detectors and—crucially—asks *what survives a
controlled, external pre-domain falsification*.

## Key findings

- **Image side (source overlap, not memorization).** A nearest-neighbour scan
  against PMC-OA flags a substantial fraction of SLAKE images as having an
  extreme same-view neighbour. Manual adjudication of every flagged pair shows
  these are same-modality, same-projection images of *different* patients, so we
  report the signal as **source/distributional overlap** between SLAKE and
  PMC-sourced collections rather than per-image memorization.
- **Text side (what survives).** A cell-internal exchangeability test isolates a
  single model-specific signal that survives an external non-medical baseline
  check and an ordering ablation; a superficially similar PathVQA signal is
  re-attributed to release-order artefacts once baselines are included.
- **Cohort-relative detectors collapse.** Min-K%++ and top-K overlap signals are
  shown to be confounded by inter-model calibration heterogeneity; we recommend
  against using them as standalone membership-inference signals on small
  domain-specialized cohorts. A synthetic calibration model reproduces the
  confound exactly.

A claims-to-evidence map in the paper records, for each claim, whether it
survives an external pre-domain control.

## Install

```bash
python -m pip install -e .            # core (numpy/scipy) — detectors + analysis
python -m pip install -e ".[hf]"      # + torch/transformers for VLM scoring
python -m pip install -e ".[embed]"   # + open_clip/faiss for image embeddings
python -m pip install -e ".[dev]"     # + pytest/ruff for development
```

Requires Python ≥ 3.10.

## Quickstart

Run an end-to-end audit of one model on one benchmark (requires the `hf`
extra and access to the benchmark data):

```bash
medvlm-audit \
  --benchmark slake_en \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --output results/qwen_slake.jsonl
```

Use the detectors directly:

```python
from medvlm_contam.detectors.exchangeability import exchangeability_test
from medvlm_contam.detectors.mink_pp import min_k_pp_scores

# per-example answer log-likelihoods produced by a VLM scorer
p_value = exchangeability_test(per_example_logprobs, n_permutations=10_000)
```

## Reproduce the analysis figures

The synthetic and analysis figures regenerate from the committed JSON summaries
in `outputs/`, on CPU, with no network access:

```bash
python scripts/confound_simulation.py    # cohort-median confound demo
python scripts/tau_sensitivity.py        # image-NN threshold sensitivity
python scripts/topk_overlap.py           # cross-model top-K Jaccard
python scripts/make_jaccard_figure.py    # render the Jaccard heatmap
```

Figures are written to `figures/`. The `confound_simulation.py` driver is fully
deterministic (fixed seed).

## Run the tests

```bash
pytest
```

## Repository layout

```
src/medvlm_contam/
  benchmarks/     dataset loaders (SLAKE, PathVQA, VQA-RAD, OmniMedVQA)
  models/         VLM scorer interface + HuggingFace backend
  detectors/      exchangeability, Min-K%++, MM-Detect-style probes, image-NN
  audit.py        end-to-end driver (loaders + scorers + detectors)
  analysis.py     aggregation / summary statistics
  cli.py          `medvlm-audit` entry point
scripts/          CPU reproducibility + figure scripts
outputs/          small JSON result summaries (inputs to the figure scripts)
tests/            unit tests for the detectors
```

The large per-example log-likelihood files, image embeddings, and
benchmark image data are not redistributed here; the benchmarks themselves are
publicly available from their original sources.

## Data and scope

This project re-analyzes **already-public** benchmarks and introduces no new
patient data. The detectors are intended as cohort-level, benchmark-integrity
tools; a positive overlap signal is statistical circumstantial evidence, not
proof of training-set inclusion, and should not be used to make definitive
claims about a specific model's training data.

## Citation

If you use this code or its findings, please cite the paper (see
[`CITATION.cff`](CITATION.cff)).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
