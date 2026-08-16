# First-Divergence Consequence Analysis for Masked Visual Generators

**Munsik Kim** · [ORCID 0009-0008-9350-2435](https://orcid.org/0009-0008-9350-2435)

**Technical preprint; not peer reviewed.** First public release: `v1.0.0`.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21965747.svg)](https://doi.org/10.5281/zenodo.21965747)

[Paper PDF](paper/FDCA_v1.0.0.pdf) · [Zenodo record](https://zenodo.org/records/21965747) · [Korean README](README_KO.md)

FDCA couples a reference masked decoder and an approximate decoder, isolates the first approximation-induced split, and decomposes endpoint risk into exact split incidence and split-conditional consequence. The visual study uses an audited branch-independent Halton schedule, coordinatewise conditional maximal coupling, event-addressed random-number generation, realistic W8 round-to-nearest quantization, and a common approximate suffix after the natural seed.

## Results at a glance

| Checkpoint | Contrast | rho 0.5 | rho 0.3 | Ratio | Paired 95% CI for difference |
|---|---:|---:|---:|---:|---:|
| H-MaskGIT-T | Conditional LPIPS | 0.03013 | 0.01013 | 2.98 | [0.01719, 0.02287] |
| H-MaskGIT-T | Incidence-weighted LPIPS | 0.00393 | 0.00221 | 1.77 | [0.00137, 0.00207] |
| H-MaskGIT-S | Conditional LPIPS | 0.03076 | 0.01038 | 2.96 | [0.01714, 0.02381] |
| H-MaskGIT-S | Incidence-weighted LPIPS | 0.00411 | 0.00214 | 1.92 | [0.00150, 0.00248] |

Headline values are operational LPIPS/DINO outcomes for two checkpoints in one model family, not human perceptual ground truth or architecture-universal guarantees. See [compact evidence](docs/RESULTS_AT_A_GLANCE.md).

## Quickstart (model-free)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r environment/requirements-model-free.txt
make verify test reproduce
```

## Repository map

- `fdca/`: theory, coupling, stable-set/first-divergence, RNG, adapters, perturbation and analysis
- `experiments/`: T0 through P3S purpose, preregistration, gate and negative findings
- `public_results/`: compact row/block evidence, tables and figures
- `paper/`: v1.0.0 PDF and LaTeX/BibTeX source
- `docs/`: architecture, stack, scope, experiment and claim ledgers
- `scripts/` and `tests/`: public validation and model-free reproduction

## Full GPU boundary

Tier-B regeneration requires official checkpoints, VQ and metric assets listed by exact revision/size/SHA-256 in [the external asset registry](docs/EXTERNAL_ASSET_REGISTRY.md). They are not redistributed. Full trajectories, datasets, Conda environments, external clones and private logs are absent.

## Negative and unsupported claims

P05R's valid influence envelope was vacuous; P2A remained incomplete; the P1M profile largely reproduced rho ordering; W4 was stress-only. No claim is made about human judgment, production controllers, persistent FP32-vs-W8 deployment, other schedules, coupling independence, or models outside this two-checkpoint H-MaskGIT family.

## Citation

```bibtex
@article{kim2026fdca, title={First-Divergence Consequence Analysis: From Quantization Onset to Perceptual Impact in Masked Visual Generators}, author={Kim, Munsik}, year={2026}, doi={10.5281/zenodo.21965747}}
```

## Licenses

Original code is MIT. Paper, figures, tables, release documentation and included audits are CC BY 4.0. See [license scope](LICENSE_SCOPE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
