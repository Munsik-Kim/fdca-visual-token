# First-Divergence Consequence Analysis for Masked Visual Generators

FDCA measures how often one quantization transition changes discrete token choices and how much the resulting paired outputs differ. It combines an explicit coupling, natural quantization seeds, a common propagation kernel, audited random numbers, and inference over paired blocks.

**Munsik Kim** · [ORCID 0009-0008-9350-2435](https://orcid.org/0009-0008-9350-2435)

**Technical preprint; not peer reviewed.** Current GitHub revision candidate: **v1.0.1, unreleased**. [Revised PDF](paper/FDCA_v1.0.1.pdf) · [Revised source](paper/source/v1.0.1/FDCA_v1.0.1.tex) · [Corrections and evidence](docs/REVISION_NOTES_v1.0.1.md) · [한국어](README_KO.md)

The archived **v1.0.0** remains available as [PDF](paper/FDCA_v1.0.0.pdf), [source](paper/source/FDCA_v1.0.0.tex), and [Zenodo record](https://zenodo.org/records/21965747). DOI [10.5281/zenodo.21965747](https://doi.org/10.5281/zenodo.21965747) identifies that archived release; no new DOI or release is assigned to this revision.

## What the experiment measures

From a common anchor state, compare **one FP32-versus-W8 seed transition**, then propagate both branches with the same W8 kernel. The schedule is fixed at 32 Halton steps. Rho is the **target unresolved token fraction**: larger rho is earlier in generation. Within each panel, the same class–generation-seed block is paired at rho .3 and .5.

For block b, `h[b]` is the analytic batch split probability under independent coordinatewise coupling; `m[b]` is the estimated split-conditional LPIPS. Report `H = mean(h)`, `M_equal = mean(m)`, and `R = mean(h*m)`. The split-weighted conditional mean is `M_split = R/H`. In general, `R = H*M_equal + population_covariance(h,m)`; multiplying the two reported means does not recover R.

## Primary CFG3 W8 results

<!-- GENERATED_HEADLINE_START -->

| Panel | Metric | n | rho .3 | rho .5 | Delta (.5-.3) | Legacy 95% CI |
|---|---|---:|---:|---:|---:|---|
| Tiny CFG3 W8 | H | 48 | 0.213227 | 0.129756 | -0.083471 | Not a frozen headline endpoint; see post-hoc supplement |
| Tiny CFG3 W8 | M_equal | 48 | 0.010126 | 0.030126 | 0.020000 | [0.017189, 0.022874] |
| Tiny CFG3 W8 | R | 48 | 0.002213 | 0.003927 | 0.001714 | [0.001368, 0.002066] |
| Small CFG3 W8 | H | 40 | 0.199666 | 0.127957 | -0.071710 | Not a frozen headline endpoint; see post-hoc supplement |
| Small CFG3 W8 | M_equal | 40 | 0.010380 | 0.030756 | 0.020376 | [0.017142, 0.023806] |
| Small CFG3 W8 | R | 40 | 0.002140 | 0.004106 | 0.001966 | [0.001499, 0.002477] |

<!-- GENERATED_HEADLINE_END -->

The table retains the original **20,000-resample, semantic-seed, higher-quantile** paired-block intervals for the original headline endpoints. A separate [post-hoc supplement](public_results/reanalysis_v1_0_1/README.md) uses 100,000 resamples, stable SHA-256 label seeds and linear quantiles, including H intervals, ratios and robustness checks. These CI rules are labeled separately throughout the revision. [Three-panel figure](public_results/reanalysis_v1_0_1/figures/main_panels.pdf) · [paired contrasts](public_results/reanalysis_v1_0_1/figures/paired_contrasts.pdf) · [all Tiny anchors](public_results/reanalysis_v1_0_1/figures/tiny_all_rho.pdf).

H decreases while M_equal and R increase in two separate checkpoints of one family. Commit batch size also changes from 14 to 8; [mean coordinate TV](public_results/reanalysis_v1_0_1/tables/contrasts.csv) does not support a general decrease in local quantization error. LPIPS/DINO measure paired output discrepancy. This is neither a human quality/preference result nor a persistent FP32-versus-W8 generation experiment.

## CPU reproduction and paper build

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r environment/requirements-revision.txt
make verify test reproduce
make reanalyse
make paper-revision              # requires Tectonic; builds and saves the revision PDF/bbl
python scripts/build_paper.py --version 1.0.1 --check
make paper                       # separate historical v1.0.0 page-count validation
```

`make reproduce` checks all eight frozen contrast intervals at **both endpoints** and reaggregates two saved OOF MAE ratios. It does not refit the original LOBO models or reconstruct features. `make reanalyse` generates JSON, CSV, LaTeX tables and PDF/SVG/PNG figures from immutable public tables; `python scripts/reanalyse_revision.py --check` verifies the recorded analysis without modifying canonical outputs. [Reproduction boundaries](docs/REPRODUCIBILITY_TIERS.md) · [compile receipt](paper/build_v1.0.1.json).

## Scope and evidence

Primary n is **48 Tiny / 40 Small** blocks, with 8 conditional repeats per anchor. Small CFG1 (16 blocks, 4 repeats) and Tiny W4 stress (24 blocks, 4 repeats) are overlapping subsets. Matched timing is separate: Tiny n=24 / Small n=40, 2 repeats. Repeats, coordinates, anchors and images are not extra independent blocks. Compact tables lack the complete class/seed manifest, preventing independent class-cluster inference and a full audit of predecessor overlap.

The frozen P2P/P3S runner sources implement **per-output-channel symmetric weight-only RTN fake quantization**. P2P's historical YAML planned per-tensor RTN; this is a documented [plan-to-source deviation](docs/evidence/QUANTIZER_SOURCE_RECEIPT.json). Historical source hashes and imports were checked; execution-time scales and weights were not replayed. No packed-kernel speed or VRAM saving is measured.

P05R's empirical influence envelope was vacuous; P1M's additional within-rho predictive value was limited; P2A remained incomplete and P2B supplied the subsequent audit closure. W4 remains stress-only. [Scientific scope](docs/SCIENTIFIC_SCOPE.md), [experiment ledger](docs/EXPERIMENT_LEDGER.md), [source-to-claim map](docs/SOURCE_TO_CLAIM_MAP.md), and [proof ledger](docs/PROOF_LEDGER_v1.0.1.md) retain these boundaries. Follow-up experiments in the revised paper are **proposals, not executed results**.

External checkpoints, VQ/metric weights, datasets, trajectories, environments and private logs are excluded. [External asset registry](docs/EXTERNAL_ASSET_REGISTRY.md). No new GPU inference, training, model download or image generation was performed for this revision.

## Citation and licenses

The release citation remains historical:

```bibtex
@article{kim2026fdca, title={First-Divergence Consequence Analysis: From Quantization Onset to Perceptual Impact in Masked Visual Generators}, author={Kim, Munsik}, year={2026}, doi={10.5281/zenodo.21965747}}
```

When referring to this unreleased revision, also identify its GitHub commit and v1.0.1 source. `CITATION.cff` and `.zenodo.json` continue to describe v1.0.0. Original code is MIT. Paper, figures, tables, release documentation and included audits are CC BY 4.0. [License scope](LICENSE_SCOPE.md) · [third-party notices](THIRD_PARTY_NOTICES.md).
