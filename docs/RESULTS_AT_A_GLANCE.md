# Results at a glance

These results concern one FP32-versus-W8 transition from a common anchor, followed by a common W8 suffix under the fixed 32-step Halton schedule and declared coordinatewise coupling. Rho is target unresolved fraction. The paired difference is always .5 minus .3.

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

The table is generated from [the shared results](../public_results/reanalysis_v1_0_1/results.json) through the reanalysis script and `scripts/update_revision_docs.py`. Its original headline CIs use 20,000 paired-block resamples, stored semantic seeds and higher quantiles. H did not have a frozen headline CI in that procedure; its intervals are in the separately labeled post-hoc supplement.

- [Post-hoc 100,000-resample linear-quantile contrasts](../public_results/reanalysis_v1_0_1/tables/contrasts.csv), with bootstrap ratios and M_split recalculated within each sample.
- [Aggregation diagnostic](../public_results/reanalysis_v1_0_1/tables/aggregation.csv): `R=mean(h*m)`, `M_equal=mean(m)`, `M_split=R/H`, and `R=H*M_equal+population_covariance(h,m)` using 1/n covariance. The Small product-of-means shortfall is substantive, not rounding.
- [Blockwise symmetric attribution](../public_results/reanalysis_v1_0_1/tables/attribution.csv): an exact arithmetic identity, without causal mediation interpretation.
- [Schedule and coordinate TV](../public_results/reanalysis_v1_0_1/tables/schedule.csv): c changes from 14 to 8. Coordinate TV is `expected_seed_count/c`; its primary-panel contrasts appear in contrasts.csv. It does not decrease generally when batch H decreases.
- [Matched timing](../public_results/reanalysis_v1_0_1/tables/matched.csv): Tiny n=24, Small n=40, distinct from natural-shock estimates. Seed identity pairing is documented in execution records but cannot be independently reconstructed from compact token/coordinate data.
- [Subset interactions](../public_results/reanalysis_v1_0_1/tables/interactions.csv): common16 CFG1 minus CFG3 and common24 W4 minus W8 only. Shared directions do not assert equal magnitudes; W4 is exploratory stress near incidence saturation.
- [Exactly six Bonferroni endpoints](../public_results/reanalysis_v1_0_1/tables/bonferroni_six.csv): H, M_equal, R for each of the two primary CFG3 W8 panels; 99.1667% marginal intervals. This does not correct study-wide selection.
- [All Tiny adjacent anchors](../public_results/reanalysis_v1_0_1/tables/tiny_adjacent.csv): R is not increasing at every adjacent rho. The .1 flush anchor is explicitly retained.

The [three-panel figure](../public_results/reanalysis_v1_0_1/figures/main_panels.pdf), [paired contrast figure](../public_results/reanalysis_v1_0_1/figures/paired_contrasts.pdf), [schedule diagnostic](../public_results/reanalysis_v1_0_1/figures/schedule_tv.pdf), and [all-rho Tiny supplement](../public_results/reanalysis_v1_0_1/figures/tiny_all_rho.pdf) also have SVG and PNG outputs.

`make reproduce` checks eight original contrasts at both CI endpoints plus two saved OOF MAE ratios. Saved predictions are not an independent LOBO training reproduction. `make reanalyse` generates this post-hoc supplement; conditional-repeat Monte Carlo error remains in the estimated block outcomes. Block inference assumes independent equally weighted blocks; the compact export cannot identify class clusters or independently establish all predecessor/class-seed disjointness.

See [corrections](REVISION_NOTES_v1.0.1.md), [scientific scope](SCIENTIFIC_SCOPE.md), [proof ledger](PROOF_LEDGER_v1.0.1.md), and [source-to-claim map](SOURCE_TO_CLAIM_MAP.md) for interpretation boundaries. Frozen P2P H1–H5 and P3S R1–R4 remain in their original summary files; v1.0.0 DOI metadata remains unchanged.
