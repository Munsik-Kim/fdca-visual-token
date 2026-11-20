# FDCA consolidated results
Public-evidence review: **2026-11-20**. Original study through LT16. No new GPU experiment.

**Controlled context experiments support context-dependent propagation. The tested reference-free precision-restoration policies have not established reliable incremental gains over their specified controls.** Positive oracle diagnostics and unresolved policy results are both retained.

[Complete Korean synthesis](RESULTS_SUMMARY_KO.md) · [Normalized result rows](research/followup_2026_11/data/consolidated_results.csv) · [Detailed chronology](research/followup_2026_11/REPORT_KO.md) · [Methods](research/followup_2026_11/METHODS_KO.md)

## Findings and limits

| Question | Evidence | Limit |
|---|---|---|
| Does shared context change propagation? | M-confirm: 8.6445 fewer mismatches/32, 98.3333% CI [7.1367,10.1992]. LT11: other-class context 6.9727 fewer, 97.5% CI [5.9570,8.1016]; matching the donor class adds 1.7207 [1.3496,2.1211]. | Different panels; LT11 resamples 32 class pairs. Context visibility, values and arrangement are specified interventions. |
| Does an early pulse have larger consequences? | LT3 early-minus-late contrasts: block 2 +0.004121, block3 +0.005000; both adjusted 97.5% CIs positive. OP5 retains the direction with five exposed coordinates. | Pulse discrepancy R is not restoration gain G. OP5 reuses LT3 classes/parents. |
| Do fixed restoration policies improve reference distance? | LT5 two primary intervals and R-test six primary intervals include zero. LT10 retains all 13 adjusted exploratory comparisons. | Inconclusive direction is not proof of zero effect, equivalence or impossibility. |
| Can reference information help? | LT12 reference-token repair +0.080290 [0.058796,0.103802]; block 2 at oracle-known onset +0.017311 [0.000361,0.035429]; 98.75% CIs. Block 3 unresolved. | Reused 64-class oracle diagnostic. Different information and intervention budgets from a deployable policy. |
| Can current W8 features select a one-step repair? | LT14 proposed-minus-ENTROPY OOF gain +0.000618, below the frozen +0.005 gate. LT15 NOT_STARTED_DEV_CRITERIA; 265 reserved classes. | Development is not independent confirmation; reserved identities are not published. |
| Does the observed best time transfer to fresh randomness? | LT16 OLD_MAX +0.058063 originally, −0.000723 on fresh draws, adjusted 98.3333% CI [−0.004644,+0.003300]. Split-half transfer −0.001295, auxiliary 95% CI [−0.003998,+0.001329]. | Reused 24 classes, 48 saved parent paths, 158 states×8 redraws. Conditional saved-state diagnosis, not online policy performance. |

R is pulse output discrepancy; G is the reduction in LPIPS distance to a generated FP32 reference. Neither is human preference or image quality. W8 is reconstructed in FP32 here, so packed-kernel speed/VRAM benefits were not measured. Studies overlap and their sample counts/effects must not be pooled.

## Reproduction and publication update

The previous public replay checks 68 evidence hashes, M-confirm/R-test/LT10 contrasts and LT16 tables plus 4,653 independently recalculated values. This update adds aggregate results and a consolidated index. Newly recovered unit-level inputs were independently reaggregated locally, including LT6, LT11, LT12, LT13 and the LT14 development gate; those new unit-level inputs and their replay code are not redistributed. The added LT6 aggregate is source-hash linked, but its unit-level replay requires the retained local inputs. See the [aggregate source manifest](research/followup_2026_11/data/aggregate_source_manifest_20261120.json) and [validation boundaries](research/followup_2026_11/EVIDENCE_AND_VALIDATION.md).

```bash
python -B research/followup_2026_11/scripts/replay_public_evidence.py --output build/followup-replay
python -B research/followup_2026_11/scripts/build_results_summary.py --check
make verify test reproduce
```

No checkpoint, full raw probability/image archive, private delivery URL or reserved holdout identity is redistributed. Scalar numerical replay does not rerun the generator, decoder or LPIPS network.

The original paper, v1.0.0 release and DOI remain historical. [PR #2](https://github.com/Munsik-Kim/fdca-visual-token/pull/2) and the [integration draft PR #4](https://github.com/Munsik-Kim/fdca-visual-token/pull/4) were open and unmerged on this review date; this publication does not merge them or create a release. A next development design would average gain labels over independent redraws and separate selection from evaluation randomness. It has not been executed.
