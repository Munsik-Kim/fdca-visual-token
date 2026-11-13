# Evidence, provenance and validation boundaries

Prepared 13 November 2026. [Portfolio](README.md) · [Korean report](REPORT_KO.md)

## What is published

This addendum contains historical summary tables and selected compact scalar inputs. It is not a redistribution of full probability laws, images, model checkpoints, datasets or every GPU execution script. The independently usable deliverable is a detailed research record and a model-free numerical replay of the principal context, restoration and repeated-randomness evidence.

[source_manifest.json](data/source_manifest.json) binds 68 published source artifacts to their original member SHA-256 and published SHA-256, and records the hashes of ten inspected source ZIPs. An unchanged member is marked byte-identical. Path-redacted completion summaries, the projected LT16 state manifest, the R-cal signed table with private-path columns removed, and the block-2 tensor metadata projection are explicitly marked as transformations. Hashes establish byte identity with the inspected artifacts; they do not independently prove when an experiment was run or whether a scientific claim is correct.

All public paths are repository-relative. Private delivery URLs and workstation paths are removed. The 265 reserved holdout class identities are not included. Analyzed class/seed/state identifiers in the numerical inputs are retained to permit paired and clustered replay; these are experimental identifiers, not independent observations merely because they occupy separate rows.

## Claim-to-evidence map

| Claim or decision | Direct source | Public validation level |
|---|---|---|
| LT1 layer ranking and baseline | [arm table](evidence/tables/lt1_arms.csv), [completion record](evidence/summaries/lt1.txt) | Source bytes and report comparison; no new trajectory execution |
| LT1-R absolute gains unresolved; selectors agree | [restoration](evidence/tables/lt1r_restoration.csv), [policies](evidence/tables/lt1r_policies.csv) | Original numerical tables preserved |
| LT2 local/state/pulse distinction | [contrasts](evidence/tables/lt2_contrasts.csv) | Original numerical table preserved |
| LT2C weighted pulse and covariance | [contrasts](evidence/tables/lt2c_contrasts.csv), [summary](evidence/tables/lt2c_pulse.csv) | Original numerical tables preserved |
| LT3 and OP5 early pulse contrasts positive | [LT3](evidence/tables/lt3_contrasts.csv), [LT4](evidence/tables/lt4_contrasts.csv) | Figure values read directly from original rows |
| LT5 policy did not establish gain | [primary](evidence/tables/lt5_primary.csv), [auxiliary](evidence/tables/lt5_auxiliary.csv) | Original numerical tables preserved |
| LT6 original 95% interval | Researcher's completion account; matching point estimate in [M-dev](evidence/tables/m_dev_contrasts.csv) | **Standalone LT6 compact unavailable here; original CI not independently replayed** |
| M-confirm context/layout/dose effects | [256 units](evidence/tables/m_confirm_units.csv), [contrasts](evidence/tables/m_confirm_contrasts.csv) | Counts from bit strings, class means, six contrasts, 20,000 paired draws and 12 CI endpoints replayed |
| R0/R-cal/R-alt diagnostics | [R0](evidence/tables/r0_contrasts.csv), [calibration](evidence/tables/r_cal_scores.csv), [R-alt](evidence/tables/r_alt_contrasts.csv) | Original source tables preserved; not all independently regenerated in this publication |
| R-test six primary intervals include zero | [5,760 endpoints](evidence/tables/r_test_endpoints.csv), [policy freeze](evidence/tables/r_test_policy.json), [primary](evidence/tables/r_test_primary.csv) | Pairing, complete 320×2×9 panel, reference identity, class aggregation and 12 CI endpoints replayed |
| LT10 adjusted 13 comparisons unresolved | [13 comparisons](evidence/tables/lt10_13_contrasts.csv) | Same R-test endpoint inputs, 13×2 intervals (unadjusted and adjusted), 52 endpoints replayed |
| LT11 shared and matched source effects | [contrasts](evidence/tables/lt11_contrasts.csv), [record](evidence/summaries/lt11.txt) | Original numerical table preserved; plotted original adjusted intervals |
| LT12 oracle edit and block-2 exception | [four primary results](evidence/tables/lt12_primary.csv), [record](evidence/summaries/lt12.txt) | All four primary rows retained, including unresolved block 3 |
| LT13 original/redraw distinction | [two panels](evidence/tables/lt13_contrasts.csv), [historical record](evidence/summaries/adaptive.txt) | Original numerical table preserved |
| LT14 gate failed; LT15 not started | [OOF summary](evidence/tables/lt14_policy_summary.csv), [gate](evidence/tables/lt14_dev_gate.json), [observed maximum](evidence/tables/lt14_observed_maximum.json) | Frozen decision and all policy rows retained; no retraining in this publication |
| LT16 maximum did not persist; selection optimism | [draws](evidence/lt16/numeric/STATE_DRAWS.csv), [selector manifest](evidence/lt16/inputs/SELECTOR_MANIFEST.csv), [expected outputs](evidence/lt16/expected/) | 16 CSVs byte-identical, numerical JSON matched, two bootstrap arrays regenerated; independent scalar implementation checked 4,653 values |

## Sample reuse is part of the evidence

| Group | Reuse relationship | Inference boundary |
|---|---|---|
| LT1-R/LT2/LT2C and R-cal | Small development panels, including reused eight-class data | Exploratory selection and calibration |
| LT3 → LT4 → LT6/M-dev | Same 32 experimental classes; parent/early-condition reuse is material | Only LT3 is the new-class step in this chain |
| LT5 → R0 | Same 32-class restoration panel | Duration diagnostic, not new confirmation |
| M-confirm | New 32-class panel relative to development | Confirmatory within specified model and intervention |
| R-test → LT10/R-alt/LT12/LT13 | R-test's 320 classes reused fully or in subsets | Follow-up comparisons are retrospective or diagnostic |
| LT11 | New 64 classes grouped into 32 pairs | Pair-cluster inference; not 512 independent classes |
| LT14 → LT16 | Same 24 development classes, 48 original parents | Frozen-state redraw diagnostic, not held-out policy performance |
| LT15 | Reserved 265 classes, not opened or executed | No performance claim is possible |

The public package does not contain the complete original cross-stage class/seed/coordinate ledger for every early phase. “New” versus “reused” is documented at the available contract/report level; this publication does not claim to have independently audited every historical cross-stage intersection. M-confirm/R-test/LT16 within-panel keys are checked by the public replay. No pooled sample size or meta-analytic effect is reported across these overlapping studies.

## Three distinct meanings of validation

1. **Historical GPU and full-law validation:** the original completion records describe generator execution, stored-law CPU replay, decoder/LPIPS checks where performed, preservation and transfer checks. These checks have different scopes. They are historical evidence, not newly executed checks here.
2. **This publication's numerical replay:** saved scalar distances, mismatch bit strings, class/parent pairing, frozen selector mappings, bootstrap RNG and intervals. No generator, full raw categorical law, decoder, LPIPS network, image or new model weight is evaluated.
3. **Repository checks:** public manifest integrity, existing tests, existing paper-result reproduction, internal file links, privacy screening and rendered figure inspection. These do not establish scientific generalization.

The M-confirm `analysis.json` contains an analysis-time status `AWAITING_INDEPENDENT_AUDIT`. That snapshot is retained byte-for-byte. The later completion account describes completed audits; the new public replay has its own explicit output. These statuses refer to different moments and are not rewritten to imply the original snapshot was a final receipt.

LT13's duplicate preservation-side full-law replay was stopped after a partial run; the earlier polling log and final drained log differ in progress count. Completed local audits and incomplete duplicate replay must not be conflated. LT14's CSV column-order mismatch was corrected after confirming identical values. LT16 included twelve selected model/decoder/LPIPS reruns in its historical technical audit; the present public replay does not repeat them.

## Reproduction commands and outputs

```bash
python -B research/followup_2026_11/scripts/replay_public_evidence.py \
  --output build/followup-replay
```

The wrapper checks all 68 published-source hashes, then writes `PUBLIC_REPLAY.json`. M-confirm uses PCG64 seed 2026111001 and R-test/LT10 uses 2026111004. Their regenerated bootstrap index byte hashes must match the historical analysis metadata. LT16 preserves its original analysis and independent audit implementations; primary and auxiliary seeds are 2026111601 and 2026111602. The original numerical values and CI levels are not retuned to obtain a different conclusion.

LT16's projected state manifest has a new public file hash because private raw paths were removed. The wrapper compares the numerical JSON subtree, not the old input-file-hash wrapper. All 16 generated CSV files must still be byte-identical to the original outputs. Separate independently implemented scalar checks validate 4,653 numerical values with absolute tolerance 2e-14 and relative tolerance 2e-12.

[PUBLIC_VALIDATION.json](data/PUBLIC_VALIDATION.json) is the publication-time numerical receipt. Continuous integration repeats the same replay on Python 3.11 and 3.12. CI status is available in the publication pull request and Actions; this static document does not predict future CI outcomes.

For figures, install matplotlib in addition to the model-free requirements and run `scripts/build_figures.py` from this directory, or its full repository-relative path. [figure_data.csv](data/figure_data.csv) records each plotted value, source row key, unit and CI level. Figures are descriptive presentations of existing evidence, not new statistical analyses.

## Preserved boundaries and remaining gaps

- The archived v1.0.0 paper, source, DOI metadata, release and original data are not replaced. The root README links this addendum and the separate correction PR.
- LT6 standalone original input and the complete seven-stage PAPER-EVIDENCE archive were unavailable during publication preparation. The LT10 delivery bundle includes inherited M-confirm and restoration numerical inputs, enabling the explicitly listed replays. This is not evidence that all missing original files were recovered.
- Independent model-level verification of every earlier categorical law, image, quantizer tensor and LOBO model fit is outside this publication. Statistical replay cannot substitute for those checks.
- Reference LPIPS discrepancy, token agreement and image quality are different objectives. No human-preference, speedup, packed-INT8, multi-model generalization or restoration impossibility claim is made.
- Later proposed plans in historical summaries describe the state at that time. The current consolidated recommendation is [NEXT_RESEARCH_KO.md](NEXT_RESEARCH_KO.md), and is explicitly unexecuted.
