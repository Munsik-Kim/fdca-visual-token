# Experiment ledger

| Phase | Question / estimand | Unit | Frozen result | Public evidence | Negative/scope finding |
|---|---|---|---|---|---|
| T0_SYNTHETIC | Exact theorem/property validation; Exact terminal laws, innovation/influence envelopes, stable set, Wasserstein and persistent fresh-innovation counterexample | exhaustive states/paths | `PASS_FDCA_VIS_T0_SYNTHETIC` | `experiments/T0_SYNTHETIC/`, `public_results/summaries/` | No claim about a learned visual model. |
| P0_CONTRACT | Sampler/source contract audit; Halton branch independence, immutability, schedule counts, RNG paths and instrumentation identity | sampler calls | `PASS_FDCA_VIS_P0_CONTRACT` | `experiments/P0_CONTRACT/`, `public_results/summaries/` | Confidence sampling was excluded from exact theorem claims. |
| P05R_INFLUENCE | Scalar influence microaudit; Direct innovation and local single-site influence under scalar FP32 | 32 blocks; 24 calibration/8 holdout | `PASS_FDCA_VIS_P05R_NEGATIVE` | `experiments/P05R_INFLUENCE/`, `public_results/summaries/` | Frozen empirical envelope was vacuous; finite profiles are not global certificates. |
| P1L_SINGLE_SHOCK | Controlled single-shock regime; Descendant propagation after one committed-token alternative shock | 48 new blocks | `PASS_FDCA_VIS_P1L_STRONG` | `experiments/P1L_SINGLE_SHOCK/`, `public_results/summaries/` | Artificial shock; not natural quantization incidence. |
| P1M_INCREMENTAL_VALUE | Model-free incremental analysis; Whether frozen profiles add within-rho predictive value | 48 blocks | `PROFILE_ORDERING_ONLY; REGIME_RESULT_ROBUST` | `experiments/P1M_INCREMENTAL_VALUE/`, `public_results/summaries/` | Profile was secondary rather than a global certificate. |
| P2N_NATURAL_SHOCK | Natural quantization-seed bridge; One FP32-vs-W8 transition followed by common W8 suffix | new blocks per original execution record; compact class/seed identities unavailable | `PASS_FDCA_VIS_P2N_STRONG` | `experiments/P2N_NATURAL_SHOCK/`, `public_results/summaries/` | Coupling- and common-suffix-specific. |
| P2A_P2B_AUDITS | RNG and MASK/clamp closure; Semantic-key duplicates, complete token pairs, MASK law and pre/post-clamp accounting | complete selected coordinates | `P2A incomplete; P2B clean` | `experiments/P2A_P2B_AUDITS/`, `public_results/summaries/` | P2A incompleteness is preserved; P2B closed the audit. |
| P2P_PERCEPTUAL | Perceptual natural-shock bridge; Split incidence times conditional LPIPS; DINO and timing corroboration | 48 primary blocks × 5 anchors × 8 conditional repeats | `PASS_FDCA_VIS_P2P_STRONG` | `experiments/P2P_PERCEPTUAL/`, `public_results/summaries/` | Operational metrics, no human ratings; W4 stress is not primary. |
| P3S_REPLICATION | Second-checkpoint replication; Mechanism-matched W8 natural seed and matched timing | 40 primary blocks × 2 anchors × 8 conditional repeats | `PASS_FDCA_VIS_P3S_STRONG_REPLICATION` | `experiments/P3S_REPLICATION/`, `public_results/summaries/` | Same architecture family; official-CFG panel was secondary. |

## v1.0.1 interpretation and reanalysis amendment

The frozen gate labels above are historical decisions, not new omnibus validations. The revision preserves every original CSV, JSON, YAML and v1.0.0 paper asset, while correcting aggregate formulas, intervention definitions, panel counts, theory assumptions and related work. [Revision notes](REVISION_NOTES_v1.0.1.md) identify each correction and its evidence.

| Panel | Paired blocks | Anchors | Conditional repeats | Dependence / purpose |
|---|---:|---:|---:|---|
| P2P Tiny CFG3 W8 | 48 | 5 | 8 | Primary; same blocks at all anchors. |
| P3S Small CFG3 W8 | 40 | 2 | 8 | Separate checkpoint panel. |
| P3S Small official CFG1 | 16 | 2 | 4 | Subset of the Small panel, not an extra independent replication. |
| P2P Tiny W4 | 24 | 2 | 4 | Subset of Tiny W8; exploratory saturated stress condition. |
| P2P matched timing | 24 | 2 | 2 | Separate recorded seed-identity timing diagnostic. |
| P3S matched timing | 40 | 2 | 2 | Separate recorded seed-identity timing diagnostic. |

P2P selected the .3/.5 contrast after preceding P2N results. The original execution records describe new blocks for P2P and a new 40-block P3S panel with a conservative power plan. Complete compact class/base-seed identities are not public, so this revision cannot independently certify all predecessor overlap or class clustering. The YAMLs record the plans; their later public availability is not an externally authenticated pre-experiment timestamp.

The original 20,000/higher analysis is replayed with both CI endpoints. The separate [100,000/linear post-hoc supplement](../public_results/reanalysis_v1_0_1/README.md) reports H, M_equal, R, M_split, blockwise attribution, opportunity/TV diagnostics and subset interactions. It does not add experimental observations, refit LOBO, close P2A retroactively, or execute the [follow-up plan](FOLLOWUP_PLAN.md).
