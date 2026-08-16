# FDCA Visual P3S Final Independent Audit

## Verdict

`PASS_P3S_INDEPENDENT_ARTIFACT_AUDIT_WITH_MINOR_REPORTING_QUALIFICATIONS`

The frozen P3S outcome is supported:

- validity: `PASS_FDCA_VIS_P3S_VALIDITY`
- R1-R4: all PASS
- replication outcome: `SECOND_CHECKPOINT_PERCEPTUAL_BRIDGE_REPLICATED`
- overall: `PASS_FDCA_VIS_P3S_STRONG_REPLICATION`
- recommendation: `RECOMMEND_PAPER_INTEGRATION_AND_FREEZE`

No load-bearing contradiction was found.

## Archive integrity

| archive | bytes | SHA-256 | CRC | package manifest |
|---|---:|---|---|---|
| Full | 22,116,505 | `58c200af8e51bc6320321422c847e91b29cc58454d541acf3403ffc7960a0298` | PASS | 1,431/1,431 |
| Summary | 12,204,312 | `cb48c2f53a5f2903b5103adfff747f18546e8472e48728d2ad3f386ce03798df` | PASS | 151/151 |

The Full run manifest verified 1,429/1,429 non-self-referential run files. All files shared by Full and Summary were byte-identical except their package-specific manifest and checksum containers.

## Preregistration and power

- P2P predecessor tree hash was unchanged before/after P3S.
- The P3S scientific contract and power analysis were hashed before outcomes.
- Candidate block counts were 32, 40, 48.
- The planning effect was fixed at 50% of the P2P paired difference.
- 40 blocks were correctly selected as the smallest candidate with >=90% planned power for R1-R4; R4 was the limiting hypothesis at n=32.
- Scientific population: 40 new blocks; smoke: 4; official-CFG sensitivity: first frozen 16 scientific blocks.
- Duplicate class-seed pairs: 0; predecessor overlap: 0.

## Checkpoint and scope

- official checkpoint: `ImageNet_256_small.pth`
- recorded size: 832,274,338 bytes
- SHA-256: `760b9937d16157e7e518fbf405efe281552933242b552d5e8262cb10507fb0e2`
- resolved Hugging Face revision: `ffc7fd4c5fc7b16010acc4aa342310484d7de62a`
- runtime parameter count: 69,344,769
- official family label: H-MaskGIT-S / 69M

The checkpoint bytes were intentionally excluded from the review archives. This audit verified ledger/hash consistency and the frozen input-verification record, but did not independently rehash the external 832 MB checkpoint file.

## Independent numerical checks

Frozen R1-R4 values are arithmetically consistent across the execution summary, replication summary CSV, gate JSON, report, and secondary implementation audit.

| test | rho=.5 | rho=.3 | difference | ratio | 95% CI | pass |
|---|---:|---:|---:|---:|---|---|
| R1 conditional LPIPS | 0.0307559003 | 0.0103801444 | 0.0203757559 | 2.96295 | [0.0171418933, 0.0238055077] | PASS |
| R2 unconditional LPIPS | 0.0041055879 | 0.0021395708 | 0.0019660171 | 1.91888 | [0.0014987960, 0.0024774863] | PASS |
| R3 conditional DINO | 0.0139895851 | 0.0045685112 | 0.0094210738 | 3.06218 | [0.0072970696, 0.0119191596] | PASS |
| R4 matched timing LPIPS | 0.0322709419 | 0.0100364194 | 0.0222345225 | 3.21538 | [0.0160275512, 0.0288621023] | PASS |

Effect retention relative to P2P Tiny was 1.019, 1.147, 0.635, and 0.886 for R1-R4 respectively. The DINO absolute effect attenuated, but its reverse-time ratio remained approximately three.

The packaged secondary implementation independently reconstructed all four mean differences with maximum absolute discrepancy 0.

## Token-to-perceptual link

- rows: 640
- blocks: 40
- LOBO leakage: 0
- S0 MAE: 0.0128024061
- S1 MAE: 0.0069640337
- S1/S0 ratio: 0.543963
- block improvement 95% CI: [0.0049141340, 0.0067896425]
- within-rho Spearman: 0.583640
- 95% CI: [0.501120, 0.663510]

The secondary token-to-perceptual result therefore replicated under the frozen rule.

## Validity and controls

- scientific pair rows: 1,040
- complete seed token-pair rows: 55,314
- terminal grid files: 1,040
- zero controls: 112
- zero-control LPIPS maximum: 0
- zero-control DINO maximum: 0
- zero-control token-grid mismatches: 0
- zero-control image-hash mismatches: 0
- selected-MASK events: 0
- clamp mismatch delta: 0
- probability non-finite count: 0
- maximum primary probability-sum error: 4.10e-7
- runtime semantic-key uses: 426,070
- unclassified duplicate keys: 0
- missing/unexpected keys: 0
- mismatch/logging audits: PASS
- generator peak VRAM: 370,888,704 bytes
- metric/decode peak VRAM: 8,493,245,440 bytes

The review packages contain complete pre/post-clamp hashes and grids for every paired output. The P2P predecessor tree remained byte-identical.

## Official-CFG sensitivity

The secondary 16-block CFG=1.0 panel was directionally consistent:

- conditional LPIPS difference: 0.0291319, ratio 3.4235, CI [0.0209855, 0.0369047]
- unconditional LPIPS difference: 0.0016860, ratio 1.9176, CI [0.0010347, 0.0022662]
- conditional DINO difference: 0.0155568, ratio 2.4066, CI [0.0057994, 0.0244906]

It remains secondary and non-gating.

## Test and reproduction qualification

Two distinct validation counts exist:

1. Actual `pytest` execution: 52 distinct parameterized test categories, 52 PASS, 0 FAIL.
2. `P3S_TEST_ASSERTIONS.json`: 60 generated PASS assertions.

The 60 generated entries are not 60 independent tests: the reproduction script cycles a shorter frozen checklist to produce 60 named assertions. Future papers should report these as `52 pytest tests + 60 generated audit assertions`, not as `60 independent tests`.

The packaged model-free reaggregation reports:

- 1,040 perceptual rows
- 112 decomposition rows
- 640 OOF rows
- maximum secondary-implementation discrepancy: 0

This audit inspected the exact reaggregation and gate code, verified all frozen CSV/JSON arithmetic and manifests, and verified the internal secondary-implementation result. The current audit environment did not independently rerun the Parquet pipeline because no Parquet engine was available; this is an audit-scope qualification, not a discrepancy in the package.

## Final scope

The replication supports checkpoint/scale robustness inside the official H-MaskGIT family. It does not establish:

- architecture-family generality;
- scheduler generality;
- coupling-universal behavior;
- persistent FP32-vs-W8 deployment behavior;
- human perceptual ground truth;
- global/tight certificate claims.

## Final recommendation

`RECOMMEND_PAPER_INTEGRATION_AND_EMPIRICAL_FREEZE`

No further same-family checkpoint or rho sweep is required before paper drafting. The final paper should distinguish 52 actual pytest categories from 60 generated audit assertions and note that external checkpoint bytes were excluded from the public review archive.
