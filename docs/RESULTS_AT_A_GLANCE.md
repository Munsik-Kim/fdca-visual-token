# Results at a glance

| Checkpoint | Contrast | rho 0.5 | rho 0.3 | Ratio | Paired 95% CI for difference |
|---|---:|---:|---:|---:|---:|
| H-MaskGIT-T | Conditional LPIPS | 0.03013 | 0.01013 | 2.98 | [0.01719, 0.02287] |
| H-MaskGIT-T | Incidence-weighted LPIPS | 0.00393 | 0.00221 | 1.77 | [0.00137, 0.00207] |
| H-MaskGIT-S | Conditional LPIPS | 0.03076 | 0.01038 | 2.96 | [0.01714, 0.02381] |
| H-MaskGIT-S | Incidence-weighted LPIPS | 0.00411 | 0.00214 | 1.92 | [0.00150, 0.00248] |

All values map to `public_results/tables/*BLOCK_LEVEL.csv` and are checked by `scripts/reproduce_public_results.py`. P2P H1-H5 and P3S R1-R4 frozen summaries are included under `public_results/summaries/`.
