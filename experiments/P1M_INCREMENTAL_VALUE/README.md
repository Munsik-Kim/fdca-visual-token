# P1M_INCREMENTAL_VALUE

- **Question:** Model-free incremental analysis
- **Frozen estimand:** Whether frozen profiles add within-rho predictive value
- **Model/checkpoint:** Frozen P1L
- **Independent unit / size:** 48 blocks
- **Sampler:** Halton `randomize=False` where a learned model was used
- **Approximation:** realistic symmetric W8 RTN where applicable
- **Frozen result:** `PROFILE_ORDERING_ONLY; REGIME_RESULT_ROBUST`
- **Preregistration:** `preregistration.yaml` (the file beside this README)
- **Compact evidence:** [`../../public_results/`](../../public_results/)
- **Model-free command:** `make reproduce`
- **Full GPU boundary:** official external assets are required and are not redistributed
- **Negative/scope finding:** Profile was secondary rather than a global certificate.
