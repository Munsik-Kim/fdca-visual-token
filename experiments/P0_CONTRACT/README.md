# P0_CONTRACT

- **Question:** Sampler/source contract audit
- **Frozen estimand:** Halton branch independence, immutability, schedule counts, RNG paths and instrumentation identity
- **Model/checkpoint:** H-MaskGIT-T smoke
- **Independent unit / size:** sampler calls
- **Sampler:** Halton `randomize=False` where a learned model was used
- **Approximation:** realistic symmetric W8 RTN where applicable
- **Frozen result:** `PASS_FDCA_VIS_P0_CONTRACT`
- **Preregistration:** `preregistration.yaml` (the file beside this README)
- **Compact evidence:** [`../../public_results/`](../../public_results/)
- **Model-free command:** `make reproduce`
- **Full GPU boundary:** official external assets are required and are not redistributed
- **Negative/scope finding:** Confidence sampling was excluded from exact theorem claims.
