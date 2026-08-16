# P1L_SINGLE_SHOCK

- **Question:** Controlled single-shock regime
- **Frozen estimand:** Descendant propagation after one committed-token alternative shock
- **Model/checkpoint:** H-MaskGIT-T
- **Independent unit / size:** 48 new blocks
- **Sampler:** Halton `randomize=False` where a learned model was used
- **Approximation:** realistic symmetric W8 RTN where applicable
- **Frozen result:** `PASS_FDCA_VIS_P1L_STRONG`
- **Preregistration:** `preregistration.yaml` (the file beside this README)
- **Compact evidence:** [`../../public_results/`](../../public_results/)
- **Model-free command:** `make reproduce`
- **Full GPU boundary:** official external assets are required and are not redistributed
- **Negative/scope finding:** Artificial shock; not natural quantization incidence.
