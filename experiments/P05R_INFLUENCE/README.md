# P05R_INFLUENCE

- **Question:** Scalar influence microaudit
- **Frozen estimand:** Direct innovation and local single-site influence under scalar FP32
- **Model/checkpoint:** H-MaskGIT-T
- **Independent unit / size:** 32 blocks; 24 calibration/8 holdout
- **Sampler:** Halton `randomize=False` where a learned model was used
- **Approximation:** realistic symmetric W8 RTN where applicable
- **Frozen result:** `PASS_FDCA_VIS_P05R_NEGATIVE`
- **Preregistration:** `preregistration.yaml` (the file beside this README)
- **Compact evidence:** [`../../public_results/`](../../public_results/)
- **Model-free command:** `make reproduce`
- **Full GPU boundary:** official external assets are required and are not redistributed
- **Negative/scope finding:** Influence envelope was valid but vacuous.
