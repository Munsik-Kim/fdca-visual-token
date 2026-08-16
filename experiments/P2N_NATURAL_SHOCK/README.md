# P2N_NATURAL_SHOCK

- **Question:** Natural quantization-seed bridge
- **Frozen estimand:** One FP32-vs-W8 transition followed by common W8 suffix
- **Model/checkpoint:** H-MaskGIT-T
- **Independent unit / size:** new disjoint blocks
- **Sampler:** Halton `randomize=False` where a learned model was used
- **Approximation:** realistic symmetric W8 RTN where applicable
- **Frozen result:** `PASS_FDCA_VIS_P2N_STRONG`
- **Preregistration:** `preregistration.yaml` (the file beside this README)
- **Compact evidence:** [`../../public_results/`](../../public_results/)
- **Model-free command:** `make reproduce`
- **Full GPU boundary:** official external assets are required and are not redistributed
- **Negative/scope finding:** Coupling- and common-suffix-specific.
