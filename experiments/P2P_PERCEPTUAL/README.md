# P2P_PERCEPTUAL

- **Question:** Perceptual natural-shock bridge
- **Frozen estimand:** Split incidence times conditional LPIPS; DINO and timing corroboration
- **Model/checkpoint:** H-MaskGIT-T W8
- **Independent unit / size:** 48 blocks, 2,496 pairs
- **Sampler:** Halton `randomize=False` where a learned model was used
- **Approximation:** realistic symmetric W8 RTN where applicable
- **Frozen result:** `PASS_FDCA_VIS_P2P_STRONG`
- **Preregistration:** `preregistration.yaml` (the file beside this README)
- **Compact evidence:** [`../../public_results/`](../../public_results/)
- **Model-free command:** `make reproduce`
- **Full GPU boundary:** official external assets are required and are not redistributed
- **Negative/scope finding:** Operational metrics, no human ratings; W4 stress is not primary.
