# P3S_REPLICATION

- **Question:** Second-checkpoint replication
- **Frozen estimand:** Mechanism-matched W8 natural seed and matched timing
- **Model/checkpoint:** H-MaskGIT-S W8
- **Independent unit / size:** 40 primary blocks, 1,040 pairs
- **Sampler:** Halton `randomize=False` where a learned model was used
- **Approximation:** realistic symmetric W8 RTN where applicable
- **Frozen result:** `PASS_FDCA_VIS_P3S_STRONG_REPLICATION`
- **Preregistration:** `preregistration.yaml` (the file beside this README)
- **Compact evidence:** [`../../public_results/`](../../public_results/)
- **Model-free command:** `make reproduce`
- **Full GPU boundary:** official external assets are required and are not redistributed
- **Negative/scope finding:** Same architecture family; official-CFG panel was secondary.
