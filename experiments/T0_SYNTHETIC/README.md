# T0_SYNTHETIC

- **Question:** Exact theorem/property validation
- **Frozen estimand:** Exact terminal laws, innovation/influence envelopes, stable set, Wasserstein and persistent fresh-innovation counterexample
- **Model/checkpoint:** tiny synthetic decoder
- **Independent unit / size:** exhaustive states/paths
- **Sampler:** Halton `randomize=False` where a learned model was used
- **Approximation:** realistic symmetric W8 RTN where applicable
- **Frozen result:** `PASS_FDCA_VIS_T0_SYNTHETIC`
- **Preregistration:** `preregistration.yaml` (the file beside this README)
- **Compact evidence:** [`../../public_results/`](../../public_results/)
- **Model-free command:** `make reproduce`
- **Full GPU boundary:** official external assets are required and are not redistributed
- **Negative/scope finding:** No claim about a learned visual model.
