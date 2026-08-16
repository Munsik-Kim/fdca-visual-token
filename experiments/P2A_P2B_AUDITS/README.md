# P2A_P2B_AUDITS

- **Question:** RNG and MASK/clamp closure
- **Frozen estimand:** Semantic-key duplicates, complete token pairs, MASK law and pre/post-clamp accounting
- **Model/checkpoint:** Frozen P2N / one-step replay
- **Independent unit / size:** complete selected coordinates
- **Sampler:** Halton `randomize=False` where a learned model was used
- **Approximation:** realistic symmetric W8 RTN where applicable
- **Frozen result:** `P2A incomplete; P2B clean`
- **Preregistration:** `preregistration.yaml` (the file beside this README)
- **Compact evidence:** [`../../public_results/`](../../public_results/)
- **Model-free command:** `make reproduce`
- **Full GPU boundary:** official external assets are required and are not redistributed
- **Negative/scope finding:** P2A incompleteness is preserved; P2B closed the audit.
