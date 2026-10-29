# FDCA v1.0.1 proof and claim ledger

Status: manual mathematical audit of the revised finite-horizon statements, not formal machine verification. The original `paper/source/FDCA_v1.0.0.tex` remains unchanged. The revised statements and their proofs are together in `paper/source/v1.0.1/FDCA_v1.0.1.tex`, Section 4. Existing exact synthetic tests provide bounded examples; they do not prove the neural assumptions.

## Definitions and assumptions read before the proofs

- A finite position set and codebook, an ordered partition into fixed branch-independent commit batches, and equal initial states define the setting. Token values are never rewritten after commit.
- Each transition law factors over its current batch. Conditional coordinate coupling events are independent when the product split formula is used. The stable-set theorem itself does not require factorization.
- A state must include every variable on which a transition depends. A cache or other branch-specific memory cannot be omitted just because a token grid is logged.
- First divergence is the first unequal state. The no-split symbol survives all finite stages. Full-path hazards condition on no previous split and a common state. Single-anchor `h[b,r]` is conditioned on its anchor state, not persistent-path survival.
- Weighted additive costs in the stable-set theorem have finite nonnegative weights, finite nonnegative coordinate costs, zero diagonal costs and finite diameters. Hamming weights in the DAG theorem are nonnegative and normalized.
- A zero-hazard product contributes zero without asserting that the conditional mean on the zero-probability event is observed or equals zero. `M_split` is undefined if `H=0`.

## Statement-by-statement proof trace

| Item | v1.0.0 issue or retained scope | Revised status | Assumption → proof step → conclusion |
|---|---|---|---|
| First-divergence definition | Classical stopping-time construction; no novelty claim warranted for the definition itself. | Verified as a definition under the specified joint law. | Equal initial states and finite stages identify the first unequal state. Coupling changes the joint law and can change the stopping time's distribution. |
| Future-stable set | Requires equality before the split and no future rewriting in either branch. | Verified as a definition. | Membership guarantees equality through every allowed continuation. The common fixed monotone schedule supplies committed positions as one such set. |
| Stable-set additive-cost theorem | The phrase “any coordinate-additive terminal cost” omitted zero diagonal, nonnegative weights/costs and bounded diameter. The proof's zero-cost and summation steps need these restrictions. Classify the original wording as **overstated**, not the corrected core argument as invalid. | **Verified under explicit revised assumptions.** | Equal states before the split → stable coordinates remain equal → diagonal costs vanish. Each other coordinate cost is bounded by a finite diameter. Multiplication by nonnegative weights and summation preserve inequality. On no split all costs vanish. |
| Full-path onset–consequence proposition | The path identity is sound with integrable cost, zero no-split cost, survival conditioning and conditional definitions. Its following empirical product sentence was ambiguous at the block-population aggregation level. | **Verified identity**, with population aggregation corrected separately. | The events `{tau=k+1}` partition all split paths. Expand expected terminal cost over these events. Condition on survival and common state, then on the split event. The tower property gives each `1[G_k] h_k(X_k) m_k(X_k)` term. It imposes no sign relationship between incidence and consequence. |
| Conditional-form maximal coupling | Coordinate-level maximality does not imply whole-batch/path/LPIPS optimality. Zero-TV residual normalization must be bypassed. | **Verified for one categorical coordinate.** | Reference draw has law `p`; accepting the overlap and sampling the positive `q-p` residual gives law `q`. Disagreement mass is `TV(p,q)`. When TV is zero, return the same draw. |
| Product-form batch split probability | Valid only under factorized marginals and conditional independence of coordinate couplings. | **Verified under those assumptions.** | Each coordinate agrees with probability `1-TV_u`. Independence gives the product probability of no coordinate mismatch; its complement is `h`. This is an analytic probability, not a binomial count estimator. |
| Innovation–influence hybrid inequality | Requires a global class containing both branches and every single-site hybrid. A local measured state panel is insufficient. | **Verified conditional inequality.** | Insert `q(.|X)` between `p(.|X)` and `q(.|Y)` and use TV triangle inequality. The first term is bounded by `s`. Telescope the second along the admissible hybrid path, adding only mismatched earlier coordinates and bounding each by `M_uv`. |
| Clipped DAG theorem | Coordinate maximal coupling must apply to the actual branch conditionals at commit. Coupling, state coverage and schedule assumptions must be stated in the theorem's scope. | **Verified conditional theorem.** | Monotonicity makes terminal coordinate mismatch equal to mismatch at commit. Conditional maximality and the hybrid inequality give `pi_u <= min(1,s_u+sum M_uv pi_v)`. Nonnegative coefficients and induction in commit order give `pi_u <= rbar_u`; nonnegative Hamming weights give the terminal bound. |
| Nilpotent inverse | Original statement already correctly used strict block-triangularity, without contraction. | **Verified; retained.** | With `K` commit blocks, a product of `K` strictly forward block influences is zero, hence `M^K=0`. Multiply `(I-M)` by `sum_{j=0}^{K-1} M^j` to obtain the identity. This proves a finite inverse, not infinite-horizon stability. |
| Single shock versus persistent innovation | Future same-state innovation is zero under a common suffix, but state-mediated influence can remain. Persistent approximation has additional innovations. | **Verified distinction.** | Set the two future kernels equal in the single-shock intervention. They can still disagree when evaluated at different states. Keeping `p` on one persistent branch and `q` on the other generally requires nonzero future source terms. The preserved finite-state counterexample demonstrates failure of a seed-only persistent bound. |
| Neural empirical envelope certificate | Finite local profiles do not establish global suprema or hybrid coverage. | **Incomplete/unproved for neural global certification**; finite diagnostic preserved. | No exhaustive neural state-space coverage is available. P05R's measured envelope is empirically vacuous, and P1M adds little within-anchor predictive value. Neither finding is removed or upgraded. |
| Hamming/token bound to LPIPS | LPIPS is not a token-coordinate additive cost. | **Unproved transfer.** | A valid decoder/metric continuity or Lipschitz bound with constants would be needed. No such constant is established. Observed LPIPS directions cannot fill that proof gap. |
| Confidence sampler, continuous diffusion, new architectures | Token-value rewriting violates monotonicity; continuous variables and caches change state/estimand assumptions. | **Outside current theorem scope.** | Their application conditions must be checked separately. No theorem is extended merely by analogy. |

## Population arithmetic is not a causal theorem

For each block and anchor, `R_b=h_b*m_b`. For equal block weighting,

`H=mean(h)`, `M_equal=mean(m)`, `R=mean(h*m)`, `M_split=R/H` if `H>0`.

Expanding `(h-H)(m-M_equal)` and averaging gives exactly

`R = H*M_equal + covariance_population(h,m)`,

using denominator `n`, not `n-1`. A product of panel means therefore need not equal the mean of block products. In the corrected attribution, the elementary identity

`h1*m1-h0*m0 = (h1-h0)*(m1+m0)/2 + (m1-m0)*(h1+h0)/2`

is applied to every paired block and then averaged. The old printed formula using panel means does not recover this estimator, and its covariance discrepancy is not rounding. This is exact arithmetic, not a causal mediation identification result.

## Code and bounded test connections

| Proof component | Public implementation / test connection | What this can validate |
|---|---|---|
| Stable-set path restriction | `fdca/synthetic/coupled_paths.py::stable_set_pathwise_violations`; `tests/model_free/test_first_divergence.py::test_residual_mask_stable_set_pathwise_violation_is_zero` | Enumerated finite synthetic coupled paths and Hamming costs, under maximal and CRN examples. No general LPIPS transfer. |
| First split / seed mass | `fdca/core/first_divergence.py`; `test_first_divergence_and_seed_distributions_normalize` | Probability normalization on the enumerated example. The full expectation identity is justified by the proof, not normalization alone. |
| Global finite-state suprema | `fdca/core/influence.py::exact_influence`, `hybrid_inequality_violations`; `tests/model_free/test_influence_and_envelopes.py` | Exhaustive single-site and hybrid checks in the finite declared synthetic state classes. Not global neural certification. |
| Clipped recursion and finite inverse | `fdca/core/dag_envelope.py::{clipped_envelope,finite_resolvent_matrix,inverse_resolvent_matrix}` and corresponding tests | Numerical consistency and envelopes for the tested finite DAG models. The algebraic proof establishes nilpotency in the stated general setting. |
| Persistent fresh innovation | `test_persistent_fresh_innovation_counterexample_exists`, `test_persistent_envelope_with_future_innovation_passes` | A preserved counterexample to applying seed-only propagation to persistent approximation, and its source-inclusive envelope. |
| Coordinate coupling | `fdca/core/coupling.py`, `tests/model_free/test_coupling.py`; `test_crn_is_not_silently_promoted_to_maximal_theorem` | Finite categorical marginal/mismatch contracts and a counterexample to silently applying maximal-coupling bounds to CRN. |
| Empirical aggregation | `scripts/reanalyse_revision.py`, generated `results.json` / `audit.json`, and `tests/public_contract/test_revision_statistics.py` | Paired data integrity and numerical identities in the actual public compact tables; empirical arithmetic is separate from theory proof. |

## Evidence classes

The stable-set theorem, expectation identity and conditional DAG envelope are mathematical statements under their declared assumptions. The two checkpoint contrasts and aggregation/schedule diagnostics are empirical/reanalysis results. Schedule opportunity as an explanation is a hypothesis because state, horizon, position and seed composition co-vary. Cross-architecture, alternative-coupling, and precision-restoration experiments are explicitly unexecuted plans. The manuscript does not use one evidence class as proof of another.
