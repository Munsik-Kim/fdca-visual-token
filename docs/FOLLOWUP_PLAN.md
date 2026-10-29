# Follow-up research plan — NOT RUN

This is a proposed research program, not results in v1.0.1. No GPU inference, training, model-weight download, new image generation, new architecture evaluation, or human study was performed for the accuracy revision. Pilot estimates, feasibility checks, intervention contracts, primary endpoints, decision rules and multiplicity families must be frozen before confirmatory outcomes. A particular block count is not guaranteed sufficient in advance.

## P1: schedule and opportunities within the same model

First compare the existing commit-batch sequence against uniform c=8 over 32 steps, crossed with Halton ordering versus a pre-frozen permutation (2×2). A changed schedule also changes prefix states; this experiment estimates a schedule intervention, not a pure causal effect of c.

Add a diagnostic arm at the *same anchor state*: compare p versus q on nested, preselected c=1,4,8 coordinates; draw all remaining transition coordinates from common p, then propagate both branches with common q. This arm controls the number of opportunities at a fixed state and is deliberately different from the natural whole-transition quantization intervention. Record coordinate identities and selection, seed counts, severity, conditional m and R.

Freeze criteria to determine whether equal-c removes the h ordering and whether m/R contrasts survive adjustment or matching for seed count and severity. A confidence interval that overlaps zero is inconclusive, not proof that the ordering disappeared or two settings are equivalent. If the predeclared equal-c criterion fails or controlled m/R differences fail to exceed a predeclared meaningful threshold, narrow the proposed schedule explanation. Remaining horizon, state, location and seed composition must remain explicit alternative explanations.

## P2: a second discrete generation structure

LlamaGen single-token AR is the first candidate. Confirm the official checkpoint/tokenizer and execution feasibility before committing resources; GPU memory and runtime are unknown. Define a c=1 single-shock/common-suffix experiment within the new architecture. The AR state includes the KV cache. For a token-mediated arm, reconstruct both caches from the relevant canonical prefix using the common propagation model; keep this distinct from a persistent-cache arm that carries hidden-state differences forward.

Report within-model paired effects and uncertainty; do not pool raw LPIPS indiscriminately across models. Failure to reproduce h↓ does not invalidate the use of separate h and m estimands, but it *does* falsify a universal replication of the current sign pattern. Do not extend the current theorem without checking fixed schedule, state and coupling assumptions. Continuous diffusion or MAR with thresholded onset would introduce a different estimand, requiring its own justification.

## P3: non-saturated quantization and coupling sensitivity

The frozen source audit identifies per-output-channel RTN; reproduce runtime granularity/weight provenance before a new quantization experiment. Select W6 **or** one alternative calibration method on development data, then freeze it before held-out outcomes. Keep saturated W4 as stress evidence.

For every alternative coupling, first check marginals and the exact self-case, then report the dependence of h and paired discrepancy on the coupling. Falsifiers include failure of marginal/self-case validity and disappearance or reversal of the claimed contrast under preselected valid alternatives. A coupling change cannot be treated as an invariant replication of the same paired estimand.

## Practical value: separate prospective restoration test

Preserve an existing layer×time plan if one is available. Distinguish (a) one module persistently W8, (b) one transition shock, and (c) restoring one module to FP32 within all-W8. Their effects need not be symmetric or additive.

Compare any proposed selection rule with local MSE, logit KL/TV, h-only, seed count/severity, simple early protection and equal-cost random selection. Do not use post-shock descendant features as online inputs. Freeze the selection policy on a development panel, then evaluate restoration gain on entirely new class/seed blocks at matched parameter bytes or measured cost. A lack of additional gain over the strongest matched-cost baseline narrows FDCA's practical contribution to measurement and diagnosis. Keep feasibility/variance pilots separate and choose confirmatory sample sizes using the effect target, power and multiplicity plan.

If human evaluation is later added, separate blind paired similarity from quality/preference and use representative sampling. Existing LPIPS/DINO discrepancies do not constitute this human validation.
