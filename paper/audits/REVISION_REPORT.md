# FDCA v1.0.0 - Initial Public Release Review

## Status

`v1.0.0` is the first public archival release. Earlier numbered manuscripts
were internal development drafts and were not published on Zenodo or released
on GitHub.

The review did not change a frozen experimental estimate or scientific gate.
It consolidated the final manuscript, clarified assumptions and notation, and
aligned the paper with its public provenance and audit package.

## Main corrections incorporated before publication

1. **Anchor notation.** The manuscript distinguishes the target anchor label
   `rho_tilde` from the attained unresolved fraction `rho_att`; the exact
   32-step schedule table is included.
2. **Theoretical scope.** Product-law factorization is required only for the
   exact split-incidence product formula. The stable-set result is stated
   independently of that assumption. The innovation-influence result explicitly
   requires hybrid-state closure.
3. **Boundary cases.** The zero-hazard convention, all-zero quantization rows,
   and the `TV=0` branch of conditional maximal coupling are explicit.
4. **Quantization contract.** The public paper records symmetric per-output-
   channel RTN, zero point, clipping range, deterministic rounding, and the
   tensors excluded from quantization.
5. **Model provenance.** The paper records the exact upstream source revision,
   checkpoint names, resolved revisions, sizes, hashes, parameter counts,
   grid size, and sampler steps.
6. **Second-checkpoint reporting.** The H-MaskGIT-S predictive results and
   intervals are complete.
7. **Audit arithmetic.** P2P/P3S terminal-grid counts, archive/run manifest
   differences, zero controls, and test/assertion inventories are reconciled.
8. **Claim discipline.** The public text avoids claims of a global certificate,
   a phase transition, source-age causality, coupling universality, persistent
   deployment behavior, or human perceptual ground truth.
9. **Public packaging.** Internal scratch material was removed. The release
   contains the manuscript, source, metadata, license, public revision notes,
   and compact independent-audit artifacts.

## Public version policy

- Zenodo record version: `1.0.0`
- Git tag/release: `v1.0.0`
- Scholarly status: technical preprint; not peer reviewed
- Paper/documentation license: CC BY 4.0
- No prior public version exists.
