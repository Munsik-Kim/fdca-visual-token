# v1.0.1 validation record

Base commit: `00c56527716c89803b79d4e1e2fcc6692afd5d78`. Dedicated branch: `fix/scientific-accuracy-v1.0.1`. This record concerns an unreleased GitHub revision candidate. It does not assert a tag, release, merge or Zenodo upload.

## Scientific and CPU checks

| Command / check | Observed outcome |
|---|---|
| `make test reproduce` | **163 tests passed**; eight original paired contrasts matched n, means, difference, ratio and **both CI endpoints** under their stored semantic seeds / 20,000 draws / higher quantiles. Two stored OOF MAE ratios matched. |
| `python scripts/reanalyse_revision.py --check` | **PASS**; temporary regeneration matched numeric results, generated CSV/Markdown/LaTeX tables, input/script hashes, panel/subset contracts and the six-endpoint family. Figure formats regenerated; cross-platform font-dependent image bytes are not asserted identical by this check. |
| `python scripts/update_revision_docs.py --check` | **PASS** for both README tables and results ledger. |
| Independent numeric audit | H/M_equal/R/M_split bootstrap endpoints independently recalculated from CSVs with declared seeds; schedule tuples compared with both preserved execution schedule audits. |
| `pytest -q tests/public_contract/test_paper_revision.py` after final compilation | **6 passed**; revision source/input-set/output receipt and error detection checked. |
| `make verify` | **PASS**; no privacy/secret/prohibited-asset/size/DOI/manifest finding; all 43 historical file hashes preserved. |
| `git diff --check` | **PASS**. |

The full test run occurred before the last manuscript-only wording/float-placement improvements; final paper build, receipt checks, six paper tests and page review were then rerun. Model-free results and analysis code were unchanged by those manuscript edits.

The [analysis audit](../public_results/reanalysis_v1_0_1/audit.json) records exact library versions, public input/script SHA-256 and seed mapping. Input means and uncertainty remain estimated on their declared block population. Complete raw class/seed identities, coordinate probability laws, decoded images and historical runtime scale/weight tensors were not reconstructed. LOBO feature extraction and model training were not replayed. The auxiliary reanalysis ZIP was unavailable in the supplied workspace; public CSVs and explicit stable seeds were used.

## Paper and historical preservation

`make paper` passed the legacy frozen check with **20 archived / 20 rebuilt pages**, without overwriting v1.0.0 PDF/source. `make paper-revision` compiled **17 pages**, regenerating BibTeX from the revision bibliography and saving the actual new PDF and .bbl. This build has **0 undefined references/citations, 0 missing glyphs and 0 overfull boxes**. Ordinary underfull/font-request notices are not presented as a warning-free build.

The final PDF SHA-256 is `99aa82981dec73d81a277cc9963edfb44a00e7cc41de14aa4a78cfea4fc5847b`. [Compile provenance](../paper/build_v1.0.1.json) binds 34 input hashes, the fresh .bbl, engine and PDF. The build rejects source changes during compilation or changes in the input set. The original v1.0.0 PDF hash remains `fb4d453af04603c989486efa10b905d1442a880acdcd668e6482a816e5451696`.

`python scripts/render_paper_review.py` rendered every page at **120 dpi**. All 17 pages were inspected for table/figure clipping, symbols, caption placement, font legibility and bibliography/reference layout. Changed pages were rechecked after the final build. Figure 4 now appears before the historical appendix/references. [Visual review receipt](../paper/visual_review_v1.0.1.json) records final PDF and per-page render hashes; render images/contact sheets remain in ignored `build/paper-review-v1.0.1/` and are reproducible locally.

[Historical asset receipt](evidence/HISTORICAL_ASSETS_v1.0.0.json) preserves 43 original files and records read-only GitHub v1.0.0 release-asset digests. Neither archived release asset was replaced. `CITATION.cff`, `.zenodo.json`, original raw tables/results, preregistrations and v1.0.0 paper assets remain byte-identical. The current checkout's non-self-referential public manifest is regenerated separately and labeled `v1.0.1-unreleased`.

No GPU inference, model training, model-weight download, image generation or new perceptual measurement was run. The revised paper's follow-up experiments are proposals marked NOT RUN. Source-level quantizer granularity is resolved; historical runtime tensor attestation, full class/seed/coordinate identity revalidation, independent LOBO refitting and held-out practical/human validation remain outside the completed checks.
