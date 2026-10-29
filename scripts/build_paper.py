#!/usr/bin/env python3
"""Build an editable revision, or independently check the frozen v1.0.0 pages.

Revision builds start without a .bbl, so BibTeX must run. Local inputs are
staged with repository-relative paths; source subdirectories are supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(version: str) -> Path:
    if version == "1.0.0":
        return ROOT / "paper/source/FDCA_v1.0.0.tex"
    if version != "1.0.1":
        raise ValueError(f"unsupported paper version: {version}")
    return ROOT / f"paper/source/v{version}/FDCA_v{version}.tex"


def input_files(version: str) -> list[Path]:
    source = source_path(version)
    paths = [source, source.parent / "references.bib"]
    if version != "1.0.0":
        generated = ROOT / "public_results/reanalysis_v1_0_1"
        paths += sorted(p for p in generated.rglob("*") if p.suffix in {".tex", ".pdf", ".png", ".csv", ".json"})
        paths += sorted(p for p in (ROOT / "paper/generated/v1.0.1").rglob("*") if p.is_file())
    return paths


def inspect_log(log: str, *, legacy: bool = False) -> dict:
    errors = re.findall(
        r"[^\n]*(?:undefined references|undefined citations|Citation[^\n]*undefined|"
        r"Reference[^\n]*undefined|Missing character:|^![^\n]*|"
        r"I couldn't open database|Warning--I didn't find a database entry)[^\n]*",
        log, re.MULTILINE | re.IGNORECASE,
    )
    overfull = re.findall(r"Overfull \\[hv]box[^\n]*", log)
    if errors or (overfull and not legacy):
        raise RuntimeError(json.dumps({"unresolved_or_missing": errors, "overfull": overfull}, indent=2))
    return {"undefined_references_or_citations": 0, "missing_glyphs": 0, "overfull_boxes": len(overfull)}


def validate_pdf(path: Path) -> int:
    reader = PdfReader(str(path), strict=True)
    if reader.is_encrypted or not reader.pages:
        raise RuntimeError("paper must be a readable nonempty PDF")
    for number, page in enumerate(reader.pages, 1):
        if len((page.extract_text() or "").strip()) < 10:
            raise RuntimeError(f"page {number} has no extractable paper text")
    return len(reader.pages)


def record_revision(version: str, built: Path, bbl: Path, log: str, engine: str, *, staged_hashes: dict | None = None) -> dict:
    if not bbl.is_file() or not bbl.read_text().strip():
        raise RuntimeError("fresh bibliography output is required")
    diagnostics = inspect_log(log)
    pages = validate_pdf(built)
    inputs = {p.relative_to(ROOT).as_posix(): digest(p) for p in input_files(version)}
    if staged_hashes is not None and inputs != staged_hashes:
        raise RuntimeError("paper inputs changed during compilation; rebuild from final sources")
    output = ROOT / f"paper/FDCA_v{version}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    if built.resolve() != output.resolve():
        shutil.copy2(built, output)
    target_bbl = source_path(version).with_suffix(".bbl")
    if bbl.resolve() != target_bbl.resolve():
        shutil.copy2(bbl, target_bbl)
    receipt = {
        "schema": "FDCA_PAPER_BUILD_V2", "version": version,
        "status": "unreleased GitHub revision candidate", "engine": engine,
        "bibliography": "BibTeX regenerated from versioned references.bib",
        "inputs": inputs,
        "bbl_sha256": digest(target_bbl), "pdf_sha256": digest(output),
        "pages": pages, "diagnostics": diagnostics,
        "legacy_page_equality_required": False,
    }
    receipt_path = ROOT / f"paper/build_v{version}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def check_receipt(version: str) -> dict:
    receipt = json.loads((ROOT / f"paper/build_v{version}.json").read_text())
    current_inputs = {p.relative_to(ROOT).as_posix() for p in input_files(version)}
    if set(receipt["inputs"]) != current_inputs:
        raise RuntimeError("paper input set changed since compilation")
    for rel, expected in receipt["inputs"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError(f"paper input changed since compilation: {rel}")
    if receipt["bbl_sha256"] != digest(source_path(version).with_suffix(".bbl")):
        raise RuntimeError("bibliography changed since compilation")
    pdf = ROOT / f"paper/FDCA_v{version}.pdf"
    if receipt["pdf_sha256"] != digest(pdf) or receipt["pages"] != validate_pdf(pdf):
        raise RuntimeError("PDF differs from compile receipt")
    return {"pass": True, "version": version, "pages": receipt["pages"], "input_hashes_checked": len(receipt["inputs"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["1.0.0", "1.0.1"], default="1.0.0")
    parser.add_argument("--check", action="store_true", help="check revision source/PDF compile receipt without rebuilding")
    parser.add_argument("--compiled", type=Path, help="import a CI latexmk build after checking its log and fresh .bbl")
    parser.add_argument("--engine-label", default="latexmk/pdflatex (CI)")
    args = parser.parse_args()
    if args.check:
        if args.version == "1.0.0":
            parser.error("--check applies to revision receipts; default build checks frozen v1.0.0 pages")
        print(json.dumps(check_receipt(args.version), sort_keys=True))
        return
    source = source_path(args.version)
    if args.compiled:
        if args.version == "1.0.0":
            parser.error("use check_paper_pages.py for an externally compiled legacy paper")
        built = args.compiled.resolve()
        log = built.with_suffix(".log").read_text(errors="replace")
        print(json.dumps(record_revision(args.version, built, built.with_suffix(".bbl"), log, args.engine_label), sort_keys=True))
        return
    engine = shutil.which("tectonic")
    if engine is None:
        raise SystemExit("tectonic is required (or use --compiled for a fresh latexmk CI build)")
    engine_version = subprocess.check_output([engine, "--version"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="fdca-paper-") as temporary:
        stage = Path(temporary)
        staged_hashes = {}
        for path in input_files(args.version):
            destination = stage / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            staged_hashes[path.relative_to(ROOT).as_posix()] = digest(destination)
        if args.version == "1.0.0":
            shutil.copy2(source.with_suffix(".bbl"), stage / source.with_suffix(".bbl").relative_to(ROOT))
        staged_source = stage / source.relative_to(ROOT)
        environment = dict(os.environ)
        environment.setdefault("SOURCE_DATE_EPOCH", "1793232000")
        result = subprocess.run(
            [engine, "-X", "compile", staged_source.name, "--keep-intermediates", "--keep-logs", "--untrusted"],
            cwd=staged_source.parent, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        build_log = ROOT / "build" / f"paper-v{args.version}.log"
        build_log.parent.mkdir(exist_ok=True)
        build_log.write_text(result.stdout)
        print(result.stdout)
        result.check_returncode()
        built = staged_source.with_suffix(".pdf")
        log = staged_source.with_suffix(".log").read_text(errors="replace")
        if args.version == "1.0.0":
            inspect_log(log, legacy=True)
            frozen_pages = validate_pdf(ROOT / "paper/FDCA_v1.0.0.pdf")
            built_pages = validate_pdf(built)
            if frozen_pages != built_pages:
                raise RuntimeError(f"legacy page mismatch: {frozen_pages} != {built_pages}")
            print(json.dumps({"pass": True, "version": "1.0.0", "frozen_pages": frozen_pages, "built_pages": built_pages, "source_sha256": digest(source)}))
        else:
            print(json.dumps(record_revision(args.version, built, staged_source.with_suffix(".bbl"), log, engine_version, staged_hashes=staged_hashes), sort_keys=True))


if __name__ == "__main__":
    main()
