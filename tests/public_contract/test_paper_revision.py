"""Revision output provenance and staging contracts, without compiling TeX."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("build_paper", ROOT / "scripts/build_paper.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_versioned_input_paths_do_not_scan_directories_as_files():
    paths = build.input_files("1.0.1")
    assert paths
    assert all(path.is_file() for path in paths)
    assert build.source_path("1.0.0").parent != build.source_path("1.0.1").parent
    assert not any(p.suffix == ".bbl" for p in paths)


@pytest.mark.parametrize("message", [
    "LaTeX Warning: There were undefined references.",
    "LaTeX Warning: Citation `absent' on page 2 undefined on input line 10.",
    "Missing character: There is no character in font",
    "Overfull \\hbox (3.0pt too wide) at lines 2--3",
])
def test_revision_rejects_unresolved_or_overflowing_output(message):
    with pytest.raises(RuntimeError):
        build.inspect_log(message)


def test_checked_in_revision_matches_compile_inputs_and_pdf():
    assert build.check_receipt("1.0.1")["pass"]
