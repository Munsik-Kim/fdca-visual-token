#!/usr/bin/env python3
"""Assert that the checked-in and freshly built papers have equal page counts."""
from pathlib import Path
import sys

from pypdf import PdfReader


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_paper_pages.py FROZEN.pdf BUILT.pdf")
    frozen, built = map(Path, sys.argv[1:])
    frozen_pages = len(PdfReader(str(frozen)).pages)
    built_pages = len(PdfReader(str(built)).pages)
    if frozen_pages != built_pages:
        raise SystemExit(f"page mismatch: {frozen_pages} != {built_pages}")
    print(f"pages {frozen_pages}")


if __name__ == "__main__":
    main()
