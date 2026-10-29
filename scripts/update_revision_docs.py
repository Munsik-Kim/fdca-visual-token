#!/usr/bin/env python3
"""Keep README and result-ledger numeric tables tied to generated analysis."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- GENERATED_HEADLINE_START -->'
END = '<!-- GENERATED_HEADLINE_END -->'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    table = (ROOT/'public_results/reanalysis_v1_0_1/tables/headline.md').read_text().strip()
    for name in ['README.md','README_KO.md','docs/RESULTS_AT_A_GLANCE.md']:
        path = ROOT/name
        text = path.read_text()
        if text.count(START) != 1 or text.count(END) != 1:
            raise ValueError(f'{name}: expected one generated table region')
        before, middle = text.split(START)
        _, after = middle.split(END)
        expected = before + START + '\n\n' + table + '\n\n' + END + after
        if args.check:
            if text != expected:
                raise SystemExit(f'{name}: generated headline table differs; run update_revision_docs.py')
        else:
            path.write_text(expected)
    print({'pass': True, 'documents': 3, 'mode': 'check' if args.check else 'generate'})


if __name__ == '__main__':
    main()
