#!/usr/bin/env python3
"""Replay published scalar evidence only. No models, downloads, or GPU calls."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=2e-12, abs_tol=2e-14):
        raise AssertionError((actual, expected))


def compare_tree(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare_tree(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            compare_tree(a, e)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        close(actual, expected)
    else:
        assert actual == expected, (actual, expected)


def indices(seed, n, expected_hash):
    draw = np.random.Generator(np.random.PCG64(seed)).integers(n, size=(20000, n), dtype=np.int64)
    assert hashlib.sha256(draw.tobytes()).hexdigest() == expected_hash
    return draw


def check_interval(x, draw, row, lower='lower', upper='upper'):
    confidence = float(row['confidence'])
    q = [(1-confidence)/2, (1+confidence)/2]
    lo, hi = np.quantile(x[draw].mean(axis=1), q, method='linear')
    close(x.mean(), row['estimate'])
    close(lo, row[lower])
    close(hi, row[upper])


def replay_context():
    path = ROOT / 'evidence/tables'
    units = rows(path / 'm_confirm_units.csv')
    metadata = json.loads((path / 'm_confirm_analysis.json').read_text())
    classes = sorted({int(r['class_id']) for r in units})
    assert classes == metadata['classes_analysis_order'] and len(units) == 256
    assert len({r['unit_id'] for r in units}) == 256
    conditions = ['SPARSE', 'REAL_H', 'SHUF_H', 'REAL_L', 'SHUF_L']
    values = {}
    for condition in conditions:
        for row in units:
            bits = row['bits_'+condition]
            assert len(bits) == 32 and set(bits) <= {'0', '1'}
            assert bits.count('1') == int(row['N_'+condition])
            close(int(row['N_'+condition])/32, row['Y_'+condition])
            assert int(row['active_H']) == 74 and int(row['active_L']) == 37
        values[condition] = np.array([
            np.mean([float(r['Y_'+condition]) for r in units if int(r['class_id']) == c])
            for c in classes])
        assert all(sum(int(r['class_id']) == c for r in units) == 8 for c in classes)
    contrasts = {
        'C_ctx': values['SPARSE']-values['REAL_H'],
        'C_layout': values['SHUF_H']-values['REAL_H'],
        'C_dose': ((values['REAL_L']-values['REAL_H'])+
                   (values['SHUF_L']-values['SHUF_H']))/2,
        'C_shuffle': values['SPARSE']-values['SHUF_H'],
        'C_layout_L': values['SHUF_L']-values['REAL_L'],
    }
    contrasts['interaction'] = contrasts['C_layout']-contrasts['C_layout_L']
    draw = indices(2026111001, 32, metadata['bootstrap_draw_sha256'])
    for row in rows(path / 'm_confirm_contrasts.csv'):
        check_interval(contrasts[row['contrast']], draw, row, 'ci_low', 'ci_high')
        close(contrasts[row['contrast']].mean()*32, row['estimate_tokens_per_32'])
    return {'classes': 32, 'paired_units': 256, 'contrasts': 6, 'CI_endpoints': 12,
            'bootstrap_index_sha256': hashlib.sha256(draw.tobytes()).hexdigest()}


def replay_restoration():
    path = ROOT / 'evidence/tables'
    endpoints = rows(path / 'r_test_endpoints.csv')
    metadata = json.loads((path / 'r_test_analysis.json').read_text())
    classes = sorted({int(r['class_id']) for r in endpoints})
    assert classes == metadata['classes'] and len(classes) == 320 and len(endpoints) == 5760
    keyed = {(int(r['class_id']), int(r['replicate_id']), r['policy_id']): r for r in endpoints}
    actions = ['Q', 'E2', 'M2', 'L2', 'U2', 'E3', 'M3', 'L3', 'U3']
    assert len(keyed) == len(endpoints)
    assert set(keyed) == {(c, rep, a) for c in classes for rep in (0, 1) for a in actions}
    assert all(r['status'] == 'COMPLETE' for r in endpoints)
    for c in classes:
        for rep in (0, 1):
            assert len({keyed[c, rep, a]['reference_terminal_hash'] for a in actions}) == 1
    gain = {a: np.array([np.mean([
        float(keyed[c, rep, 'Q']['D'])-float(keyed[c, rep, a]['D']) for rep in (0, 1)
    ]) for c in classes]) for a in actions if a != 'Q'}
    policies = json.loads((path / 'r_test_policy.json').read_text())['policies']
    assert policies == {'Bstar': 'U3', 'D': 'E2', 'F': 'E2', 'L': 'M3', 'S': 'M3'}
    contrasts = {'A_F': gain['E2'], 'B_F': gain['E2']-gain['M3'],
                 'A_S': gain['M3'], 'B_S': gain['M3']-gain['U3'],
                 'A_D': gain['E2'], 'B_D': gain['E2']-gain['U3']}
    draw = indices(2026111004, 320, metadata['bootstrap_draw_sha256'])
    for row in rows(path / 'r_test_primary.csv'):
        check_interval(contrasts[row['contrast']], draw, row)
    matched = {}
    for block in (2, 3):
        for a, b in [('E', 'M'), ('E', 'L'), ('M', 'L')]:
            matched[f'{a}{block}_minus_{b}{block}'] = gain[f'{a}{block}']-gain[f'{b}{block}']
        matched[f'E{block}_minus_U{block}'] = gain[f'E{block}']-gain[f'U{block}']
    for period in 'EMLU':
        matched[f'{period}3_minus_{period}2'] = gain[f'{period}3']-gain[f'{period}2']
    matched['block_by_early_mid'] = (gain['E3']-gain['M3'])-(gain['E2']-gain['M2'])
    for row in rows(path / 'lt10_13_contrasts.csv'):
        check_interval(matched[row['contrast']], draw, row, 'ci_low', 'ci_high')
        adjusted = dict(row, confidence=row['bonferroni_confidence'])
        check_interval(matched[row['contrast']], draw, adjusted,
                       'bonferroni_13_lower', 'bonferroni_13_upper')
    return {'classes': 320, 'units': 640, 'endpoints': 5760,
            'R_test_primary_contrasts': 6, 'LT10_contrasts': 13,
            'CI_endpoints': 64, 'bootstrap_index_sha256': hashlib.sha256(draw.tobytes()).hexdigest()}


def replay_lt16(output):
    source = ROOT / 'evidence/lt16'
    spec = importlib.util.spec_from_file_location('frozen_lt16', source / 'analyse_lt16.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tables, numerical, primary, auxiliary = module.compute(
        rows(source / 'inputs/SELECTOR_MANIFEST.csv'),
        rows(source / 'inputs/UNIQUE_STATE_MANIFEST.csv'),
        rows(source / 'numeric/STATE_DRAWS.csv'))
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        module.write_csv(output / name, table)
        # Frozen scalar tables must be byte-identical, not just rounded matches.
        assert digest(output / name) == digest(source / 'expected' / name), name
    expected = json.loads((source / 'expected/analysis.json').read_text())['numerical']
    compare_tree(numerical, expected)
    (output / 'analysis.json').write_text(json.dumps({'numerical': numerical}, indent=2)+'\n')
    np.save(output / 'primary_bootstrap_index.npy', primary, allow_pickle=False)
    np.save(output / 'auxiliary_bootstrap_index.npy', auxiliary, allow_pickle=False)
    subprocess.run([sys.executable, '-B', str(source / 'independent_statistics_lt16.py'),
                    '--root', str(source), '--analysis', str(output),
                    '--output', str(output / 'independent_statistics.json')], check=True)
    independent = json.loads((output / 'independent_statistics.json').read_text())
    return {'tables_byte_identical': len(tables), 'numerical_JSON': 'MATCH',
            'bootstrap_arrays': 2, 'independent_values': independent['numeric_values_compared'],
            'counts': numerical['counts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert not out.is_relative_to(ROOT), 'Write generated results outside the published evidence folder.'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'data/source_manifest.json').read_text())
    for entry in manifest['sources']:
        assert digest(ROOT / entry['published_path']) == entry['published_sha256'], entry['published_path']
    result = {'status': 'PASS', 'source_files_sha256': len(manifest['sources']),
              'M_confirm': replay_context(), 'R_test_and_LT10': replay_restoration(),
              'LT16': replay_lt16(out / 'lt16'),
              'scope': 'Published scalar values, paired aggregation, bootstrap draws and intervals only.',
              'exclusions': ['no model/decoder/LPIPS execution', 'no full raw-law audit',
                             'no new experiments', 'no reserved holdout access']}
    (out / 'PUBLIC_REPLAY.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
