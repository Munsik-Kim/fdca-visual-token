#!/usr/bin/env python3
"""Replay eight frozen paired contrasts and two stored-OOF MAE ratios.

This does not reconstruct image metrics, original features, or retrain LOBO models.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def paired_values(frame, col, rhos=(.3, .5)):
    """Require complete, unique within-panel block pairs; never silently join."""
    selected = frame[frame.anchor_target_rho.isin(rhos)]
    if selected.duplicated(['block_id', 'anchor_target_rho']).any():
        raise ValueError('duplicate block/anchor pair')
    if selected.empty:
        raise ValueError('empty panel')
    p = selected.pivot(index='block_id', columns='anchor_target_rho', values=col).sort_index()
    if set(p.columns) != set(rhos) or p.isna().any().any():
        raise ValueError('missing block/anchor or non-finite metric')
    if not np.isfinite(p.to_numpy()).all():
        raise ValueError('non-finite metric')
    return p


def boot(frame, col, seed):
    p = paired_values(frame, col)
    d = (p[.5] - p[.3]).to_numpy()
    rng = np.random.default_rng(seed)
    draw = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    return {'n': len(d), 'left': float(p[.5].mean()), 'right': float(p[.3].mean()),
            'difference': float(d.mean()), 'ratio': float(p[.5].mean() / p[.3].mean()),
            'ci_lower': float(np.quantile(draw, .025, method='higher')),
            'ci_upper': float(np.quantile(draw, .975, method='higher'))}


def close(a, b, tol=1e-12):
    if not np.isfinite(a) or not np.isfinite(b) or abs(a-b) > tol:
        raise AssertionError((a, b))


def reproduce():
    tests = {}
    specifications = [
        ('P2P', 'P2P_HYPOTHESIS_TESTS.json', [('H1','lpips_alex'), ('H2','unconditional_lpips_alex'), ('H4','dino_cosine_distance'), ('H5','lpips_alex')]),
        ('P3S', 'P3S_REPLICATION_TESTS.json', [('R1','lpips_alex'), ('R2','unconditional_lpips_alex'), ('R3','dino_cosine_distance'), ('R4','lpips_alex')])]
    for phase, summary, specs in specifications:
        frame = pd.read_csv(ROOT / f'public_results/tables/{phase}_BLOCK_LEVEL.csv')
        frame = frame[frame.cell_id.eq('W8_L1.00')] if phase == 'P2P' else frame[frame.panel.eq('primary_cfg3')]
        matched = pd.read_csv(ROOT / f'public_results/tables/{phase}_MATCHED_BLOCK_LEVEL.csv')
        stored = json.loads((ROOT / 'public_results/summaries' / summary).read_text())
        for key, col in specs:
            z = boot(matched if key in ('H5', 'R4') else frame, col, stored[key]['semantic_seed'])
            for actual, expected in [('n','block_count'), ('left','left_mean'), ('right','right_mean'), ('difference','difference_mean'), ('ratio','ratio_of_block_means'), ('ci_lower','ci95_lower'), ('ci_upper','ci95_upper')]:
                close(z[actual], stored[key][expected])
            tests[f'{phase}_{key}'] = z
        oof = pd.read_csv(ROOT / f'public_results/tables/{phase}_TOKEN_OOF.csv')
        expected = stored['H3'] if phase == 'P2P' else json.loads((ROOT/'public_results/summaries/P3S_TOKEN_LINK_TEST.json').read_text())
        ratio = float(oof.abs_error_S1.mean() / oof.abs_error_S0.mean())
        close(ratio, expected['mae_ratio_S1_over_S0'])
        tests[f'{phase}_TOKEN'] = {'rows':len(oof), 'blocks':int(oof.block_id.nunique()), 'mae_ratio':ratio, 'scope':'stored OOF error reaggregation; no feature reconstruction or LOBO refit'}
    return tests


def main():
    tests = reproduce()
    out = ROOT/'build'
    out.mkdir(exist_ok=True)
    result = {'pass':True, 'method':{'resamples':20000, 'quantile':'higher', 'rng':'NumPy default_rng / PCG64; stored semantic seeds', 'sample_unit':'paired block'}, 'tests':tests}
    (out/'reproduced_results.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'pass':True, 'checks':len(tests), 'both_ci_endpoints_checked':8, 'output':'build/reproduced_results.json'}))


if __name__ == '__main__':
    main()
