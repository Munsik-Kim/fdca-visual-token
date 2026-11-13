#!/usr/bin/env python3
"""Plot historical estimates directly from the published evidence tables."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'evidence/tables'
LT16 = ROOT / 'evidence/lt16/expected'
FIGURES = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'svg.hashsalt': 'fdca-followup-2026-11',
                     'savefig.facecolor': '#ffffff'})
BLUE = '#165a82'
ORANGE = '#ad5a18'
PURPLE = '#775395'
RECORDS = []


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def point(path, key, value, estimate='estimate', lower='lower', upper='upper',
          confidence='confidence', scale=1, label=None, unit='LPIPS', filters=None):
    rr = [r for r in rows(path) if r[key] == value and
          all(r[k] == v for k, v in (filters or {}).items())]
    assert len(rr) == 1, (path, key, value, len(rr))
    r = rr[0]
    if lower == 'ci':
        lo, hi = json.loads(r['ci'])
    else:
        lo, hi = float(r[lower]), float(r[upper])
    e = float(r[estimate])*scale
    c = float(r[confidence]) if confidence else .95
    row = dict(label=label or value, estimate=e, lower=lo*scale, upper=hi*scale,
               confidence=c, unit=unit, source=str(path.relative_to(ROOT)),
               source_key=json.dumps({key: value, **(filters or {})}, sort_keys=True))
    RECORDS.append(row)
    return row


def forest(ax, data, color=BLUE, threshold=None):
    y = np.arange(len(data))[::-1]
    e = np.array([r['estimate'] for r in data])
    lo = np.array([r['lower'] for r in data])
    hi = np.array([r['upper'] for r in data])
    ax.errorbar(e, y, xerr=[e-lo, hi-e], fmt='o', ms=6, color=color,
                elinewidth=2, capsize=4, zorder=3)
    ax.set_yticks(y, [r['label'] for r in data])
    ax.axvline(0, color='#555555', lw=1)
    if threshold is not None:
        ax.axvline(threshold, color='#888888', lw=1, ls='--')
    ax.grid(axis='x', color='#e5e9ec', lw=.7)
    ax.set_ylim(-.6, len(data)-.4)
    ax.tick_params(axis='y', length=0, pad=8)


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(FIGURES / (name+'.'+ext), dpi=180,
                    metadata={'Creator': 'FDCA source-table figure builder', 'Date': None} if ext == 'svg' else None)
        if ext == 'svg':
            target = FIGURES / (name+'.svg')
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


def context():
    m = [point(TABLES/'m_confirm_contrasts.csv', 'contrast', key,
               lower='ci_low', upper='ci_high', scale=32, label=label,
               unit='tokens per 32') for key, label in [
                   ('C_ctx', 'Add recorded context'), ('C_layout', 'Original vs shuffled layout'),
                   ('C_dose', 'High vs low visibility\n(average of both layouts)')]]
    l = [point(TABLES/'lt11_contrasts.csv', 'contrast', key, scale=32,
               label=label, unit='tokens per 32', filters={'horizon': 'W32'}) for key, label in [
                   ('C_shared', 'Add other-class context'), ('C_match', 'Matched vs other-class source')]]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.3), gridspec_kw={'height_ratios': [3, 2]})
    forest(axes[0], m, threshold=1)
    forest(axes[1], l, color=ORANGE, threshold=1)
    axes[0].set_title('M-confirm: 32 new classes, 256 pairs | adjusted 98.33% CIs', loc='left', pad=12)
    axes[1].set_title('LT11: 32 class pairs, 512 units | adjusted 97.5% CIs', loc='left', pad=12)
    for ax in axes:
        ax.set_xlim(-.3, 11)
    axes[1].set_xlabel('Reduction in new mismatches per 32 future coordinates (positive = fewer)')
    fig.suptitle('Shared context changes propagation', fontsize=17, fontweight='bold', x=.31, ha='left')
    fig.text(.31, .025, 'Dashed line: one-token practical threshold. Separate experiments; no pooled effect.', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.31, right=.98, bottom=.12, top=.86, hspace=.55)
    save(fig, 'context_effects')


def restoration():
    a = [point(TABLES/'lt5_primary.csv', 'endpoint', key, estimate='mean', lower='ci_low', upper='ci_high',
               confidence='confidence_level', label=label) for key, label in [
                   ('A_FDCA_ABSOLUTE', 'FDCA vs all-W8'), ('B_FDCA_MINUS_LOCAL_TV', 'FDCA vs local TV')]]
    b = [point(TABLES/'r_test_primary.csv', 'contrast', key, label=label) for key, label in [
        ('A_F', 'A_F = A_D: E2 vs all-W8'), ('B_F', 'B_F: E2 vs M3'),
        ('A_S', 'A_S: M3 vs all-W8'), ('B_S', 'B_S: M3 vs U3'), ('B_D', 'B_D: E2 vs U3')]]
    c = [point(TABLES/'lt12_primary.csv', 'contrast', key, label=label) for key, label in [
        ('A_ref', 'Reference-token edit vs all-W8'), ('A_b2', 'Block 2 at oracle time vs all-W8'),
        ('A_b3', 'Block 3 at oracle time vs all-W8'), ('B_dir', 'Reference-token vs alternate edit')]]
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 11), gridspec_kw={'height_ratios': [2, 4, 3.5]})
    for ax, data, color in zip(axes, [a, b, c], [BLUE, BLUE, PURPLE]):
        forest(ax, data, color=color, threshold=.01)
    axes[0].set_title('LT5 | 32 new classes | adjusted 97.5% CIs', loc='left', pad=12)
    axes[1].set_title('R-test | 320 new classes | adjusted 99.17% CIs', loc='left', pad=12)
    axes[2].set_title('LT12 oracle diagnostic | 64 reused classes | adjusted 98.75% CIs', loc='left', pad=12)
    for ax in axes[:2]:
        ax.set_xlim(-.03, .03)
    axes[2].set_xlim(-.012, .112)
    axes[-1].set_xlabel('Signed reduction in FP32-reference LPIPS discrepancy')
    fig.suptitle('Policy gains unresolved; oracle exception retained', fontsize=16, fontweight='bold', x=.38, ha='left')
    fig.text(.38, .022, 'Positive = closer to the reference, not better image quality.\nDashed line: +0.01. LT12 uses reference information; x-axis scales differ.', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.38, right=.98, top=.91, bottom=.1, hspace=.6)
    save(fig, 'restoration_evidence')


def pulse():
    data = []
    for b in (2, 3):
        data.append(point(TABLES/'lt3_contrasts.csv', 'contrast', f'C{b}', estimate='mean',
                          lower='ci', confidence='confidence_level', filters={'metric': 'R'},
                          label=f'Block {b}: LT3 FULL', unit='pulse R'))
        data.append(point(TABLES/'lt4_contrasts.csv', 'contrast', f'C5_{b}', estimate='mean',
                          lower='ci', confidence='confidence_level', filters={'metric': 'R'},
                          label=f'Block {b}: LT4 OP5', unit='pulse R'))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    forest(ax, data)
    ax.set_xlim(-.0003, .0076)
    ax.set_xlabel('R at step 0 minus mean R at steps 15 and 23')
    ax.set_title('Early pulse discrepancy persists after matching five coordinates', loc='left', pad=20, fontsize=14)
    fig.text(.25, .025, 'Adjusted 97.5% CIs. Same 32 classes; LT4 reuses the LT3 parents and early condition.\nPulse discrepancy is not signed restoration gain or an isolated causal effect of time.', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.25, right=.97, top=.81, bottom=.24)
    save(fig, 'pulse_timing')


def optimism():
    selectors = rows(LT16/'selector_summary.csv')
    old = next(r for r in selectors if r['selector'] == 'OLD_MAX')
    new = point(LT16/'primary_contrasts.csv', 'contrast', 'A', lower='ci_low', upper='ci_high',
                label='Fresh draws at OLD_MAX states')
    aux = {r['contrast']: r for r in rows(LT16/'auxiliary_contrasts.csv')}
    selfrow = aux['SPLIT_HALF_SELF_VALUE']
    transfer = aux['SPLIT_HALF_TRANSFER_VALUE']
    gap = aux['SPLIT_HALF_SELECTION_OPTIMISM']
    for key in ('SPLIT_HALF_SELF_VALUE', 'SPLIT_HALF_TRANSFER_VALUE', 'SPLIT_HALF_SELECTION_OPTIMISM'):
        point(LT16/'auxiliary_contrasts.csv', 'contrast', key, lower='ci_low', upper='ci_high')
    RECORDS.append(dict(label='LT14 observed pathwise maximum', estimate=float(old['old_G']), lower='', upper='',
                        confidence='', unit='LPIPS; post-selection descriptive',
                        source='evidence/lt16/expected/selector_summary.csv', source_key='selector=OLD_MAX;old_G'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    axes[0].bar([0, 1], [float(old['old_G']), new['estimate']], color=[ORANGE, BLUE], width=.55)
    axes[0].errorbar(1, new['estimate'], yerr=[[new['estimate']-new['lower']], [new['upper']-new['estimate']]],
                     fmt='none', color='#222222', capsize=5, lw=2)
    axes[0].set_xticks([0, 1], ['Original pathwise max\n(selected on outcomes)', 'New 8 draws/state\n98.33% CI'])
    axes[0].set_title('A large observed maximum did not persist', loc='left', fontsize=12, pad=18)
    axes[0].text(0, .060, '+0.05806', ha='center', color=ORANGE, fontweight='bold')
    axes[0].text(1, .006, '−0.00072', ha='center', color=BLUE, fontweight='bold')
    axes[0].set_ylim(-.009, .071)
    vals = [float(selfrow['estimate']), float(transfer['estimate'])]
    axes[1].bar([0, 1], vals, color=[ORANGE, BLUE], width=.55)
    for i, r in enumerate([selfrow, transfer]):
        e, lo, hi = (float(r[k]) for k in ['estimate', 'ci_low', 'ci_high'])
        axes[1].errorbar(i, e, yerr=[[e-lo], [hi-e]], fmt='none', color='#222222', capsize=5, lw=2)
    axes[1].set_xticks([0, 1], ['Select + evaluate\non same draw half', 'Evaluate selection\non other draw half'])
    axes[1].set_title('Split-half evaluation exposes optimism', loc='left', fontsize=12, pad=18)
    axes[1].set_ylim(-.006, .018)
    axes[1].text(.5, .0157, f"Paired gap: +{float(gap['estimate']):.5f}\n95% CI [{float(gap['ci_low']):.5f}, {float(gap['ci_high']):.5f}]", ha='center', fontsize=10)
    for ax in axes:
        ax.axhline(0, color='#555555', lw=1)
        ax.set_ylabel('Signed LPIPS reduction')
        ax.grid(axis='y', color='#e5e9ec', lw=.7)
        ax.set_axisbelow(True)
    fig.suptitle('LT16: separate selection from evaluation randomness', x=.07, ha='left', fontsize=16, fontweight='bold')
    fig.text(.07, .025, '24 reused development classes, 48 original parents. Stored-state conditional diagnostics, not new-class policy evaluation.\nLeft: no confidence interval claimed for the original selected maximum. Right: auxiliary 95% CIs; panel scales differ.', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.075, right=.98, top=.82, bottom=.24, wspace=.3)
    save(fig, 'selection_optimism')


if __name__ == '__main__':
    context()
    restoration()
    pulse()
    optimism()
    target = ROOT / 'data/figure_data.csv'
    with target.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(RECORDS[0]))
        writer.writeheader()
        writer.writerows(RECORDS)
    print(f'Built 4 figures (PNG + SVG) from {len(RECORDS)} source-bound plotted values.')
