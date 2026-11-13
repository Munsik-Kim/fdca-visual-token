#!/usr/bin/env python3
"""Render plain-language figures from saved evidence; never fit or rerun a model.

Both languages share the same source values. Intervals are copied from their
original comparisons, not newly selected or recomputed. The output CSV records
each plotted quantity and its source/aggregation. SVG text is converted to paths
so the Korean figures do not require a reader to install the author's fonts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREY = "#17628c", "#b95d25", "#8795a1"
FIELDS = ["figure", "quantity", "estimate", "lower", "upper", "confidence",
          "unit", "source", "source_key", "aggregation"]


def rows(relative):
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def exact(relative, key, value):
    found = [r for r in rows(relative) if r[key] == value]
    if len(found) != 1:
        raise ValueError((relative, key, value, len(found)))
    return found[0]


def evidence():
    records = []

    def add(figure, quantity, estimate, unit, source, key, aggregation,
            lower="", upper="", confidence=""):
        records.append(dict(zip(FIELDS, [figure, quantity, float(estimate), lower,
                       upper, confidence, unit, source, key, aggregation])))

    source = "evidence/tables/m_confirm_units.csv"
    units = rows(source)
    grouped = defaultdict(list)
    for row in units:
        grouped[row["class_id"]].append(row)
    if len(grouped) != 32 or len(units) != 256 or {len(v) for v in grouped.values()} != {8}:
        raise ValueError("Unexpected M-confirm panel")
    means = {}
    for condition in ["SPARSE", "REAL_H", "SHUF_H", "REAL_L", "SHUF_L"]:
        for row in units:
            n = int(row["N_" + condition])
            bits = row["bits_" + condition]
            if len(bits) != 32 or bits.count("1") != n or float(row["Y_" + condition]) != n / 32:
                raise ValueError("Mismatch count, rate or bit string disagree")
        means[condition] = float(np.mean([
            np.mean([int(r["N_" + condition]) for r in group])
            for group in grouped.values()]))
        add("context_conditions", condition, means[condition], "mismatches per 32 coordinates",
            source, "N_" + condition, "Within-class mean, then equal mean over 32 classes; 8 pairs/class")
    contrasts_source = "evidence/tables/m_confirm_contrasts.csv"
    calculated = {
        "C_ctx": means["SPARSE"] - means["REAL_H"],
        "C_layout": means["SHUF_H"] - means["REAL_H"],
        "C_dose": ((means["REAL_L"] - means["REAL_H"]) +
                   (means["SHUF_L"] - means["SHUF_H"])) / 2,
    }
    for key, value in calculated.items():
        r = exact(contrasts_source, "contrast", key)
        if not np.isclose(value, float(r["estimate"]) * 32, rtol=0, atol=1e-12):
            raise ValueError("Condition means disagree with original contrast")

    draw_source = "evidence/lt16/expected/selector_draws.csv"
    draws = [r for r in rows(draw_source) if r["selector"] == "OLD_MAX"]
    parent_counts = defaultdict(int)
    for r in draws:
        parent_counts[r["trajectory_id"]] += 1
    if len(draws) != 384 or len(parent_counts) != 48 or set(parent_counts.values()) != {8}:
        raise ValueError("Unexpected OLD_MAX redraw panel")
    values = np.array([float(r["G_new"]) for r in draws])
    counts = [int((values > 0).sum()), int((values < 0).sum()), int((values == 0).sum())]
    contributions = [float(values[values > 0].sum() / len(values)),
                     float(values[values < 0].sum() / len(values)), float(values.mean())]
    original = exact("evidence/lt16/expected/signed_contributions.csv", "selector", "OLD_MAX")
    for value, key in zip(contributions, ["positive_contribution", "negative_contribution", "combined"]):
        if not np.isclose(value, float(original[key]), rtol=0, atol=2e-14):
            raise ValueError("Signed contribution disagrees with historical table")
    for name, value in zip(["positive", "negative", "zero"], counts):
        add("signed_outcomes", name + "_count", value, "draws", draw_source,
            "selector=OLD_MAX;G_new sign=" + name, "Count within 48 parents x 8 draws; not independent classes")
    for name, value in zip(["positive", "negative", "net"], contributions):
        add("signed_outcomes", name + "_contribution", value, "signed LPIPS reduction",
            draw_source, "selector=OLD_MAX;" + name, "Sum of group signed G divided by all 384 draws")

    selector_source = "evidence/lt16/expected/selector_summary.csv"
    old = exact(selector_source, "selector", "OLD_MAX")
    old_value = float(old["old_G"])
    add("selection_recheck", "original_selected_maximum", old_value, "signed LPIPS reduction",
        selector_source, "selector=OLD_MAX;old_G", "Original pathwise selected maximum; descriptive; no CI claimed")
    primary_source = "evidence/lt16/expected/primary_contrasts.csv"
    primary = exact(primary_source, "contrast", "A")
    new = tuple(float(primary[k]) for k in ["estimate", "ci_low", "ci_high"])
    if not np.isclose(contributions[2], new[0], rtol=0, atol=2e-14):
        raise ValueError("Signed net mean disagrees with the primary estimate")
    add("selection_recheck", "fresh_draws_at_selected_states", new[0], "signed LPIPS reduction",
        primary_source, "contrast=A", "Historical primary class-bootstrap interval", new[1], new[2],
        float(primary["confidence"]))
    aux_source = "evidence/lt16/expected/auxiliary_contrasts.csv"
    split = []
    for key in ["SPLIT_HALF_SELF_VALUE", "SPLIT_HALF_TRANSFER_VALUE", "SPLIT_HALF_SELECTION_OPTIMISM"]:
        r = exact(aux_source, "contrast", key)
        v = tuple(float(r[k]) for k in ["estimate", "ci_low", "ci_high"])
        split.append(v)
        add("selection_recheck", key, v[0], "signed LPIPS reduction", aux_source,
            "contrast=" + key, "Historical auxiliary class-bootstrap interval", v[1], v[2], float(r["confidence"]))
    return records, means, counts, contributions, old_value, new, split


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=175, facecolor="white")
    fig.savefig(output / (name + ".svg"), facecolor="white",
                metadata={"Creator": "FDCA saved-evidence explainer", "Date": None})
    p = output / (name + ".svg")
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    plt.close(fig)


def setup(language):
    if language == "ko":
        candidates = [f.name for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name]
        if not candidates:
            raise RuntimeError("Korean figures require a Noto Sans CJK system font")
        font = sorted(set(candidates))[0]
    else:
        font = "DejaVu Sans"
    plt.rcParams.update({"font.family": font, "font.size": 12, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "svg.fonttype": "path", "svg.hashsalt": "fdca-plain-evidence-v1"})


def context(output, language, means):
    ko = language == "ko"
    conditions = ["SPARSE", "REAL_L", "SHUF_L", "REAL_H", "SHUF_H"]
    labels = (["해당 문맥을 추가하지 않음", "37개 · 원래 위치", "37개 · 위치를 섞음", "74개 · 원래 위치", "74개 · 위치를 섞음"] if ko else
              ["No added recorded context", "37 tokens · original layout", "37 tokens · shuffled layout", "74 tokens · original layout", "74 tokens · shuffled layout"])
    fig, ax = plt.subplots(figsize=(11, 6.4))
    y = np.arange(5)[::-1]
    vals = [means[k] for k in conditions]
    ax.barh(y, vals, color=[GREY, BLUE, ORANGE, BLUE, ORANGE], height=.55)
    for yy, value in zip(y, vals):
        ax.text(value + .13, yy, f"{value:.3f}", va="center", fontsize=13, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(0, 14)
    ax.set_xticks([0, 4, 8, 12])
    ax.grid(axis="x", color="#e4e9ef")
    ax.set_axisbelow(True)
    ax.set_xlabel("다음 32개 위치에서 새로 달라진 토큰 수의 평균 (작을수록 두 경로가 더 비슷함)" if ko else
                  "Mean new mismatches in the next 32 positions (smaller = more agreement)", labelpad=12)
    fig.suptitle("같은 주변 토큰을 주면 이후의 차이가 얼마나 줄었을까?" if ko else
                 "How much does shared context change later disagreement?", x=.04, ha="left", fontsize=19, fontweight="bold")
    fig.text(.04, .86, "M-confirm · 새 실험 클래스 32개 · 짝지은 비교 256개" if ko else
             "M-confirm · 32 new experimental classes · 256 paired units", fontsize=12, color="#435466")
    fig.text(.04, .055, "조건별 평균을 표시했습니다. 오차막대는 생략했으며 비교의 보정 구간은 별도 표를 따릅니다.\n이는 토큰의 일치 정도입니다. 이미지 품질이나 256개의 독립 클래스 성과를 뜻하지 않습니다." if ko else
             "Condition means only; no mean intervals are claimed. Adjusted contrast intervals are reported separately.\nThe endpoint is token agreement, not image quality. The 256 pairs are clustered in 32 classes.", fontsize=10, color="#435466")
    fig.subplots_adjust(left=.32, right=.96, top=.78, bottom=.24)
    save(fig, output, "context_conditions_" + language)


def signed(output, language, counts, contributions, new):
    ko = language == "ko"
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 9.2))
    labels = ["참조에 가까워짐", "참조에서 멀어짐", "거리 변화 0"] if ko else ["Closer to reference", "Farther from reference", "Zero distance change"]
    for ax in axes:
        ax.tick_params(axis="y", length=0, pad=10)
        ax.grid(axis="x", color="#e4e9ef")
        ax.set_axisbelow(True)
    axes[0].barh([2, 1, 0], counts, color=[BLUE, ORANGE, GREY], height=.55)
    axes[0].set_yticks([2, 1, 0], labels)
    axes[0].set_xlim(0, 400)
    for y, n in zip([2, 1, 0], counts):
        axes[0].text(n + 5, y, f"{n}/384  ({100*n/384:.1f}%)", va="center", fontsize=11)
    axes[0].set_xlabel("반복 결과 수" if ko else "Number of redraw outcomes")
    axes[0].set_title("384개 결과에서 이득·손실·변화 없음이 함께 나타났습니다" if ko else
                      "Benefits, losses and zero change coexist in 384 outcomes", loc="left", pad=14)
    axes[1].barh([2, 1, 0], contributions, color=[BLUE, ORANGE, GREY], height=.55)
    axes[1].set_yticks([2, 1, 0], ["양의 결과가 더한 값", "음의 결과가 더한 값", "전체 평균 이득"] if ko else
                      ["Positive contribution", "Negative contribution", "Net mean gain"])
    axes[1].set_xlim(-.0072, .0066)
    axes[1].set_xticks([-.006, -.004, -.002, 0, .002, .004, .006])
    axes[1].axvline(0, color="#465261", lw=1)
    for y, value in zip([2, 1, 0], contributions):
        axes[1].text(value + (.00015 if value >= 0 else -.00015), y, f"{value:+.6f}",
                     va="center", ha="left" if value >= 0 else "right", fontsize=11)
    axes[1].set_xlabel("참조와의 LPIPS 거리 감소량 (양수 = 참조에 가까워짐)" if ko else
                      "Reduction in reference LPIPS distance (positive = closer)")
    axes[1].set_title("각 집단의 이득 합계를 전체 384개로 나누면" if ko else
                      "Each group sum is divided by all 384 outcomes", loc="left", pad=14)
    fig.suptitle("가끔 크게 좋아져도 평균 복원 이득은 남지 않을 수 있습니다" if ko else
                 "Occasional large gains can be offset by losses", x=.04, ha="left", fontsize=18, fontweight="bold")
    fig.text(.04, .90, "LT16 · 과거 최대 이득 시점(OLD_MAX) · 48개 원 경로 × 새 난수 8회" if ko else
             "LT16 · previously selected maximum (OLD_MAX) · 48 parents × 8 fresh draws", fontsize=11, color="#435466")
    interval = f"[{new[1]:+.6f}, {new[2]:+.6f}]"
    fig.text(.04, .035, "같은 개발 클래스 24개를 재사용한 기술 통계입니다. 새 클래스 성능이나 이미지 품질의 판정이 아닙니다.\n"
             f"아래의 순이득은 점추정입니다. 주 분석의 보정 98.3333% 구간은 {interval}으로 0을 포함합니다." if ko else
             "Descriptive statistics on 24 reused development classes; not new-class performance or an image-quality verdict.\n"
             f"The net gain is a point estimate. Its primary adjusted 98.3333% CI is {interval}, including zero.", fontsize=10, color="#435466")
    fig.subplots_adjust(left=.29, right=.97, top=.83, bottom=.14, hspace=.62)
    save(fig, output, "signed_outcomes_" + language)


def selection(output, language, old, new, split):
    ko = language == "ko"
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 9.2))
    groups = [([old, new[0]], [None, new],
               ["원 결과를 보고 고른 최대", "그 시점을 새 난수로 재평가"] if ko else
               ["Maximum selected on original outcomes", "Same states, fresh randomness"]),
              ([split[0][0], split[1][0]], split[:2],
               ["고를 때 쓴 난수에서 평가", "나머지 난수에서 평가"] if ko else
               ["Evaluate on selection draws", "Evaluate on other draws"])]
    for ax, (values, intervals, labels) in zip(axes, groups):
        ax.barh([1, 0], values, color=[ORANGE, BLUE], height=.44)
        for y, value, interval in zip([1, 0], values, intervals):
            if interval is not None:
                e, lo, hi = interval
                ax.errorbar(e, y, xerr=[[e-lo], [hi-e]], fmt="none", color="#1a2633", capsize=5, lw=1.8)
            ax.text(.99, y, f"{value:+.6f}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=12)
        ax.set_yticks([1, 0], labels)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.axvline(0, color="#465261", lw=1)
        ax.grid(axis="x", color="#e4e9ef")
        ax.set_axisbelow(True)
        ax.set_ylim(-.65, 1.65)
        ax.set_xlabel("참조와의 LPIPS 거리 감소량" if ko else "Reduction in reference LPIPS distance")
    axes[0].set_xlim(-.01, .076)
    axes[0].set_xticks([-.01, 0, .01, .02, .03, .04, .05, .06, .07])
    axes[1].set_xlim(-.006, .020)
    axes[1].set_xticks([-.005, 0, .005, .010, .015, .020])
    axes[0].set_title("선택된 최대값은 새 난수에서 유지되지 않았습니다" if ko else
                      "The selected maximum did not persist", loc="left", pad=14)
    axes[1].set_title("선택과 평가에 같은 자료를 쓰면 값이 커졌습니다" if ko else
                      "Reusing selection draws inflated the evaluation", loc="left", pad=14)
    fig.suptitle("잘 나온 결과를 고르는 것과, 다음 결과를 개선하는 것은 다릅니다" if ko else
                 "Picking a good observed outcome does not validate a policy", x=.04, ha="left", fontsize=17, fontweight="bold")
    fig.text(.04, .90, "LT16 · 개발 클래스 24개 · 원 경로 48개 · 저장 상태에서의 조건부 진단" if ko else
             "LT16 · 24 development classes · 48 original parents · saved-state conditional diagnostic", fontsize=11, color="#435466")
    gap = f"{split[2][0]:+.6f} [{split[2][1]:.6f}, {split[2][2]:.6f}]"
    fig.text(.04, .035, "위: 새 난수 결과에만 주 분석의 보정 98.3333% 구간을 표시. 원 최대값에는 구간을 주장하지 않습니다.\n"
             f"아래: 보조 95% 구간. 두 평가의 짝지은 차이는 {gap}. 두 패널의 축 범위가 다릅니다." if ko else
             "Top: adjusted 98.3333% interval for fresh draws only; no interval claimed for the selected original maximum.\n"
             f"Bottom: auxiliary 95% intervals; paired gap {gap}. Panel axis ranges differ.", fontsize=10, color="#435466")
    fig.subplots_adjust(left=.36, right=.97, top=.83, bottom=.15, hspace=.65)
    save(fig, output, "selection_recheck_" + language)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures/explainer")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, means, counts, contributions, old, new, split = evidence()
    for language in ["ko", "en"]:
        setup(language)
        context(args.output, language, means)
        signed(args.output, language, counts, contributions, new)
        selection(args.output, language, old, new, split)
    with (args.output / "plotted_values.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    sources = sorted({r["source"] for r in records} | {
        "evidence/tables/m_confirm_contrasts.csv", "evidence/lt16/expected/signed_contributions.csv"})
    manifest = {
        "kind": "Descriptive presentation of existing evidence; no new inference",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "matplotlib": matplotlib.__version__, "plotted_quantities": len(records),
        "languages": ["ko", "en"], "logical_figures": 3,
        "checks": ["M-confirm bit strings/counts/rates and equal class aggregation",
                   "M-confirm means recover original three primary contrasts",
                   "LT16 OLD_MAX 48 parents x 8 draws; signed contributions recover original table"],
        "scope": "Stored scalar values only. No GPU/model/decoder/metric run or reserved holdout access.",
    }
    (args.output / "sources.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "figures": 3, "languages": 2, "plotted_quantities": len(records)}))


if __name__ == "__main__":
    main()
