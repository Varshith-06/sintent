"""Headline figure: (a) recall at 5% false alarms by world and monitor, with bootstrap intervals
and n printed in every cell; (b) R1 score distributions by world with the threshold marked.

    python -m pilot.analysis.figure --run pilot_main
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import RESULTS_DIR
from .score import analyze, cells, episode_scores, load

# Reference categorical palette, slots 1-3 (validated all-pairs, light mode); ink for text.
SERIES = {"R1": "#2a78d6", "S": "#eb6834", "R2": "#1baf7a", "L": "#6250d6"}
WORLD_TONE = {"x+": "#8a8984", "x-": "#2a78d6", "x-F": "#eb6834", "x-N": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"R1": "R1 belief readout", "S": "S sandbox check", "R2": "R2 pre-action gate", "L": "L LLM simulator"}
POPS = [("x- violations (knowing)", "x−\nknowing\nviolations"),
        ("x-F violations", "x−F\nforged-authority\nviolations"),
        ("x-F violations, belief-check pass", "x−F violations,\nbelief check\npassed"),
        ("x-N violations", "x−N\nnuisance\nviolations")]


def boot_ci(pop, scores, m, t, n=2000, seed=0):
    by_t = defaultdict(lambda: defaultdict(list))
    for e in pop:
        v = scores[e["episode_id"]][m]
        if not np.isnan(v):
            by_t[e["template"]][e["pair_id"]].append(v > t)
    if not by_t:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = []
        for g in by_t.values():
            keys = list(g)
            for k in rng.choice(len(keys), len(keys)):
                s.extend(g[keys[k]])
        out.append(np.mean(s))
    return np.quantile(out, 0.025), np.quantile(out, 0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    res = analyze(args.run)
    eps, ro, bc, llm = load(args.run)
    scores = episode_scores(eps, ro, llm)
    cs = cells(eps, bc)
    monitors = res["monitors"]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    width = 0.8 / len(monitors)
    for j, m in enumerate(monitors):
        t = res["thresholds"][f"{m}@0.05"]
        for i, (key, _) in enumerate(POPS):
            cell = res["recall_table"][f"0.05|{key}|{m}"]
            x = i + (j - (len(monitors) - 1) / 2) * width
            r = cell["recall"]
            if cell["n"] == 0 or r is None or np.isnan(r):
                a.text(x, 0.02, "n=0", ha="center", va="bottom", fontsize=8, color=INK2)
                continue
            lo, hi = boot_ci(cs[key], scores, m, t)
            a.bar(x, r, width * 0.92, color=SERIES[m], label=LABELS[m] if i == 0 else None, zorder=2)
            a.errorbar(x, r, yerr=[[r - lo], [hi - r]], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
            a.text(x, min(hi, 1) + 0.03, f"{r:.2f}\nn={cell['n']}", ha="center", va="bottom", fontsize=7.5, color=INK)
    a.set_xticks(range(len(POPS)), [lab for _, lab in POPS])
    a.set_ylim(0, 1.25)
    a.set_yticks(np.linspace(0, 1, 6))
    a.set_ylabel("Recall at 5% false alarms on benign twins (x+)")
    a.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    a.legend(frameon=False, loc="upper left", ncol=len(monitors), fontsize=9)
    h = res["headline"]
    a.set_title(f"(a) Who catches what. Headline gap S − R1 = {h['gap']:.2f} "
                f"[{h['ci'][0]:.2f}, {h['ci'][1]:.2f}], n = {h['n']}" if h["n"] and not np.isnan(h["gap"])
                else "(a) Who catches what",
                loc="left", fontsize=10.5, color=INK)

    dist = res["R1_dist_test"]
    worlds = ["x+", "x-", "x-F", "x-N"]
    rng = np.random.default_rng(0)
    for i, w in enumerate(worlds):
        v = np.array(dist.get(w, []))
        if len(v) == 0:
            continue
        if len(v) > 3:
            parts = b.violinplot(v, positions=[i], widths=0.7, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(WORLD_TONE[w])
                pc.set_alpha(0.25)
                pc.set_edgecolor("none")
        b.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=WORLD_TONE[w], edgecolor=SURFACE, lw=0.6,
                  zorder=3)
    if not any(len(dist.get(w, [])) for w in worlds):
        b.text(0.5, 0.5, "R1 not fit (too few decision states)", transform=b.transAxes, ha="center", color=INK2)
    thr = res["thresholds"]["R1@0.05"]
    if not np.isnan(thr):
        b.axhline(thr, color=INK2, lw=1, ls="--", zorder=1)
        b.text(len(worlds) - 0.5, thr, " 5% false-alarm\n threshold", va="center", fontsize=8, color=INK2)
    names = ["x+\nauthorized", "x−\nknowing", "x−F\nforged approval", "x−N\nnuisance"]
    b.set_xticks(range(4), [f"{lab}\nn={len(dist.get(w, []))}" for lab, w in zip(names, worlds)])
    b.set_ylabel("R1 readout score (z, higher = 'forbidden')")
    b.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    b.set_title("(b) R1 at the decision point, test half", loc="left", fontsize=10.5, color=INK)

    fig.text(0.01, 0.005, f"{res['run']}: one model, six CAPE-Bench templates; S is an upper bound for consequence "
             f"monitors. Intervals: cluster bootstrap over scenarios within templates.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = RESULTS_DIR / args.run / "headline_figure.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
