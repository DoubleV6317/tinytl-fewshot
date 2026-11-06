"""Redraw Figure 1 (accuracy relative to LB vs k) and Figure 4 (imprinting gain) from the analysis output.

Reads ../实验设计/分析输出_补充.txt, the output of analyze_supp.py, so the plotted numbers are the ones
the report quotes. Hollow marks are exploratory comparisons, as in Figure 2.

    python make_figures.py [--preview DIR]
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / '实验设计' / '分析输出_补充.txt'
DATASETS = [('flowers102', 'Flowers102'), ('aircraft', 'FGVC-Aircraft')]
KS = [2, 5, 10]

# Categorical slots 1-4 of the reference palette, in fixed order; LB is the reference, so it gets neutral ink.
COLOR = {'T4': '#2a78d6', 'T3': '#eb6834', 'T2': '#1baf7a', 'last': '#eda100', 'LB': '#52514e'}
INK, MUTED, GRID, BASE, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'

ROW = r'n=(\d+) d=([+-][\d.]+) sd=[\d.]+ 93% CI \[([+-][\d.]+), ([+-][\d.]+)\]'


def sections(text):
    """{header line: body} for the '=== ... ===' and '--- ... ---' blocks."""
    parts = re.split(r'^((?:===|---) .*)$', text, flags=re.M)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def depth_rows(secs):
    """(dataset, k, arm) -> (d, L, U, exploratory) for the paired arm - LB rows."""
    out = {}
    for head, body in secs.items():
        m = re.match(r'--- (\w+) k=(\d+):', head)
        if not m:
            continue
        for line in body.splitlines():
            r = re.match(r'\s+(T2|T3|T4|last)\s+' + ROW + r'\s+(.*)$', line)
            if r:
                out[(m.group(1), int(m.group(2)), r.group(1))] = (
                    float(r.group(3)), float(r.group(4)), float(r.group(5)), 'exploratory' in r.group(6))
    return out


def imprint_rows(secs):
    """(dataset, k, arm) -> (e, L, U, exploratory) for imprinting - random head, 50 epochs."""
    out = {}
    for head, body in secs.items():
        m = re.match(r'=== Q2: (\w+) imprinted vs random head', head)
        if not m:
            continue
        for line in body.splitlines():
            r = re.match(r'\s+k=(\d+) (LB|T4)\s+' + ROW + r'(.*)$', line)
            if r:
                out[(m.group(1), int(r.group(1)), r.group(2))] = (
                    float(r.group(4)), float(r.group(5)), float(r.group(6)), 'exploratory' in r.group(7))
    return out


def style(ax, title):
    ax.set_title(title, loc='left', fontsize=8, color=INK, pad=4)
    ax.grid(axis='y', color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BASE)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0, labelsize=7)
    ax.set_xticks(range(len(KS)), [str(k) for k in KS])


def figure_shots(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.45), sharey=True)
    offset = {'T4': -0.09, 'T3': 0.0, 'T2': 0.09, 'last': 0.0}
    for ax, (ds, title) in zip(axes, DATASETS):
        style(ax, title)
        ax.axhline(0, color=BASE, linewidth=0.8, zorder=1)
        ax.axhline(-2, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        for arm in ['last', 'T2', 'T3', 'T4']:
            pts = [(i + offset[arm], rows[(ds, k, arm)]) for i, k in enumerate(KS) if (ds, k, arm) in rows]
            if not pts:
                continue
            xs = [x for x, _ in pts]
            ds_ = [v[0] for _, v in pts]
            ax.plot(xs, ds_, color=COLOR[arm], linewidth=1.3, solid_capstyle='round', zorder=2)
            for x, (d, lo, hi, expl) in pts:
                ax.plot([x, x], [lo, hi], color=COLOR[arm], linewidth=0.9, zorder=2)
                ax.plot(x, d, marker='o', markersize=5.5, zorder=3, color=COLOR[arm],
                        markerfacecolor=SURFACE if expl else COLOR[arm],
                        markeredgecolor=COLOR[arm] if expl else SURFACE,
                        markeredgewidth=1.2 if expl else 0.8)
        ax.set_xlim(-0.45, 2.75)
        ax.set_xlabel('Training images per class, k', fontsize=7, color=INK)
    # Label the Aircraft k=10 depth results, which the text discusses. T2 and T3 sit together, so their labels
    # carry the arm name and stay between the zero line and the margin.
    ax = axes[1]
    for arm, x, y in [('T2', 2.22, -0.45), ('T3', 2.22, -1.4), ('T4', 1.98, None), ('last', 2.07, None)]:
        d = rows[('aircraft', 10, arm)][0]
        ax.text(x, d if y is None else y, f'{arm} {d:+.2f}'.replace('-', '−'), ha='left', va='center',
                fontsize=6.5, color=INK)
    axes[0].set_ylabel('Difference from LB, pp', fontsize=7, color=INK)
    axes[0].set_ylim(-11.8, 2.6)
    axes[0].set_yticks(range(-10, 3, 2))
    handles = [Line2D([], [], color=COLOR[a], marker='o', markersize=5, markeredgecolor=SURFACE, linewidth=1.3,
                      label=a) for a in ['T4', 'T3', 'T2', 'last']]
    handles.append(Line2D([], [], color=MUTED, marker='o', markersize=5, markerfacecolor=SURFACE, linewidth=0,
                          markeredgewidth=1.2, label='exploratory'))
    handles.append(Line2D([], [], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)), label='−2 pp margin'))
    fig.legend(handles=handles, loc='upper center', ncol=6, frameon=False, fontsize=7, handlelength=1.6,
               columnspacing=1.4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
    fig.savefig(path)
    return fig


def figure_imprint(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.05), sharey=True)
    width, shift = 0.32, {'LB': -0.17, 'T4': 0.17}
    for ax, (ds, title) in zip(axes, DATASETS):
        style(ax, title)
        ax.axhline(0, color=BASE, linewidth=0.8, zorder=1)
        for i, k in enumerate(KS):
            for arm in ['LB', 'T4']:
                x = i + shift[arm]
                if (ds, k, arm) not in rows:
                    ax.text(x, 0.08, 'not run', rotation=90, ha='center', va='bottom', fontsize=5.5, color=MUTED)
                    continue
                e, lo, hi, expl = rows[(ds, k, arm)]
                ax.bar(x, e, width, color=SURFACE if expl else COLOR[arm], edgecolor=COLOR[arm],
                       linewidth=1.0 if expl else 0, zorder=2)
                ax.plot([x, x], [lo, hi], color=INK, linewidth=0.8, zorder=3)
        ax.set_xlim(-0.6, 2.6)
        ax.set_xlabel('k', fontsize=7, color=INK)
    axes[0].set_ylabel('Imprinting gain, pp', fontsize=7, color=INK)
    axes[0].set_ylim(-0.6, 3.9)
    handles = [Patch(facecolor=COLOR['LB'], label='LB'), Patch(facecolor=COLOR['T4'], label='T4'),
               Patch(facecolor=SURFACE, edgecolor=MUTED, linewidth=1.0, label='exploratory')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=7, handlelength=1.2,
               columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.8)
    fig.savefig(path)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preview', type=Path, help='also write PNG previews here')
    args = p.parse_args()
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'axes.unicode_minus': True})
    secs = sections(ANALYSIS.read_text(encoding='utf-8'))
    depth, imprint = depth_rows(secs), imprint_rows(secs)
    for key in sorted(depth):
        print('depth', key, depth[key])
    for key in sorted(imprint):
        print('imprint', key, imprint[key])
    figs = {'fig1_k_vs_accuracy': figure_shots(depth, HERE / 'fig1_k_vs_accuracy.pdf'),
            'fig4_imprinting_gain': figure_imprint(imprint, HERE / 'fig4_imprinting_gain.pdf')}
    if args.preview:
        for name, fig in figs.items():
            fig.savefig(args.preview / f'{name}.png', dpi=220)


if __name__ == '__main__':
    main()
