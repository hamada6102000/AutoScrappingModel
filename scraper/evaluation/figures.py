"""
Paper figures (PNG, 300 dpi) drawn from the report — every value plotted is a
value of summary.json / the CSV tables, nothing else.

Colours: one fixed categorical slot per module (validated reference palette,
light mode); text in neutral ink, recessive grid, direct labels where few
marks are shown. Requires matplotlib (optional dependency).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .config import COND_INFO

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4
MODULE_COLOR = {"static": SLOTS[0], "rule_static": SLOTS[1], "rule_based": SLOTS[2],
                "self_healing": SLOTS[3]}


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "figure.dpi": 100,
    })
    return plt


def _pct_axis(ax):
    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _no_xgrid(ax):
    ax.grid(axis="x", visible=False)


def _legend_top(ax):
    """Legend in the band above the plot, right-aligned (never over data)."""
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=4, fontsize=8, borderaxespad=0.2)


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so end labels never overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def _short(cond: str) -> str:
    return COND_INFO[cond]["short"]


def module_f1(plt, rep, path: Path) -> Path | None:
    mods = [m for m in rep.get("modules") or [] if m["overall_f1"] is not None]
    if not mods:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.36
    xs = range(len(mods))
    series = [("initial_f1", "Original page (t0)", SLOTS[0]), ("changed_f1", "After layout changes", SLOTS[1])]
    for i, (key, label, color) in enumerate(series):
        vals = [m[key] if m[key] is not None else 0 for m in mods]
        pos = [x + (i - 0.5) * width for x in xs]
        bars = ax.bar(pos, vals, width - 0.04, color=color, label=label)
        for b, v, m in zip(bars, vals, mods):
            if m[key] is not None:
                ax.annotate(f"{v * 100:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(list(xs), [_short(m["condition"]) for m in mods])
    ax.set_ylim(0, 1.08)
    _pct_axis(ax)
    ax.set_ylabel("Field-level F1")
    ax.set_title("Extraction F1 by module", loc="left")
    _no_xgrid(ax)
    _legend_top(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def f1_by_step(plt, rep, path: Path) -> Path | None:
    rows = rep.get("f1_by_step") or []
    conds = rep.get("conditions_run") or []
    if not rows or not conds:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    steps = [r["step"] for r in rows]
    ends = []
    for cond in conds:
        ys = [r.get(f"{cond}_f1") for r in rows]
        pts = [(s, y) for s, y in zip(steps, ys) if y is not None]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=MODULE_COLOR[cond], linewidth=2,
                marker="o", markersize=4, label=_short(cond))
        ends.append((cond, pts[-1]))
    lo_all = min((r.get(f"{c}_f1") for r in rows for c in conds if r.get(f"{c}_f1") is not None), default=0)
    gap = max(0.03, (1.02 - max(0, lo_all - 0.05)) * 0.06)
    placed = _spread([p[1] for _, p in ends], gap)
    for (cond, (x, _y)), y in zip(ends, placed):
        ax.annotate(f"{_short(cond)} {_y * 100:.1f}%", (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK_2)
    ax.set_xticks(steps)
    ax.set_xlabel("Step (0 = original page; 1-8 = cumulative simulated changes)")
    ax.set_ylabel("Field-level F1")
    _pct_axis(ax)
    lo = min((r.get(f"{c}_f1") for r in rows for c in conds if r.get(f"{c}_f1") is not None), default=0)
    ax.set_ylim(max(0, lo - 0.05), 1.02)
    ax.set_xlim(min(steps) - 0.3, max(steps) + 2.2)
    ax.set_title("F1 over the change sequence", loc="left")
    _legend_top(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def healing_by_operator(plt, data, rep, path: Path) -> Path | None:
    conds = [c for c in rep.get("conditions_run") or [] if COND_INFO[c]["heal"]]
    by = defaultdict(lambda: [0, 0])
    ops = set()
    for r in data["runs"].values():
        if r["condition"] in conds and r["step"] and r.get("breaking") and r["detected"]:
            ops.add(r["op"])
            by[(r["condition"], r["op"])][0] += 1
            by[(r["condition"], r["op"])][1] += 1 if r.get("gt_recovered") else 0
    if not ops:
        return None
    ops = sorted(ops)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = min(0.35, 0.8 / len(conds))
    for i, cond in enumerate(conds):
        vals, ns = [], []
        for op in ops:
            n, k = by.get((cond, op), [0, 0])
            vals.append(k / n if n else 0)
            ns.append(n)
        pos = [x + (i - (len(conds) - 1) / 2) * width for x in range(len(ops))]
        bars = ax.bar(pos, vals, width - 0.04, color=MODULE_COLOR[cond], label=_short(cond))
        for b, v, n in zip(bars, vals, ns):
            if n:
                ax.annotate(f"n={n}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=6, color=INK_2)
    ax.set_xticks(range(len(ops)), [o.replace("_", "\n") for o in ops], fontsize=7)
    ax.set_ylim(0, 1.12)
    _pct_axis(ax)
    ax.set_ylabel("GT-verified healing success")
    ax.set_title("Healing success by change operator\n(detected breaking changes)", loc="left")
    _no_xgrid(ax)
    _legend_top(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def token_reduction(plt, data, path: Path) -> Path | None:
    rows = sorted((b for b in data["baselines"] if b["raw_tokens"]), key=lambda b: b["site"])
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    xs = range(len(rows))
    width = 0.38
    for i, (key, label) in enumerate((("raw_tokens", "Raw HTML"), ("llm_view_tokens", "LLM view"))):
        pos = [x + (i - 0.5) * width for x in xs]
        ax.bar(pos, [b[key] for b in rows], width - 0.04, color=SLOTS[i], label=label)
    ax.set_yscale("log")
    ax.set_xticks(list(xs), [b["site"].replace("site_", "") for b in rows], fontsize=7)
    ax.set_xlabel("Site")
    ax.set_ylabel("Tokens (log scale)")
    ax.set_title("Page size before and after pruning", loc="left")
    _no_xgrid(ax)
    _legend_top(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def write_figures(rep: dict, data: dict, out_dir: Path) -> dict:
    """Draw every figure that has data; returns {name: path or reason}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt = _setup()
    except ImportError:
        return {"error": "matplotlib is not installed (pip install matplotlib)"}
    made = {}
    for name, fn in (("module_f1", lambda p: module_f1(plt, rep, p)),
                     ("f1_by_step", lambda p: f1_by_step(plt, rep, p)),
                     ("healing_by_operator", lambda p: healing_by_operator(plt, data, rep, p)),
                     ("token_reduction", lambda p: token_reduction(plt, data, p))):
        path = out_dir / f"{name}.png"
        try:
            result = fn(path)
            made[name] = str(result) if result else "no data"
        except Exception as exc:  # a figure must never break the report
            made[name] = f"failed: {type(exc).__name__}: {exc}"
    return made
