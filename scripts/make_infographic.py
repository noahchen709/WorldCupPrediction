"""Render a polished, print-ready PDF infographic for the 2026 World Cup forecast.

Reads the Monte Carlo simulation output (and, when available, the historical
backtests) and lays out an editorial, poster-style PDF plus per-page PNGs that
are easy to share on LinkedIn or attach to a resume.

Usage:
    python3 scripts/make_infographic.py
    python3 scripts/make_infographic.py --out reports/world_cup_2026_infographic.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

# Matplotlib writes a font cache on first import; point it at a writable dir
# so the script runs cleanly in sandboxes and CI.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wc_mpl_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"

# Publication palette: cleaner, cooler, and less earthy.
PAPER = "#e8edf0"
CARD = "#f9faf7"
CARD_ALT = "#edf1ec"
INK = "#13213a"
INK2 = "#4b5968"
MUTED = "#7b8994"
LINE = "#cbd4d9"
ACCENT = "#167a7f"
ACCENT_INK = "#0e565a"
ACCENT_SOFT = "#dcebec"
PANEL_HEAD = "#1b2e4b"
CLAY = "#d65a3a"
GOLD = "#b98922"
TEAL = "#3b88c3"
PAPER_LINE = "#d6dde1"

# Categorical palette for confederations (muted, warm, non-clashing).
CONF_COLORS = {
    "UEFA": "#167a7f",
    "CONMEBOL": "#d65a3a",
    "CAF": "#b98922",
    "CONCACAF": "#3b88c3",
    "AFC": "#7464a8",
    "OFC": "#5f9d78",
    "Other": "#88919a",
}

# 4:5 portrait works well for LinkedIn feeds/carousels and still drops neatly
# into research reports as a full-page figure plate.
PAGE_W, PAGE_H = 9.0, 11.25


def resolve_font(candidates: list[str], fallback: str) -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return fallback


SERIF = resolve_font(["Fraunces", "Georgia", "Charter", "Palatino", "DejaVu Serif"], "serif")
SANS = resolve_font(
    ["Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "sans-serif"
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": SANS,
            "text.color": INK,
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.facecolor": CARD,
            "svg.fonttype": "none",
        }
    )


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def pct(value: float, digits: int = 1) -> str:
    value = value or 0.0
    if 0 < value < 0.001:
        return "<0.1%"
    return f"{value * 100:.{digits}f}%"


def new_page() -> plt.Figure:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patches.append(
        plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor=PAPER, zorder=-10)
    )
    fig.patches.append(
        plt.Rectangle((0.026, 0.026), 0.948, 0.948, transform=fig.transFigure,
                      facecolor="none", edgecolor=PAPER_LINE, linewidth=1.0, zorder=-8)
    )
    # Quiet pitch/grid lines give the pages texture without looking decorative.
    for yy in (0.20, 0.50, 0.80):
        fig.add_artist(
            plt.Line2D([0.03, 0.97], [yy, yy], transform=fig.transFigure,
                       color=PAPER_LINE, lw=0.6, alpha=0.45, zorder=-7)
        )
    for xx in (0.18, 0.50, 0.82):
        fig.add_artist(
            plt.Line2D([xx, xx], [0.03, 0.97], transform=fig.transFigure,
                       color=PAPER_LINE, lw=0.6, alpha=0.35, zorder=-7)
        )
    return fig


def panel(fig, x, y, w, h, *, facecolor=CARD, edgecolor=LINE, lw=1.0, radius=0.006):
    """Draw a rounded card behind a region (figure-fraction coordinates)."""
    shadow = FancyBboxPatch(
        (x + 0.006, y - 0.006),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor="#81909a",
        edgecolor="none",
        alpha=0.12,
        mutation_aspect=PAGE_W / PAGE_H,
        zorder=-1,
        clip_on=False,
    )
    fig.patches.append(shadow)
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=fig.transFigure,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        mutation_aspect=PAGE_W / PAGE_H,
        zorder=0,
        clip_on=False,
    )
    fig.patches.append(box)
    return box


def header(fig, eyebrow: str, title: str, subtitle: str, kicker: str) -> None:
    fig.text(0.06, 0.948, eyebrow.upper(), color=GOLD, fontsize=9.6,
             fontweight="bold", family=SANS)
    fig.text(0.06, 0.912, title, color=INK, fontsize=26, family=SERIF, fontweight="bold",
             va="top")
    fig.text(0.06, 0.868, subtitle, color=INK2, fontsize=10.0, family=SANS, va="top")
    fig.text(0.94, 0.906, kicker.upper(), color=INK2, fontsize=7.2, family=SANS,
             fontweight="bold", ha="right", va="top", linespacing=1.35)
    line = plt.Line2D([0.06, 0.94], [0.832, 0.832], transform=fig.transFigure,
                      color=GOLD, lw=1.1, alpha=0.85)
    fig.add_artist(line)
    fig.add_artist(plt.Line2D([0.82, 0.94], [0.948, 0.948], transform=fig.transFigure,
                              color=TEAL, lw=5.0, solid_capstyle="butt"))


def footer(fig, text: str, page: str) -> None:
    fig.text(0.06, 0.037, text, color=INK2, fontsize=7.8, family=SANS, va="center")
    fig.text(0.50, 0.037, "WORLD CUP FORECAST 2026", color=GOLD, fontsize=7.4,
             family=SANS, va="center", ha="center", fontweight="bold")
    fig.text(0.94, 0.037, page, color=INK2, fontsize=7.8, family=SANS,
             va="center", ha="right", fontweight="bold")


def axes_in(fig, x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor("none")
    ax.set_zorder(3)  # draw plot content above the rounded panel cards
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return ax


# --------------------------------------------------------------------------- #
# Page 1 - the headline forecast
# --------------------------------------------------------------------------- #
def kpi_strip(fig, results, meta, generated) -> None:
    favourite = results[0]
    runner = results[1]
    cells = [
        ("Title favourite", favourite["team"], pct(favourite["champion_probability"]) + " to win"),
        ("Closest challenger", runner["team"], pct(runner["champion_probability"]) + " to win"),
        ("Field", f"{meta['teamCount']} teams", "12 groups of four"),
        ("Simulations", f"{meta['iterations']:,}", f"as of {generated}"),
    ]
    x0, w, gap = 0.06, 0.2025, 0.0167
    y, h = 0.724, 0.086
    for i, (label, value, note) in enumerate(cells):
        x = x0 + i * (w + gap)
        panel(fig, x, y, w, h)
        fig.patches.append(
            plt.Rectangle((x, y), 0.010, h, transform=fig.transFigure,
                          facecolor=GOLD if i == 0 else ACCENT, edgecolor="none", zorder=2)
        )
        fig.text(x + 0.018, y + h - 0.018, label.upper(), color=MUTED, fontsize=7.6,
                 fontweight="bold", va="top")
        fig.text(x + 0.018, y + h - 0.040, value, color=INK, fontsize=15.5, family=SERIF,
                 fontweight="bold", va="top")
        fig.text(x + 0.018, y + 0.013, note, color=ACCENT_INK, fontsize=8, va="bottom")


def champion_bars(fig, results, top_n=14) -> None:
    x, y, w, h = 0.06, 0.335, 0.88, 0.365
    panel(fig, x, y, w, h)
    fig.patches.append(
        plt.Rectangle((x, y + h - 0.070), w, 0.070, transform=fig.transFigure,
                      facecolor=PANEL_HEAD, edgecolor="none", zorder=1)
    )
    fig.text(x + 0.025, y + h - 0.023, "Who lifts the trophy?", color=CARD, fontsize=15,
             family=SERIF, fontweight="bold", va="top")
    fig.text(x + 0.025, y + h - 0.052, "Estimated probability of winning the 2026 World Cup",
             color="#d7e0e6", fontsize=9.2, va="top")
    top3 = sum(r["champion_probability"] for r in results[:3])
    fig.text(x + w - 0.030, y + h - 0.025, pct(top3), color=GOLD, fontsize=14.5,
             family=SERIF, fontweight="bold", ha="right", va="top")
    fig.text(x + w - 0.030, y + h - 0.052, "title share of top three", color="#d7e0e6",
             fontsize=7.6, ha="right", va="top", fontweight="bold")

    ax = axes_in(fig, x + 0.235, y + 0.035, w - 0.30, h - 0.125)
    teams = [r["team"] for r in results[:top_n]][::-1]
    probs = [r["champion_probability"] for r in results[:top_n]][::-1]
    colors = [CLAY if i == len(teams) - 1 else (TEAL if i >= len(teams) - 3 else ACCENT)
              for i in range(len(teams))]

    bars = ax.barh(range(len(teams)), probs, color=colors, height=0.66, zorder=3)
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams, fontsize=10.5, color=INK)
    ax.set_xlim(0, max(probs) * 1.16)
    ax.set_xticks([])
    ax.grid(axis="x", color=LINE, linewidth=0.6, alpha=0.55, zorder=1)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + max(probs) * 0.012, bar.get_y() + bar.get_height() / 2,
                pct(prob), va="center", ha="left", fontsize=9.5, color=INK2,
                fontweight="bold")
    # faint baseline
    ax.axvline(0, color=LINE, lw=1.0, zorder=2)


def confederation_donut(fig, results, conf_by_team) -> None:
    x, y, w, h = 0.06, 0.078, 0.42, 0.230
    panel(fig, x, y, w, h)
    fig.text(x + 0.025, y + h - 0.026, "Title geography", color=INK, fontsize=13,
             family=SERIF, fontweight="bold", va="top")
    fig.text(x + 0.025, y + h - 0.048, "Champion probability by confederation", color=MUTED,
             fontsize=8.6, va="top")

    totals: dict[str, float] = {}
    for r in results:
        conf = conf_by_team.get(r["team"], "Other")
        totals[conf] = totals.get(conf, 0.0) + r["champion_probability"]
    items = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    items = [(k, v) for k, v in items if v > 0.001]

    ax = axes_in(fig, x + 0.02, y + 0.012, 0.18, h - 0.085)
    ax.set_aspect("equal")
    sizes = [v for _, v in items]
    colors = [CONF_COLORS.get(k, "#9b9488") for k, _ in items]
    ax.pie(sizes, colors=colors, startangle=90, counterclock=False,
           wedgeprops={"width": 0.36, "edgecolor": CARD, "linewidth": 2.4})
    fig.text(x + 0.11, y + 0.094, pct(items[0][1], 0), color=INK, fontsize=13,
             family=SERIF, fontweight="bold", ha="center", va="center")

    # legend column
    lx = x + 0.225
    ly = y + h - 0.075
    for (name, value), color in zip(items, colors):
        ax.figure.patches.append(
            plt.Rectangle((lx, ly), 0.014, 0.014, transform=fig.transFigure,
                          facecolor=color, edgecolor="none")
        )
        fig.text(lx + 0.022, ly + 0.007, name, color=INK2, fontsize=8.6, va="center")
        fig.text(x + w - 0.022, ly + 0.007, pct(value), color=INK, fontsize=8.6,
                 va="center", ha="right", fontweight="bold")
        ly -= 0.030


def favourite_funnel(fig, results) -> None:
    x, y, w, h = 0.52, 0.078, 0.42, 0.230
    panel(fig, x, y, w, h)
    fav = results[0]
    fig.text(x + 0.025, y + h - 0.026, f"{fav['team']}'s survival curve", color=INK, fontsize=13,
             family=SERIF, fontweight="bold", va="top")
    fig.text(x + 0.025, y + h - 0.048, "Probability of reaching each stage", color=MUTED,
             fontsize=8.6, va="top")

    stages = [
        ("Round of 16", fav["round_of_16_probability"]),
        ("Quarter-final", fav["quarterfinal_probability"]),
        ("Semi-final", fav["semifinal_probability"]),
        ("Final", fav["final_probability"]),
        ("Champion", fav["champion_probability"]),
    ]
    ax = axes_in(fig, x + 0.15, y + 0.022, w - 0.175, h - 0.092)
    labels = [s for s, _ in stages][::-1]
    vals = [v for _, v in stages][::-1]
    bars = ax.barh(range(len(labels)), vals, color=ACCENT, height=0.52, zorder=3)
    bars[-1].set_color(CLAY)  # champion stage
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.3, color=INK)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([])
    for bar, v in zip(bars, vals):
        ax.text(min(bar.get_width() + 0.02, 0.86), bar.get_y() + bar.get_height() / 2,
                pct(v), va="center", ha="left", fontsize=8.8, color=INK2, fontweight="bold")


def render_page1(fig, results, meta, conf_by_team, generated) -> None:
    header(
        fig,
        "2026 FIFA World Cup · Quantitative Forecast",
        "Who Wins the World Cup?",
        "A Monte Carlo simulation of the 48-team tournament, driven by World Football Elo "
        "ratings\nand xG-adjusted match outcomes.",
        f"{meta['iterations']:,} simulations\n48-team tournament",
    )
    kpi_strip(fig, results, meta, generated)
    champion_bars(fig, results)
    confederation_donut(fig, results, conf_by_team)
    favourite_funnel(fig, results)
    footer(fig, f"Source: eloratings.net · generated {generated}", "01 / 03")


# --------------------------------------------------------------------------- #
# Page 2 - stage progression matrix + group favourites
# --------------------------------------------------------------------------- #
def _blend(t: float) -> tuple:
    """Blend paper -> accent green for a sequential heatmap cell."""
    import matplotlib.colors as mcolors

    c0 = mcolors.to_rgb("#e7edf0")
    c1 = mcolors.to_rgb(ACCENT)
    t = max(0.0, min(1.0, t)) ** 0.7
    return tuple(c0[i] + (c1[i] - c0[i]) * t for i in range(3))


def progression_matrix(fig, results, top_n=14) -> None:
    x, y, w, h = 0.06, 0.455, 0.88, 0.385
    panel(fig, x, y, w, h)
    fig.patches.append(
        plt.Rectangle((x, y + h - 0.067), w, 0.067, transform=fig.transFigure,
                      facecolor=PANEL_HEAD, edgecolor="none", zorder=1)
    )
    fig.text(x + 0.025, y + h - 0.024, "Knockout pressure map", color=CARD,
             fontsize=15, family=SERIF, fontweight="bold", va="top")
    fig.text(x + 0.025, y + h - 0.052,
             "Probability of reaching each stage. Darker means more likely.",
             color="#d7e0e6", fontsize=9.2, va="top")
    fig.text(x + w - 0.030, y + h - 0.027, "SURVIVAL", color=GOLD, fontsize=8.0,
             ha="right", va="top", fontweight="bold")
    fig.text(x + w - 0.030, y + h - 0.049, "round-by-round odds", color="#d7e0e6",
             fontsize=7.3, ha="right", va="top", fontweight="bold")

    cols = [
        ("Round of 16", "round_of_16_probability"),
        ("Quarter", "quarterfinal_probability"),
        ("Semi", "semifinal_probability"),
        ("Final", "final_probability"),
        ("Win", "champion_probability"),
    ]
    ax = axes_in(fig, x + 0.025, y + 0.02, w - 0.05, h - 0.130)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, top_n)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    label_w = 2.05  # in data units used for the team name column
    ax.set_xlim(-label_w, len(cols))

    # column headers
    for j, (name, _) in enumerate(cols):
        ax.text(j + 0.5, -0.28, name, ha="center", va="bottom", fontsize=8.7,
                color=MUTED, fontweight="bold")

    for i, r in enumerate(results[:top_n]):
        ax.text(-label_w + 0.04, i + 0.5, f"{i + 1}", ha="left", va="center",
                fontsize=8.5, color=MUTED, fontweight="bold")
        ax.text(-label_w + 0.30, i + 0.5, r["team"], ha="left", va="center",
                fontsize=9.6, color=INK)
        for j, (_, key) in enumerate(cols):
            v = r.get(key, 0.0)
            ax.add_patch(
                plt.Rectangle((j + 0.05, i + 0.12), 0.90, 0.76, facecolor=_blend(v),
                              edgecolor=CARD, linewidth=1.3)
            )
            txt_color = "#ffffff" if v > 0.42 else INK2
            ax.text(j + 0.5, i + 0.5, pct(v, 0) if v >= 0.1 else pct(v), ha="center",
                    va="center", fontsize=8.2, color=txt_color, fontweight="bold")


def group_favourites(fig, structure, results) -> None:
    x, y, w, h = 0.06, 0.078, 0.88, 0.340
    panel(fig, x, y, w, h)
    fig.text(x + 0.025, y + h - 0.026, "Group favourites", color=INK, fontsize=15,
             family=SERIF, fontweight="bold", va="top")
    fig.text(x + 0.025, y + h - 0.052,
             "Two most likely sides to advance from each group.", color=MUTED,
             fontsize=9.5, va="top")

    prob_by_team = {r["team"]: r for r in results}
    groups = structure.get("groups", [])

    cols, rows = 4, 3
    cell_w = (w - 0.05) / cols
    cell_h = (h - 0.105) / rows
    gx0 = x + 0.025
    gy0 = y + 0.020

    for idx, group in enumerate(groups[: rows * cols]):
        c = idx % cols
        rr = idx // cols
        cx = gx0 + c * cell_w
        cy = gy0 + (rows - 1 - rr) * cell_h
        fig.patches.append(
            plt.Rectangle((cx + 0.006, cy + 0.006), cell_w - 0.014, cell_h - 0.014,
                          transform=fig.transFigure, facecolor=CARD_ALT, edgecolor=LINE,
                          linewidth=0.8, zorder=1)
        )
        fig.patches.append(
            plt.Rectangle((cx + 0.006, cy + cell_h - 0.022), cell_w - 0.014, 0.016,
                          transform=fig.transFigure, facecolor=ACCENT_INK, edgecolor="none",
                          zorder=2)
        )
        fig.text(cx + 0.016, cy + cell_h - 0.014, f"GROUP {group['group']}", color=CARD,
                 fontsize=7.1, fontweight="bold", va="center", zorder=3)

        ranked = sorted(
            group["teams"],
            key=lambda t: prob_by_team.get(t, {}).get("round_of_16_probability", 0.0),
            reverse=True,
        )
        yy = cy + cell_h - 0.040
        for pos, team in enumerate(ranked[:2]):
            adv = prob_by_team.get(team, {}).get("round_of_16_probability", 0.0)
            marker = ACCENT if pos == 0 else GOLD
            fig.text(cx + 0.016, yy, team, color=INK, fontsize=7.6, va="center",
                     fontweight="bold")
            fig.text(cx + cell_w - 0.024, yy, pct(adv, 0), color=INK2, fontsize=7.4,
                     va="center", ha="right", fontweight="bold")
            fig.patches.append(
                plt.Rectangle((cx + 0.016, yy - 0.012), cell_w - 0.056, 0.005,
                              transform=fig.transFigure, facecolor=LINE,
                              edgecolor="none", zorder=2)
            )
            fig.patches.append(
                plt.Rectangle((cx + 0.016, yy - 0.012), (cell_w - 0.056) * adv, 0.005,
                              transform=fig.transFigure, facecolor=marker, edgecolor="none",
                              zorder=3)
            )
            yy -= 0.022


def render_page2(fig, results, structure, meta, generated) -> None:
    header(
        fig,
        "2026 FIFA World Cup · Stage Probabilities",
        "The Path Through the Bracket",
        "Each team's chance of surviving to the round of 16, quarter-finals, semi-finals, "
        "final and title.",
        "Stage survival\nTop 14 teams",
    )
    progression_matrix(fig, results)
    group_favourites(fig, structure, results)
    footer(fig, f"Source: eloratings.net · {meta['iterations']:,} simulations", "02 / 03")


# --------------------------------------------------------------------------- #
# Page 3 - model credibility / backtests
# --------------------------------------------------------------------------- #
def backtest_panel(fig, tournaments) -> None:
    x, y, w, h = 0.06, 0.395, 0.88, 0.445
    panel(fig, x, y, w, h)
    fig.patches.append(
        plt.Rectangle((x, y + h - 0.070), w, 0.070, transform=fig.transFigure,
                      facecolor=PANEL_HEAD, edgecolor="none", zorder=1)
    )
    fig.text(x + 0.025, y + h - 0.024, "Backtested under pressure", color=CARD,
             fontsize=15, family=SERIF, fontweight="bold", va="top")
    fig.text(x + 0.025, y + h - 0.052,
             "Re-running the model with pre-tournament ratings only — no hindsight.",
             color="#d7e0e6", fontsize=9.2, va="top")
    fig.text(x + w - 0.030, y + h - 0.027, "3 TOURNAMENTS", color=GOLD, fontsize=8.0,
             ha="right", va="top", fontweight="bold")
    fig.text(x + w - 0.030, y + h - 0.049, "2014 · 2018 · 2022", color="#d7e0e6",
             fontsize=7.3, ha="right", va="top", fontweight="bold")

    rows = []
    for t in tournaments:
        s = t["summary"]
        rows.append(
            (
                s["tournament"].replace(" FIFA World Cup", ""),
                s["actual_champion"],
                s["actual_champion_probability"],
                s["actual_champion_rank"],
                s["top_pick"],
            )
        )

    n = len(rows)
    if not n:
        return
    top = y + h - 0.095
    bottom = y + 0.03
    row_h = (top - bottom) / n
    max_prob = max(r[2] for r in rows) or 1.0

    # column headers
    fig.text(x + 0.04, top + 0.012, "WORLD CUP", color=INK2, fontsize=7.8, fontweight="bold")
    fig.text(x + 0.20, top + 0.012, "ACTUAL CHAMPION", color=INK2, fontsize=7.8, fontweight="bold")
    fig.text(x + 0.40, top + 0.012, "PRE-TOURNAMENT ODDS", color=INK2, fontsize=7.8,
             fontweight="bold")
    fig.text(x + w - 0.04, top + 0.012, "RANK", color=INK2, fontsize=7.8, fontweight="bold",
             ha="right")

    ax = axes_in(fig, x + 0.40, bottom, 0.34, top - bottom)
    ax.set_xlim(0, max_prob * 1.25)
    ax.set_ylim(0, n)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    for i, (name, champ, prob, rank, top_pick) in enumerate(rows):
        cy = top - (i + 0.5) * row_h
        if i % 2 == 0:
            fig.patches.append(
                plt.Rectangle((x + 0.025, cy - row_h * 0.34), w - 0.05, row_h * 0.68,
                              transform=fig.transFigure, facecolor=ACCENT_SOFT,
                              edgecolor="none", zorder=1)
            )
        fig.text(x + 0.04, cy, name, color=INK, fontsize=11, family=SERIF,
                 fontweight="bold", va="center")
        fig.text(x + 0.20, cy, champ, color=INK2, fontsize=9.6, va="center")
        # rank badge
        good = rank <= 5
        fig.text(x + w - 0.04, cy, f"#{rank}", color=ACCENT_INK if good else CLAY,
                 fontsize=11, fontweight="bold", va="center", ha="right", family=SERIF)
        # bar
        ax.barh(i + 0.5, prob, height=0.30, color=ACCENT if rank <= 3 else TEAL, zorder=3)
        ax.text(prob + max_prob * 0.02, i + 0.5, pct(prob), va="center", ha="left",
                fontsize=8.8, color=INK2, fontweight="bold")


def takeaways_panel(fig, tournaments) -> None:
    x, y, w, h = 0.06, 0.078, 0.88, 0.280
    panel(fig, x, y, w, h)
    fig.text(x + 0.025, y + h - 0.026, "Receipts", color=INK, fontsize=14,
             family=SERIF, fontweight="bold", va="top")

    ranks = [t["summary"]["actual_champion_rank"] for t in tournaments]
    cal = [t["summary"]["calibration_error"] for t in tournaments]
    avg_rank = sum(ranks) / len(ranks) if ranks else 0
    avg_cal = sum(cal) / len(cal) if cal else 0
    top5 = sum(1 for r in ranks if r <= 5)

    stats = [
        (f"#{avg_rank:.1f}", "average pre-tournament rank\nthe model gave the eventual winner"),
        (f"{top5}/{len(ranks)}", "champions the model placed\ninside its top five favourites"),
        (f"{avg_cal:.3f}", "mean calibration error\n(predicted vs. observed rates)"),
    ]
    cw = (w - 0.05) / 3
    for i, (big, note) in enumerate(stats):
        cx = x + 0.025 + i * cw
        if i:
            fig.add_artist(
                plt.Line2D([cx - 0.010, cx - 0.010], [y + 0.055, y + h - 0.075],
                           transform=fig.transFigure, color=LINE, lw=1.0)
            )
        fig.text(cx + 0.01, y + h - 0.105, big, color=ACCENT_INK, fontsize=31, family=SERIF,
                 fontweight="bold", va="center")
        fig.text(cx + 0.01, y + h - 0.175, note, color=INK2, fontsize=9.4, va="top",
                 linespacing=1.5)


def render_page3(fig, backtests, meta, generated) -> None:
    header(
        fig,
        "2026 FIFA World Cup · Model Validation",
        "Does the Model Actually Work?",
        "Backtests replay each past tournament from its pre-tournament ratings and score the "
        "forecast.",
        "Historical replay\nNo hindsight",
    )
    tournaments = backtests.get("tournaments", [])
    backtest_panel(fig, tournaments)
    takeaways_panel(fig, tournaments)
    footer(fig, "Source: pre-tournament Elo ratings", "03 / 03")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_confederation_map(strengths_path: Path) -> dict[str, str]:
    if not strengths_path.exists():
        return {}
    payload = load_json(strengths_path)
    return {t["name"]: t.get("confederation", "Other") for t in payload.get("teams", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the 2026 World Cup PDF infographic.")
    parser.add_argument("--results", type=Path, default=REPORTS_DIR / "monte_carlo_results.json")
    parser.add_argument("--strengths", type=Path, default=DATA_DIR / "derived_team_strengths.json")
    parser.add_argument("--backtests", type=Path, default=REPORTS_DIR / "world-cup-backtests.json")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "world_cup_2026_infographic.pdf")
    parser.add_argument("--png", action="store_true", help="Also export each page as a PNG.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(
            f"Missing {args.results}. Run scripts/run_simulation.py first to generate it."
        )

    apply_style()

    report = load_json(args.results)
    results = report["results"]
    structure = report.get("tournamentStructure", {})
    meta = {
        "iterations": report.get("iterations", 0),
        "teamCount": report.get("teamCount", len(results)),
    }
    generated = datetime.now().strftime("%B %-d, %Y")
    conf_by_team = build_confederation_map(args.strengths)
    backtests = load_json(args.backtests) if args.backtests.exists() else {"tournaments": []}

    args.out.parent.mkdir(parents=True, exist_ok=True)

    pages = []
    fig1 = new_page()
    render_page1(fig1, results, meta, conf_by_team, generated)
    pages.append(("page1", fig1))

    fig2 = new_page()
    render_page2(fig2, results, structure, meta, generated)
    pages.append(("page2", fig2))

    if backtests.get("tournaments"):
        fig3 = new_page()
        render_page3(fig3, backtests, meta, generated)
        pages.append(("page3", fig3))

    with PdfPages(args.out) as pdf:
        for _, fig in pages:
            pdf.savefig(fig, dpi=args.dpi)

    if args.png:
        stem = args.out.with_suffix("")
        for name, fig in pages:
            png_path = Path(f"{stem}_{name}.png")
            fig.savefig(png_path, dpi=args.dpi)
            print(f"Wrote: {png_path}")

    for _, fig in pages:
        plt.close(fig)

    print(f"Champion favourite: {results[0]['team']} ({pct(results[0]['champion_probability'])})")
    print(f"Wrote: {args.out} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
