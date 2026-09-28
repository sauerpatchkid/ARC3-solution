"""Build docs/reports/goose_options_comparison.png: every option tried in
semester 2, compared with the novelty-label agent measured in the same test.

    uv run python docs/reports/make_options_chart.py docs/reports/goose_options_comparison.png

Numbers are the total levels per arm read from each test's manifest (see
docs/plans/*results*.md). Colours validated with the dataviz skill's checker:
blue 600 #184f95 (adopted) and blue 350 #5598e7 (ingredient) pass as an ordinal
pair; both separate from the grey (worst CVD dE 20.8, normal 23.8); every bar is
value-labelled because the lighter marks sit under 3:1 contrast.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import IdentityTransform

OUT = sys.argv[1]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ADOPTED, INGREDIENT, DROPPED = "#184f95", "#5598e7", "#c3c2b7"
DPI = 200
RADIUS_PX = 4 * DPI / 96          # a 4 CSS-px data-end radius at this resolution

# --------------------------------------------------------------------------------
# data: (label, levels, novelty_levels_same_test, class)
A, I, D = "adopted", "ingredient", "dropped"
GROUPS = [
    ("Plan B, full test: 25 games × 3 seeds × 100k moves", [
        ("Original Goose (“did the screen change?”)", 54, 79, D)]),
    ("Plan B, first test: 6 games × 3 seeds × 100k moves", [
        ("Don’t-repeat mask", 27, 32, D),
        ("Novelty + don’t-repeat mask", 27, 32, D)]),
    ("Return map: 8 games × 3 seeds, then 25 games × 2 seeds, 100k moves", [
        ("Return map (8-game test)", 45, 32, I),
        ("Return map (25-game test)", 58, 50, I)]),
    ("Upgrade screen, round 1: 8 games × 2 seeds × 50k moves", [
        ("Map + bars fix", 30, 14, I),
        ("Map, walk back only when it pays", 30, 14, I),
        ("Map, object-level clicks", 28, 14, D),
        ("Map, walk back to less-visited places", 28, 14, D),
        ("Return map", 26, 14, I),
        ("Bars fix + half credit per attempt", 20, 14, I),
        ("Skip dead clicks", 18, 14, D),
        ("Keep network across levels", 18, 14, D),
        ("Bars fix alone", 16, 14, I),
        ("Map, only when stuck", 15, 14, D),
        ("Softer reward (1/√visits)", 11, 14, D)]),
    ("Upgrade screen, round 2: 11 games × 2 seeds × 50k moves", [
        ("Map + bars + chooser + half credit", 41, 23, A),
        ("All four + skip dead clicks", 40, 23, D),
        ("Map + bars fix", 38, 23, I),
        ("Map + bars + chooser", 38, 23, I),
        ("Map + bars + half credit", 38, 23, I),
        ("Map, walk back only when it pays", 37, 23, I),
        ("Map + bars + chooser + skip dead clicks", 36, 23, D)]),
    ("Final test: 25 games × 3 seeds × 100k moves", [
        ("Map + bars + chooser + half credit", 112, 79, A)]),
]
COLOR = {A: ADOPTED, I: INGREDIENT, D: DROPPED}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": INK2})

n_rows = sum(len(r) + 1 for _, r in GROUPS)
fig = plt.figure(figsize=(11, 3.2 + 0.30 * n_rows), dpi=DPI, facecolor=SURFACE)
gs = fig.add_gridspec(2, 1, height_ratios=[1.7, 0.30 * n_rows], hspace=0.28,
                      left=0.36, right=0.95, top=0.92, bottom=0.095)


def rounded_bars(ax, ys, x0s, x1s, colors, height):
    """Bars with a 4px rounded data-end and a square baseline end, drawn in
    display space so the radius is a true circle whatever the axis scales."""
    fig.canvas.draw()
    to_disp = ax.transData.transform
    for y, xa, xb, c in zip(ys, x0s, x1s, colors):
        (dx0, dy0), (dx1, dy1) = to_disp((xa, y - height / 2)), to_disp((xb, y + height / 2))
        left, right = min(dx0, dx1), max(dx0, dx1)
        bottom, top = min(dy0, dy1), max(dy0, dy1)
        w, h = right - left, top - bottom
        if w <= 0:
            continue
        r = min(RADIUS_PX, w / 2, h / 2)
        ax.add_patch(FancyBboxPatch((left, bottom), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    transform=IdentityTransform(), facecolor=c, edgecolor="none",
                                    zorder=3, clip_on=False))
        base = dx0                                   # xa is always the baseline (0)
        sq_left = base if dx1 >= dx0 else base - min(w, r * 2)
        ax.add_patch(Rectangle((sq_left, bottom), min(w, r * 2), h, transform=IdentityTransform(),
                               facecolor=c, edgecolor="none", zorder=3, clip_on=False))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------------
# Panel A: the headline, on the full 25-game test
axA = fig.add_subplot(gs[0])
style(axA)
head = [("Original Goose", 54, D, "54"),
        ("Novelty label (adopted after Plan B)", 79, I, "79  (+46% vs original)"),
        ("Final combination (adopted now)", 112, A, "112  (+42% vs novelty label)")]
ys = list(range(len(head)))[::-1]
axA.set_ylim(-0.6, len(head) - 0.4)
axA.set_xlim(0, 170)
axA.set_xticks([0, 20, 40, 60, 80, 100, 120])
axA.set_yticks(ys)
axA.set_yticklabels([h[0] for h in head], fontsize=10)
rounded_bars(axA, ys, [0] * 3, [h[1] for h in head], [COLOR[h[2]] for h in head], 0.5)
for y, h in zip(ys, head):
    axA.text(h[1] + 2, y, h[3], va="center", fontsize=9.5, color=INK)
axA.set_xlabel("Levels finished across all 25 games × 3 seeds × 100,000 moves (75 runs)", fontsize=9)
axA.set_title("The headline: levels finished on the full 25-game test", loc="left", fontsize=13,
              fontweight="bold", color=INK, pad=10, x=-0.53)

# --------------------------------------------------------------------------------
# Panel B: every option vs the novelty agent in the same test
axB = fig.add_subplot(gs[1])
style(axB)
labels, ys, pcts, cols, tips = [], [], [], [], []
y = n_rows
headers = []
for title, rows in GROUPS:
    y -= 1
    headers.append((y, title))
    for label, lv, ref, cls in rows:
        y -= 1
        pct = 100.0 * (lv - ref) / ref
        labels.append(label)
        ys.append(y)
        pcts.append(pct)
        cols.append(COLOR[cls])
        tips.append(f"{pct:+.0f}%  ({lv} vs {ref})".replace("-", "\u2212"))
axB.set_xlim(-60, 150)
axB.set_ylim(-0.7, n_rows - 0.3)
axB.set_yticks(ys)
axB.set_yticklabels(labels, fontsize=9)
axB.axvline(0, color=MUTED, linewidth=1.0, zorder=2)
rounded_bars(axB, ys, [0] * len(ys), pcts, cols, 0.62)
for yy, p, t in zip(ys, pcts, tips):
    # positive bars: value at the tip; negative bars: value just right of the
    # baseline, because the left side has no room before the row names
    axB.text(max(p, 0) + 2, yy, t, va="center", fontsize=8.5, color=INK2)
for yy, title in headers:
    axB.text(-0.53, yy, title, transform=axB.get_yaxis_transform(), va="center", ha="left",
             fontsize=9.5, fontweight="bold", color=INK)
    axB.axhline(yy + 0.5, color=GRID, linewidth=0.8, zorder=1)
axB.set_xticks([-50, 0, 50, 100, 150])
axB.set_xticklabels(["−50%", "0\n(novelty agent)", "+50%", "+100%", "+150%"])
axB.set_xlabel("Levels finished, compared with the novelty-label agent run in the same test", fontsize=9)
axB.set_title("Every option we tried, against the novelty-label agent in the same test", loc="left",
              fontsize=13, fontweight="bold", color=INK, pad=24, x=-0.53)

# legend (always present: three classes)
handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in (ADOPTED, INGREDIENT, DROPPED)]
axB.legend(handles, ["The adopted Goose", "An ingredient of it", "Not carried forward"],
           loc="lower center", bbox_to_anchor=(0.30, 1.0), ncol=3, frameon=False, fontsize=9,
           handlelength=1.2, handleheight=1.0, borderaxespad=0.2)

fig.text(0.02, 0.012,
         "Short screens use 2 seeds and 50,000 moves, so their percentages are larger and noisier than the "
         "full tests’. Compare options within a test, not across tests.",
         fontsize=8.5, color=INK2)
fig.savefig(OUT, dpi=DPI, facecolor=SURFACE)
print("wrote", OUT)
