"""Figure 1: per-repository gain in recall@10 from adding co-change edges (results/heldout.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
KEY = "samefile+static+cc - samefile+static"
POOLED = (0.079, 0.073, 0.085)  # results/heldout.txt, all 15 repositories
COMPETITION = {"httpx", "rich", "requests", "fastapi"}

rows = json.load(open(ROOT / "results" / "heldout.json"))
data = sorted(((r["repo"], *r["delta_r@10"][KEY], r["test_commits"]) for r in rows), key=lambda t: t[1])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ys = list(range(1, len(data) + 1))
for y, (name, d, lo, hi, n) in zip(ys, data):
    ax.plot([lo, hi], [y, y], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
    ax.plot(d, y, "o", ms=7, color=BLUE, mec=SURFACE, mew=1.5, zorder=3)
ax.plot([POOLED[1], POOLED[2]], [0, 0], color=INK, lw=2, solid_capstyle="round", zorder=2)
ax.plot(POOLED[0], 0, "D", ms=7, color=INK, mec=SURFACE, mew=1.5, zorder=3)
ax.text(POOLED[2] + 0.004, 0, "+0.079", va="center", ha="left", color=INK, fontsize=10)
ax.axvline(0, color=MUTED, lw=1, zorder=1)
labels = ["all 15 repositories"] + [f"{n}*" if n in COMPETITION else n for n, *_ in data]
ax.set_yticks([0] + ys, labels)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.set_ylim(-0.8, len(data) + 0.7)
ax.set_xlim(-0.005, 0.16)
ax.set_xticks([0, 0.05, 0.10, 0.15], ["0", "+0.05", "+0.10", "+0.15"])
ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="both", length=0, colors=MUTED)
for t in ax.get_yticklabels():
    t.set_color(INK)
ax.set_xlabel("Change in recall@10 from adding co-change edges", color=MUTED, labelpad=8)
fig.tight_layout(rect=(0, 0.04, 1, 0.88))
fig.text(0.03, 0.955, "History helps in every repository", color=INK, fontsize=13, fontweight="bold", ha="left")
fig.text(0.03, 0.91, "Gain over same-file + static-graph neighbours on 3,088 held-out commits, with 95% bootstrap intervals",
         color=MUTED, fontsize=8.5, ha="left")
fig.text(0.03, 0.025, "* repository behind the competition's public tasks", color=MUTED, fontsize=8.5, ha="left")
out = ROOT / "paper" / "figures" / "fig1_per_repo_gain.png"
fig.savefig(out, facecolor=SURFACE)
print(out)
for name, d, lo, hi, n in data:
    print(f"{name:14s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  commits={n}")
