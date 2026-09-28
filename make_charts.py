# makes the 2 charts in the readme: python make_charts.py
# epoch_scores.csv is each library's own val mAP@50 per epoch, pulled from our training logs
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = Path(__file__).parent / "assets"
INK, MUTED, GRID = "#172321", "#56645F", "#E3E8E6"
RF, DF, YO, TEST = "#1F7A63", "#B8741A", "#5361B5", "#8A3B2E"

scores = {}
for r in csv.DictReader(open(ASSETS / "epoch_scores.csv")):
    scores.setdefault(r["model"], []).append((int(r["epoch"]), float(r["val_map50"])))

plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})

# val score after every epoch
fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
fig.patch.set_facecolor("white")
series = (
    ("rfdetr_large", RF, "RF-DETR Large · 50 epochs · 31 min"),
    ("dfine_x", DF, "D-FINE-X · 80 epochs · 89 min"),
    ("yolo26x", YO, "YOLO26-X 1280 px · 100 epochs · 2.3 h"),
)
for name, color, label in series:
    pts = scores[name]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=2, label=label, solid_capstyle="round")
    pe, pm = max(pts, key=lambda p: p[1])
    ax.plot([pe], [pm], "o", color=color, ms=7, zorder=5)
    ax.annotate(f"peak {pm:.3f}\nepoch {pe}", (pe, pm), xytext=(0, 12), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
    le, lm = pts[-1]
    ax.annotate(f"{lm:.3f}", (le, lm), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=color)
ax.axvline(25, color=RF, ls="--", lw=1.2, alpha=0.8)
ax.text(26, 0.712, "we train RF-DETR\nfor 25 epochs", color=RF, fontsize=9, va="bottom")
ax.set_xlim(0, 108)
ax.set_ylim(0.70, 0.935)
ax.set_xlabel("epoch")
ax.set_ylabel("validation mAP@50\n(each library's own COCO-style metric)")
ax.set_title("Every model plateaued, then slipped by its last epoch", loc="left", fontsize=13, fontweight="bold", pad=12)
ax.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(ASSETS / "epoch_curves.png", facecolor="white")

# how the score went up, on the organizers' 11-point metric
steps = [
    ("YOLO26-X, one run", 0.7981, YO),
    ("D-FINE-X, one run", 0.8388, DF),
    ("RF-DETR Base, one run", 0.8456, MUTED),
    ("RF-DETR Large, one run", 0.8499, RF),
    ("+ two runs averaged (model soup)", 0.8622, RF),
    ("+ flip at prediction time = final", 0.8737, RF),
    ("final model on the hidden test (171 pictures)", 0.8502, TEST),
]
fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
fig.patch.set_facecolor("white")
lo = 0.78
for i, (label, v, color) in enumerate(steps):
    y = len(steps) - 1 - i
    ax.plot([lo, v], [y, y], color=color, lw=2.2, alpha=0.35 if i < 3 else 0.6, solid_capstyle="butt")
    ax.plot([v], [y], "o", color=color, ms=9, zorder=5)
    ax.text(v + 0.0015, y, f"{v:.4f}", va="center", fontsize=10, color=color, fontweight="bold" if i >= 5 else "normal")
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
ax.set_yticks(range(len(steps)))
ax.set_yticklabels([s[0] for s in steps][::-1])
ax.set_xlim(lo, 0.892)
ax.set_xlabel("mAP@50 on the organizers' metric (validation unless noted)")
ax.set_title("How the score climbed", loc="left", fontsize=13, fontweight="bold", pad=12)
ax.grid(True, axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(ASSETS / "score_progress.png", facecolor="white")
