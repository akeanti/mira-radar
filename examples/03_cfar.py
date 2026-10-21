"""CFAR thresholds, and why ordered-statistic CFAR exists.

A strong target six cells away from a weak one drags the cell-averaging
threshold up and hides the weak target. Ordered-statistic CFAR ranks the
training cells and ignores the outliers.

    python examples/03_cfar.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mira.processing import cfar_2d
from mira.viz import style

style.apply()
rng = np.random.default_rng(7)
x = rng.exponential(1.0, 200)
x[60] += 10 ** 2.0  # isolated target, 20 dB
x[130] += 10 ** 3.2  # strong target, 32 dB
x[136] += 10 ** 2.0  # weak target right next to it, 20 dB

fig, ax = plt.subplots(figsize=(13, 4.6), dpi=110)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.13, top=0.86)
db = lambda v: 10 * np.log10(v)  # noqa: E731
ax.plot(db(x), color=style.DIM, lw=1, label="power")
for method, col in (("ca", style.ACCENT_2), ("os", style.ACCENT)):
    res = cfar_2d(x[None, :], (0, 2), (0, 12), 1e-5, method, wrap0=False, peaks_only=False)
    thr = res.threshold[0]
    ax.plot(db(thr), color=col, lw=1.6, label=f"{method.upper()}-CFAR threshold, Pfa = 1e-5")
    hits = np.nonzero(res.mask[0])[0]
    ax.scatter(hits, db(x[hits]) + (2.5 if method == "ca" else 5), marker="v", color=col, s=40, zorder=5)
ax.annotate("weak target hidden by\nits neighbour (CA)", xy=(136, db(x[136])), xytext=(150, 30),
            color=style.TEXT, size=9, arrowprops=dict(arrowstyle="->", color=style.DIM))
ax.set_xlabel("range bin")
ax.set_ylabel("dB")
ax.set_ylim(-15, 45)
ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=style.TEXT)
ax.set_title("Cell-averaging vs ordered-statistic CFAR  ·  ▼ = detection", loc="left")
out = Path(__file__).resolve().parents[1] / "docs" / "media" / "cfar.png"
fig.savefig(out)
print(f"wrote {out}")
