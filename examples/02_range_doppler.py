"""Range-Doppler map of the intersection, before and after static clutter removal.

    python examples/02_range_doppler.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mira import FMCWSimulator, get_preset
from mira.processing import doppler_fft, range_doppler_power, range_fft, remove_static_clutter, to_db
from mira.scenarios import get_scenario
from mira.viz import style

style.apply()
scene, preset, _ = get_scenario("intersection")
cfg = get_preset(preset)
for _ in range(60):  # three seconds in, when everyone is on screen
    scene.step(cfg.frame_period)
cube = FMCWSimulator(cfg, seed=0).frame(scene)

rfft = range_fft(cube)
maps = {
    "raw: walls, posts and parked cars swamp everything": range_doppler_power(doppler_fft(rfft)),
    "after static clutter removal: only what moves": range_doppler_power(doppler_fft(remove_static_clutter(rfft))),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=110, sharey=True)
fig.subplots_adjust(left=0.05, right=0.93, bottom=0.12, top=0.84, wspace=0.06)
# dB above the thermal noise floor, which is the same in both maps
floor = np.median(to_db(maps["after static clutter removal: only what moves"]))
for ax, (title, p) in zip(axes, maps.items()):
    im = ax.imshow((to_db(p) - floor).T, origin="lower", aspect="auto", cmap=style.heat(), vmin=-3, vmax=60,
                   extent=[-cfg.max_velocity, cfg.max_velocity, 0, cfg.max_range])
    ax.set_title(title, loc="left")
    ax.set_xlabel("radial velocity (m/s)")
axes[0].set_ylabel("range (m)")
cb = fig.colorbar(im, cax=fig.add_axes([0.945, 0.12, 0.012, 0.72]))
cb.set_label("dB above noise")
fig.suptitle(f"Range-Doppler map, {cfg.name}: 2D FFT of {cfg.n_chirps} chirps × {cfg.n_samples} samples × "
             f"{cfg.n_virtual} channels", x=0.05, ha="left", size=13, weight="bold", color=style.TEXT)
out = Path(__file__).resolve().parents[1] / "docs" / "media" / "range-doppler.png"
fig.savefig(out)
print(f"wrote {out}")
