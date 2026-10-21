"""Micro-Doppler signatures: a pedestrian, a cyclist and a drone, frame by frame.

Each column is the Doppler spectrum at the object's range bin in one frame.
The torso draws a steady line, the swinging legs and arms wrap around it at
the gait rhythm, the bike wheels add a fainter fan, and the drone's rotor
blades smear energy across every velocity.

    python examples/05_micro_doppler.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mira import Cyclist, Drone, FMCWSimulator, Pedestrian, Scene, get_preset
from mira.processing import doppler_fft, range_fft, remove_static_clutter
from mira.viz import style

style.apply()
cfg = get_preset("MIRA-60")
seconds = 4.0
objects = {  # title: (object, range bins either side of its centre to integrate)
    "pedestrian walking towards the radar": (Pedestrian(position=(0.5, 9.0), velocity=(0.0, -1.4)), 5),
    "cyclist riding away": (Cyclist(position=(-0.5, 2.0), velocity=(0.0, 2.5)), 10),
    "quadcopter drifting sideways": (Drone(position=(0.0, 4.0), velocity=(0.5, 0.3), rotor_hz=80.0), 3),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=110, sharey=True)
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.13, top=0.82, wspace=0.05)
for ax, (title, (obj, half)) in zip(axes, objects.items()):
    scene = Scene([obj], fluctuation=False)
    sim = FMCWSimulator(cfg, seed=1, leakage_db=None)
    cols = []
    for _ in range(int(seconds / cfg.frame_period)):
        rd = doppler_fft(remove_static_clutter(range_fft(sim.frame(scene))))
        rbin = int(round(np.hypot(*obj.position) / cfg.range_axis()[1]))
        band = np.abs(rd[:, :, max(rbin - half, 0):rbin + half + 1]) ** 2
        cols.append(band.sum(axis=(1, 2)))
        scene.step(cfg.frame_period)
    spec = 10 * np.log10(np.array(cols).T + 1e-30)
    spec -= spec.max()
    floor = max(-40.0, float(np.median(spec)) + 3)  # keep the noise black
    ax.imshow(spec, origin="lower", aspect="auto", cmap=style.heat(), vmin=floor, vmax=0,
              extent=[0, seconds, -cfg.max_velocity, cfg.max_velocity])
    ax.set_title(title, loc="left")
    ax.set_xlabel("time (s)")
axes[0].set_ylabel("radial velocity (m/s)")
fig.suptitle(f"Micro-Doppler, {cfg.name} ({cfg.center_freq / 1e9:.0f} GHz): what moves inside the object",
             x=0.05, ha="left", size=13, weight="bold", color=style.TEXT)
out = Path(__file__).resolve().parents[1] / "docs" / "media" / "micro-doppler.png"
fig.savefig(out)
print(f"wrote {out}")
