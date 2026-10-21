"""Two targets 8 degrees apart, seen by an 8-element virtual array (Rayleigh limit about 14 degrees).

The angle FFT (Bartlett) sees one blob. MUSIC with forward-backward spatial
smoothing splits them, from a single snapshot.

    python examples/04_angle_music.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mira import FMCWSimulator, RadarConfig
from mira.processing import doppler_fft, range_doppler_power, range_fft, remove_static_clutter, tdm_phase_correction
from mira.processing.angle import bartlett_spectrum, music_spectrum
from mira.scenarios import two_targets
from mira.viz import style

style.apply()
cfg = RadarConfig()
scene = two_targets(separation_deg=8.0)
rd = doppler_fft(remove_static_clutter(range_fft(FMCWSimulator(cfg, seed=2).frame(scene))))
d, r = np.unravel_index(np.argmax(range_doppler_power(rd)), (cfg.n_chirps, cfg.n_samples))
x = rd[d, :, r] * tdm_phase_correction(cfg, np.array([d]))[0]

ang = np.radians(np.linspace(-40, 40, 1601))
b = bartlett_spectrum(x, cfg.virtual_positions(), ang)
m = music_spectrum(x, cfg.element_spacing, ang, n_sources=2)

fig, ax = plt.subplots(figsize=(11, 4.6), dpi=110)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.86)
ax.plot(np.degrees(ang), 10 * np.log10(b / b.max()), color=style.ACCENT_2, lw=1.6, label="angle FFT (Bartlett)")
ax.plot(np.degrees(ang), 10 * np.log10(m / m.max()), color=style.ACCENT, lw=1.6, label="MUSIC + spatial smoothing")
for a in (-4, 4):
    ax.axvline(a, color=style.DIM, lw=0.8, ls=":")
ax.text(0, -16, "true\nangles", ha="center", size=8, color=style.DIM)
ax.set_ylim(-30, 2)
ax.set_xlabel("azimuth (deg)")
ax.set_ylabel("dB")
ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=style.TEXT)
ax.set_title(f"Super-resolution: two targets at ±4°, {cfg.n_virtual}-element virtual array "
             f"(Rayleigh limit {cfg.angle_resolution_deg:.0f}°)", loc="left")
out = Path(__file__).resolve().parents[1] / "docs" / "media" / "music-vs-fft.png"
fig.savefig(out)
print(f"wrote {out}")
