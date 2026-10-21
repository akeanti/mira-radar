"""FMCW in one picture: the chirp, its echo, the beat tone, and the range FFT.

    python examples/01_fmcw_basics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mira import RadarConfig
from mira.config import C
from mira.viz import style

style.apply()
cfg = RadarConfig()
targets = [(8.0, 1.0), (21.0, 0.5), (33.0, 0.25)]  # range m, relative amplitude

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=110)
fig.subplots_adjust(left=0.05, right=0.985, bottom=0.16, top=0.83, wspace=0.28)

# 1. instantaneous frequency of TX and the delayed echoes
ax = axes[0]
t = np.linspace(0, cfg.ramp_time, 400)
ax.plot(t * 1e6, (cfg.start_freq + cfg.slope * t) / 1e9, color=style.ACCENT, lw=2, label="transmitted chirp")
stretch = 40  # real delays are 50-220 ns against a 32 us ramp: drawn 40x longer so you can see them
for (r, a), col in zip(targets, ["#ff9e4a", "#ff4d8d", "#a78bfa"]):
    tau = 2 * r / C * stretch
    tt = t[t > tau]
    ax.plot(tt * 1e6, (cfg.start_freq + cfg.slope * (tt - tau)) / 1e9, color=col, lw=1.2, ls="--",
            label=f"echo from {r:g} m")
ax.axvspan(cfg.adc_start * 1e6, (cfg.adc_start + cfg.n_samples / cfg.sample_rate) * 1e6, color=style.GRID, alpha=0.6,
           label="ADC samples")
ax.set_xlabel("time in chirp (µs)")
ax.set_ylabel("frequency (GHz)")
ax.set_title(f"1  ·  chirp and echoes (delays ×{stretch})", loc="left")
ax.legend(fontsize=7, frameon=False, labelcolor=style.TEXT)

# 2. beat signal after the mixer: one tone per target, frequency = 2 S R / c
ax = axes[1]
ts = np.arange(cfg.n_samples) / cfg.sample_rate
# complex (I/Q) sampling, like the simulator: the band is the full sample rate, not half of it
beat = sum(a * np.exp(2j * np.pi * 2 * cfg.slope * r / C * ts) for r, a in targets)
ax.plot(ts * 1e6, beat.real, color=style.ACCENT, lw=1)
ax.set_xlabel("time (µs)")
ax.set_ylabel("IF amplitude, I channel")
ax.set_ylim(-2.2, 3.6)
ax.set_title("2  ·  beat signal = TX × echo", loc="left")
f_text = "   ".join(f"{r:g} m → {2 * cfg.slope * r / C / 1e6:.1f} MHz" for r, _ in targets)
ax.text(0.5, 0.95, f_text, transform=ax.transAxes, ha="center", va="top", size=8, color=style.TEXT)

# 3. FFT of the beat signal: frequency axis is a range axis
ax = axes[2]
n_fft = 1024
spec = np.abs(np.fft.fft(beat * np.hanning(cfg.n_samples), n_fft))
f = np.arange(n_fft) * cfg.sample_rate / n_fft
rng = f * C / (2 * cfg.slope)
ax.plot(rng, 20 * np.log10(spec / spec.max() + 1e-6), color=style.ACCENT, lw=1.3)
for r, _ in targets:
    ax.axvline(r, color=style.ACCENT_2, lw=0.8, ls=":")
ax.set_xlim(0, 40)
ax.set_ylim(-60, 3)
ax.set_xlabel("range (m)  =  f × c / 2S")
ax.set_ylabel("dB")
ax.set_title(f"3  ·  range FFT (ΔR = c/2B = {cfg.range_resolution * 100:.1f} cm)", loc="left")

fig.suptitle("FMCW radar: a delay becomes a frequency", x=0.05, ha="left", size=13, weight="bold",
             color=style.TEXT)
out = Path(__file__).resolve().parents[1] / "docs" / "media" / "fmcw-basics.png"
fig.savefig(out)
print(f"wrote {out}")
