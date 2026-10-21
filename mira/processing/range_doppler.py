"""Range FFT, static clutter removal, Doppler FFT.

Data cube convention everywhere in MIRA: axis 0 is slow time (chirp loop) or
Doppler, axis 1 is the virtual channel, axis 2 is fast time (ADC sample) or
range.
"""

from __future__ import annotations

import numpy as np


def window(name: str | None, n: int) -> np.ndarray:
    if name in (None, "rect", "none"):
        return np.ones(n)
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    if name.startswith("kaiser"):
        beta = float(name[6:] or 6.0)
        return np.kaiser(n, beta)
    raise ValueError(f"unknown window {name!r}")


def range_fft(cube: np.ndarray, win: str | None = "hann", n_fft: int | None = None) -> np.ndarray:
    """FFT along fast time. Output bin k is range k * range_axis step."""
    w = window(win, cube.shape[-1])
    w = w / w.sum()  # a tone of amplitude A comes out with peak A
    return np.fft.fft(cube * w, n=n_fft, axis=-1)


def remove_static_clutter(rfft: np.ndarray) -> np.ndarray:
    """Subtract the mean over chirps in every range bin: a zero-Doppler notch (two-pulse MTI's big brother)."""
    return rfft - rfft.mean(axis=0, keepdims=True)


def doppler_fft(rfft: np.ndarray, win: str | None = "hann", n_fft: int | None = None) -> np.ndarray:
    """FFT along slow time, zero velocity moved to the centre bin."""
    w = window(win, rfft.shape[0])
    w = w / w.sum()
    return np.fft.fftshift(np.fft.fft(rfft * w[:, None, None], n=n_fft, axis=0), axes=0)


def range_doppler_power(rd_cube: np.ndarray) -> np.ndarray:
    """Non-coherent integration over channels: |X|^2 summed. Shape (doppler, range)."""
    return np.sum(np.abs(rd_cube) ** 2, axis=1)


def to_db(power: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return 10 * np.log10(np.maximum(power, floor))


def range_profile(rfft: np.ndarray) -> np.ndarray:
    """Average power per range bin, over chirps and channels."""
    return np.mean(np.abs(rfft) ** 2, axis=(0, 1))
