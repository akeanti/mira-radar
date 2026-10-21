"""A pretend laptop in a pretend room, for testing the sonar without hardware.

The microphone hears the chirp through the direct path, off a few static
surfaces and off a moving hand, all with fractional delays, plus noise and an
arbitrary audio latency.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .processor import SonarConfig, chirp


def waving_hand(t: float) -> float:
    """Hand moving towards and away from the screen, 15 to 60 cm, with a pause."""
    phase = (t % 6.0) / 6.0
    if phase < 0.4:
        return 0.15 + 0.45 * 0.5 * (1 - np.cos(2 * np.pi * phase / 0.4))
    if phase < 0.6:
        return 0.15
    return 0.15 + 0.2 * np.sin(2 * np.pi * (phase - 0.6) / 0.4 * 2) ** 2


class SimulatedRoom:
    def __init__(
        self,
        cfg: SonarConfig,
        hand: Callable[[float], float] | None = waving_hand,
        hand_gain: float = 0.05,
        static: tuple[tuple[float, float], ...] = ((0.45, 0.08), (0.95, 0.06), (1.35, 0.04)),
        direct_gain: float = 1.0,
        speaker_mic_distance: float = 0.12,
        latency: int = 1777,
        noise: float = 2e-3,
        seed: int = 0,
    ):
        """static: (range m, gain) of surfaces that do not move; latency in samples."""
        self.cfg = cfg
        self.hand = hand
        self.hand_gain = hand_gain
        self.static = static
        self.direct_gain = direct_gain
        self.d_sm = speaker_mic_distance
        self.latency = latency
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.n = 0  # samples generated so far

    def _delay(self, rng: float) -> float:
        """Samples of delay for a reflector at range rng (speaker and mic both at the laptop)."""
        return (2 * rng - self.d_sm) / self.cfg.speed_of_sound * self.cfg.sample_rate if rng > 0 else 0.0

    def read(self, frames: int) -> np.ndarray:
        """Next block of microphone samples."""
        cfg = self.cfg
        n = self.n + np.arange(frames) - self.latency
        fs = cfg.sample_rate
        d0 = self.d_sm / cfg.speed_of_sound * fs
        y = self.direct_gain * chirp(cfg, n - d0)
        for r, g in self.static:
            y += g * chirp(cfg, n - d0 - self._delay(r))
        if self.hand is not None:
            t = (self.n + np.arange(frames)) / fs
            r = np.array([self.hand(tt) for tt in t[:: 64]])
            r = np.interp(np.arange(frames), np.arange(0, frames, 64)[: len(r)], r)
            gain = self.hand_gain * (0.3 / np.maximum(r, 0.1)) ** 2
            y += gain * chirp(cfg, n - d0 - (2 * r - self.d_sm) / cfg.speed_of_sound * fs)
        y += self.noise * self.rng.standard_normal(frames)
        self.n += frames
        return y

    def hand_range(self, t: float) -> float | None:
        return None if self.hand is None else self.hand(t)
