"""FMCW sonar with a laptop speaker and microphone.

The speaker plays a linear chirp over and over, back to back. Because the
transmitted signal is periodic, any window of exactly one period of the
microphone signal contains every echo, circularly shifted by its delay. A
circular cross-correlation of that window with the chirp (one FFT, one
multiply, one inverse FFT) therefore gives the whole range profile, without
ever knowing where the period starts.

The strongest early peak is the direct path from speaker to microphone. Its
position absorbs all the unknown audio latency, so every range is measured
relative to it. Subtracting the previous complex profile (a two-pulse
canceller, the oldest trick in radar) removes everything that stands still,
the desk, the screen, the walls, and leaves what moves: your hand.

A moving hand Doppler-shifts its echo, and with a linear chirp a frequency
shift is indistinguishable from a delay: the hand reads closer than it is
when it approaches and further when it recedes (range-Doppler coupling,
r_measured = r + v * f_c * T / B, about 6 cm at 0.5 m/s here). A small
Kalman filter on the measured range gives the hand's velocity, and the
coupling is subtracted, the same correction radar trackers apply.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SonarConfig:
    sample_rate: int = 48_000
    f0: float = 18_000.0  # Hz, most adults barely hear this, laptop speakers still play it
    f1: float = 21_500.0
    period: int = 1024  # samples per chirp, 21.3 ms at 48 kHz
    taper: float = 0.1  # Tukey taper fraction: fades through the frequency jump so it does not click
    amplitude: float = 0.4
    speed_of_sound: float = 343.0  # m/s at 20 C
    max_range: float = 1.5  # m shown and searched
    min_range: float = 0.06  # m, ignore the direct path region
    mti_lag: int = 1  # profiles between the two that are subtracted

    def __post_init__(self) -> None:
        if not 0 < self.f0 < self.f1 <= self.sample_rate / 2:
            raise ValueError("need 0 < f0 < f1 <= sample_rate / 2")
        if self.max_range >= self.max_unambiguous_range:
            raise ValueError("max_range must be below period * c / (2 * sample_rate)")

    @property
    def bandwidth(self) -> float:
        return self.f1 - self.f0

    @property
    def center_freq(self) -> float:
        return 0.5 * (self.f0 + self.f1)

    @property
    def duration(self) -> float:
        return self.period / self.sample_rate

    @property
    def coupling(self) -> float:
        """Seconds: range error per m/s of radial velocity, f_c * T / B."""
        return self.center_freq * self.duration / self.bandwidth

    @property
    def effective_coupling(self) -> float:
        """Coupling seen after the canceller, whose output sits half the lag behind the newest profile."""
        return self.coupling - 0.5 * self.mti_lag * self.duration

    @property
    def range_resolution(self) -> float:
        return self.speed_of_sound / (2 * self.bandwidth)

    @property
    def bin_size(self) -> float:
        """Range covered by one sample of delay."""
        return self.speed_of_sound / (2 * self.sample_rate)

    @property
    def n_bins(self) -> int:
        return int(np.ceil(self.max_range / self.bin_size))

    @property
    def profile_rate(self) -> float:
        return self.sample_rate / self.period

    @property
    def max_unambiguous_range(self) -> float:
        return self.period * self.bin_size

    def range_axis(self) -> np.ndarray:
        return np.arange(self.n_bins) * self.bin_size


def chirp(cfg: SonarConfig, n: np.ndarray | None = None) -> np.ndarray:
    """The transmitted waveform at (possibly fractional) sample times n. Periodic in cfg.period."""
    N = cfg.period
    if n is None:
        n = np.arange(N)
    t = np.mod(n, N) / cfg.sample_rate
    k = cfg.bandwidth / cfg.duration
    phase = 2 * np.pi * (cfg.f0 * t + 0.5 * k * t * t)
    u = np.mod(n, N) / N
    a = cfg.taper
    w = np.ones_like(u)
    if a > 0:
        lo = u < a / 2
        hi = u > 1 - a / 2
        w[lo] = 0.5 * (1 - np.cos(2 * np.pi * u[lo] / a))
        w[hi] = 0.5 * (1 - np.cos(2 * np.pi * (1 - u[hi]) / a))
    return cfg.amplitude * w * np.cos(phase)


@dataclass
class SonarFrame:
    profile: np.ndarray  # |echo| per range bin, static scene included
    moving: np.ndarray  # |echo - echo mti_lag profiles ago| per range bin
    distance: float | None  # m, nearest mover after coupling correction, None if nothing moves
    velocity: float | None  # m/s of that mover, positive moving away
    direct_index: int


class _RangeFilter:
    """Constant-velocity Kalman filter on one range."""

    def __init__(self, dt: float, accel: float = 4.0, sigma: float = 0.012):
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.Q = accel**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        self.R = sigma**2
        self.x: np.ndarray | None = None
        self.P = np.eye(2)

    def step(self, z: float | None) -> np.ndarray | None:
        if self.x is None:
            if z is not None:
                self.x = np.array([z, 0.0])
                self.P = np.diag([self.R, 1.0])
            return self.x
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if z is not None:
            S = self.P[0, 0] + self.R
            K = self.P[:, 0] / S
            self.x = self.x + K * (z - self.x[0])
            self.P = self.P - np.outer(K, self.P[0])
        return self.x


class SonarProcessor:
    def __init__(self, cfg: SonarConfig, track_window: int = 3, detect_snr_db: float = 12.0,
                 coast: float = 0.3):
        """coast: seconds without a detection before the hand is forgotten."""
        self.cfg = cfg
        N = cfg.period
        ref = chirp(cfg)
        f = np.fft.fftfreq(N, 1 / cfg.sample_rate)
        # Matched filter on positive frequencies only, for an analytic (complex) range profile,
        # with a Hann taper across the band: range sidelobes drop from -13 dB to about -31 dB.
        # Everything outside the band (voices, fans, music) is thrown away.
        u = (f - cfg.f0) / cfg.bandwidth
        weight = np.where((u >= 0) & (u <= 1), 0.5 - 0.5 * np.cos(2 * np.pi * np.clip(u, 0, 1)), 0.0)
        self.H = 2 * weight * np.conj(np.fft.fft(ref)) / np.sum(ref * ref)
        self.track_window = track_window
        self.detect_snr = 10 ** (detect_snr_db / 10)
        self.max_coast = int(round(coast * cfg.profile_rate))
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0)
        self.direct: int | None = None
        self._direct_f = 0.0
        self._history: deque = deque(maxlen=self.cfg.mti_lag)
        self._filter = _RangeFilter(1 / self.cfg.profile_rate)
        self._missed = 0

    def profile(self, frame: np.ndarray) -> np.ndarray:
        """Complex circular matched-filter output for one period of microphone samples."""
        return np.fft.ifft(np.fft.fft(frame) * self.H)

    def _align(self, y: np.ndarray) -> np.ndarray:
        mag = np.abs(y)
        N = len(y)
        if self.direct is None:
            self.direct = int(np.argmax(mag))
            self._direct_f = float(self.direct)
        else:
            # follow the direct path slowly, in case the clocks drift (separate mic and speaker
            # devices); a hand right next to the laptop must not drag it along
            w = self.track_window
            idx = self.direct + np.arange(-w, w + 1)
            k = int(idx[np.argmax(mag[idx % N])])
            self._direct_f += 0.02 * (k - self._direct_f)
            self.direct = int(round(self._direct_f)) % N
        return np.roll(y, -self.direct)[: self.cfg.n_bins]

    def process_frame(self, frame: np.ndarray) -> SonarFrame:
        cfg = self.cfg
        y = self._align(self.profile(frame))
        m = np.abs(y - self._history[0]) if len(self._history) == cfg.mti_lag else np.zeros(len(y))
        self._history.append(y)

        lo = int(np.ceil(cfg.min_range / cfg.bin_size))
        noise = np.median(m[lo:]) + 1e-12
        k = lo + int(np.argmax(m[lo:]))
        z = None
        if m[k] ** 2 > self.detect_snr * noise**2:
            off = 0.0
            if 0 < k < len(m) - 1:
                a, b, c = m[k - 1], m[k], m[k + 1]
                den = a - 2 * b + c
                off = 0.5 * (a - c) / den if den < 0 else 0.0
            z = (k + off) * cfg.bin_size
            self._missed = 0
        else:
            self._missed += 1
            if self._missed > self.max_coast:
                self._filter.x = None

        state = self._filter.step(z)
        dist = vel = None
        if z is not None and state is not None:
            vel = float(state[1])
            dist = float(state[0] - vel * cfg.effective_coupling)
        return SonarFrame(np.abs(y), m, dist, vel, self.direct)

    def push(self, samples: np.ndarray) -> list[SonarFrame]:
        """Feed microphone samples of any block size; returns one SonarFrame per completed period."""
        self._buf = np.concatenate([self._buf, np.asarray(samples, float).ravel()])
        out = []
        N = self.cfg.period
        while len(self._buf) >= N:
            out.append(self.process_frame(self._buf[:N]))
            self._buf = self._buf[N:]
        return out
