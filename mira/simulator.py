"""Raw ADC data from a TDM-MIMO FMCW radar.

After the mixer, every point scatterer contributes one complex tone per chirp:

    s[l, p, i] = a * exp(j * (2*pi*f_b*t_i + 2*pi*f_d*tau_{l,m} - 2*pi*x_p*sin(theta) + phi0))

with
    f_b      = 2*S*R/c + f_d      beat frequency (range, plus a little Doppler)
    f_d      = 2*v_r/lambda       Doppler frequency, v_r > 0 moving away
    tau_{l,m}                      start time of loop l on TX m (TDM-MIMO)
    x_p                            virtual element position in wavelengths
    a                              from the radar equation and the antenna pattern

The three phase terms only depend on fast time, slow time and channel
respectively, so a whole frame is one matrix product per frame instead of a
loop over scatterers, chirps and samples.
"""

from __future__ import annotations

import numpy as np

from .config import C, RadarConfig
from .scene import Scatterers, Scene


def antenna_gain(cfg: RadarConfig, azimuth: np.ndarray) -> np.ndarray:
    """Linear one-way power gain: Gaussian main beam with the configured -3 dB width."""
    g0 = 10 ** (cfg.antenna_gain_dbi / 10)
    half = np.radians(cfg.beamwidth_deg) / 2
    pattern = 0.5 ** ((azimuth / half) ** 2)
    return g0 * np.maximum(pattern, 1e-3)  # -30 dB floor stands in for sidelobes


def received_amplitude(cfg: RadarConfig, rng: np.ndarray, azimuth: np.ndarray, rcs: np.ndarray) -> np.ndarray:
    """Amplitude (sqrt of watts) of each echo at the ADC, from the radar equation."""
    pt = 1e-3 * 10 ** (cfg.tx_power_dbm / 10)
    g = antenna_gain(cfg, azimuth)
    lam = cfg.wavelength
    pr = pt * g * g * lam**2 * rcs / ((4 * np.pi) ** 3 * np.maximum(rng, 0.1) ** 4)
    return np.sqrt(pr)


class FMCWSimulator:
    """Turns a :class:`Scene` into radar data cubes of shape (chirps, virtual channels, samples)."""

    def __init__(
        self,
        cfg: RadarConfig,
        seed: int | None = 0,
        noise: bool = True,
        leakage_db: float | None = -25.0,
        adc_bits: int | None = None,
    ):
        """
        leakage_db: power of the TX to RX coupling relative to 1 mW, shows up as a
            strong static return at the first range bins like on real hardware.
            None to disable.
        adc_bits: quantise I and Q to this many bits (full scale set from the noise
            floor and leakage), None keeps floats.
        """
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.leakage_db = leakage_db
        self.adc_bits = adc_bits

        self._t_fast = cfg.adc_start + np.arange(cfg.n_samples) / cfg.sample_rate  # (Ns,)
        loops = np.arange(cfg.n_chirps)[:, None] * cfg.loop_period
        tx_offset = cfg.virtual_tx_index()[None, :] * cfg.chirp_period
        self._t_slow = loops + tx_offset  # (Nl, P) start of each chirp
        self._x = cfg.virtual_positions()  # (P,)

    def scatterer_geometry(self, sc: Scatterers) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Range, azimuth and radial velocity of each scatterer."""
        x, y = sc.position[:, 0], sc.position[:, 1]
        rng = np.hypot(x, y)
        az = np.arctan2(x, y)
        vr = np.einsum("ij,ij->i", sc.position, sc.velocity) / np.maximum(rng, 1e-6)
        return rng, az, vr

    def frame_from_scatterers(self, sc: Scatterers, t0: float = 0.0) -> np.ndarray:
        cfg = self.cfg
        Nl, P, Ns = cfg.n_chirps, cfg.n_virtual, cfg.n_samples
        cube = np.zeros((Nl * P, Ns), np.complex128)

        if len(sc):
            rng, az, vr = self.scatterer_geometry(sc)
            # only what is in front of the radar and inside the unambiguous range
            keep = (np.abs(az) < np.pi / 2) & (rng < cfg.max_range * 0.98) & (rng > 0.05)
            rng, az, vr = rng[keep], az[keep], vr[keep]
            amp = received_amplitude(cfg, rng, az, sc.rcs[keep])
            vib_a, vib_f, vib_p = sc.vib_amp[keep], sc.vib_freq[keep], sc.vib_phase[keep]

            lam = cfg.wavelength
            fd = 2 * vr / lam
            fb = 2 * cfg.slope * rng / C + fd
            tau = 2 * rng / C
            # carrier phase at the start of the ramp, minus the residual video phase
            phi0 = 2 * np.pi * cfg.start_freq * tau - np.pi * cfg.slope * tau**2

            fast = np.exp(2j * np.pi * fb[:, None] * self._t_fast[None, :])  # (K, Ns)

            t_slow = t0 + self._t_slow  # (Nl, P)
            slow_phase = 2 * np.pi * fd[:, None, None] * self._t_slow[None]  # (K, Nl, P)
            if np.any(vib_a):
                slow_phase = slow_phase + (4 * np.pi / lam) * vib_a[:, None, None] * np.sin(
                    2 * np.pi * vib_f[:, None, None] * t_slow[None] + vib_p[:, None, None]
                )
            spatial = -2 * np.pi * self._x[None, :] * np.sin(az)[:, None]  # (K, P)
            sp = amp[:, None, None] * np.exp(1j * (slow_phase + spatial[:, None, :] + phi0[:, None, None]))
            cube += sp.reshape(len(amp), Nl * P).T @ fast

        cube = cube.reshape(Nl, P, Ns)

        if self.leakage_db is not None:
            # coupling at about 5 cm, identical on every chirp
            a = np.sqrt(1e-3 * 10 ** (self.leakage_db / 10))
            fb = 2 * cfg.slope * 0.05 / C
            cube += a * np.exp(2j * np.pi * fb * self._t_fast)[None, None, :] * np.exp(
                1j * np.linspace(0, 2, P)
            )[None, :, None]

        if self.noise:
            sigma = np.sqrt(cfg.noise_power / 2)
            cube += sigma * (self.rng.standard_normal(cube.shape) + 1j * self.rng.standard_normal(cube.shape))

        if self.adc_bits:
            cube = self._quantise(cube)

        return cube.astype(np.complex64)

    def frame(self, scene: Scene) -> np.ndarray:
        """One frame of raw data for the scene at its current time."""
        return self.frame_from_scatterers(scene.scatterers(), scene.t)

    def _quantise(self, cube: np.ndarray) -> np.ndarray:
        levels = 2 ** (self.adc_bits - 1)
        full_scale = 8 * np.sqrt(self.cfg.noise_power)
        if self.leakage_db is not None:
            full_scale = max(full_scale, 4 * np.sqrt(1e-3 * 10 ** (self.leakage_db / 10)))
        lsb = full_scale / levels
        q = lambda v: np.clip(np.round(v / lsb), -levels, levels - 1) * lsb  # noqa: E731
        return q(cube.real) + 1j * q(cube.imag)


def ideal_snr_db(cfg: RadarConfig, rng: float, rcs: float, azimuth: float = 0.0) -> float:
    """SNR after full coherent integration (range FFT x Doppler FFT x channels), no window loss."""
    amp = received_amplitude(cfg, np.array([rng]), np.array([azimuth]), np.array([rcs]))[0]
    gain = cfg.n_samples * cfg.n_chirps * cfg.n_virtual
    return float(10 * np.log10(amp**2 * gain / cfg.noise_power))
