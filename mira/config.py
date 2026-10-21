"""Radar parameters and everything that follows from them.

A radar is mostly defined by a handful of chirp parameters. Range resolution,
maximum range, velocity resolution and so on are not free choices, they are
consequences of those parameters, so they are exposed here as derived
properties rather than separate settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

C = 299_792_458.0  # speed of light, m/s
K_BOLTZMANN = 1.380649e-23  # J/K


@dataclass(frozen=True)
class RadarConfig:
    """A TDM-MIMO FMCW radar, defaults close to a TI AWR1843 in a short-range mode.

    Coordinates: the radar sits at the origin and looks along +y. Azimuth is
    measured from boresight, positive towards +x. Radial velocity is positive
    for targets moving away from the radar.
    """

    name: str = "MIRA-77"
    description: str = "77 GHz urban / short-range automotive"

    # Chirp
    start_freq: float = 77e9  # Hz
    slope: float = 30e12  # Hz/s (30 MHz/us)
    idle_time: float = 7e-6  # s, between chirps
    adc_start: float = 4e-6  # s, from ramp start to first ADC sample
    ramp_time: float = 32e-6  # s
    sample_rate: float = 10e6  # complex samples per second
    n_samples: int = 256  # ADC samples per chirp
    n_chirps: int = 128  # chirp loops per frame (each loop fires every TX once)
    frame_period: float = 0.05  # s

    # Array: TX and RX elements on a line, RX spaced d, TX spaced n_rx * d,
    # which gives a filled virtual array of n_tx * n_rx elements spaced d.
    n_tx: int = 2
    n_rx: int = 4
    element_spacing: float = 0.5  # in wavelengths

    # Link budget
    tx_power_dbm: float = 12.0
    antenna_gain_dbi: float = 10.0  # per antenna, TX and RX
    beamwidth_deg: float = 120.0  # -3 dB, one way, azimuth
    noise_figure_db: float = 12.0
    temperature: float = 290.0  # K

    extras: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.adc_start + self.n_samples / self.sample_rate > self.ramp_time + 1e-12:
            raise ValueError(
                f"{self.name}: ADC window ({(self.adc_start + self.n_samples / self.sample_rate) * 1e6:.1f} us) "
                f"does not fit in the ramp ({self.ramp_time * 1e6:.1f} us)"
            )
        if self.n_tx < 1 or self.n_rx < 1:
            raise ValueError("need at least one TX and one RX")
        if self.frame_period < self.active_frame_time:
            raise ValueError(
                f"{self.name}: frame period {self.frame_period * 1e3:.1f} ms is shorter than the "
                f"chirps it contains ({self.active_frame_time * 1e3:.1f} ms)"
            )

    # ------------------------------------------------------------------ derived
    @property
    def center_freq(self) -> float:
        """Carrier frequency at the middle of the ADC window."""
        return self.start_freq + self.slope * (self.adc_start + 0.5 * self.n_samples / self.sample_rate)

    @property
    def wavelength(self) -> float:
        return C / self.center_freq

    @property
    def bandwidth(self) -> float:
        """Bandwidth actually swept while the ADC samples. This sets range resolution."""
        return self.slope * self.n_samples / self.sample_rate

    @property
    def range_resolution(self) -> float:
        return C / (2.0 * self.bandwidth)

    @property
    def max_range(self) -> float:
        """Largest range whose beat frequency fits in the complex IF band."""
        return self.sample_rate * C / (2.0 * self.slope)

    @property
    def chirp_period(self) -> float:
        return self.idle_time + self.ramp_time

    @property
    def loop_period(self) -> float:
        """Time between two chirps from the same TX."""
        return self.n_tx * self.chirp_period

    @property
    def max_velocity(self) -> float:
        """Unambiguous radial velocity, +/- this value."""
        return self.wavelength / (4.0 * self.loop_period)

    @property
    def velocity_resolution(self) -> float:
        return self.wavelength / (2.0 * self.n_chirps * self.loop_period)

    @property
    def n_virtual(self) -> int:
        return self.n_tx * self.n_rx

    @property
    def angle_resolution_deg(self) -> float:
        """Rayleigh resolution at boresight for the filled virtual array."""
        return float(np.degrees(2.0 / (self.n_virtual * 2.0 * self.element_spacing)))

    @property
    def active_frame_time(self) -> float:
        return self.n_chirps * self.loop_period

    @property
    def frame_rate(self) -> float:
        return 1.0 / self.frame_period

    @property
    def duty_cycle(self) -> float:
        return self.active_frame_time / self.frame_period

    @property
    def noise_power(self) -> float:
        """Thermal noise power per complex ADC sample, in watts."""
        return K_BOLTZMANN * self.temperature * 10 ** (self.noise_figure_db / 10) * self.sample_rate

    @property
    def raw_data_rate(self) -> float:
        """ADC output while chirping, bits/s, assuming 16-bit I and Q."""
        return self.n_rx * self.sample_rate * 32

    # ------------------------------------------------------------------- axes
    def range_axis(self, n_fft: int | None = None) -> np.ndarray:
        n = n_fft or self.n_samples
        return np.arange(n) * self.sample_rate / n * C / (2.0 * self.slope)

    def velocity_axis(self, n_fft: int | None = None) -> np.ndarray:
        n = n_fft or self.n_chirps
        return (np.arange(n) - n // 2) * self.wavelength / (2.0 * n * self.loop_period)

    def virtual_positions(self) -> np.ndarray:
        """Virtual element positions in wavelengths, in channel order (tx-major)."""
        tx = np.arange(self.n_tx) * self.n_rx * self.element_spacing
        rx = np.arange(self.n_rx) * self.element_spacing
        return (tx[:, None] + rx[None, :]).ravel()

    def virtual_tx_index(self) -> np.ndarray:
        """Which TX each virtual channel came from."""
        return np.repeat(np.arange(self.n_tx), self.n_rx)

    # ------------------------------------------------------------------ misc
    def with_(self, **changes) -> "RadarConfig":
        return replace(self, **changes)

    def specs(self) -> list[tuple[str, str]]:
        """Human readable spec sheet, every number computed from the parameters."""
        return [
            ("Carrier", f"{self.center_freq / 1e9:.2f} GHz"),
            ("Swept bandwidth", f"{self.bandwidth / 1e6:.0f} MHz"),
            ("Chirp slope", f"{self.slope / 1e12:.1f} MHz/us"),
            ("Range resolution", f"{self.range_resolution * 100:.1f} cm"),
            ("Max range", f"{self.max_range:.1f} m"),
            ("Velocity resolution", f"{self.velocity_resolution:.2f} m/s ({self.velocity_resolution * 3.6:.2f} km/h)"),
            ("Max velocity", f"+/-{self.max_velocity:.1f} m/s (+/-{self.max_velocity * 3.6:.0f} km/h)"),
            ("Antennas", f"{self.n_tx} TX x {self.n_rx} RX = {self.n_virtual} virtual"),
            ("Angular resolution", f"{self.angle_resolution_deg:.1f} deg at boresight"),
            ("Field of view", f"+/-{self.beamwidth_deg / 2:.0f} deg azimuth (-3 dB)"),
            ("Frame", f"{self.n_chirps} loops x {self.n_samples} samples, {self.frame_rate:.0f} fps"),
            ("Raw ADC rate", f"{self.raw_data_rate / 1e6:.0f} Mbit/s while chirping"),
        ]


PRESETS: dict[str, RadarConfig] = {
    "MIRA-77": RadarConfig(),
    "MIRA-77LR": RadarConfig(
        name="MIRA-77LR",
        description="77 GHz long range, narrow beam",
        slope=8e12,
        sample_rate=15e6,
        n_samples=256,
        adc_start=0.9e-6,
        ramp_time=18e-6,
        idle_time=4e-6,
        n_chirps=128,
        n_tx=2,
        n_rx=4,
        tx_power_dbm=13.0,
        antenna_gain_dbi=16.0,
        beamwidth_deg=40.0,
        frame_period=0.05,
    ),
    "MIRA-60": RadarConfig(
        name="MIRA-60",
        description="60 GHz indoor / people sensing",
        start_freq=60e9,
        slope=60e12,
        sample_rate=6e6,
        n_samples=192,
        adc_start=3e-6,
        ramp_time=36e-6,
        idle_time=10e-6,
        n_chirps=96,
        n_tx=3,
        n_rx=4,
        tx_power_dbm=10.0,
        antenna_gain_dbi=6.0,
        beamwidth_deg=140.0,
        frame_period=0.05,
    ),
}


def get_preset(name: str) -> RadarConfig:
    try:
        return PRESETS[name.upper()]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}, choose from {', '.join(PRESETS)}") from None
