"""Direction of arrival from the virtual array.

Each range-Doppler detection comes with one complex sample per virtual
channel. A target at azimuth theta puts a linear phase ramp across them,
exactly like a tone across the samples of an FFT, so finding the angle is
spectral estimation again. Two estimators:

* Bartlett (delay-and-sum), the angle FFT that every radar SDK ships.
* MUSIC with forward-backward spatial smoothing, which can split two targets
  closer than the array's Rayleigh limit from a single snapshot.
"""

from __future__ import annotations

import numpy as np

from ..config import RadarConfig


def steering_matrix(positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """(n_angles, n_elements) steering vectors; positions in wavelengths, angles in radians."""
    return np.exp(-2j * np.pi * np.sin(angles)[:, None] * positions[None, :])


def tdm_phase_correction(cfg: RadarConfig, doppler_bins: np.ndarray, n_doppler: int | None = None) -> np.ndarray:
    """Phase to multiply each virtual channel by to undo TDM-MIMO motion.

    TX m fires m chirp periods after TX 0, so a moving target has rotated its
    phase by 2*pi*f_d*m*T_c by then. Without this the angle estimate of every
    moving target is biased. doppler_bins are fftshifted bin indices.
    Returns (len(doppler_bins), n_virtual).
    """
    n = n_doppler or cfg.n_chirps
    k = np.asarray(doppler_bins) - n // 2
    m = cfg.virtual_tx_index()
    return np.exp(-2j * np.pi * np.outer(k, m) / (n * cfg.n_tx))


def bartlett_spectrum(x: np.ndarray, positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Delay-and-sum power for channel vector(s) x of shape (..., n_elements)."""
    a = steering_matrix(positions, angles)
    return np.abs(x @ a.conj().T) ** 2 / len(positions) ** 2


def _smoothed_covariance(x: np.ndarray, sub: int) -> np.ndarray:
    """Forward-backward spatially smoothed covariance from one snapshot of a uniform array."""
    n = len(x)
    J = np.eye(sub)[::-1]
    R = np.zeros((sub, sub), complex)
    for i in range(n - sub + 1):
        s = x[i:i + sub, None]
        R += s @ s.conj().T
    R /= n - sub + 1
    return 0.5 * (R + J @ R.conj() @ J)


def music_spectrum(
    x: np.ndarray,
    spacing: float,
    angles: np.ndarray,
    n_sources: int | None = None,
    sub: int | None = None,
) -> np.ndarray:
    """MUSIC pseudo-spectrum for a uniform linear array (spacing in wavelengths).

    n_sources: number of signals, estimated from the eigenvalues when None.
    sub: subarray length for spatial smoothing, defaults to 3/4 of the array.
    """
    n = len(x)
    sub = sub or max(2, int(round(0.75 * n)))
    R = _smoothed_covariance(np.asarray(x), sub)
    w, V = np.linalg.eigh(R)  # ascending
    w = w[::-1]
    V = V[:, ::-1]
    if n_sources is None:
        # signal eigenvalues stand well above the noise ones; at most two sources
        # are expected, so the rest of the spectrum gives the noise level
        noise = np.mean(w[2:]) if sub > 3 else w[-1]
        n_sources = int(np.sum(w > noise * 10 ** (13 / 10)))
        n_sources = min(max(n_sources, 1), sub - 1)
    En = V[:, n_sources:]
    a = steering_matrix(np.arange(sub) * spacing, angles)
    denom = np.sum(np.abs(a.conj() @ En) ** 2, axis=1)
    return 1.0 / np.maximum(denom, 1e-12)


def find_peaks(spectrum: np.ndarray, angles: np.ndarray, max_peaks: int = 2, rel_db: float = 6.0) -> list[tuple[float, float]]:
    """Up to max_peaks local maxima within rel_db of the strongest, with parabolic refinement.

    A maximum sitting on the edge of the search grid is not a real peak (the
    target is outside the field of view, or near endfire where +90 and -90
    degrees alias into each other) and is dropped.
    Returns [(angle_rad, power), ...] strongest first, possibly empty.
    """
    s = spectrum
    interior = (s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])
    idx = np.nonzero(interior)[0] + 1
    if not len(idx):
        return []
    idx = idx[np.argsort(s[idx])[::-1]]
    if max(s[0], s[-1]) > s[idx[0]]:
        return []
    top = s[idx[0]]
    out = []
    step = angles[1] - angles[0]
    for i in idx[:max_peaks]:
        if s[i] < top * 10 ** (-rel_db / 10):
            break
        theta = angles[i]
        if 0 < i < len(s) - 1:
            y0, y1, y2 = np.log(np.maximum(s[i - 1:i + 2], 1e-300))
            den = y0 - 2 * y1 + y2
            if den < 0:
                theta = angles[i] + 0.5 * (y0 - y2) / den * step
        out.append((float(theta), float(s[i])))
    return out


class AngleEstimator:
    """Angle of arrival for detections, with TDM-MIMO compensation built in."""

    def __init__(self, cfg: RadarConfig, method: str = "bartlett", fov_deg: float = 70.0, step_deg: float = 0.5,
                 max_targets_per_cell: int = 2):
        self.cfg = cfg
        self.method = method
        self.fov_deg = fov_deg
        self.angles = np.radians(np.arange(-fov_deg, fov_deg + step_deg / 2, step_deg))
        self.positions = cfg.virtual_positions()
        self.max_targets = max_targets_per_cell
        self.uniform = np.allclose(np.diff(self.positions), cfg.element_spacing)

    def spectrum(self, x: np.ndarray) -> np.ndarray:
        if self.method == "music" and self.uniform:
            return music_spectrum(x, self.cfg.element_spacing, self.angles)
        return bartlett_spectrum(x, self.positions, self.angles)

    def estimate(self, x: np.ndarray, doppler_bin: int, n_doppler: int | None = None) -> list[tuple[float, float]]:
        """Angles (rad) and relative power of the target(s) in one range-Doppler cell."""
        x = x * tdm_phase_correction(self.cfg, np.array([doppler_bin]), n_doppler)[0]
        return find_peaks(self.spectrum(x), self.angles, self.max_targets)

    def range_azimuth(self, rd_cube: np.ndarray, angles: np.ndarray | None = None) -> np.ndarray:
        """Power per (angle, range), integrated over Doppler. rd_cube is (doppler, channel, range).

        sum_d |a^H x_d|^2 = a^H (sum_d x_d x_d^H) a, so one P x P covariance per
        range bin replaces beamforming every Doppler bin separately.
        """
        angles = self.angles if angles is None else angles
        corr = tdm_phase_correction(self.cfg, np.arange(rd_cube.shape[0]), rd_cube.shape[0])
        x = (rd_cube * corr[:, :, None]).astype(np.complex64).transpose(2, 1, 0)  # (R, P, D)
        cov = x @ x.conj().transpose(0, 2, 1)  # (R, P, P)
        a = steering_matrix(self.positions, angles).astype(np.complex64)  # (A, P)
        power = np.einsum("ap,rpq,aq->ar", a.conj(), cov, a, optimize=True).real
        return np.maximum(power, 0) / len(self.positions) ** 2
