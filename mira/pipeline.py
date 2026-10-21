"""The full receive chain, one frame at a time.

    ADC cube -> range FFT -> clutter removal -> Doppler FFT -> CFAR
             -> angle per detection -> point cloud -> DBSCAN -> EKF tracker
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import RadarConfig
from .processing.angle import AngleEstimator
from .processing.cfar import cfar_2d
from .processing.clustering import dbscan
from .processing.range_doppler import doppler_fft, range_doppler_power, range_fft, remove_static_clutter
from .processing.tracking import PROFILES, Track, Tracker

POINT_DTYPE = np.dtype([
    ("range", "f4"), ("azimuth", "f4"), ("velocity", "f4"), ("x", "f4"), ("y", "f4"),
    ("snr", "f4"), ("doppler_bin", "i4"), ("range_bin", "i4"), ("cluster", "i4"),
])


@dataclass
class PipelineSettings:
    range_window: str = "hann"
    doppler_window: str = "hann"
    clutter_removal: bool = True
    cfar_method: str = "ca"
    cfar_guard: tuple[int, int] = (2, 2)
    cfar_train: tuple[int, int] = (6, 8)
    pfa: float = 1e-7
    min_range: float = 0.6  # m, skip the leakage region
    angle_method: str = "bartlett"
    max_targets_per_cell: int = 2
    cluster_eps: float = 2.0  # m in (x, y, v / velocity_weight) space, for display
    cluster_min_points: int = 1
    velocity_weight: float = 2.0  # m/s that count as one metre for clustering
    track: bool = True
    tracker_profile: str = "vehicles"  # see processing.tracking.PROFILES
    range_azimuth_map: bool = True


@dataclass
class FrameResult:
    t: float
    rd_power: np.ndarray  # (doppler, range) linear
    rd_threshold: np.ndarray
    points: np.ndarray  # POINT_DTYPE
    clusters: list[dict]
    tracks: list[Track]
    ra_power: np.ndarray | None  # (angle, range) linear
    timings: dict = field(default_factory=dict)


class RadarPipeline:
    def __init__(self, cfg: RadarConfig, settings: PipelineSettings | None = None):
        self.cfg = cfg
        self.s = settings or PipelineSettings()
        self.ranges = cfg.range_axis()
        self.velocities = cfg.velocity_axis()
        self.angle = AngleEstimator(cfg, self.s.angle_method, max_targets_per_cell=self.s.max_targets_per_cell)
        self.ra_angles = np.radians(np.arange(-75, 75.1, 1.5))
        self.tracker = self._make_tracker() if self.s.track else None
        self.min_range_bin = int(np.ceil(self.s.min_range / self.ranges[1]))

    def _make_tracker(self) -> Tracker:
        return Tracker(dt=self.cfg.frame_period, max_range=self.cfg.max_range,
                       max_azimuth_deg=self.angle.fov_deg, **PROFILES[self.s.tracker_profile])

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker = self._make_tracker()

    def _interp(self, power: np.ndarray, d: int, r: int) -> tuple[float, float]:
        """Sub-bin peak position along Doppler and range by parabolic fit on log power."""
        nd, nr = power.shape

        def off(a, b, c):
            den = a - 2 * b + c
            return 0.0 if den >= 0 else 0.5 * (a - c) / den

        lp = np.log(np.maximum(power, 1e-300))
        dd = off(lp[(d - 1) % nd, r], lp[d, r], lp[(d + 1) % nd, r])
        dr = off(lp[d, r - 1], lp[d, r], lp[d, r + 1]) if 0 < r < nr - 1 else 0.0
        return dd, dr

    def process(self, cube: np.ndarray, t: float = 0.0) -> FrameResult:
        cfg, s = self.cfg, self.s
        tm = {}
        t0 = time.perf_counter()
        rfft = range_fft(cube, s.range_window)
        if s.clutter_removal:
            rfft = remove_static_clutter(rfft)
        rd = doppler_fft(rfft, s.doppler_window)
        power = range_doppler_power(rd)
        tm["fft"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        res = cfar_2d(power, s.cfar_guard, s.cfar_train, s.pfa, s.cfar_method,
                      n_integrated=cfg.n_virtual, min_index1=self.min_range_bin)
        tm["cfar"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        dr = self.ranges[1]
        dv = self.velocities[1] - self.velocities[0]
        rows = []
        for (d, r), snr in zip(res.detections, res.snr_db):
            dd, drr = self._interp(power, d, r)
            rng = (r + drr) * dr
            vel = self.velocities[d] + dd * dv
            for az, _ in self.angle.estimate(rd[d, :, r], d):
                rows.append((rng, az, vel, rng * np.sin(az), rng * np.cos(az), snr, d, r, -1))
        points = np.array(rows, dtype=POINT_DTYPE)
        tm["angle"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        clusters: list[dict] = []
        if len(points):
            feats = np.column_stack([points["x"], points["y"], points["velocity"] / s.velocity_weight])
            labels = dbscan(feats, s.cluster_eps, s.cluster_min_points)
            points["cluster"] = labels
            for c in range(labels.max() + 1):
                p = points[labels == c]
                w = 10 ** (p["snr"] / 20)
                cx, cy = np.average(p["x"], weights=w), np.average(p["y"], weights=w)
                ext = float(np.hypot(np.ptp(p["x"]), np.ptp(p["y"])))
                clusters.append(dict(
                    x=float(cx), y=float(cy), range=float(np.hypot(cx, cy)), azimuth=float(np.arctan2(cx, cy)),
                    velocity=float(np.average(p["velocity"], weights=w)), n=len(p), extent=ext,
                    snr=float(p["snr"].max()),
                ))
        tm["cluster"] = time.perf_counter() - t3

        t4 = time.perf_counter()
        tracks: list[Track] = []
        if self.tracker is not None:
            z = np.column_stack([points["range"], points["azimuth"], points["velocity"]]) if len(points) else np.zeros((0, 3))
            tracks = self.tracker.step(z, 10 ** (points["snr"] / 20) if len(points) else None)
        tm["track"] = time.perf_counter() - t4

        ra = None
        if s.range_azimuth_map:
            t5 = time.perf_counter()
            ra = self.angle.range_azimuth(rd, self.ra_angles)
            tm["ra_map"] = time.perf_counter() - t5

        tm["total"] = time.perf_counter() - t0
        return FrameResult(t, power, res.threshold, points, clusters, tracks, ra, tm)
