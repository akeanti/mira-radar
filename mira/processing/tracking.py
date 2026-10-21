"""Multi-target tracking: extended Kalman filter + global nearest neighbour.

State per track is [x, y, vx, vy] with a constant velocity model. The radar
measures range, azimuth and radial velocity, which are non-linear in that
state, hence the EKF. Using the radial velocity directly is what makes radar
tracks lock on in a couple of frames instead of a couple of seconds.

Association works on the raw detection points, not on pre-made clusters: a
car lights up several range bins and its points do not always cluster
cleanly, so each point is gated against every track (the gate grows with the
track's estimated size and Doppler spread) and joins the closest one. Each
track is then updated with the centroid of its points. Points nobody claims
are clustered with DBSCAN and start new tentative tracks, which are confirmed
after a few hits; tracks that stop getting points coast for a while and are
then dropped. This is the same idea as the "group tracker" in TI's mmWave SDK.

`linear_assignment` (Hungarian algorithm) is used to resolve clusters that
several tracks compete for.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .clustering import dbscan


def linear_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum cost assignment (Hungarian / Kuhn-Munkres with potentials), O(n^3).

    Works on rectangular matrices; returns (rows, cols) like
    scipy.optimize.linear_sum_assignment.
    """
    cost = np.asarray(cost, float)
    if cost.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, int)  # p[j] = row (1-based) assigned to column j
    way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used
            free[0] = False
            cur = cost[i0 - 1] - u[i0] - v[1:]
            better = free[1:] & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free[1:], minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    cols = np.nonzero(p[1:])[0]
    rows = p[1:][cols] - 1
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    if transposed:
        rows, cols = cols, rows
        order = np.argsort(rows)
        rows, cols = rows[order], cols[order]
    return rows, cols


def measurement_model(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """h(s) = [range, azimuth, radial velocity] and its Jacobian."""
    x, y, vx, vy = s
    r2 = max(x * x + y * y, 1e-6)
    r = np.sqrt(r2)
    vr = (x * vx + y * vy) / r
    z = np.array([r, np.arctan2(x, y), vr])
    cross = vx * y - vy * x
    H = np.array([
        [x / r, y / r, 0.0, 0.0],
        [y / r2, -x / r2, 0.0, 0.0],
        [y * cross / (r2 * r), -x * cross / (r2 * r), x / r, y / r],
    ])
    return z, H


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Track:
    id: int
    x: np.ndarray  # [x, y, vx, vy]
    P: np.ndarray
    status: str = "tentative"  # tentative | confirmed
    hits: int = 1
    misses: int = 0
    age: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=60))
    extent: float = 1.0  # m, rough size of the object
    doppler_spread: float = 0.5  # m/s, spread of radial velocities over its points
    n_points: int = 0

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:]

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.x[2:]))

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"


def _centroid(pts: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Weighted centroid of polar points as a polar measurement, plus their size and Doppler spread."""
    r, az, vr = pts[:, 0], pts[:, 1], pts[:, 2]
    xy = np.column_stack([r * np.sin(az), r * np.cos(az)])
    w = w / w.sum()
    c = w @ xy
    v = float(w @ vr)
    size = float(np.hypot(*np.ptp(xy, axis=0))) if len(pts) > 1 else 0.0
    spread = float(np.sqrt(w @ (vr - v) ** 2))
    return np.array([np.hypot(*c), np.arctan2(c[0], c[1]), v]), size, spread


PROFILES: dict[str, dict] = {
    # cars are big and fast, their points spread over several metres
    "vehicles": dict(),
    # long range: trucks up to 15 m long, coarse range bins
    "highway": dict(rigid_radius=14.0, merge_distance=4.0, exclusion_margin=2.5, birth_eps=4.0,
                    sigma_range=0.6, min_extent=3.0),
    # people are small, slow and walk close to each other
    "people": dict(accel_noise=1.5, min_extent=0.6, min_spread=0.6, birth_eps=0.9, merge_distance=0.4,
                   micro_doppler_tolerance=3.0, exclusion_margin=0.3, init_tangential_speed=1.5, rigid_radius=0.0),
}


class Tracker:
    def __init__(
        self,
        dt: float,
        accel_noise: float = 2.5,
        sigma_range: float = 0.25,
        sigma_azimuth_deg: float = 1.5,
        sigma_velocity: float = 0.2,
        gate: float = 14.0,
        min_extent: float = 1.5,
        min_spread: float = 0.8,
        confirm_hits: int = 3,
        confirm_window: int = 5,
        max_misses_tentative: int = 1,
        max_misses: int = 8,
        init_tangential_speed: float = 6.0,
        birth_eps: float = 2.8,
        birth_velocity_weight: float = 2.0,
        merge_distance: float = 2.5,
        micro_doppler_tolerance: float = 4.5,
        exclusion_margin: float = 1.5,
        rigid_radius: float = 5.0,
        rigid_doppler: float = 0.6,
        max_range: float = np.inf,
        max_azimuth_deg: float = 90.0,
    ):
        """
        dt: frame period. accel_noise: white acceleration (m/s^2) of the motion model.
        sigma_*: noise of a single detection point.
        gate: chi-square gate (3 degrees of freedom) for point-to-track association.
        min_extent, min_spread: floors on the object size (m) and Doppler spread (m/s)
            used for gating, so a fresh track can still collect all of its points.
        birth_eps, birth_velocity_weight: DBSCAN for unclaimed points, with
            birth_velocity_weight m/s counting as one metre.
        merge_distance: two confirmed tracks closer than this plus half their
            sizes, moving alike, are one object and get merged.
        micro_doppler_tolerance, exclusion_margin: unclaimed points within
            exclusion_margin + size of a confirmed track and within
            micro_doppler_tolerance m/s of its Doppler are taken as its
            micro-Doppler (wheels, limbs, rotors) and do not start new tracks.
        rigid_radius, rigid_doppler: points or tracks within rigid_radius (m) of
            a confirmed track and within rigid_doppler (m/s) of its velocity are
            parts of the same rigid object (the far end of a car up close).
            Set rigid_radius to 0 when separate objects move side by side.
        max_range, max_azimuth_deg: tracks that leave this field of view are dropped.

        See PROFILES for settings that suit vehicles or people.
        """
        self.dt = dt
        self.R = np.diag([sigma_range**2, np.radians(sigma_azimuth_deg) ** 2, sigma_velocity**2])
        self.gate = gate
        self.min_extent = min_extent
        self.min_spread = min_spread
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.max_misses_tentative = max_misses_tentative
        self.max_misses = max_misses
        self.init_tangential_speed = init_tangential_speed
        self.birth_eps = birth_eps
        self.birth_velocity_weight = birth_velocity_weight
        self.merge_distance = merge_distance
        self.micro_doppler_tolerance = micro_doppler_tolerance
        self.exclusion_margin = exclusion_margin
        self.rigid_radius = rigid_radius
        self.rigid_doppler = rigid_doppler
        self.max_range = max_range
        self.max_azimuth = np.radians(max_azimuth_deg)
        self.tracks: list[Track] = []
        self._next_id = 1

        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        self.F = F
        G = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = G
        Q[np.ix_([1, 3], [1, 3])] = G
        self.Q = Q * accel_noise**2

    # ------------------------------------------------------------- filtering
    def _predict(self, t: Track) -> None:
        t.x = self.F @ t.x
        t.P = self.F @ t.P @ self.F.T + self.Q
        t.age += 1

    def _spread_cov(self, t: Track, rng: float) -> np.ndarray:
        """Where a single point of this object may land around its centroid, in (r, az, vr)."""
        half = 0.5 * max(t.extent, self.min_extent)
        dv = max(t.doppler_spread, self.min_spread)
        return np.diag([half**2, (half / max(rng, 1.0)) ** 2, dv**2])

    def _update(self, t: Track, z: np.ndarray, R: np.ndarray) -> None:
        zp, H = measurement_model(t.x)
        nu = z - zp
        nu[1] = _wrap(nu[1])
        S = H @ t.P @ H.T + R
        K = t.P @ H.T @ np.linalg.inv(S)
        t.x = t.x + K @ nu
        I_KH = np.eye(4) - K @ H
        t.P = I_KH @ t.P @ I_KH.T + K @ R @ K.T  # Joseph form, stays symmetric positive
        t.hits += 1
        t.misses = 0

    def _group_noise(self, z: np.ndarray, n: int, size: float, spread: float) -> np.ndarray:
        """Noise of a centroid of n points from an object of the given size and Doppler spread."""
        wander = 0.25 * size
        return self.R / max(n, 1) + np.diag([wander**2, (wander / max(z[0], 1.0)) ** 2, (0.3 * spread) ** 2]) \
            + np.diag([0.05**2, np.radians(0.3) ** 2, 0.05**2])

    def _new_track(self, z: np.ndarray, R: np.ndarray, size: float, spread: float) -> Track:
        r, az, vr = z
        u = np.array([np.sin(az), np.cos(az)])  # radial unit vector
        w = np.array([np.cos(az), -np.sin(az)])  # tangential unit vector
        x = np.concatenate([r * u, vr * u])
        J = np.column_stack([u, r * w])  # d(x, y) / d(r, az)
        P = np.zeros((4, 4))
        P[:2, :2] = J @ R[:2, :2] @ J.T + np.eye(2) * 0.1
        P[2:, 2:] = np.outer(u, u) * (R[2, 2] + 0.1) + np.outer(w, w) * self.init_tangential_speed**2
        t = Track(self._next_id, x, P, extent=max(size, 0.5), doppler_spread=spread)
        t.history.append(x[:2].copy())
        self._next_id += 1
        return t

    # ------------------------------------------------------------------ step
    def step(self, points: np.ndarray, weights: np.ndarray | None = None) -> list[Track]:
        """Advance one frame.

        points: (N, 3) detections as range (m), azimuth (rad), radial velocity (m/s).
        weights: (N,) relative strength of each point, used for centroids.
        Returns the confirmed tracks.
        """
        pts = np.asarray(points, float).reshape(-1, 3)
        N = len(pts)
        w = np.ones(N) if weights is None else np.asarray(weights, float)

        for t in self.tracks:
            self._predict(t)

        # 1. gate every point against every track, tentative tracks pay a small penalty
        owner = np.full(N, -1)
        if N and self.tracks:
            D = np.full((len(self.tracks), N), np.inf)
            for i, t in enumerate(self.tracks):
                zp, H = measurement_model(t.x)
                S = H @ t.P @ H.T + self.R + self._spread_cov(t, zp[0])
                nu = pts - zp
                nu[:, 1] = (nu[:, 1] + np.pi) % (2 * np.pi) - np.pi
                d2 = np.einsum("ni,ij,nj->n", nu, np.linalg.inv(S), nu)
                d2 = d2 + (0.0 if t.confirmed else 2.0)
                D[i] = np.where(d2 < self.gate, d2, np.inf)
            best = np.argmin(D, axis=0)
            ok = np.isfinite(D[best, np.arange(N)])
            owner[ok] = best[ok]

        # 2. update each track with the centroid of its points
        for i, t in enumerate(self.tracks):
            idx = np.nonzero(owner == i)[0]
            t.n_points = len(idx)
            if not len(idx):
                t.misses += 1
                continue
            z, size, spread = _centroid(pts[idx], w[idx])
            self._update(t, z, self._group_noise(z, len(idx), max(size, t.extent), spread))
            t.extent = 0.85 * t.extent + 0.15 * max(size, 0.5)
            t.doppler_spread = 0.85 * t.doppler_spread + 0.15 * spread

        # 3. unclaimed points start new tracks, unless they hug a confirmed track
        # with a similar Doppler (the far end of a big object, not a new one)
        free = np.nonzero(owner < 0)[0]
        if len(free) and self.tracks:
            r, az, vr = pts[free].T
            xy = np.column_stack([r * np.sin(az), r * np.cos(az)])
            near = np.zeros(len(free), bool)
            for t in self.tracks:
                if not t.confirmed:
                    continue
                zp, _ = measurement_model(t.x)
                d = np.hypot(*(xy - t.position).T)
                dv = np.abs(vr - zp[2])
                # wheels, limbs and rotors scatter Doppler by several m/s around the body
                near |= (d < self.exclusion_margin + t.extent) & (dv < self.micro_doppler_tolerance)
                # a rigid body moves as one: within a vehicle length and moving like it, it is the vehicle
                near |= (d < self.rigid_radius) & (dv < self.rigid_doppler)
            free = free[~near]
        if len(free):
            r, az, vr = pts[free].T
            feats = np.column_stack([r * np.sin(az), r * np.cos(az), vr / self.birth_velocity_weight])
            labels = dbscan(feats, self.birth_eps, 1)
            for c in range(labels.max() + 1):
                sel = free[labels == c]
                z, size, spread = _centroid(pts[sel], w[sel])
                self.tracks.append(self._new_track(z, self._group_noise(z, len(sel), size, spread), size, spread))

        # 4. confirm, drop, merge duplicates
        survivors = []
        for t in self.tracks:
            if not t.confirmed and t.hits >= self.confirm_hits:
                t.status = "confirmed"
            if t.confirmed:
                dead = t.misses > self.max_misses
            else:
                dead = t.misses > self.max_misses_tentative or t.age > self.confirm_window + self.confirm_hits
            if not dead and np.trace(t.P[:2, :2]) > 400:
                dead = True
            if not dead and (np.hypot(*t.position) > self.max_range
                             or abs(np.arctan2(t.position[0], t.position[1])) > self.max_azimuth):
                dead = True
            if not dead:
                survivors.append(t)
        survivors.sort(key=lambda t: (not t.confirmed, -t.hits))

        def same(t: Track, k: Track) -> bool:
            d = np.hypot(*(t.position - k.position))
            dv = np.hypot(*(t.velocity - k.velocity))
            return (d < self.merge_distance + 0.5 * (t.extent + k.extent) and dv < 1.2) or \
                (d < self.rigid_radius and dv < self.rigid_doppler)

        kept: list[Track] = []
        for t in survivors:
            if not any(same(t, k) for k in kept):
                t.history.append(t.x[:2].copy())
                kept.append(t)
        self.tracks = kept
        return [t for t in self.tracks if t.confirmed]

    @property
    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]
