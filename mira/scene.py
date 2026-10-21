"""Things for the radar to look at.

Every object is reduced to a few point scatterers, each with a position, a
velocity, a radar cross section and an optional vibration. That is enough to
reproduce the effects that matter for processing: extended targets that light
up several cells, self-occlusion, micro-Doppler from swinging limbs and
spinning rotors, and a static background that clutter removal has to deal
with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


@dataclass
class Scatterers:
    """Flat arrays describing every point scatterer in the scene at one instant."""

    position: np.ndarray  # (N, 2) m
    velocity: np.ndarray  # (N, 2) m/s
    rcs: np.ndarray  # (N,) m^2
    vib_amp: np.ndarray  # (N,) m, radial vibration amplitude
    vib_freq: np.ndarray  # (N,) Hz
    vib_phase: np.ndarray  # (N,) rad
    owner: np.ndarray  # (N,) object id, -1 for clutter

    @staticmethod
    def empty() -> "Scatterers":
        z = np.zeros(0)
        return Scatterers(np.zeros((0, 2)), np.zeros((0, 2)), z, z, z, z, np.zeros(0, int))

    @staticmethod
    def concat(parts: list["Scatterers"]) -> "Scatterers":
        parts = [p for p in parts if len(p.rcs)]
        if not parts:
            return Scatterers.empty()
        return Scatterers(
            *(np.concatenate([getattr(p, f) for p in parts]) for f in
              ("position", "velocity", "rcs", "vib_amp", "vib_freq", "vib_phase", "owner"))
        )

    def __len__(self) -> int:
        return len(self.rcs)


def _points(pos, vel, rcs, owner, vib_amp=None, vib_freq=None, vib_phase=None) -> Scatterers:
    pos = np.atleast_2d(np.asarray(pos, float))
    n = len(pos)
    vel = np.broadcast_to(np.asarray(vel, float), (n, 2)).copy()
    rcs = np.broadcast_to(np.asarray(rcs, float), (n,)).copy()
    zeros = np.zeros(n)
    return Scatterers(
        pos, vel, rcs,
        zeros if vib_amp is None else np.broadcast_to(np.asarray(vib_amp, float), (n,)).copy(),
        zeros if vib_freq is None else np.broadcast_to(np.asarray(vib_freq, float), (n,)).copy(),
        zeros if vib_phase is None else np.broadcast_to(np.asarray(vib_phase, float), (n,)).copy(),
        np.full(n, owner, int),
    )


def _rot(heading: float) -> np.ndarray:
    """Rotation taking body coordinates (x forward, y left) to world coordinates."""
    c, s = np.cos(heading), np.sin(heading)
    return np.array([[c, -s], [s, c]])


@dataclass
class Maneuver:
    """Constant yaw rate and/or acceleration between two times."""

    t_start: float
    t_end: float
    yaw_rate: float = 0.0  # rad/s, positive is counter-clockwise
    accel: float = 0.0  # m/s^2 along the heading


@dataclass(kw_only=True)
class SceneObject:
    kind: ClassVar[str] = "object"
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    label: str = ""
    maneuvers: list[Maneuver] = field(default_factory=list)
    t_spawn: float = 0.0  # invisible before this time
    id: int = -1
    # body heading for objects that may stand still
    _heading: float = field(default=np.nan, repr=False)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, float)
        self.velocity = np.asarray(self.velocity, float)
        if np.isnan(self._heading):
            self._heading = float(np.arctan2(self.velocity[1], self.velocity[0])) if np.any(self.velocity) else np.pi / 2

    @property
    def heading(self) -> float:
        if np.hypot(*self.velocity) > 0.05:
            self._heading = float(np.arctan2(self.velocity[1], self.velocity[0]))
        return self._heading

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))

    def active(self, t: float) -> bool:
        return t >= self.t_spawn

    def step(self, t: float, dt: float) -> None:
        if not self.active(t):
            return
        yaw, acc = 0.0, 0.0
        for m in self.maneuvers:
            if m.t_start <= t < m.t_end:
                yaw += m.yaw_rate
                acc += m.accel
        speed = self.speed
        heading = self.heading
        if yaw or acc:
            heading += yaw * dt
            speed = max(0.0, speed + acc * dt)
            self.velocity = speed * np.array([np.cos(heading), np.sin(heading)])
            self._heading = heading
        self.position = self.position + self.velocity * dt

    def scatterers(self, t: float) -> Scatterers:  # pragma: no cover - abstract
        raise NotImplementedError


def _visible(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Facets whose outward normal points towards the radar at the origin."""
    to_radar = -points
    return np.einsum("ij,ij->i", normals, to_radar) > 0


@dataclass(kw_only=True)
class Car(SceneObject):
    kind: ClassVar[str] = "car"
    length: float = 4.5
    width: float = 1.8
    rcs_total: float = 10.0  # m^2, about 10 dBsm

    def scatterers(self, t: float) -> Scatterers:
        hl, hw = self.length / 2, self.width / 2
        # corners, bumper centres, side panels, wheel wells
        body = np.array([
            [hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw],
            [hl, 0], [-hl, 0], [0, hw], [0, -hw],
            [hl * 0.6, hw], [hl * 0.6, -hw], [-hl * 0.6, hw], [-hl * 0.6, -hw],
        ])
        normals = np.array([
            [1, 1], [1, -1], [-1, 1], [-1, -1],
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [0, 1], [0, -1], [0, 1], [0, -1],
        ], float)
        weights = np.array([1.5, 1.5, 1.2, 1.2, 2.0, 1.6, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6])
        R = _rot(self.heading)
        pos = self.position + body @ R.T
        nrm = normals @ R.T
        vis = _visible(pos, nrm)
        w = weights[vis]
        rcs = self.rcs_total * w / w.sum() if w.size else w
        return _points(pos[vis], self.velocity, rcs, self.id)


@dataclass(kw_only=True)
class Pedestrian(SceneObject):
    """Torso, two legs and two arms. The limbs swing, which spreads the Doppler."""

    kind: ClassVar[str] = "pedestrian"
    rcs_torso: float = 0.5
    gait_phase: float = 0.0

    def scatterers(self, t: float) -> Scatterers:
        v = self.velocity
        speed = self.speed
        R = _rot(self.heading)
        fwd = R[:, 0]
        left = R[:, 1]
        f_gait = 0.45 + 0.4 * speed  # gait cycles per second
        ph = 2 * np.pi * f_gait * t + self.gait_phase + np.array([0.0, np.pi, np.pi, 0.0])
        # left leg, right leg, left arm, right arm: each arm swings with the opposite leg
        amp_v = speed * np.array([1.4, 1.4, 0.7, 0.7])
        limb_v = (speed + amp_v * np.sin(ph))[:, None] * fwd
        limb_off = -amp_v / (2 * np.pi * f_gait) * np.cos(ph)
        lateral = np.array([0.12, -0.12, 0.22, -0.22])
        limb_pos = self.position + limb_off[:, None] * fwd + lateral[:, None] * left
        pos = np.vstack([self.position, limb_pos])
        vel = np.vstack([v, limb_v if speed > 0.05 else np.zeros((4, 2))])
        rcs = np.array([self.rcs_torso, 0.12, 0.12, 0.05, 0.05])
        # chest wall moves a few mm with breathing, which a 60 GHz radar can see
        vib_amp = np.array([0.004, 0, 0, 0, 0])
        vib_freq = np.array([0.25, 0, 0, 0, 0])
        return _points(pos, vel, rcs, self.id, vib_amp, vib_freq, np.zeros(5))


@dataclass(kw_only=True)
class Cyclist(SceneObject):
    kind: ClassVar[str] = "cyclist"

    def scatterers(self, t: float) -> Scatterers:
        R = _rot(self.heading)
        # rider, front hub, rear hub, pedalling legs, a spoke on each wheel
        body = np.array([[0.0, 0.0], [0.55, 0.0], [-0.55, 0.0], [0.1, 0.0], [0.55, 0.0], [-0.55, 0.0]])
        pos = self.position + body @ R.T
        rcs = np.array([0.8, 0.2, 0.2, 0.15, 0.03, 0.03])
        wheel_hz = self.speed / (np.pi * 0.7)
        cadence = wheel_hz / 2.5
        vib_amp = np.array([0.0, 0.0, 0.0, 0.08, 0.3, 0.3])
        vib_freq = np.array([0.0, 0.0, 0.0, cadence, wheel_hz, wheel_hz])
        vib_phase = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.3])
        return _points(pos, self.velocity, rcs, self.id, vib_amp, vib_freq, vib_phase)


@dataclass(kw_only=True)
class Drone(SceneObject):
    """Small quadcopter. The spinning blades smear energy across Doppler."""

    kind: ClassVar[str] = "drone"
    rotor_hz: float = 90.0

    def scatterers(self, t: float) -> Scatterers:
        arms = 0.18 * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) / np.sqrt(2)
        pos = np.vstack([self.position, self.position + arms])
        rcs = np.array([0.08, 0.001, 0.001, 0.001, 0.001])
        vib_amp = np.array([0.0, 0.11, 0.11, 0.11, 0.11])
        vib_freq = np.array([0.0, 1, 1, 1, 1]) * self.rotor_hz
        vib_phase = np.array([0.0, 0.0, 1.6, 3.1, 4.7])
        return _points(pos, self.velocity, rcs, self.id, vib_amp, vib_freq, vib_phase)


@dataclass(kw_only=True)
class Reflector(SceneObject):
    """A single point scatterer, like a corner reflector. Handy for tests and calibration."""

    kind: ClassVar[str] = "reflector"
    rcs: float = 1.0

    def scatterers(self, t: float) -> Scatterers:
        return _points(self.position, self.velocity, self.rcs, self.id)


@dataclass(kw_only=True)
class Static(SceneObject):
    """Anything that does not move: posts, walls, parked cars, the TX-RX leakage."""

    kind: ClassVar[str] = "static"
    points: np.ndarray = field(default_factory=lambda: np.zeros((1, 2)))
    rcs: np.ndarray | float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.points = np.atleast_2d(np.asarray(self.points, float))

    def step(self, t: float, dt: float) -> None:
        pass

    def scatterers(self, t: float) -> Scatterers:
        return _points(self.points, (0.0, 0.0), self.rcs, self.id)


# footprint (length, width) used to draw ground truth
SIZES = {"pedestrian": (0.5, 0.5), "cyclist": (1.8, 0.6), "drone": (0.5, 0.5)}


@dataclass
class Scene:
    objects: list[SceneObject] = field(default_factory=list)
    t: float = 0.0
    name: str = "scene"
    # polylines drawn under the bird's eye view (road edges, lane marks)
    map_lines: list[np.ndarray] = field(default_factory=list)
    fluctuation: bool = True  # Swerling-like RCS fluctuation frame to frame
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        for i, o in enumerate(self.objects):
            if o.id < 0:
                o.id = i

    def add(self, obj: SceneObject) -> SceneObject:
        obj.id = max((o.id for o in self.objects), default=-1) + 1
        self.objects.append(obj)
        return obj

    def movers(self) -> list[SceneObject]:
        return [o for o in self.objects if o.kind != "static" and o.active(self.t)]

    def scatterers(self) -> Scatterers:
        parts = [o.scatterers(self.t) for o in self.objects if o.active(self.t)]
        sc = Scatterers.concat(parts)
        if self.fluctuation and len(sc):
            static_ids = [o.id for o in self.objects if o.kind == "static"]
            moving = ~np.isin(sc.owner, static_ids)
            # chi-square with 4 degrees of freedom, mean 1 (Swerling III)
            fluct = self._rng.gamma(2.0, 0.5, len(sc))
            sc.rcs = np.where(moving, sc.rcs * fluct, sc.rcs)
        return sc

    def step(self, dt: float) -> None:
        for o in self.objects:
            o.step(self.t, dt)
        self.t += dt

    def truth(self) -> list[dict]:
        """Ground truth for the moving objects currently in the scene."""
        return [
            dict(id=o.id, kind=o.kind, label=o.label, position=o.position.copy(),
                 velocity=o.velocity.copy(), heading=o.heading, size=SIZES.get(o.kind, (0.5, 0.5))
                 if not isinstance(o, Car) else (o.length, o.width))
            for o in self.movers()
        ]
