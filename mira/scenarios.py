"""Ready-made scenes. Each function returns a fresh Scene at t = 0."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .scene import Car, Cyclist, Drone, Maneuver, Pedestrian, Reflector, Scene, Static


def _line(p0, p1, spacing: float) -> np.ndarray:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.hypot(*(p1 - p0)) / spacing) + 1)
    return p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0)


def _ground_clutter(rng: np.random.Generator, n: int, max_range: float, fov_deg: float, mean_rcs: float) -> Static:
    r = max_range * np.sqrt(rng.uniform(0.02, 1, n))
    az = np.radians(rng.uniform(-fov_deg / 2, fov_deg / 2, n))
    pts = np.column_stack([r * np.sin(az), r * np.cos(az)])
    return Static(points=pts, rcs=rng.exponential(mean_rcs, n), label="ground")


def intersection(seed: int = 7) -> Scene:
    """A radar on a lamp post at a city intersection, looking up the street.

    Main road along y (x from -8 to -1 m), cross street along x (y from 24 to
    31 m), a crosswalk at y = 14 m. Cars in every direction including one
    turning left, pedestrians on the crosswalk and the sidewalk, a cyclist in
    the bike lane and a drone hovering over the pavement. Fits the MIRA-77
    preset (50 m, +/-12 m/s).
    """
    rng = np.random.default_rng(seed)
    objs = [
        Car(label="car", position=(-6.25, 49.0), velocity=(0.0, -9.5)),
        Car(label="car", position=(-2.75, 2.0), velocity=(0.0, 5.5), t_spawn=1.0,
            maneuvers=[Maneuver(1.0, 7.0, accel=0.8)]),
        Car(label="car", position=(-40.0, 26.25), velocity=(9.0, 0.0), length=4.8, width=1.9, rcs_total=14.0),
        Car(label="car", position=(31.0, 28.75), velocity=(-6.0, 0.0), t_spawn=1.0,
            maneuvers=[Maneuver(5.875, 5.875 + np.pi / 2 / 0.75, yaw_rate=0.75)]),
        Pedestrian(label="pedestrian", position=(-9.0, 13.6), velocity=(1.4, 0.0), gait_phase=0.0),
        Pedestrian(label="pedestrian", position=(0.5, 14.4), velocity=(-1.2, 0.0), t_spawn=2.0, gait_phase=1.7),
        Pedestrian(label="pedestrian", position=(0.8, 30.0), velocity=(0.0, -1.3), gait_phase=0.6),
        Cyclist(label="cyclist", position=(-1.4, 40.0), velocity=(0.0, -5.0), t_spawn=0.5),
        Drone(label="drone", position=(9.0, 16.0), velocity=(-1.0, 2.2),
              maneuvers=[Maneuver(3.0, 7.0, yaw_rate=0.25)]),
    ]

    posts = [(1.6, y) for y in (8, 20, 34, 46)] + [(-9.6, y) for y in (12, 22, 34, 46)]
    walls = np.vstack([
        _line((-11, 4), (-11, 23), 0.8), _line((-11, 32), (-11, 49), 0.8),
        _line((3.0, 4), (3.0, 23), 0.8), _line((3.0, 32), (3.0, 49), 0.8),
        _line((-30, 22.5), (-11, 22.5), 1.2), _line((3, 22.5), (30, 22.5), 1.2),
        _line((-30, 32.5), (-11, 32.5), 1.2), _line((3, 32.5), (30, 32.5), 1.2),
    ])
    parked = np.vstack([
        _line((4.4, 36.0), (4.4, 40.5), 0.9), _line((4.4, 42.0), (4.4, 46.5), 0.9),
        _line((-9.2, 38.0), (-9.2, 42.5), 0.9),
    ])
    objs += [
        Static(points=posts, rcs=2.0, label="lamp post"),
        Static(points=walls, rcs=rng.exponential(0.4, len(walls)), label="building"),
        Static(points=parked, rcs=2.5, label="parked car"),
        _ground_clutter(rng, 220, 50.0, 130.0, 0.02),
    ]

    lines = [
        np.array([[-8, 0], [-8, 24]]), np.array([[-8, 31], [-8, 50]]),
        np.array([[-1, 0], [-1, 24]]), np.array([[-1, 31], [-1, 50]]),
        np.array([[-45, 24], [-8, 24]]), np.array([[-1, 24], [45, 24]]),
        np.array([[-45, 31], [-8, 31]]), np.array([[-1, 31], [45, 31]]),
        np.array([[1.5, 0], [1.5, 24]]), np.array([[1.5, 31], [1.5, 50]]),
    ]
    dashes = [np.array([[-4.5, y], [-4.5, y + 1.5]]) for y in np.arange(1, 50, 4) if not 23 < y < 31]
    dashes += [np.array([[x, 27.5], [x + 1.5, 27.5]]) for x in np.arange(-44, 44, 4) if not -9 < x < 0]
    zebra = [np.array([[x, 13.2], [x, 14.8]]) for x in np.arange(-7.5, -1.2, 0.7)]
    return Scene(objs, name="intersection", map_lines=lines + dashes + zebra, seed=seed)


def two_targets(separation_deg: float = 8.0, rng_m: float = 25.0, seed: int = 0) -> Scene:
    """Two point targets at the same range and speed, a few degrees apart: an angular resolution test."""
    half = np.radians(separation_deg / 2)
    objs = [
        Reflector(position=rng_m * np.array([np.sin(s * half), np.cos(s * half)]), velocity=(0.0, -5.0), rcs=5.0)
        for s in (-1, 1)
    ]
    return Scene(objs, name="two-targets", fluctuation=False, seed=seed)


def indoor(seed: int = 3) -> Scene:
    """People in a 10 m room for the MIRA-60 preset: two walking, one sitting and breathing."""
    rng = np.random.default_rng(seed)
    objs = [
        Pedestrian(position=(-3.5, 6.0), velocity=(0.9, 0.35), gait_phase=0.0,
                   maneuvers=[Maneuver(4.0, 6.5, yaw_rate=-0.9)]),
        Pedestrian(position=(2.5, 1.5), velocity=(0.0, 1.1), gait_phase=2.0,
                   maneuvers=[Maneuver(3.0, 5.0, yaw_rate=1.2)]),
        Pedestrian(position=(1.8, 4.2), velocity=(0.0, 0.0), label="sitting"),
    ]
    walls = np.vstack([_line((-5, 0.5), (-5, 9.5), 0.3), _line((5, 0.5), (5, 9.5), 0.3),
                       _line((-5, 9.5), (5, 9.5), 0.3)])
    furniture = np.array([[-2.0, 8.5], [3.5, 8.0], [-4.0, 2.0], [0.0, 7.0]])
    objs += [
        Static(points=walls, rcs=rng.exponential(0.3, len(walls)), label="wall"),
        Static(points=furniture, rcs=1.0, label="furniture"),
    ]
    room = [np.array([[-5, 0], [-5, 9.5], [5, 9.5], [5, 0]])]
    return Scene(objs, name="indoor", map_lines=room, seed=seed)


def rural(seed: int = 11) -> Scene:
    """Two-lane country road for the MIRA-77LR preset: traffic out to 200 m."""
    rng = np.random.default_rng(seed)
    objs = [
        Car(position=(-1.8, 190.0), velocity=(0.0, -19.0)),
        Car(position=(-1.8, 120.0), velocity=(0.0, -17.0), rcs_total=40.0, length=12.0, width=2.5, label="truck"),
        Car(position=(1.8, 20.0), velocity=(0.0, 16.0)),
        Car(position=(1.8, 70.0), velocity=(0.0, 14.0)),
        Cyclist(position=(3.6, 45.0), velocity=(0.0, 6.0)),
    ]
    guard = _line((5.5, 5), (5.5, 220), 2.0)
    trees = np.column_stack([rng.uniform(-30, -8, 60), rng.uniform(10, 220, 60)])
    objs += [Static(points=guard, rcs=0.5, label="guardrail"),
             Static(points=trees, rcs=rng.exponential(1.0, 60), label="trees"),
             _ground_clutter(rng, 250, 220.0, 40.0, 0.05)]
    lines = [np.array([[-3.6, 0], [-3.6, 230]]), np.array([[3.6, 0], [3.6, 230]])]
    lines += [np.array([[0, y], [0, y + 3]]) for y in range(0, 230, 9)]
    return Scene(objs, name="rural", map_lines=lines, seed=seed)


SCENARIOS: dict[str, tuple[Callable[[], Scene], str, str]] = {
    # name: (builder, radar preset, tracker profile)
    "intersection": (intersection, "MIRA-77", "vehicles"),
    "indoor": (indoor, "MIRA-60", "people"),
    "rural": (rural, "MIRA-77LR", "highway"),
    "two-targets": (two_targets, "MIRA-77", "vehicles"),
}


def get_scenario(name: str) -> tuple[Scene, str, str]:
    """Build a scenario: returns the scene, the radar preset and the tracker profile it was designed for."""
    try:
        make, preset, profile = SCENARIOS[name]
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}, choose from {', '.join(SCENARIOS)}") from None
    return make(), preset, profile
