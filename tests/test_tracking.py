import itertools

import numpy as np
import pytest

from mira.processing.clustering import dbscan
from mira.processing.tracking import Tracker, linear_assignment, measurement_model


def _brute_force(cost: np.ndarray) -> float:
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
    n0, n1 = cost.shape
    return min(sum(cost[i, p[i]] for i in range(n0)) for p in itertools.permutations(range(n1), n0))


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (4, 6), (6, 4), (5, 5)])
def test_hungarian_matches_brute_force(rng, shape):
    for _ in range(20):
        cost = rng.uniform(0, 10, shape)
        rows, cols = linear_assignment(cost)
        assert len(rows) == min(shape)
        assert len(set(rows)) == len(rows) and len(set(cols)) == len(cols)
        assert cost[rows, cols].sum() == pytest.approx(_brute_force(cost))


def test_measurement_jacobian_is_correct():
    s = np.array([3.0, 12.0, -1.5, 4.0])
    _, H = measurement_model(s)
    eps = 1e-6
    num = np.zeros((3, 4))
    for i in range(4):
        d = np.zeros(4)
        d[i] = eps
        num[:, i] = (measurement_model(s + d)[0] - measurement_model(s - d)[0]) / (2 * eps)
    assert np.allclose(H, num, atol=1e-6)


def test_dbscan_groups_and_noise():
    a = np.array([[0, 0], [0.5, 0], [1.0, 0.2]])
    b = np.array([[10, 10], [10.4, 10.3]])
    lone = np.array([[30, -5]])
    labels = dbscan(np.vstack([a, b, lone]), eps=1.0, min_samples=2)
    assert len(set(labels[:3])) == 1 and len(set(labels[3:5])) == 1
    assert labels[0] != labels[3]
    assert labels[5] == -1


def _polar(p, v):
    r = np.hypot(*p)
    return np.array([r, np.arctan2(p[0], p[1]), p @ v / r])


def test_tracker_follows_a_crossing_target():
    """A target crossing the beam: radial velocity alone does not give the tangential speed, the EKF must."""
    dt = 0.05
    vel_err, pos_err = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        tr = Tracker(dt)
        p, v = np.array([-15.0, 20.0]), np.array([8.0, 0.0])
        for k in range(80):
            z = _polar(p, v) + rng.normal(0, [0.1, np.radians(1.0), 0.1])
            confirmed = tr.step(z[None, :])
            p = p + v * dt
        assert len(confirmed) == 1
        t = confirmed[0]
        pos_err.append(np.hypot(*(t.position - (p - v * dt))))
        vel_err.append(np.hypot(*(t.velocity - v)))
    assert np.median(pos_err) < 0.25
    assert np.median(vel_err) < 0.5


def test_tracker_keeps_two_targets_apart_and_drops_lost_ones(rng):
    dt = 0.05
    tr = Tracker(dt)
    a, va = np.array([-5.0, 30.0]), np.array([0.0, -8.0])
    b, vb = np.array([6.0, 10.0]), np.array([0.0, 6.0])
    for k in range(60):
        pts = [_polar(a, va), _polar(b, vb)] if k < 40 else [_polar(a, va)]
        tr.step(np.array(pts) + rng.normal(0, [0.1, np.radians(1.0), 0.1], (len(pts), 3)))
        a, b = a + va * dt, b + vb * dt
    confirmed = tr.confirmed
    assert len(confirmed) == 1
    assert np.hypot(*(confirmed[0].position - (a - va * dt))) < 1.0


def test_false_alarms_do_not_become_tracks(rng):
    tr = Tracker(0.05)
    for _ in range(100):
        n = rng.integers(0, 3)
        pts = np.column_stack([rng.uniform(2, 50, n), rng.uniform(-1, 1, n), rng.uniform(-12, 12, n)])
        tr.step(pts)
    assert len(tr.confirmed) == 0
