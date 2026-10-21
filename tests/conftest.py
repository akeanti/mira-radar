import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mira.scene import Scatterers, _points  # noqa: E402


def point_targets(targets, rcs=1.0) -> Scatterers:
    """Scatterers from (range m, azimuth deg, radial velocity m/s) tuples."""
    parts = []
    for r, az_deg, vr in targets:
        az = np.radians(az_deg)
        u = np.array([np.sin(az), np.cos(az)])
        parts.append(_points(r * u, vr * u, rcs, 0))
    return Scatterers.concat(parts)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
