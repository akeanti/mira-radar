import numpy as np
import pytest

from mira.config import RadarConfig
from mira.processing import ca_cfar_scale, cfar_2d, doppler_fft, os_cfar_scale, range_doppler_power, range_fft
from mira.processing.cfar import betainc
from mira.simulator import FMCWSimulator

from .conftest import point_targets


@pytest.mark.parametrize("a,b,x", [(1, 8, 0.1), (8, 256, 0.05), (4, 400, 0.02), (2.5, 3.5, 0.7), (30, 900, 0.04)])
def test_betainc_matches_scipy(a, b, x):
    sp = pytest.importorskip("scipy.special")
    assert betainc(a, b, x) == pytest.approx(sp.betainc(a, b, x), rel=1e-10, abs=1e-14)


def test_ca_scale_closed_form():
    for n in (8, 16, 32, 100):
        assert ca_cfar_scale(n, 1e-6) == pytest.approx(n * (1e-6 ** (-1 / n) - 1))


def test_os_scale_reduces_to_closed_form():
    n, k = 24, 18
    a = os_cfar_scale(n, k, 1e-4, 1)
    i = np.arange(k)
    assert np.prod((n - i) / (n - i + a)) == pytest.approx(1e-4, rel=1e-6)


@pytest.mark.parametrize("method", ["ca", "os"])
def test_false_alarm_rate_on_noise(method):
    """Measured Pfa on real pipeline noise (8 channels integrated) is close to the design value."""
    cfg = RadarConfig()
    sim = FMCWSimulator(cfg, seed=11, leakage_db=None)
    pfa, hits, cells = 1e-3, 0, 0
    for _ in range(6):
        p = range_doppler_power(doppler_fft(range_fft(sim.frame_from_scatterers(point_targets([])))))
        r = cfar_2d(p, (2, 2), (4, 8), pfa, method, n_integrated=cfg.n_virtual, peaks_only=False)
        hits += r.mask.sum()
        cells += r.mask.size
    assert 0.4 * pfa < hits / cells < 2.5 * pfa


def test_os_cfar_survives_a_neighbouring_strong_target(rng):
    """The textbook masking case: a weak target 6 cells from a strong one."""
    p = rng.exponential(1.0, (1, 256))
    p[0, 100] += 1e3
    p[0, 106] += 1e2
    ca = cfar_2d(p, (0, 2), (0, 12), 1e-5, "ca", wrap0=False, peaks_only=False)
    os_ = cfar_2d(p, (0, 2), (0, 12), 1e-5, "os", wrap0=False, peaks_only=False)
    assert ca.mask[0, 100] and os_.mask[0, 100]
    assert not ca.mask[0, 106]
    assert os_.mask[0, 106]


def test_detections_are_peaks():
    cfg = RadarConfig()
    cube = FMCWSimulator(cfg, seed=5).frame_from_scatterers(point_targets([(20, 0, -3), (30, 10, 4)]))
    p = range_doppler_power(doppler_fft(range_fft(cube)))
    res = cfar_2d(p, n_integrated=cfg.n_virtual, min_index1=4)
    assert len(res.detections) >= 2
    assert np.all(res.snr_db > 0)
