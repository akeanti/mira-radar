import numpy as np
import pytest

from mira.processing.angle import bartlett_spectrum, find_peaks, music_spectrum

POS = np.arange(8) * 0.5
ANG = np.radians(np.arange(-70, 70.25, 0.25))


def snapshot(rng, angles_deg, noise=0.03, amps=None):
    th = np.radians(angles_deg)
    amps = np.ones(len(th)) if amps is None else np.asarray(amps)
    s = amps * np.exp(2j * np.pi * rng.uniform(size=len(th)))
    x = np.exp(-2j * np.pi * np.outer(np.sin(th), POS)).T @ s
    return x + noise * (rng.standard_normal(8) + 1j * rng.standard_normal(8))


@pytest.mark.parametrize("az", [-55, -20, 0, 7.3, 33, 60])
def test_bartlett_single_target(rng, az):
    peaks = find_peaks(bartlett_spectrum(snapshot(rng, [az]), POS, ANG), ANG, 1)
    assert np.degrees(peaks[0][0]) == pytest.approx(az, abs=0.5)


def test_music_resolves_below_the_rayleigh_limit(rng):
    """Two targets 8 degrees apart with an 8-element array (Rayleigh limit ~14 deg)."""
    music_ok = bartlett_ok = 0
    for _ in range(30):
        x = snapshot(rng, [-4, 4])
        pb = find_peaks(bartlett_spectrum(x, POS, ANG), ANG, 2, 10)
        pm = find_peaks(music_spectrum(x, 0.5, ANG, n_sources=2), ANG, 2, 25)

        def ok(p):
            return len(p) == 2 and all(abs(np.degrees(a) - t) < 1.5 for (a, _), t in zip(sorted(p), (-4, 4)))

        music_ok += ok(pm)
        bartlett_ok += ok(pb)
    assert music_ok >= 27
    assert bartlett_ok == 0


def test_music_counts_one_source(rng):
    for az in (-30, 0, 25):
        p = find_peaks(music_spectrum(snapshot(rng, [az]), 0.5, ANG), ANG, 2, 20)
        assert len(p) == 1
        assert np.degrees(p[0][0]) == pytest.approx(az, abs=0.5)


def test_edge_peak_is_rejected():
    """A target outside the search grid must not be reported at the grid edge."""
    x = np.exp(-2j * np.pi * np.sin(np.radians(85)) * POS)
    assert find_peaks(bartlett_spectrum(x, POS, ANG), ANG) == []
