import numpy as np
import pytest

from mira.config import RadarConfig
from mira.pipeline import PipelineSettings, RadarPipeline
from mira.processing import AngleEstimator, doppler_fft, range_doppler_power, range_fft
from mira.simulator import FMCWSimulator, ideal_snr_db

from .conftest import point_targets

TARGETS = [(20.0, 15.0, -5.0), (35.0, -30.0, 3.3), (12.3, 0.0, 0.8), (42.0, 40.0, -9.1)]


def test_point_targets_are_measured_accurately():
    cfg = RadarConfig()
    cube = FMCWSimulator(cfg, seed=0).frame_from_scatterers(point_targets(TARGETS))
    res = RadarPipeline(cfg, PipelineSettings(track=False, range_azimuth_map=False)).process(cube)
    pts = res.points
    assert len(pts) >= len(TARGETS)
    for r, az, v in TARGETS:
        k = np.argmin(np.abs(pts["range"] - r) + np.abs(pts["velocity"] - v))
        assert pts["range"][k] == pytest.approx(r, abs=cfg.range_resolution / 4)
        assert pts["velocity"][k] == pytest.approx(v, abs=cfg.velocity_resolution / 2)
        assert np.degrees(pts["azimuth"][k]) == pytest.approx(az, abs=1.5)


def test_snr_follows_the_radar_equation():
    """Peak power after the 2D FFT and coherent channel sum, over the noise, matches the link budget."""
    cfg = RadarConfig()
    sim = FMCWSimulator(cfg, seed=1, leakage_db=None)
    r = 25.0
    cube = sim.frame_from_scatterers(point_targets([(r, 0.0, 0.0)], rcs=1.0))
    rd = doppler_fft(range_fft(cube, None), None)
    coherent = np.abs(rd.sum(axis=1)) ** 2
    signal = coherent.max()
    noise_cube = FMCWSimulator(cfg, seed=2, leakage_db=None).frame_from_scatterers(point_targets([]))
    noise = np.mean(np.abs(doppler_fft(range_fft(noise_cube, None), None).sum(axis=1)) ** 2)
    measured = 10 * np.log10(signal / noise)
    # rectangular windows; the target sits a little off the bin centre, so allow scalloping loss
    assert measured == pytest.approx(ideal_snr_db(cfg, r, 1.0), abs=4.0)


def test_tdm_mimo_motion_compensation_is_needed():
    """A fast target's angle is wrong without the TDM phase correction, right with it."""
    cfg = RadarConfig()
    r, az, v = 20.0, 20.0, -11.0
    cube = FMCWSimulator(cfg, seed=3, noise=False, leakage_db=None).frame_from_scatterers(point_targets([(r, az, v)]))
    rd = doppler_fft(range_fft(cube))
    d, rbin = np.unravel_index(np.argmax(range_doppler_power(rd)), (cfg.n_chirps, cfg.n_samples))
    est = AngleEstimator(cfg)
    good = np.degrees(est.estimate(rd[d, :, rbin], d)[0][0])
    from mira.processing.angle import bartlett_spectrum, find_peaks

    raw = find_peaks(bartlett_spectrum(rd[d, :, rbin], cfg.virtual_positions(), est.angles), est.angles)
    bad = np.degrees(raw[0][0]) if raw else np.nan
    assert good == pytest.approx(az, abs=1.0)
    assert not abs(bad - az) < 3.0


def test_quantised_adc_still_detects():
    cfg = RadarConfig()
    cube = FMCWSimulator(cfg, seed=4, adc_bits=12).frame_from_scatterers(point_targets(TARGETS[:2]))
    res = RadarPipeline(cfg, PipelineSettings(track=False, range_azimuth_map=False)).process(cube)
    assert len(res.points) >= 2
