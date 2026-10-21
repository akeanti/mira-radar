import numpy as np
import pytest

from mira.config import C, PRESETS, RadarConfig, get_preset


def test_textbook_relations():
    cfg = RadarConfig()
    B = cfg.slope * cfg.n_samples / cfg.sample_rate
    assert cfg.bandwidth == pytest.approx(B)
    assert cfg.range_resolution == pytest.approx(C / (2 * B))
    assert cfg.max_range == pytest.approx(cfg.n_samples * cfg.range_resolution)
    assert cfg.max_velocity == pytest.approx(cfg.wavelength / (4 * cfg.n_tx * cfg.chirp_period))
    assert cfg.velocity_resolution == pytest.approx(2 * cfg.max_velocity / cfg.n_chirps)


def test_axes_match_resolutions():
    cfg = RadarConfig()
    assert np.diff(cfg.range_axis())[0] == pytest.approx(cfg.range_resolution)
    v = cfg.velocity_axis()
    assert v[cfg.n_chirps // 2] == 0
    assert np.diff(v)[0] == pytest.approx(cfg.velocity_resolution)


def test_virtual_array_is_filled():
    cfg = RadarConfig(n_tx=3, n_rx=4)
    assert np.allclose(np.diff(cfg.virtual_positions()), cfg.element_spacing)
    assert list(cfg.virtual_tx_index()) == [0] * 4 + [1] * 4 + [2] * 4


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_are_consistent(name):
    cfg = get_preset(name)
    assert cfg.duty_cycle <= 1
    assert len(cfg.specs()) > 8


def test_adc_window_must_fit_in_ramp():
    with pytest.raises(ValueError):
        RadarConfig(ramp_time=20e-6)


def test_unknown_preset():
    with pytest.raises(KeyError):
        get_preset("nope")
