import numpy as np
import pytest

from mira.sonar import SimulatedRoom, SonarConfig, SonarProcessor, chirp


def _run(cfg, room, block, seconds=12.0):
    """Distance errors and (measured, true) speeds while the simulated hand is moving."""
    proc = SonarProcessor(cfg)
    errs, speeds = [], []
    n = 0
    period_s = cfg.period / cfg.sample_rate
    while room.n < seconds * cfg.sample_rate:
        for fr in proc.push(room.read(block)):
            n += 1
            t = n * period_s - period_s / 2
            truth = room.hand_range(t) - room.d_sm / 2
            v = (room.hand_range(t + 0.01) - room.hand_range(t - 0.01)) / 0.02
            if t > 2 and abs(v) > 0.1 and fr.distance is not None:
                errs.append(fr.distance - truth)
                speeds.append((fr.velocity, v))
    return np.array(errs), np.array(speeds)


@pytest.mark.parametrize("block,latency", [(480, 1777), (1024, 0), (333, 5000), (2048, 123)])
def test_hand_distance_any_block_size_and_latency(block, latency):
    cfg = SonarConfig()
    errs, speeds = _run(cfg, SimulatedRoom(cfg, latency=latency), block)
    assert len(errs) > 250
    # a hand waving fast, pausing and changing direction: most estimates within a few cm
    assert np.median(np.abs(errs)) < 0.035
    assert np.mean(np.abs(errs) < 0.05) > 0.55
    assert np.corrcoef(speeds[:, 0], speeds[:, 1])[0, 1] > 0.9


def test_steady_approach_range_unbiased_and_speed_measured():
    """A hand approaching at a steady 0.6 m/s reads ~7 cm short through range-Doppler coupling; the velocity
    from the range filter is used to take that out."""
    cfg = SonarConfig()
    room = SimulatedRoom(cfg, hand=lambda t: 0.5 - 0.6 * (t % 1.0) if (t % 2.0) < 1.0 else 0.5, latency=900)
    proc = SonarProcessor(cfg)
    period_s = cfg.period / cfg.sample_rate
    errs, speeds, n = [], [], 0
    while room.n < 8 * cfg.sample_rate:
        for fr in proc.push(room.read(512)):
            n += 1
            t = n * period_s - period_s / 2
            if t > 2 and 0.2 < (t % 2.0) < 0.9 and fr.distance is not None:
                errs.append(fr.distance - (room.hand_range(t) - room.d_sm / 2))
                speeds.append(fr.velocity)
    assert abs(np.median(errs)) < 0.015
    assert np.median(speeds) == pytest.approx(-0.6, abs=0.1)


def test_chirp_is_periodic_and_in_band():
    cfg = SonarConfig()
    x = chirp(cfg, np.arange(3 * cfg.period))
    assert np.allclose(x[: cfg.period], x[cfg.period: 2 * cfg.period])
    spec = np.abs(np.fft.rfft(x[: cfg.period])) ** 2
    f = np.fft.rfftfreq(cfg.period, 1 / cfg.sample_rate)
    in_band = spec[(f >= cfg.f0 - 500) & (f <= cfg.f1 + 500)].sum()
    assert in_band / spec.sum() > 0.97


def test_nothing_moves_nothing_detected():
    cfg = SonarConfig()
    room = SimulatedRoom(cfg, hand=None)
    proc = SonarProcessor(cfg)
    frames = []
    while room.n < 4 * cfg.sample_rate:
        frames += proc.push(room.read(512))
    assert sum(f.distance is not None for f in frames[50:]) < 0.05 * len(frames[50:])
