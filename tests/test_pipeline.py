import numpy as np

from mira.config import get_preset
from mira.pipeline import PipelineSettings, RadarPipeline
from mira.scenarios import get_scenario
from mira.simulator import FMCWSimulator


def _score(name: str, frames: int, warmup: int = 20):
    scene, preset, profile = get_scenario(name)
    cfg = get_preset(preset)
    sim = FMCWSimulator(cfg, seed=0)
    pipe = RadarPipeline(cfg, PipelineSettings(tracker_profile=profile, range_azimuth_map=False))
    seen = found = false = 0
    for i in range(frames):
        res = pipe.process(sim.frame(scene), scene.t)
        truth = [o for o in scene.truth()
                 if np.hypot(*o["position"]) < 0.95 * cfg.max_range
                 and abs(np.degrees(np.arctan2(*o["position"]))) < min(cfg.beamwidth_deg / 2, 60)]
        used = set()
        for o in truth:
            d = [(np.hypot(*(t.position - o["position"])), t.id) for t in res.tracks if t.id not in used]
            if d and min(d)[0] < 3.0:
                used.add(min(d)[1])
                if i >= warmup:
                    found += 1
        if i >= warmup:
            seen += len(truth)
            false += len(res.tracks) - len(used)
        scene.step(cfg.frame_period)
    return found / seen, false / (frames - warmup)


def test_intersection_end_to_end():
    recall, false_per_frame = _score("intersection", 120)
    assert recall > 0.85
    assert false_per_frame < 1.5


def test_indoor_people():
    recall, false_per_frame = _score("indoor", 100)
    # two walkers are tracked; the person sitting still is invisible after static clutter removal
    assert recall > 0.6
    assert false_per_frame < 0.5


def test_range_azimuth_map_shape():
    scene, preset, _ = get_scenario("two-targets")
    cfg = get_preset(preset)
    pipe = RadarPipeline(cfg)
    res = pipe.process(FMCWSimulator(cfg).frame(scene))
    assert res.ra_power.shape == (len(pipe.ra_angles), cfg.n_samples)
    assert np.all(res.ra_power >= 0)
