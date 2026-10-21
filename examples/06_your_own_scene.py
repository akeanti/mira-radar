"""Use MIRA as a library: build a scene, simulate the radar, read out tracks.

    python examples/06_your_own_scene.py
"""

import numpy as np

from mira import Car, FMCWSimulator, Maneuver, Pedestrian, RadarPipeline, Scene, Static, get_preset

cfg = get_preset("MIRA-77")
scene = Scene([
    Car(position=(-3.0, 45.0), velocity=(0.0, -10.0)),                       # oncoming car
    Car(position=(20.0, 20.0), velocity=(-7.0, 0.0),                         # car turning towards us
        maneuvers=[Maneuver(1.5, 3.6, yaw_rate=0.75)]),
    Pedestrian(position=(4.0, 12.0), velocity=(-1.2, 0.3)),
    Static(points=[(6.0, 15.0), (6.0, 30.0)], rcs=3.0),                       # two lamp posts
])

sim = FMCWSimulator(cfg, seed=0)
radar = RadarPipeline(cfg)

for frame in range(60):
    result = radar.process(sim.frame(scene), scene.t)
    if frame % 20 == 19:
        print(f"t = {scene.t:4.2f} s   {len(result.points):2d} detections   {len(result.tracks)} tracks")
        for t in result.tracks:
            r, az = np.hypot(*t.position), np.degrees(np.arctan2(*t.position))
            print(f"    track {t.id:2d}: {r:5.1f} m at {az:+5.1f} deg, {t.speed * 3.6:4.0f} km/h")
    scene.step(cfg.frame_period)
