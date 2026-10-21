"""Command line: `mira demo`, `mira render`, `mira sonar`, `mira specs`, `mira bench`."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _scenario(args):
    from .config import get_preset
    from .pipeline import PipelineSettings
    from .scenarios import get_scenario

    scene, preset, profile = get_scenario(args.scenario)
    cfg = get_preset(args.preset or preset)
    settings = PipelineSettings(tracker_profile=profile, angle_method=args.angle, cfar_method=args.cfar)
    return scene, cfg, settings


def cmd_demo(args) -> None:
    from .viz.render import live

    scene, cfg, settings = _scenario(args)
    live(scene, cfg, settings, show_truth=not args.no_truth)


def cmd_render(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from .viz.render import render_frames, save_gif, save_mp4

    scene, cfg, settings = _scenario(args)
    every = max(1, int(round(cfg.frame_rate / args.fps)))
    fps = cfg.frame_rate / every
    n = int(args.seconds * cfg.frame_rate)
    skip = int(args.start * cfg.frame_rate)
    print(f"rendering {args.scenario} on {cfg.name}: {n // every} frames at {fps:g} fps")
    images = render_frames(scene, cfg, n, every=every, skip=skip, settings=settings, show_truth=not args.no_truth,
                           dpi=args.dpi)
    for out in args.out:
        if out.endswith(".gif"):
            path = save_gif(images, out, fps, colors=args.colors, scale=args.scale)
        elif out.endswith(".mp4"):
            path = save_mp4(images, out, fps)
        else:
            from PIL import Image

            Image.fromarray(images[-1]).save(out)
            path = out
        print(f"wrote {path}")


def cmd_sonar(args) -> None:
    from .sonar.processor import SonarConfig

    cfg = SonarConfig(f0=args.f0, f1=args.f1, max_range=args.max_range)
    if args.audible:
        cfg = SonarConfig(f0=3000.0, f1=9000.0, max_range=args.max_range)
    if args.render:
        import matplotlib

        matplotlib.use("Agg")
        from .sonar.live import render_simulated
        from .viz.render import save_gif

        images = render_simulated(cfg, seconds=args.seconds, fps=args.fps)
        print(f"wrote {save_gif(images, args.render, args.fps, colors=96, scale=args.scale)}")
    elif args.simulate:
        from .sonar.live import run_simulated

        run_simulated(cfg)
    else:
        from .sonar.live import run_live

        run_live(cfg, device=args.device)


def cmd_specs(args) -> None:
    from .config import PRESETS

    presets = list(PRESETS.values())
    rows = [k for k, _ in presets[0].specs()]
    table = {p.name: dict(p.specs()) for p in presets}
    if args.markdown:
        print("| | " + " | ".join(f"**{p.name}**" for p in presets) + " |")
        print("|---|" + "---|" * len(presets))
        print("| Use | " + " | ".join(p.description for p in presets) + " |")
        for r in rows:
            print(f"| {r} | " + " | ".join(table[p.name][r] for p in presets) + " |")
        return
    w = max(len(r) for r in rows) + 2
    cols = [max(len(p.name), max(len(v) for v in table[p.name].values())) + 2 for p in presets]
    print(" " * w + "".join(p.name.ljust(c) for p, c in zip(presets, cols)))
    for r in rows:
        print(r.ljust(w) + "".join(table[p.name][r].ljust(c) for p, c in zip(presets, cols)))


def cmd_bench(args) -> None:
    from .pipeline import RadarPipeline
    from .simulator import FMCWSimulator

    scene, cfg, settings = _scenario(args)
    sim = FMCWSimulator(cfg)
    pipe = RadarPipeline(cfg, settings)
    acc: dict[str, list[float]] = {}
    sims = []
    for i in range(args.frames):
        t0 = time.perf_counter()
        cube = sim.frame(scene)
        sims.append(time.perf_counter() - t0)
        res = pipe.process(cube, scene.t)
        for k, v in res.timings.items():
            acc.setdefault(k, []).append(v)
        scene.step(cfg.frame_period)
    print(f"{cfg.name}, {args.scenario}, {args.frames} frames, data cube {cfg.n_chirps}x{cfg.n_virtual}x{cfg.n_samples}")
    print(f"  {'simulate':10s} {np.mean(sims) * 1e3:7.1f} ms")
    for k, v in acc.items():
        print(f"  {k:10s} {np.mean(v) * 1e3:7.1f} ms")
    print(f"  real-time budget at {cfg.frame_rate:.0f} fps: {cfg.frame_period * 1e3:.0f} ms")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mira", description="MIRA: a complete MIMO FMCW radar, from chirp to track.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def scenario_args(sp):
        from .scenarios import SCENARIOS

        sp.add_argument("--scenario", default="intersection", choices=list(SCENARIOS))
        sp.add_argument("--preset", default=None, help="radar preset, defaults to the one the scenario was made for")
        sp.add_argument("--angle", default="bartlett", choices=["bartlett", "music"], help="angle estimator")
        sp.add_argument("--cfar", default="ca", choices=["ca", "os"], help="CFAR flavour")
        sp.add_argument("--no-truth", action="store_true", help="hide the simulator's ground truth")

    sp = sub.add_parser("demo", help="live radar console on a simulated scene")
    scenario_args(sp)
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("render", help="render the console to GIF / MP4 / PNG")
    scenario_args(sp)
    sp.add_argument("--out", nargs="+", default=["mira.gif"])
    sp.add_argument("--seconds", type=float, default=8.0)
    sp.add_argument("--start", type=float, default=2.0, help="seconds simulated before recording starts")
    sp.add_argument("--fps", type=float, default=10.0)
    sp.add_argument("--dpi", type=int, default=80, help="80 gives 1280x720")
    sp.add_argument("--colors", type=int, default=128, help="GIF palette size")
    sp.add_argument("--scale", type=float, default=1.0, help="GIF downscale factor")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("sonar", help="turn your laptop's speaker and microphone into a sonar")
    sp.add_argument("--simulate", action="store_true", help="no sound card: a simulated room and hand")
    sp.add_argument("--f0", type=float, default=18000.0)
    sp.add_argument("--f1", type=float, default=21500.0)
    sp.add_argument("--audible", action="store_true", help="3-9 kHz chirp, for speakers that cannot play 20 kHz")
    sp.add_argument("--max-range", type=float, default=1.5)
    sp.add_argument("--device", default=None, help="sounddevice device id or name")
    sp.add_argument("--render", metavar="GIF", default=None, help="render the simulated room to a GIF instead")
    sp.add_argument("--seconds", type=float, default=8.0, help="with --render")
    sp.add_argument("--fps", type=float, default=10.0, help="with --render")
    sp.add_argument("--scale", type=float, default=0.75, help="with --render: GIF downscale factor")
    sp.set_defaults(func=cmd_sonar)

    sp = sub.add_parser("specs", help="spec sheet of the radar presets")
    sp.add_argument("--markdown", action="store_true")
    sp.set_defaults(func=cmd_specs)

    sp = sub.add_parser("bench", help="time every stage of the pipeline")
    scenario_args(sp)
    sp.add_argument("--frames", type=int, default=60)
    sp.set_defaults(func=cmd_bench)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
