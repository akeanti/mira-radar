"""Run a scenario through simulator + pipeline + dashboard, live or to a file."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from ..config import RadarConfig
from ..pipeline import FrameResult, PipelineSettings, RadarPipeline
from ..scene import Scene
from ..simulator import FMCWSimulator


def run(scene: Scene, cfg: RadarConfig, settings: PipelineSettings | None = None, n_frames: int | None = None,
        seed: int = 0) -> Iterator[tuple[int, FrameResult, list[dict], float]]:
    """Yield (frame index, result, ground truth, simulation ms) forever or for n_frames."""
    sim = FMCWSimulator(cfg, seed=seed)
    pipe = RadarPipeline(cfg, settings)
    i = 0
    while n_frames is None or i < n_frames:
        t0 = time.perf_counter()
        cube = sim.frame(scene)
        sim_ms = (time.perf_counter() - t0) * 1e3
        res = pipe.process(cube, scene.t)
        yield i, res, scene.truth(), sim_ms
        scene.step(cfg.frame_period)
        i += 1


def make_dashboard(scene: Scene, cfg: RadarConfig, settings: PipelineSettings | None = None, **kw):
    from .dashboard import Dashboard

    pipe = RadarPipeline(cfg, settings)
    return Dashboard(cfg, pipe.ra_angles, map_lines=scene.map_lines, scene_name=scene.name, **kw)


def live(scene: Scene, cfg: RadarConfig, settings: PipelineSettings | None = None, show_truth: bool = True) -> None:
    """Open a window and run the scenario in (simulated) real time."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    dash = make_dashboard(scene, cfg, settings, show_truth=show_truth, dpi=72)
    frames = run(scene, cfg, settings)

    def tick(_):
        i, res, truth, sim_ms = next(frames)
        dash.update(res, truth, i, sim_ms)
        return []

    anim = FuncAnimation(dash.fig, tick, interval=cfg.frame_period * 1000, cache_frame_data=False)
    dash.fig._mira_anim = anim  # keep a reference alive while the window is open
    plt.show()


def render_frames(scene: Scene, cfg: RadarConfig, n_frames: int, every: int = 1, skip: int = 0,
                  settings: PipelineSettings | None = None, show_truth: bool = True, dpi: int = 80,
                  figsize: tuple[float, float] = (16, 9), progress: bool = True) -> list[np.ndarray]:
    """RGB frames of the dashboard.

    Simulates every radar frame (the tracker needs them) but only draws every
    `every`-th one, after `skip` warm-up frames that fill the waterfall and let
    the tracks settle.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    dash = make_dashboard(scene, cfg, settings, show_truth=show_truth, dpi=dpi, figsize=figsize)
    images = []
    for i, res, truth, sim_ms in run(scene, cfg, settings, skip + n_frames):
        dash.update(res, truth, i, sim_ms)
        if i >= skip and (i - skip) % every == 0:
            images.append(dash.rgb())
        if progress and i % 20 == 0:
            print(f"  frame {i}/{skip + n_frames}", flush=True)
    dash.close()
    return images


def save_gif(images: list[np.ndarray], path: str | Path, fps: float, colors: int = 128, scale: float = 1.0) -> Path:
    """Write an animated GIF with one shared palette (smaller, no colour flicker)."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = [Image.fromarray(im) for im in images]
    if scale != 1.0:
        size = (int(pil[0].width * scale) // 2 * 2, int(pil[0].height * scale) // 2 * 2)
        pil = [im.resize(size, Image.LANCZOS) for im in pil]
    sample = pil[:: max(1, len(pil) // 8)]
    swatch = _ui_swatch(sample[0].width)
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample) + swatch.height))
    for k, im in enumerate(sample):
        strip.paste(im, (0, k * im.height))
    # make sure the palette keeps the colour scales and accents even if they cover few pixels
    strip.paste(swatch, (0, sample[0].height * len(sample)))
    palette = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in pil]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(round(1000 / fps)), loop=0,
                   optimize=True, disposal=1)
    return path


def _ui_swatch(width: int):
    from PIL import Image
    from matplotlib.colors import to_rgb

    from . import style

    cols = [style.velocity_cmap()(x)[:3] for x in np.linspace(0, 1, 24)]
    cols += [style.heat()(x)[:3] for x in np.linspace(0, 1, 16)]
    cols += [to_rgb(c) for c in (style.TRACK, style.ACCENT, style.ACCENT_2, style.TEXT, style.DIM, style.TRUTH)]
    block = 40
    rows = []
    for c in cols:
        rows.append(np.tile((np.array(c) * 255).astype(np.uint8), (block, width, 1)))
    return Image.fromarray(np.concatenate(rows, axis=0))


def save_mp4(images: list[np.ndarray], path: str | Path, fps: float) -> Path:
    """Write an H.264 MP4 through ffmpeg (must be on PATH)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = images[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow",
           "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for im in images:
        proc.stdin.write(np.ascontiguousarray(im[: h // 2 * 2, : w // 2 * 2]).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return path
