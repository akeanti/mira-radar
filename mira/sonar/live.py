"""Sonar console, fed by the sound card or by the simulated room."""

from __future__ import annotations

import queue
import time

import numpy as np
import matplotlib.pyplot as plt

from ..viz import style
from .processor import SonarConfig, SonarFrame, SonarProcessor, chirp
from .room import SimulatedRoom


class SonarDashboard:
    def __init__(self, cfg: SonarConfig, history_s: float = 8.0, simulated: bool = False, dpi: int = 80,
                 figsize: tuple[float, float] = (16, 9)):
        style.apply()
        self.cfg = cfg
        self.n_hist = int(history_s * cfg.profile_rate)
        self.wf = np.zeros((cfg.n_bins, self.n_hist))
        self.trace = np.full(self.n_hist, np.nan)
        self._gain = np.sqrt(cfg.range_axis() + 0.1)
        self._ref = None
        self.dyn = 28.0  # dB shown in the waterfall

        fig = plt.figure(figsize=figsize, dpi=dpi)
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], left=0.05, right=0.985, top=0.86, bottom=0.08,
                              wspace=0.14, hspace=0.35)
        ax_wf = fig.add_subplot(gs[:, 0])
        ax_pr = fig.add_subplot(gs[0, 1])
        ax_d = fig.add_subplot(gs[1, 1])
        self.ax_wf, self.ax_pr, self.ax_d = ax_wf, ax_pr, ax_d

        title = fig.text(0.05, 0.93, "MIRA SONAR", size=24, weight="bold", color=style.ACCENT, va="center")
        x1 = title.get_window_extent(fig.canvas.get_renderer()).x1 / fig.bbox.width + 0.012
        fig.text(x1, 0.942, "YOUR LAPTOP AS AN FMCW SONAR", size=11.5, weight="bold", color=style.TEXT, va="center")
        fig.text(x1, 0.915,
                 f"{cfg.f0 / 1e3:.1f}–{cfg.f1 / 1e3:.1f} kHz chirp · ΔR {cfg.range_resolution * 100:.1f} cm · "
                 f"{cfg.profile_rate:.0f} profiles/s · same matched-filter maths as the radar",
                 size=9, color=style.DIM, va="center")
        fig.text(0.985, 0.942, "● SIMULATED ROOM" if simulated else "● LIVE · SPEAKER → HAND → MIC", size=10,
                 weight="bold", color=style.ACCENT_2, ha="right", va="center")
        self.status = fig.text(0.985, 0.915, "", size=9.5, color=style.TEXT, ha="right", va="center")
        fig.text(0.985, 0.025, "github.com/akeanti/mira-radar", size=8.5, color=style.FAINT, ha="right")

        ax_wf.set_title("RANGE-TIME WATERFALL  ·  what moves in front of the screen", color=style.TEXT)
        self.wf_img = ax_wf.imshow(self.wf, origin="lower", aspect="auto", cmap=style.heat(), vmin=0, vmax=1,
                                   extent=[-history_s, 0, 0, cfg.max_range * 100], interpolation="nearest")
        (self.trace_line,) = ax_wf.plot(np.linspace(-history_s, 0, self.n_hist), self.trace, color=style.ACCENT,
                                        lw=1.6, alpha=0.9)
        ax_wf.set_xlabel("time (s)")
        ax_wf.set_ylabel("distance from the laptop (cm)")

        ax_pr.set_title("RANGE PROFILE  ·  matched filter output", color=style.TEXT)
        r = cfg.range_axis() * 100
        (self.raw_line,) = ax_pr.plot(r, np.zeros_like(r), color=style.DIM, lw=1.0, label="everything (desk, walls)")
        (self.mov_line,) = ax_pr.plot(r, np.zeros_like(r), color=style.ACCENT, lw=1.4, label="moving only")
        ax_pr.set_xlim(0, cfg.max_range * 100)
        ax_pr.set_ylim(-60, 2)
        ax_pr.set_xlabel("distance (cm)")
        ax_pr.set_ylabel("dB")
        ax_pr.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=style.DIM)

        ax_d.set_title("NEAREST MOVING OBJECT", color=style.TEXT)
        ax_d.set_xlim(0, cfg.max_range * 100)
        ax_d.set_ylim(0, 1)
        ax_d.set_yticks([])
        ax_d.set_xlabel("distance (cm)")
        ax_d.barh([0.18], [cfg.max_range * 100], height=0.12, color=style.GRID)
        self.bar = ax_d.barh([0.18], [0], height=0.12, color=style.ACCENT)[0]
        self.readout = ax_d.text(0.5, 0.68, "--.- cm", transform=ax_d.transAxes, ha="center", va="center", size=40,
                                 weight="bold", color=style.ACCENT)
        self.speed = ax_d.text(0.5, 0.42, "", transform=ax_d.transAxes, ha="center", va="center", size=12,
                               color=style.TEXT)

    def update(self, frames: list[SonarFrame], direct: int | None = None, truth: float | None = None) -> None:
        if not frames:
            return
        for fr in frames:
            self.wf = np.roll(self.wf, -1, axis=1)
            self.trace = np.roll(self.trace, -1)
            # dB, partly compensating the spreading loss so a far hand stays visible
            col = 20 * np.log10(fr.moving * self._gain + 1e-12)
            peak = float(col.max())
            self._ref = peak if self._ref is None else max(self._ref - 0.1, peak)
            self.wf[:, -1] = np.clip((col - self._ref + self.dyn) / self.dyn, 0, 1)
            self.trace[-1] = np.nan if fr.distance is None else fr.distance * 100
        last = frames[-1]
        self.wf_img.set_data(self.wf)
        self.trace_line.set_ydata(self.trace)
        ref = np.max(last.profile) + 1e-12
        self.raw_line.set_ydata(20 * np.log10(last.profile / ref + 1e-6))
        self.mov_line.set_ydata(20 * np.log10(last.moving / ref + 1e-6))
        if last.distance is not None:
            self.readout.set_text(f"{last.distance * 100:5.1f} cm")
            self.bar.set_width(last.distance * 100)
            v = last.velocity or 0.0
            self.speed.set_text(("approaching" if v < 0 else "moving away") + f"  {abs(v) * 100:3.0f} cm/s"
                                if abs(v) > 0.05 else "holding still")
        else:
            self.readout.set_text("--.- cm")
            self.bar.set_width(0)
            self.speed.set_text("")
        extra = f"  |  TRUTH {truth * 100:5.1f} cm" if truth is not None else ""
        self.status.set_text(f"DIRECT PATH AT SAMPLE {last.direct_index:4d} (latency absorbed){extra}")

    def rgb(self) -> np.ndarray:
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()


def run_simulated(cfg: SonarConfig | None = None, block: int = 480, realtime: bool = True):
    """Simulated room in a window, at the speed sound would arrive."""
    from matplotlib.animation import FuncAnimation

    cfg = cfg or SonarConfig()
    room = SimulatedRoom(cfg)
    proc = SonarProcessor(cfg)
    dash = SonarDashboard(cfg, simulated=True, dpi=72)
    per_tick = max(1, int(cfg.sample_rate * 0.04 / block))

    def tick(_):
        frames = []
        for _ in range(per_tick):
            frames += proc.push(room.read(block))
        t = room.n / cfg.sample_rate
        truth = room.hand_range(t)
        dash.update(frames, truth=None if truth is None else truth - room.d_sm / 2)
        return []

    anim = FuncAnimation(dash.fig, tick, interval=40, cache_frame_data=False)
    dash.fig._mira_anim = anim
    plt.show()


def render_simulated(cfg: SonarConfig | None = None, seconds: float = 8.0, fps: float = 12.0, warmup: float = 4.0,
                     block: int = 480) -> list[np.ndarray]:
    """Frames of the sonar console on the simulated room, for GIFs."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    cfg = cfg or SonarConfig()
    room = SimulatedRoom(cfg)
    proc = SonarProcessor(cfg)
    dash = SonarDashboard(cfg, simulated=True)
    images = []
    next_shot = warmup
    while room.n < (warmup + seconds) * cfg.sample_rate:
        frames = proc.push(room.read(block))
        t = room.n / cfg.sample_rate
        if frames:
            truth = room.hand_range(t)
            dash.update(frames, truth=None if truth is None else truth - room.d_sm / 2)
        if t >= next_shot:
            images.append(dash.rgb())
            next_shot += 1 / fps
    plt.close(dash.fig)
    return images


def run_live(cfg: SonarConfig | None = None, device=None, block: int = 512) -> None:
    """Play the chirp on the default output and listen on the default input."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:  # OSError: PortAudio missing
        raise SystemExit(f"live sonar needs the sounddevice package and PortAudio: pip install sounddevice ({e})")
    from matplotlib.animation import FuncAnimation

    cfg = cfg or SonarConfig()
    tx = chirp(cfg).astype(np.float32)
    N = cfg.period
    q: queue.Queue = queue.Queue()
    pos = 0

    def callback(indata, outdata, frames, t, status):  # runs on the audio thread
        nonlocal pos
        idx = (pos + np.arange(frames)) % N
        outdata[:] = tx[idx, None]
        pos = (pos + frames) % N
        q.put(indata[:, 0].copy())

    proc = SonarProcessor(cfg)
    dash = SonarDashboard(cfg, simulated=False, dpi=72)
    # some output devices refuse mono, so play the same chirp on up to two channels
    out_channels = max(1, min(2, int(sd.query_devices(device, "output")["max_output_channels"])))
    stream = sd.Stream(samplerate=cfg.sample_rate, blocksize=block, dtype="float32", channels=(1, out_channels),
                       callback=callback, device=device, latency="low")

    def tick(_):
        frames = []
        while True:
            try:
                frames += proc.push(q.get_nowait())
            except queue.Empty:
                break
        dash.update(frames)
        return []

    print(f"MIRA sonar: playing {cfg.f0 / 1e3:.1f}-{cfg.f1 / 1e3:.1f} kHz chirps. Move your hand in front of the "
          "laptop. Close the window to stop.")
    with stream:
        time.sleep(0.3)
        anim = FuncAnimation(dash.fig, tick, interval=40, cache_frame_data=False)
        dash.fig._mira_anim = anim
        plt.show()
