"""The radar console: bird's-eye view, range-Doppler map, micro-Doppler waterfall.

One Dashboard object owns a matplotlib figure. Call update() with each
FrameResult; the same code drives the live window and the GIF/MP4 renderer.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, Wedge

from ..config import RadarConfig
from ..pipeline import FrameResult
from . import style


def _db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(x, 1e-30))


class Dashboard:
    def __init__(
        self,
        cfg: RadarConfig,
        ra_angles: np.ndarray,
        *,
        map_lines: list[np.ndarray] | None = None,
        scene_name: str = "",
        show_truth: bool = True,
        figsize: tuple[float, float] = (16, 9),
        dpi: int = 80,
        history_s: float = 6.0,
        dynamic_range_db: float = 42.0,
        footer: str = "github.com/akeanti/mira-radar",
        simulated: bool = True,
    ):
        style.apply()
        self.cfg = cfg
        self.show_truth = show_truth
        self.dyn = dynamic_range_db
        self.vmax = cfg.max_velocity
        self.rmax = cfg.max_range
        self._floor_rd = None
        self._floor_ra = None
        self._transient: list = []

        fig = plt.figure(figsize=figsize, dpi=dpi)
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.62, 1], left=0.035, right=0.985, top=0.875, bottom=0.065,
                              wspace=0.10, hspace=0.28)
        self.ax_bev = fig.add_subplot(gs[:, 0])
        self.ax_rd = fig.add_subplot(gs[0, 1])
        self.ax_wf = fig.add_subplot(gs[1, 1])

        self._header(scene_name, footer, simulated)
        self._setup_bev(ra_angles, map_lines or [])
        self._setup_rd()
        self._setup_waterfall(history_s)

    # ------------------------------------------------------------------ setup
    def _header(self, scene_name: str, footer: str, simulated: bool) -> None:
        cfg, f = self.cfg, self.fig
        title = f.text(0.035, 0.935, cfg.name, size=24, weight="bold", color=style.ACCENT, va="center")
        x1 = title.get_window_extent(f.canvas.get_renderer()).x1 / f.bbox.width + 0.012
        f.text(x1, 0.947, f"{cfg.start_freq / 1e9:.0f} GHz TDM-MIMO FMCW RADAR", size=11.5, weight="bold",
               color=style.TEXT, va="center")
        f.text(x1, 0.92,
               f"{cfg.n_tx}TX×{cfg.n_rx}RX · {cfg.bandwidth / 1e6:.0f} MHz · "
               f"ΔR {cfg.range_resolution * 100:.1f} cm · Δv {cfg.velocity_resolution:.2f} m/s · "
               f"Δθ {cfg.angle_resolution_deg:.0f}° · {cfg.frame_rate:.0f} fps",
               size=9, color=style.DIM, va="center")
        tag = "SIMULATED" if simulated else "LIVE"
        f.text(0.985, 0.947, f"● {tag}" + (f"  ·  {scene_name.upper()}" if scene_name else ""),
               size=10, weight="bold", color=style.ACCENT_2, ha="right", va="center")
        self.status = f.text(0.985, 0.92, "", size=9.5, color=style.TEXT, ha="right", va="center")
        if footer:
            f.text(0.985, 0.022, footer, size=8.5, color=style.FAINT, ha="right", va="center")
        f.text(0.035, 0.022, "range-Doppler FFT  >  2D CFAR  >  MIMO beamforming  >  DBSCAN  >  EKF tracking",
               size=8.5, color=style.FAINT, va="center")

    def _setup_bev(self, ra_angles: np.ndarray, map_lines: list[np.ndarray]) -> None:
        ax, cfg = self.ax_bev, self.cfg
        ax.set_title("BIRD'S-EYE VIEW  ·  range-azimuth image, detections, tracks", color=style.TEXT)
        half_fov = min(cfg.beamwidth_deg / 2, 70.0)
        self.half_fov = half_fov
        r = self.rmax
        # equal scale on both axes: fit the fan plus room for the legend above it
        bb = ax.get_position(original=True)
        fw, fh = self.fig.get_size_inches()
        ratio = (bb.height * fh) / (bb.width * fw)
        xmax = max(1.04 * r * np.sin(np.radians(max(half_fov, 35))), 1.42 * r / (2 * ratio))
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-0.04 * r, -0.04 * r + 2 * xmax * ratio)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(style.FRAME)

        # field of view and grid
        fan = Wedge((0, 0), r, 90 - half_fov, 90 + half_fov, facecolor="#0d1a1f", edgecolor="none", zorder=0)
        ax.add_patch(fan)
        self._fan = fan
        step = 10 if r <= 60 else (25 if r <= 150 else 50)
        if r <= 20:
            step = 2.5 if r <= 12 else 5
        for rr in np.arange(step, r + 1e-6, step):
            th = np.radians(np.linspace(-half_fov, half_fov, 120))
            ax.plot(rr * np.sin(th), rr * np.cos(th), color=style.GRID, lw=0.8, zorder=1)
            ax.text(rr * np.sin(np.radians(half_fov)) + 0.012 * r, rr * np.cos(np.radians(half_fov)),
                    f"{rr:g} m", size=7.5, color=style.DIM, va="center", zorder=6)
        for a in np.arange(-np.floor(half_fov / 30) * 30, half_fov + 1, 30):
            t = np.radians(a)
            ax.plot([0, r * np.sin(t)], [0, r * np.cos(t)], color=style.GRID, lw=0.8, zorder=1)
            ax.text(1.035 * r * np.sin(t), 1.035 * r * np.cos(t), f"{a:+.0f}°", size=7.5, color=style.DIM,
                    ha="center", va="center", zorder=6)
        for s in (-1, 1):
            t = np.radians(s * half_fov)
            ax.plot([0, r * np.sin(t)], [0, r * np.cos(t)], color=style.FRAME, lw=1.0, zorder=1)

        if map_lines:
            roads = ax.add_collection(LineCollection(map_lines, colors="#2a3a48", linewidths=1.0, zorder=2))
            roads.set_clip_path(fan)

        # range-azimuth image on its polar grid
        rng_edges = np.concatenate([cfg.range_axis() - cfg.range_axis()[1] / 2, [cfg.max_range]])
        da = ra_angles[1] - ra_angles[0]
        ang_edges = np.concatenate([ra_angles - da / 2, [ra_angles[-1] + da / 2]])
        A, Rr = np.meshgrid(ang_edges, np.clip(rng_edges, 0, None), indexing="ij")
        self.ra_mesh = ax.pcolormesh(Rr * np.sin(A), Rr * np.cos(A), np.zeros((len(ra_angles), len(rng_edges) - 1)),
                                     cmap=style.radar_glow(), vmin=0, vmax=1, shading="flat", zorder=3,
                                     rasterized=True, antialiased=False)

        ax.plot([0], [0], marker="^", ms=11, color=style.ACCENT, zorder=9)
        ax.text(0, -0.025 * r, "MIRA", size=7.5, color=style.ACCENT, ha="center", va="top", zorder=9)

        self.det = ax.scatter([], [], s=14, c=[], cmap=style.velocity_cmap(), vmin=-self.vmax, vmax=self.vmax,
                              zorder=7, linewidths=0)
        # legend and colour key in the empty top corners of the panel
        y_top = ax.get_ylim()[1]
        x_left = ax.get_xlim()[0]
        x_right = ax.get_xlim()[1]
        ax.text(x_left + 0.02 * r, y_top - 0.03 * r, "TRACKS", size=9, weight="bold", color=style.TRACK, va="top",
                zorder=10)
        self.track_table = ax.text(x_left + 0.02 * r, y_top - 0.075 * r, "", size=8, color=style.TEXT, va="top",
                                   linespacing=1.35, zorder=10)
        leg = [
            ("□", style.TRACK, "track: id·km/h, line = 1 s ahead"),
            ("●", "#f2f2f2", "detection, colour = radial velocity"),
            ("▭", style.TRUTH, "ground truth (simulator)"),
        ]
        x_leg = x_right - 0.56 * r
        for i, (sym, col, txt) in enumerate(leg if self.show_truth else leg[:2]):
            y = y_top - 0.03 * r - i * 0.045 * r
            ax.text(x_leg, y, sym, size=9, color=col, va="top", zorder=10)
            ax.text(x_leg + 0.04 * r, y, txt, size=7.5, color=style.DIM, va="top", zorder=10)
        cax = ax.inset_axes([0.69, 0.80, 0.27, 0.012])
        cb = self.fig.colorbar(self.det, cax=cax, orientation="horizontal")
        cb.outline.set_edgecolor(style.FRAME)
        cb.set_ticks([-self.vmax, 0, self.vmax])
        cb.set_ticklabels([f"-{self.vmax:.0f} m/s\napproaching", "0", f"+{self.vmax:.0f} m/s\nreceding"])
        cax.tick_params(labelsize=6.5, colors=style.DIM, length=2)

    def _setup_rd(self) -> None:
        ax, cfg = self.ax_rd, self.cfg
        ax.set_title("RANGE-DOPPLER MAP  ·  2D CFAR detections", color=style.TEXT)
        self.rd_img = ax.imshow(np.zeros((cfg.n_samples, cfg.n_chirps)), origin="lower", aspect="auto",
                                extent=[-self.vmax, self.vmax, 0, self.rmax], cmap=style.heat(),
                                vmin=0, vmax=self.dyn, interpolation="nearest")
        self.rd_det = ax.scatter([], [], s=34, facecolors="none", edgecolors=style.ACCENT, linewidths=0.9)
        ax.set_xlabel("radial velocity (m/s)   ← approaching | receding →")
        ax.set_ylabel("range (m)")
        ax.axvline(0, color=style.FAINT, lw=0.6, ls=":")

    def _setup_waterfall(self, history_s: float) -> None:
        ax, cfg = self.ax_wf, self.cfg
        n = max(2, int(round(history_s / cfg.frame_period)))
        self.wf = np.zeros((cfg.n_chirps, n))
        ax.set_title("MICRO-DOPPLER WATERFALL  ·  strongest return per velocity", color=style.TEXT)
        self.wf_img = ax.imshow(self.wf, origin="lower", aspect="auto", extent=[-history_s, 0, -self.vmax, self.vmax],
                                cmap=style.heat(), vmin=0, vmax=self.dyn, interpolation="nearest")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("radial velocity (m/s)")
        ax.axhline(0, color=style.FAINT, lw=0.6, ls=":")

    # ----------------------------------------------------------------- update
    def update(self, res: FrameResult, truth: list[dict] | None = None, frame: int = 0,
               sim_ms: float | None = None) -> None:
        cfg = self.cfg
        for a in self._transient:
            a.remove()
        self._transient = []

        # range-Doppler map, scaled from its own noise floor
        rd_db = _db(res.rd_power)
        floor = float(np.median(rd_db))
        self._floor_rd = floor if self._floor_rd is None else 0.8 * self._floor_rd + 0.2 * floor
        self.rd_img.set_data((rd_db - self._floor_rd + 4).T)
        pts = res.points
        if len(pts):
            self.rd_det.set_offsets(np.column_stack([pts["velocity"], pts["range"]]))
        else:
            self.rd_det.set_offsets(np.zeros((0, 2)))

        # waterfall: strongest cell per Doppler bin, beyond the leakage region
        spec = rd_db[:, max(3, int(0.5 / cfg.range_axis()[1])):].max(axis=1)
        self.wf = np.roll(self.wf, -1, axis=1)
        self.wf[:, -1] = spec - self._floor_rd - 8
        self.wf_img.set_data(self.wf)

        # range-azimuth glow
        if res.ra_power is not None:
            ra_db = _db(res.ra_power)
            f = float(np.median(ra_db))
            self._floor_ra = f if self._floor_ra is None else 0.8 * self._floor_ra + 0.2 * f
            self.ra_mesh.set_array(np.clip((ra_db - self._floor_ra - 6) / 24, 0, 1).ravel())

        if len(pts):
            self.det.set_offsets(np.column_stack([pts["x"], pts["y"]]))
            self.det.set_array(pts["velocity"])
        else:
            self.det.set_offsets(np.zeros((0, 2)))
            self.det.set_array(np.zeros(0))

        ax = self.ax_bev
        if self.show_truth and truth:
            for o in truth:
                patch = ax.add_patch(self._truth_patch(o))
                patch.set_clip_path(self._fan)  # the radar's field of view only
                self._transient.append(patch)

        rows = []
        for t in sorted(res.tracks, key=lambda t: t.id):
            x, y = t.position
            hist = np.array(t.history)
            if len(hist) > 1:
                segs = np.stack([hist[:-1], hist[1:]], axis=1)
                alpha = np.linspace(0.05, 0.7, len(segs))
                cols = np.zeros((len(segs), 4))
                cols[:, :3] = (1.0, 0.81, 0.35)
                cols[:, 3] = alpha
                self._transient.append(ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.6, zorder=8)))
            half = float(np.clip(0.35 * t.extent, 0.6, 2.0))
            self._transient.append(ax.add_patch(Polygon(
                [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]],
                closed=True, fill=False, edgecolor=style.TRACK, lw=1.3, zorder=8)))
            vx, vy = t.velocity
            self._transient += ax.plot([x, x + vx], [y, y + vy], color=style.TRACK, lw=1.2, zorder=8)
            kmh = t.speed * 3.6
            self._transient.append(ax.text(x + half + 0.3, y + half, f"{t.id:02d}·{kmh:.0f}", size=7,
                                           color=style.TRACK, weight="bold", zorder=9, va="bottom"))
            rng = np.hypot(x, y)
            az = np.degrees(np.arctan2(x, y))
            rows.append(f"{t.id:02d}  {rng:5.1f} m  {az:+5.1f}°  {kmh:4.0f} km/h")
        self.track_table.set_text("\n".join(rows[:12]) if rows else "none")

        ms = res.timings.get("total", 0) * 1e3
        sim = f"  |  SIM {sim_ms:4.0f} ms" if sim_ms is not None else ""
        self.status.set_text(
            f"T+{res.t:05.2f} s  |  FRAME {frame:04d}  |  POINTS {len(pts):3d}  |  TRACKS {len(res.tracks):2d}  |  "
            f"DSP {ms:4.0f} ms{sim}")

    def _truth_patch(self, o: dict):
        x, y = o["position"]
        L, W = o.get("size", (0.5, 0.5))
        if o["kind"] in ("pedestrian", "drone"):
            return Circle((x, y), 0.45, fill=False, edgecolor=style.TRUTH, lw=0.9, alpha=0.75, zorder=4,
                          ls="--" if o["kind"] == "drone" else "-")
        h = o.get("heading", 0.0)
        c, s = np.cos(h), np.sin(h)
        corners = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
        rot = corners @ np.array([[c, s], [-s, c]])
        return Polygon(rot + [x, y], closed=True, fill=False, edgecolor=style.TRUTH, lw=0.9, alpha=0.75, zorder=4)

    # ----------------------------------------------------------------- output
    def rgb(self) -> np.ndarray:
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self) -> None:
        plt.close(self.fig)
