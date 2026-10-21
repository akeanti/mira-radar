"""Constant false alarm rate detection.

A fixed threshold either drowns in clutter or misses weak targets. CFAR
estimates the noise around every cell from a ring of training cells (skipping
a few guard cells so the target does not raise its own threshold) and scales
it by a factor chosen for the false alarm probability you asked for.

The scale factor is computed exactly for a power map made by summing |X|^2
over `n_integrated` channels, which is what the pipeline produces: noise in
each cell is then Gamma(n_integrated) distributed rather than exponential.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, lgamma, log

import numpy as np


# --------------------------------------------------------------- statistics
def _betacf(a: float, b: float, x: float, max_iter: int = 400, eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return exp(ln_front) * _betacf(a, b, x) / a
    return 1.0 - exp(ln_front) * _betacf(b, a, 1.0 - x) / b


@lru_cache(maxsize=256)
def ca_cfar_scale(n_train: int, pfa: float, n_integrated: int = 1) -> float:
    """Threshold factor alpha so that P(cell > alpha * mean(training)) = pfa under noise only.

    For n_integrated = 1 this is the textbook N * (pfa^(-1/N) - 1).
    """
    if n_integrated == 1:
        return n_train * (pfa ** (-1.0 / n_train) - 1.0)
    p, q = float(n_integrated), float(n_integrated * n_train)

    def pfa_of(t: float) -> float:  # t = alpha / N
        return 1.0 - betainc(p, q, t / (1.0 + t))

    lo, hi = 0.0, 1.0
    while pfa_of(hi) > pfa:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if pfa_of(mid) > pfa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) * n_train


def _gamma_sf(x: np.ndarray, k: int) -> np.ndarray:
    """Survival function of a unit-scale Gamma with integer shape k."""
    term = np.ones_like(x)
    total = np.ones_like(x)
    for j in range(1, k):
        term = term * x / j
        total = total + term
    return np.exp(-x) * total


@lru_cache(maxsize=256)
def os_cfar_scale(n_train: int, rank: int, pfa: float, n_integrated: int = 1) -> float:
    """Threshold factor for ordered-statistic CFAR using the rank-th smallest training cell (1-based)."""
    n, k = n_train, rank
    if n_integrated == 1:
        def pfa_of(alpha: float) -> float:
            i = np.arange(k)
            return float(np.prod((n - i) / (n - i + alpha)))
    else:
        P = n_integrated
        y = np.linspace(1e-6, P * 12.0 + 40.0, 6000)
        sf = _gamma_sf(y, P)
        cdf = 1.0 - sf
        logpdf = (P - 1) * np.log(y) - y - lgamma(P)
        log_coef = lgamma(n + 1) - lgamma(k) - lgamma(n - k + 1)
        with np.errstate(divide="ignore"):
            f_k = np.exp(log_coef + (k - 1) * np.log(cdf) + (n - k) * np.log(sf) + logpdf)
        f_k = np.nan_to_num(f_k)
        dy = y[1] - y[0]

        def pfa_of(alpha: float) -> float:
            return float(np.sum(_gamma_sf(alpha * y, P) * f_k) * dy)

    lo, hi = 0.0, 1.0
    while pfa_of(hi) > pfa:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pfa_of(mid) > pfa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ helpers
def _box_sum(x: np.ndarray, h0: int, h1: int, wrap0: bool) -> np.ndarray:
    """Sum of x over a (2*h0+1) x (2*h1+1) box around every cell. Axis 0 optionally circular, axis 1 zero padded."""
    xp = np.pad(x, ((h0, h0), (0, 0)), mode="wrap" if wrap0 else "constant")
    xp = np.pad(xp, ((0, 0), (h1, h1)))
    s = np.zeros((xp.shape[0] + 1, xp.shape[1] + 1))
    s[1:, 1:] = xp.cumsum(0).cumsum(1)
    w0, w1 = 2 * h0 + 1, 2 * h1 + 1
    return s[w0:, w1:] - s[:-w0, w1:] - s[w0:, :-w1] + s[:-w0, :-w1]


def local_maxima(power: np.ndarray, wrap0: bool = True) -> np.ndarray:
    """True where a cell is >= its 8 neighbours."""
    p = np.pad(power, ((1, 1), (0, 0)), mode="wrap" if wrap0 else "edge")
    p = np.pad(p, ((0, 0), (1, 1)), mode="constant", constant_values=-np.inf)
    centre = p[1:-1, 1:-1]
    peak = np.ones_like(centre, bool)
    for d0 in (-1, 0, 1):
        for d1 in (-1, 0, 1):
            if d0 == 0 and d1 == 0:
                continue
            peak &= centre >= p[1 + d0:p.shape[0] - 1 + d0, 1 + d1:p.shape[1] - 1 + d1]
    return peak


# --------------------------------------------------------------------- CFAR
@dataclass
class CFARResult:
    detections: np.ndarray  # (M, 2) int, (axis0 index, axis1 index)
    threshold: np.ndarray  # same shape as the input map
    noise: np.ndarray  # noise estimate per cell
    mask: np.ndarray  # bool, cells above threshold (before peak grouping)
    snr_db: np.ndarray  # (M,) detection power over local noise estimate


def cfar_2d(
    power: np.ndarray,
    guard: tuple[int, int] = (2, 2),
    train: tuple[int, int] = (4, 8),
    pfa: float = 1e-6,
    method: str = "ca",
    n_integrated: int = 1,
    os_rank: float = 0.75,
    wrap0: bool = True,
    peaks_only: bool = True,
    min_index1: int = 0,
) -> CFARResult:
    """2D CFAR on a power map (linear units).

    guard, train: half-widths along (axis 0, axis 1). For a range-Doppler map
        axis 0 is Doppler (circular, so wrap0=True) and axis 1 is range.
    method: "ca" cell averaging, or "os" ordered statistic (robust when two
        targets sit close together).
    min_index1: ignore cells with axis-1 index below this (e.g. leakage in the
        first range bins).
    """
    g0, g1 = guard
    t0, t1 = train
    h0, h1 = g0 + t0, g1 + t1

    if method == "ca":
        ones = np.ones_like(power, dtype=float)
        s_outer = _box_sum(power, h0, h1, wrap0)
        s_inner = _box_sum(power, g0, g1, wrap0)
        n_outer = _box_sum(ones, h0, h1, wrap0)
        n_inner = _box_sum(ones, g0, g1, wrap0)
        n_train = np.rint(n_outer - n_inner).astype(int)
        noise = (s_outer - s_inner) / np.maximum(n_train, 1)
        alpha = np.zeros_like(noise)
        for n in np.unique(n_train):
            if n > 0:
                alpha[n_train == n] = ca_cfar_scale(int(n), pfa, n_integrated)
        threshold = alpha * noise
    elif method == "os":
        if wrap0:
            pp = np.pad(power, ((h0, h0), (0, 0)), mode="wrap")
        else:
            pp = np.pad(power, ((h0, h0), (0, 0)), constant_values=np.nan)
        pp = np.pad(pp, ((0, 0), (h1, h1)), constant_values=np.nan)
        win = np.lib.stride_tricks.sliding_window_view(pp, (2 * h0 + 1, 2 * h1 + 1))
        ring = np.ones((2 * h0 + 1, 2 * h1 + 1), bool)
        ring[t0:t0 + 2 * g0 + 1, t1:t1 + 2 * g1 + 1] = False
        cells = win[..., ring]  # (N0, N1, n_ring)
        cells = np.sort(cells, axis=-1)  # NaNs last
        n_valid = np.sum(~np.isnan(cells), axis=-1)
        rank = np.clip(np.rint(os_rank * n_valid).astype(int), 1, None)
        noise = np.take_along_axis(cells, (rank - 1)[..., None], axis=-1)[..., 0]
        alpha = np.zeros_like(noise)
        for n, k in set(zip(n_valid.ravel().tolist(), rank.ravel().tolist())):
            alpha[(n_valid == n) & (rank == k)] = os_cfar_scale(int(n), int(k), pfa, n_integrated)
        threshold = alpha * noise
    else:
        raise ValueError(f"unknown CFAR method {method!r}")

    mask = power > threshold
    if min_index1:
        mask[:, :min_index1] = False
    hits = mask & local_maxima(power, wrap0) if peaks_only else mask
    det = np.argwhere(hits)
    snr = 10 * np.log10(power[det[:, 0], det[:, 1]] / np.maximum(noise[det[:, 0], det[:, 1]], 1e-300))
    return CFARResult(det, threshold, noise, mask, snr)


def ca_cfar_1d(x: np.ndarray, guard: int, train: int, pfa: float, n_integrated: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """1D cell-averaging CFAR, edges handled with the cells that exist. Returns (mask, threshold)."""
    res = cfar_2d(x[None, :], guard=(0, guard), train=(0, train), pfa=pfa, n_integrated=n_integrated,
                  wrap0=False, peaks_only=False)
    return res.mask[0], res.threshold[0]
