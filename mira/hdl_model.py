"""Bit-exact models of the Verilog cores in hdl/, used as their golden reference."""

from __future__ import annotations

import numpy as np


def cfar_ca_fixed(
    lines: list[np.ndarray],
    guard: int = 2,
    train_log2: int = 3,
    alpha: int = 40,
    alpha_frac: int = 4,
) -> list[tuple[int, int, int, int, int]]:
    """Integer CA-CFAR exactly as hdl/cfar_ca.v computes it.

    lines: unsigned integer power samples, one array per line.
    Returns (index, data, threshold, detect, last) for every cell whose full
    window lies inside its line, in streaming order.
    """
    train = 1 << train_log2
    c = train + guard
    out = []
    for x in lines:
        x = np.asarray(x, np.int64)
        n = len(x)
        for j in range(c, n - c):
            lead = int(x[j + guard + 1:j + c + 1].sum())  # newer samples
            lag = int(x[j - c:j - guard].sum())  # older samples
            noise = (lead + lag) >> (train_log2 + 1)
            thr = (noise * alpha) >> alpha_frac
            out.append((j, int(x[j]), thr, int(x[j] > thr), int(j == n - c - 1)))
    return out


def alpha_fixed(alpha: float, alpha_frac: int = 4, alpha_w: int = 8) -> int:
    """Quantise a CFAR threshold factor to the core's fixed-point format."""
    q = int(round(alpha * (1 << alpha_frac)))
    if not 0 < q < (1 << alpha_w):
        raise ValueError(f"alpha {alpha} does not fit in {alpha_w} bits with {alpha_frac} fractional bits")
    return q
