"""The Verilog CFAR core against its bit-exact Python model, in Icarus Verilog."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mira.hdl_model import alpha_fixed, cfar_ca_fixed

HDL = Path(__file__).resolve().parents[1] / "hdl"
pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None, reason="needs Icarus Verilog")


@pytest.mark.parametrize("guard,train_log2,alpha,gap", [(2, 3, 2.5, 20), (1, 2, 4.0, 0), (3, 4, 1.75, 50)])
def test_cfar_core_is_bit_exact(tmp_path, guard, train_log2, alpha, gap):
    rng = np.random.default_rng(guard * 10 + train_log2)
    lines = []
    for n in rng.integers(2 * ((1 << train_log2) + guard) + 3, 140, 8):
        x = rng.exponential(300, n)
        x[rng.integers(0, n, 4)] += rng.uniform(300, 20000, 4)
        lines.append(np.clip(x, 0, 65535).astype(int))
    a = alpha_fixed(alpha)
    with open(tmp_path / "in.txt", "w") as f:
        for x in lines:
            for k, v in enumerate(x):
                f.write(f"{v:04x} {int(k == len(x) - 1)}\n")
    exe = tmp_path / "tb"
    subprocess.run(["iverilog", "-g2005", "-o", str(exe), f"-Ptb_cfar_ca.GUARD={guard}",
                    f"-Ptb_cfar_ca.TRAIN_LOG2={train_log2}", str(HDL / "cfar_ca.v"), str(HDL / "tb_cfar_ca.v")],
                   check=True)
    subprocess.run(["vvp", "-n", str(exe), f"+IN={tmp_path / 'in.txt'}", f"+OUT={tmp_path / 'out.txt'}",
                    f"+ALPHA={a}", f"+GAP={gap}"], check=True, capture_output=True)
    got = [tuple(map(int, line.split())) for line in (tmp_path / "out.txt").read_text().splitlines()]
    want = cfar_ca_fixed(lines, guard, train_log2, a, 4)
    assert len(got) == len(want)
    assert got == want
    assert sum(w[3] for w in want) > 0
