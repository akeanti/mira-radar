# HDL

Verilog cores for the parts of the radar chain that belong in an FPGA, each
with a bit-exact Python model in [`mira/hdl_model.py`](../mira/hdl_model.py)
used as its golden reference.

## `cfar_ca.v`: streaming CA-CFAR

```
in_data ─►┌──────────────── shift register, 2·(TRAIN+GUARD)+1 cells ────────────────┐
          │  TRAIN leading │ GUARD │ CUT │ GUARD │ TRAIN lagging                    │
          └───────┬────────────────────┬─────────────────────┬───────────────────────┘
                  │ +new −oldest       │                     │ +entering −oldest
               sum_lead             cell under test        sum_lag
                  └──────────┬─────────┘                     │
                             └──── (sum >> log2(2·TRAIN)) · α >> FRAC ──► threshold ──► CUT > threshold
```

| Parameter | Default | Meaning |
|---|---|---|
| `DATA_W` | 16 | width of the unsigned power samples |
| `GUARD` | 2 | guard cells per side |
| `TRAIN_LOG2` | 3 | 2^TRAIN_LOG2 training cells per side (power of two, so the mean is a shift) |
| `ALPHA_W`, `ALPHA_FRAC` | 8, 4 | threshold factor α in unsigned fixed point (Q4.4) |
| `INDEX_W` | 12 | lines up to 4096 cells |

- One sample per clock when `in_valid` is high; idle cycles are fine.
- `in_last` marks the end of a line (a range line of the range-Doppler map, for example).
- One result per cell whose whole window lies inside its line, `out_index` giving its position, 2 clocks after the newest sample of its window.
- Threshold: `((sum_lead + sum_lag) >> (TRAIN_LOG2 + 1)) * alpha >> ALPHA_FRAC`, detection if the cell is strictly greater. `mira.hdl_model.alpha_fixed(2.5)` converts a real α to the register value.

### Simulate

```bash
pytest tests/test_hdl.py -v        # needs iverilog on PATH
```

The test generates random lines with exponential noise and targets, streams
them through the core with random idle cycles, and compares every output word
with `cfar_ca_fixed()`. By hand:

```bash
iverilog -g2005 -o tb hdl/cfar_ca.v hdl/tb_cfar_ca.v
vvp tb +IN=in.txt +OUT=out.txt +ALPHA=40 +GAP=20
```

### Synthesise

```bash
yosys -p "read_verilog hdl/cfar_ca.v; synth_xilinx -family xc7 -top cfar_ca; stat"
yosys -p "read_verilog hdl/cfar_ca.v; synth_ecp5 -top cfar_ca; stat"
yosys -p "read_verilog hdl/cfar_ca.v; synth_ice40 -top cfar_ca; stat"
```

Yosys 0.33, default parameters:

| Target | Logic | Flip-flops | Other |
|---|---|---|---|
| Xilinx 7-series | 121 LUT + 18 INV | 442 | 1 DSP48E1, 23 CARRY4 |
| Lattice ECP5 | 107 LUT4 + 4 PFUMX | 442 | 1 MULT18X18D, 58 CCU2C |
| Lattice iCE40 | 511 LUT4 | 442 | 125 SB_CARRY |

Most flip-flops are the 21-cell window. No timing closure has been done yet;
the critical path is the multiply-compare in stage 2, which a third pipeline
stage would cut if needed.

## Next

A streaming radix-2 range FFT feeding this core, and a cocotb testbench that
pushes a whole simulated frame from `mira.FMCWSimulator` through both.
