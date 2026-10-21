# Contributing

Thanks for wanting to make MIRA better. Bug reports, questions, figures that
explain something better, new scenarios and new processing stages are all
welcome.

## Getting set up

```bash
git clone https://github.com/akeanti/mira-radar
cd mira-radar
pip install -e ".[dev]"
pytest
```

`tests/test_hdl.py` needs [Icarus Verilog](https://steveicarus.github.io/iverilog/)
(`apt install iverilog`, `brew install icarus-verilog`) and is skipped without it.

## Ground rules

- **Keep it readable.** MIRA is meant to be read. Prefer a clear loop over a
  clever one-liner, and say *why* in a comment when the maths isn't obvious.
- **NumPy and Matplotlib only** in the core package. Optional extras (like
  `sounddevice` for the sonar) are imported lazily.
- **Measure it.** A new stage comes with a test that checks it against theory
  or ground truth, not just that it runs. Look at `tests/test_cfar.py` for
  the spirit.
- **Change the console, show it.** If you touch `viz/`, attach a screenshot
  or a short GIF (`mira render`) to the pull request.

## Good first issues

- A new scenario in `mira/scenarios.py` (a roundabout, a parking lot, a drone swarm)
- A new example script in `examples/` that explains one concept with one figure
- Running `mira sonar` on your laptop and reporting whether it works, with the model
- Window comparisons (Hann, Blackman, Kaiser) for range and Doppler sidelobes

## Bigger pieces

See the roadmap in the README. If you want to take one on, open an issue first
so we can agree on the shape before you write a lot of code.
