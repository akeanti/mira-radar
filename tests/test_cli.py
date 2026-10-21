from mira.cli import main


def test_specs(capsys):
    main(["specs", "--markdown"])
    out = capsys.readouterr().out
    assert "MIRA-77" in out and "Range resolution" in out


def test_render_png(tmp_path):
    out = tmp_path / "frame.png"
    main(["render", "--scenario", "two-targets", "--seconds", "0.2", "--start", "0", "--out", str(out)])
    assert out.stat().st_size > 10_000
