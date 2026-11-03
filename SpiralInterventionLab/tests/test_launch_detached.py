from pathlib import Path

from SpiralInterventionLab.examples import launch_detached


def test_script_runs_commands_in_order_and_stops_at_the_first_failure(tmp_path):
    text = launch_detached.script([["-m", "pkg.first", "--x", "a b"], ["-m", "pkg.second"]], tmp_path)
    lines = text.splitlines()
    first = next(i for i, line in enumerate(lines) if "pkg.first" in line)
    second = next(i for i, line in enumerate(lines) if "pkg.second" in line)
    assert first < second and "caffeinate -i -s" in lines[first] and "< /dev/null" in lines[first]
    assert "'a b'" in lines[first]
    assert any(line.startswith("if [ $rc -ne 0 ]; then exit") for line in lines[first:second])
    assert str(tmp_path / "exit_status.txt") in text and lines[-1].startswith("echo done")


def test_script_records_exit_codes_under_zsh(tmp_path):
    import shutil
    import subprocess
    import pytest
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is not installed")
    text = launch_detached.script([["-c", "raise SystemExit(3)"], ["-c", "print('never')"]], tmp_path)
    # caffeinate exists only on macOS; the exit bookkeeping is what is under test.
    path = tmp_path / "run.zsh"
    path.write_text(text.replace("caffeinate -i -s ", ""))
    assert subprocess.run([zsh, str(path)]).returncode == 3
    assert (tmp_path / "exit_status.txt").read_text() == "step=0 exit=3\n"
