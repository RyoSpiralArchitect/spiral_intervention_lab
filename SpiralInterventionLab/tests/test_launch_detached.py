from pathlib import Path

from SpiralInterventionLab.examples import launch_detached


def test_script_runs_commands_in_order_and_stops_at_the_first_failure(tmp_path):
    text = launch_detached.script([["-m", "pkg.first", "--x", "a b"], ["-m", "pkg.second"]], tmp_path)
    lines = text.splitlines()
    first = next(i for i, line in enumerate(lines) if "pkg.first" in line)
    second = next(i for i, line in enumerate(lines) if "pkg.second" in line)
    assert first < second and "caffeinate -i -s" in lines[first] and "< /dev/null" in lines[first]
    assert "'a b'" in lines[first]
    assert any(line.startswith("if [ $status -ne 0 ]; then exit") for line in lines[first:second])
    assert str(tmp_path / "exit_status.txt") in text and lines[-1].startswith("echo done")
