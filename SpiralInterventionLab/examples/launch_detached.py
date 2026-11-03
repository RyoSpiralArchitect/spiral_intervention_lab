"""Run one or more runner commands detached from the launching session.

A session's end kills its background children, and a child that inherits the
session's stdout dies at its next print. This launcher starts a new session
with stdin from /dev/null, logs to files, keeps the Mac awake with
``caffeinate -i -s``, runs the commands in order, stops at the first failure,
and writes each command's exit status. It returns as soon as the session starts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]


def script(commands: list[list[str]], log_dir: Path) -> str:
    lines = ["#!/bin/zsh", f"cd {shlex.quote(str(REPO))}"]
    for index, argv in enumerate(commands):
        out, err = log_dir / f"step_{index}.stdout.log", log_dir / f"step_{index}.stderr.log"
        lines += [f"caffeinate -i -s {shlex.join([sys.executable, '-u', *argv])} "
                  f"< /dev/null > {shlex.quote(str(out))} 2> {shlex.quote(str(err))}",
                  # zsh reserves $status (read-only), so the exit code lives in rc.
                  "rc=$?", f"echo \"step={index} exit=$rc\" >> {shlex.quote(str(log_dir / 'exit_status.txt'))}",
                  "if [ $rc -ne 0 ]; then exit $rc; fi"]
    lines.append(f"echo done >> {shlex.quote(str(log_dir / 'exit_status.txt'))}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True,
                        help="JSON list of argv lists, each run as python3 -u <argv>, e.g. ['-m', 'pkg.mod', ...]")
    args = parser.parse_args(argv)
    commands = json.loads(args.commands.read_text())
    if not commands or not all(isinstance(c, list) and c and all(isinstance(a, str) for a in c) for c in commands):
        raise ValueError("commands must be a non-empty JSON list of argv lists")
    log_dir = args.log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    path = log_dir / "run.zsh"
    path.write_text(script(commands, log_dir))
    child = subprocess.Popen(["/bin/zsh", str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    (log_dir / "launcher.json").write_text(json.dumps({"pid": child.pid, "script": str(path)}, indent=2) + "\n")
    print(json.dumps({"pid": child.pid, "log_dir": str(log_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
