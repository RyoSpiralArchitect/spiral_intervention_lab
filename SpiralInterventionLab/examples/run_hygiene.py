"""Preflight checks and provenance for long model runs on this Mac.

A run refuses to start on battery, below its disk headroom, or from a modified
git tree, and records what started it: argv, PID, session, HEAD, power, the
PYTORCH_MPS_* environment and the user-site startup files that every Python
process executes. Telemetry rows capture memory, swap and disk per case.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EVALUATOR_SOURCES = ("SpiralInterventionLab/tasks/structural_rescue.py", "SpiralInterventionLab/tasks/role_binding.py",
                     "SpiralInterventionLab/tasks/language_tasks.py")
# The floor protects the machine; swap grows by about 5 GiB while the 3B model is
# loaded. The user set 5 GiB of launch headroom on 2026-11-03, while another
# session's build was writing to disk; a run that reaches the floor stops cleanly.
LOAD_HEADROOM_GIB = 5


def evaluator_sha256() -> str:
    digest = hashlib.sha256()
    for relative in EVALUATOR_SOURCES:
        digest.update(relative.encode() + b"\0" + (REPO / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def power_source() -> str:
    try:
        first = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return "ac" if "AC Power" in first else "battery" if "Battery Power" in first else "unknown"


def startup_files() -> dict[str, str]:
    """sha256 of the sitecustomize and .pth files Python runs before any repo code."""
    roots = [Path(p) for p in [site.getusersitepackages(), *site.getsitepackages()] if p]
    found = {}
    for root in roots:
        for path in sorted([*root.glob("*.pth"), *root.glob("sitecustomize.py"), *root.glob("usercustomize.py")]):
            if path.is_file():
                found[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def swap_usage() -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def preflight(output: Path, *, floor_gib: float, allow_dirty: bool = False) -> None:
    free = shutil.disk_usage(output).free / 2**30
    if free < floor_gib + LOAD_HEADROOM_GIB:
        raise RuntimeError(f"free_disk_{free:.1f}_gib_below_floor_plus_load_headroom_{floor_gib + LOAD_HEADROOM_GIB:g}")
    if power_source() != "ac":
        raise RuntimeError("not_on_ac_power")
    if not allow_dirty and _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("modified_git_tree")


def provenance() -> dict[str, Any]:
    return {"argv": list(sys.argv), "pid": os.getpid(), "sid": os.getsid(0), "python": sys.version,
            "git_head": _git("rev-parse", "HEAD"),
            "git_clean_tracked": not _git("status", "--porcelain", "--untracked-files=no"),
            "power": power_source(), "evaluator_sha256": evaluator_sha256(),
            "pytorch_mps_env": {k: v for k, v in os.environ.items() if k.startswith("PYTORCH_MPS")},
            "startup_files_sha256": startup_files()}


def telemetry(output: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"free_disk_gib": round(shutil.disk_usage(output).free / 2**30, 3), "swap": swap_usage()}
    try:
        import torch
        if torch.backends.mps.is_available():
            row.update(mps_allocated_gib=round(torch.mps.current_allocated_memory() / 2**30, 3),
                       mps_driver_gib=round(torch.mps.driver_allocated_memory() / 2**30, 3))
    except Exception as exc:  # telemetry must never stop a run
        row["telemetry_error"] = type(exc).__name__
    return row
