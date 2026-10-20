"""One bounded local replay; provide the local checkpoint path on the CLI."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-model-path", required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    output = here / "specificity.json"
    log_path = here / "runtime.stdout.log"
    if output.exists() or log_path.exists():
        raise FileExistsError("use a new directory for a retry")
    free = lambda: shutil.disk_usage("/System/Volumes/Data").free / 2**30
    if free() < 12:
        raise RuntimeError("preflight requires at least 12 GiB free")
    command = [
        sys.executable, "-u", "-m", "SpiralInterventionLab.examples.replay_source_clause_diagnostic",
        "--source-jsonl", "results/entity_probe_refresh_llama_20261014_retry1/live/c1.jsonl",
        "--source-manifest", "results/entity_probe_refresh_llama_20261014_retry1/manifest.json",
        "--worker-model-path", args.worker_model_path, "--positions", "8",
        "--measure-source-operators", "--source-specificity-controls", "--output", str(output),
    ]
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "disk_stop_gib": 8, "timeout_seconds": 900, "preflight_free_gib": free(),
        "scope": "normal_cap_source_specificity_not_task_or_controller_uplift",
        "controller_calls": 0, "production_apply_allowed": False,
        "predeclared_controls": ["self_copy", "radial_shrink"],
        "physical_replay_bound_one_site": 25,
        "physical_replay_bound_two_sites": 37,
        "diagnostic_slot_bound": 7,
    }
    (here / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start, reason, process = time.monotonic(), None, None
    with log_path.open("x") as log:
        def event(kind, **details):
            row = {"event": kind, **details}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                elapsed = int(time.monotonic() - start)
                available = free()
                if available < 8 or elapsed > 900:
                    reason = "low_disk" if available < 8 else "timeout"
                    break
                if elapsed % 30 < 5:
                    event("resource_checkpoint", elapsed_seconds=elapsed, disk_free_gib=available)
                time.sleep(5)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            status = {"returncode": process.returncode, "stop_reason": reason,
                      "elapsed_seconds": int(time.monotonic() - start), "disk_free_gib": free()}
            event("supervised_end", **status)
            (here / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return 0 if process.returncode == 0 and reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
