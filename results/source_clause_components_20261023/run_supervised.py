"""One fixed seven-condition experiment per numerical configuration, no retries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-model-path", required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    cpu = args.device == "cpu"
    dtype, initial_floor, stop_floor, time_limit = ("float32", 32, 16, 900) if cpu else ("float16", 20, 12, 600)
    target = directory / ("cpu32" if cpu else "mps16")
    target.mkdir(exist_ok=False)
    gib = 1024 ** 3
    command = ["python3", "-u", "-m", "SpiralInterventionLab.examples.replay_source_clause_trajectory",
               "--source-jsonl", "results/entity_probe_refresh_llama_20261014_retry1/live/c1.jsonl",
               "--source-manifest", "results/entity_probe_refresh_llama_20261014_retry1/manifest.json",
               "--reference-receipt", "results/source_clause_specificity_llama_20261021_aligned/specificity.json",
               "--worker-model-path", args.worker_model_path, "--position", "8",
               "--source-token-ids", "11,1418", "--decompose-components", "--device", args.device,
               "--dtype", dtype, "--torch-threads", "4", "--output-dir", str(target / "measurement")]
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
              "wall_time_limit_seconds": time_limit, "stop_free_disk_gib": stop_floor,
              "preflight_free_disk_gib": shutil.disk_usage(root).free / gib,
              "status": "preflight", "resource_samples": [], "controller_calls": 0,
              "production_apply_allowed": False}

    def save():
        (target / "supervision.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    if report["preflight_free_disk_gib"] < initial_floor:
        report.update(status="blocked", stop_reason="preflight_disk_floor")
        save()
        return 2
    start = time.monotonic()
    with (target / "run.log").open("x") as log:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        report.update(status="running", child_pid=child.pid)
        try:
            while child.poll() is None:
                elapsed, free = time.monotonic() - start, shutil.disk_usage(root).free / gib
                report["resource_samples"].append({"seconds": round(elapsed, 2), "free_disk_gib": round(free, 2),
                    "swap": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()})
                if free < stop_floor or elapsed > time_limit:
                    report["stop_reason"] = "disk_floor" if free < stop_floor else "time_limit"
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
                save()
                time.sleep(10)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            report["stop_reason"] = "supervisor_interrupted"
            raise
        finally:
            report.update(status="complete" if child.returncode == 0 else "stopped_or_failed",
                          returncode=child.returncode, elapsed_seconds=round(time.monotonic() - start, 2),
                          final_free_disk_gib=round(shutil.disk_usage(root).free / gib, 2))
            save()
    print(json.dumps({k: v for k, v in report.items() if k != "resource_samples"}), flush=True)
    return child.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
