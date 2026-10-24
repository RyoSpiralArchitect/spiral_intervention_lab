"""Resource-bounded, single-child offline diagnostic; never terminates other jobs."""
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
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    root = output.parents[1]
    log_path = output / "run.log"
    if log_path.exists() or (output / "measurement").exists():
        raise FileExistsError("refusing to overwrite a prior attempt")
    gib = 1024 ** 3
    if shutil.disk_usage(root).free < 32 * gib:
        raise RuntimeError("CPU float32 preflight requires 32 GiB free disk")
    command = ["python3", "-u", "-m", "SpiralInterventionLab.examples.replay_source_clause_trajectory",
               "--source-jsonl", "results/entity_probe_refresh_llama_20261014_retry1/live/c1.jsonl",
               "--source-manifest", "results/entity_probe_refresh_llama_20261014_retry1/manifest.json",
               "--reference-receipt", "results/source_clause_specificity_llama_20261021_aligned/specificity.json",
               "--worker-model-path", args.worker_model_path, "--position", "8",
               "--source-token-ids", "11,1418", "--device", "cpu", "--dtype", "float32",
               "--torch-threads", "4", "--output-dir", str(output / "measurement")]
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
              "wall_time_limit_seconds": 900, "stop_free_disk_gib": 16,
              "controller_calls": 0, "production_apply_allowed": False,
              "status": "starting", "resource_samples": []}

    def save():
        (output / "supervision.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    start = time.monotonic()
    with log_path.open("x") as log:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        report.update(status="running", child_pid=child.pid)
        save()
        try:
            while child.poll() is None:
                elapsed, free = time.monotonic() - start, shutil.disk_usage(root).free / gib
                report["resource_samples"].append({"seconds": round(elapsed, 2), "free_disk_gib": round(free, 2)})
                reason = "disk_floor" if free < 16 else "time_limit" if elapsed > 900 else None
                if reason:
                    report["stop_reason"] = reason
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
    print(json.dumps({k: v for k, v in report.items() if k != "resource_samples"}))
    return child.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
