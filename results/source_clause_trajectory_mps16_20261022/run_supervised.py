"""One labelled-capture MPS replication with a resource-bounded owned child."""
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
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    if (directory / "run.log").exists() or (directory / "measurement").exists():
        raise FileExistsError("prior attempt exists")
    gib = 1024 ** 3
    if shutil.disk_usage(root).free < 20 * gib:
        raise RuntimeError("MPS preflight requires 20 GiB free disk")
    command = ["python3", "-u", "-m", "SpiralInterventionLab.examples.replay_source_clause_trajectory",
               "--source-jsonl", "results/entity_probe_refresh_llama_20261014_retry1/live/c1.jsonl",
               "--source-manifest", "results/entity_probe_refresh_llama_20261014_retry1/manifest.json",
               "--reference-receipt", "results/source_clause_specificity_llama_20261021_aligned/specificity.json",
               "--worker-model-path", args.worker_model_path, "--position", "8",
               "--source-token-ids", "11,1418", "--device", "mps", "--dtype", "float16",
               "--torch-threads", "4", "--output-dir", str(directory / "measurement")]
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
              "wall_time_limit_seconds": 600, "stop_free_disk_gib": 12,
              "status": "starting", "resource_samples": [],
              "controller_calls": 0, "production_apply_allowed": False}

    def save():
        (directory / "supervision.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    start = time.monotonic()
    with (directory / "run.log").open("x") as log:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        report.update(status="running", child_pid=child.pid)
        try:
            while child.poll() is None:
                elapsed, free = time.monotonic() - start, shutil.disk_usage(root).free / gib
                report["resource_samples"].append({"seconds": round(elapsed, 2), "free_disk_gib": round(free, 2)})
                if free < 12 or elapsed > 600:
                    report["stop_reason"] = "disk_floor" if free < 12 else "time_limit"
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
