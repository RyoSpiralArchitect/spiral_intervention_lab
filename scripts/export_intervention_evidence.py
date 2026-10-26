"""Export the explicitly selected October pilot evidence without modifying originals.

Only synthetic-task receipts and named derived audits are exported. Model weights,
raw vectors, provider supervision logs and the unrelated live runs stay local.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import tarfile


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public_bytes(data: bytes, root: Path) -> bytes:
    text = data.decode().replace(str(root.resolve()) + "/", "")
    if re.search(r"/Users/|/home/|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._-]{20,}", text):
        raise ValueError("unreviewed_local_path_or_secret_like_content")
    return text.encode()


def export(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    lineage = {}

    def copy(relative: Path, destination: Path) -> None:
        original = (root / relative).read_bytes()
        published = public_bytes(original, root)
        path = output / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(published)
        lineage[destination.as_posix()] = {"original_path": relative.as_posix(),
            "original_sha256": sha(original), "published_sha256": sha(published),
            "transform": "repository_prefix_removed" if published != original else "byte_identical"}

    completed = Path("results/structural_rescue_pilot_20261027_retry1")
    for name in ("manifest.json", "results.json", "execution_plan.json", "environment.json"):
        copy(completed / name, Path("pilot") / name)
    for path in sorted((root / completed).glob("*_observation.json")):
        copy(path.relative_to(root), Path("pilot") / path.name)
    for name in ("manifest.json", "results.json", "execution_plan.json"):
        copy(Path("results/structural_rescue_pilot_20261027") / name,
             Path("initial_attempt") / name)
    choice = Path("results/source_radial_controller_binding_20261026")
    for name in ("summary.json", "selected_source_summary.json", "fresh_choice_summary.json",
                 "choice_audit.jsonl", "selected_source_audit.jsonl", "fresh_choice_audit.jsonl"):
        copy(choice / name, Path("source_choice") / name)

    # Publish only the four historical source files named in the execution plan,
    # preserving their exact bytes. The full original archive remains local.
    analyzer = runpy.run_path(str(root / "SpiralInterventionLab/examples/summarize_structural_rescue.py"))
    execution = json.loads((output / "pilot/execution_plan.json").read_text())
    archived = root / completed / "source_snapshot.tgz"
    destination = output / "pilot/source_snapshot.tgz"
    with tarfile.open(archived, "r:gz") as source, tarfile.open(destination, "w:gz") as target:
        for path, expected in sorted(execution["code_sha256"].items()):
            name = analyzer["snapshot_member"](path)
            item = source.extractfile(name)
            if item is None:
                raise ValueError("missing_archived_source:" + name)
            data = item.read()
            if sha(data) != expected or public_bytes(data, root) != data:
                raise ValueError("source_not_exact_or_not_public_safe:" + name)
            member = tarfile.TarInfo(name)
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))
    lineage["pilot/source_snapshot.tgz"] = {"original_path": str(completed / "source_snapshot.tgz"),
        "original_sha256": sha(archived.read_bytes()), "published_sha256": sha(destination.read_bytes()),
        "transform": "four_hash_verified_source_members_only"}
    analyzer["main"]([str(output / "pilot")])
    publication = {"schema": "intervention_evidence_publication_v1", "lineage": lineage,
        "scope": "historical_pilot_and_derived_choice_audits_not_new_measurements",
        "not_published": ["raw tensors", "full historical source archive", "source-choice raw receipts and host logs"],
        "production_apply_allowed": False}
    (output / "publication.json").write_text(json.dumps(publication, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.repo_root.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
