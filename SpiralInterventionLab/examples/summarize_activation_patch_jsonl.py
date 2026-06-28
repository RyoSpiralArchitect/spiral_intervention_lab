from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _iter_jsonl_paths(paths: Sequence[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            resolved.extend(sorted(item for item in path.rglob("*.jsonl") if item.is_file()))
        elif path.is_file():
            resolved.append(path)
    return resolved


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        return int(value)
    except Exception:
        return None


def _is_activation_patch_row(row: Mapping[str, Any]) -> bool:
    looks_like_candidate = any(
        row.get(key) not in (None, "", [])
        for key in (
            "activation_patch_site",
            "activation_patch_layer",
            "activation_patch_surface_id",
            "activation_patch_source_localization",
        )
    )
    return looks_like_candidate and (
        str(row.get("diagnostic_family") or "") == "activation_patch"
        or str(row.get("activation_patch_op_kind") or "") == "activation_patch"
        or str(row.get("evidence_kind") or "") == "activation_patch_certification"
    )


def _walk_activation_patch_rows(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if _is_activation_patch_row(value):
            yield value
        pool = value.get("activation_patch_candidate_pool")
        if isinstance(pool, Sequence) and not isinstance(pool, (str, bytes, bytearray)):
            for row in pool:
                if isinstance(row, Mapping) and _is_activation_patch_row(row):
                    yield row
        evidence_rows = value.get("evidence_rows")
        if isinstance(evidence_rows, Sequence) and not isinstance(evidence_rows, (str, bytes, bytearray)):
            for row in evidence_rows:
                if isinstance(row, Mapping) and _is_activation_patch_row(row):
                    yield row
        for inner in value.values():
            yield from _walk_activation_patch_rows(inner)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _walk_activation_patch_rows(item)


def _row_identity(path: Path, line_no: int, row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(path),
        int(line_no),
        row.get("objective_bundle_key") or row.get("bundle_key"),
        row.get("recipe_name"),
        row.get("operator_recipe_id"),
        row.get("activation_patch_surface_id"),
        row.get("activation_patch_site"),
        row.get("activation_patch_layer"),
        row.get("activation_patch_source_localization"),
        row.get("activation_patch_alpha"),
        row.get("activation_patch_step_size"),
        row.get("actual_delta_class"),
        row.get("target_mass_delta"),
        row.get("target_top20_hit_delta"),
        row.get("target_top20_threshold_gap_delta"),
    )


def _compact_row(path: Path, line_no: int, row: Mapping[str, Any]) -> dict[str, Any]:
    target_mass = _as_float(row.get("target_mass_delta"))
    gap_delta = _as_float(row.get("target_top20_threshold_gap_delta"))
    return {
        "path": str(path),
        "line": int(line_no),
        "objective_bundle_key": row.get("objective_bundle_key") or row.get("bundle_key"),
        "actuator_bundle_key": row.get("actuator_bundle_key"),
        "intended_term": row.get("intended_term") or row.get("objective_term"),
        "site": row.get("activation_patch_site"),
        "layer": _as_int(row.get("activation_patch_layer")),
        "source_localization": row.get("activation_patch_source_localization"),
        "operator_axis": row.get("operator_axis"),
        "operator_recipe_expansion_mode": row.get("operator_recipe_expansion_mode"),
        "seed_source": row.get("activation_patch_seed_source"),
        "forced_seed": bool(row.get("activation_patch_forced_seed", False)),
        "base_localization": row.get("activation_patch_base_localization"),
        "contrast_mode": row.get("activation_patch_contrast_mode"),
        "contrast_scale": _as_float(row.get("activation_patch_contrast_scale")),
        "alpha": _as_float(row.get("activation_patch_alpha")),
        "step_size": _as_float(row.get("activation_patch_step_size")),
        "surface_step_size_cap": _as_float(
            row.get("activation_patch_surface_step_size_cap", row.get("surface_step_size_cap"))
        ),
        "step_size_cap_release_allowed": bool(
            row.get(
                "activation_patch_step_size_cap_release_allowed",
                row.get("step_size_cap_release_allowed", False),
            )
        ),
        "hook_call_count": _as_int(row.get("activation_patch_hook_call_count")),
        "selected_token_count": _as_int(row.get("activation_patch_selected_token_count")),
        "source_norm": _as_float(row.get("activation_patch_source_norm")),
        "target_norm_before": _as_float(row.get("activation_patch_target_norm_before")),
        "target_norm_after": _as_float(row.get("activation_patch_target_norm_after")),
        "source_target_cosine": _as_float(row.get("activation_patch_source_target_cosine")),
        "source_target_delta_norm": _as_float(row.get("activation_patch_source_target_delta_norm")),
        "raw_blend_delta_norm": _as_float(row.get("activation_patch_raw_blend_delta_norm")),
        "blend_delta_norm": _as_float(row.get("activation_patch_blend_delta_norm", row.get("blend_delta_norm"))),
        "step_size_clip_saturated": row.get(
            "activation_patch_step_size_clip_saturated",
            row.get("step_size_clip_saturated"),
        ),
        "status": row.get("status"),
        "actual_delta_class": row.get("actual_delta_class"),
        "actuator_class": row.get("actuator_class"),
        "effect_role": row.get("effect_role"),
        "safety_role": row.get("safety_role"),
        "target_mass_delta": target_mass,
        "target_top20_hit_delta": _as_int(row.get("target_top20_hit_delta")),
        "gap_delta": gap_delta,
        "focus_rank_delta": _as_int(row.get("focus_rank_delta")),
        "repeat_delta": _as_float(row.get("repeat_delta")),
        "entropy_delta": _as_float(row.get("entropy_delta")),
        "top1_margin_delta": _as_float(row.get("top1_margin_delta")),
        "self_delta": _as_float(row.get("self_delta")),
        "cross_delta": _as_float(row.get("cross_delta")),
        "alignment_margin": _as_float(row.get("alignment_margin")),
        "positive_traits": list(row.get("positive_traits") or [])
        if isinstance(row.get("positive_traits"), Sequence) and not isinstance(row.get("positive_traits"), (str, bytes, bytearray))
        else [],
        "blocked_by": list(row.get("blocked_by") or [])
        if isinstance(row.get("blocked_by"), Sequence) and not isinstance(row.get("blocked_by"), (str, bytes, bytearray))
        else [],
        "recipe_name": row.get("recipe_name"),
        "operator_recipe_id": row.get("operator_recipe_id"),
    }


def _best_rows(rows: Sequence[Mapping[str, Any]], *, top_k: int) -> dict[str, list[dict[str, Any]]]:
    def finite_value(row: Mapping[str, Any], key: str, default: float) -> float:
        value = _as_float(row.get(key))
        return default if value is None else value

    return {
        "target_mass_delta": [
            dict(row) for row in sorted(rows, key=lambda item: finite_value(item, "target_mass_delta", -1e9), reverse=True)[:top_k]
        ],
        "gap_delta": [
            dict(row) for row in sorted(rows, key=lambda item: finite_value(item, "gap_delta", 1e9))[:top_k]
        ],
        "focus_rank_delta": [
            dict(row) for row in sorted(rows, key=lambda item: finite_value(item, "focus_rank_delta", -1e9), reverse=True)[:top_k]
        ],
    }


def summarize_activation_patch_jsonl(paths: Sequence[str | Path], *, top_k: int = 8) -> dict[str, Any]:
    jsonl_paths = _iter_jsonl_paths(paths)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    parse_errors = 0
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                try:
                    event = json.loads(line)
                except Exception:
                    parse_errors += 1
                    continue
                for row in _walk_activation_patch_rows(event):
                    identity = _row_identity(path, line_no, row)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    rows.append(_compact_row(path, line_no, row))

    by_actual_delta_class = Counter(str(row.get("actual_delta_class") or "unknown") for row in rows)
    by_actuator_class = Counter(str(row.get("actuator_class") or "unknown") for row in rows)
    by_site = Counter(str(row.get("site") or "unknown") for row in rows)
    by_source_localization = Counter(str(row.get("source_localization") or "unknown") for row in rows)
    by_operator_axis = Counter(str(row.get("operator_axis") or "unknown") for row in rows)

    grouped: dict[tuple[str, int | None, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("site") or "unknown"),
                row.get("layer") if isinstance(row.get("layer"), int) else None,
                str(row.get("source_localization") or "unknown"),
                str(row.get("actual_delta_class") or "unknown"),
            )
        ].append(row)

    matrix = []
    for (site, layer, source_localization, actual_delta_class), group_rows in sorted(grouped.items()):
        target_mass_values = [value for value in (_as_float(row.get("target_mass_delta")) for row in group_rows) if value is not None]
        gap_values = [value for value in (_as_float(row.get("gap_delta")) for row in group_rows) if value is not None]
        blend_values = [value for value in (_as_float(row.get("blend_delta_norm")) for row in group_rows) if value is not None]
        step_size_values = [value for value in (_as_float(row.get("step_size")) for row in group_rows) if value is not None]
        cosine_values = [value for value in (_as_float(row.get("source_target_cosine")) for row in group_rows) if value is not None]
        matrix.append(
            {
                "site": site,
                "layer": layer,
                "source_localization": source_localization,
                "actual_delta_class": actual_delta_class,
                "count": len(group_rows),
                "max_target_mass_delta": max(target_mass_values) if target_mass_values else None,
                "min_gap_delta": min(gap_values) if gap_values else None,
                "max_blend_delta_norm": max(blend_values) if blend_values else None,
                "max_step_size": max(step_size_values) if step_size_values else None,
                "mean_source_target_cosine": (sum(cosine_values) / len(cosine_values)) if cosine_values else None,
                "hooked_rows": sum(1 for row in group_rows if (_as_int(row.get("hook_call_count")) or 0) > 0),
                "clip_saturated_rows": sum(1 for row in group_rows if bool(row.get("step_size_clip_saturated"))),
                "top20_hits": sum(1 for row in group_rows if (_as_int(row.get("target_top20_hit_delta")) or 0) > 0),
            }
        )

    return _json_safe(
        {
            "input_paths": [str(path) for path in jsonl_paths],
            "jsonl_file_count": len(jsonl_paths),
            "parse_errors": parse_errors,
            "activation_patch_row_count": len(rows),
            "by_actual_delta_class": dict(sorted(by_actual_delta_class.items())),
            "by_actuator_class": dict(sorted(by_actuator_class.items())),
            "by_site": dict(sorted(by_site.items())),
            "by_source_localization": dict(sorted(by_source_localization.items())),
            "by_operator_axis": dict(sorted(by_operator_axis.items())),
            "matrix": matrix,
            "response_curve_points": [
                dict(row)
                for row in sorted(
                    (
                        row
                        for row in rows
                        if str(row.get("operator_axis") or "")
                        == "activation_patch_cap_release_response_curve"
                    ),
                    key=lambda item: (
                        str(item.get("site") or ""),
                        _as_float(item.get("step_size")) or 0.0,
                    ),
                )
            ],
            "best_rows": _best_rows(rows, top_k=max(1, int(top_k))),
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize activation_patch diagnostic rows from JSONL logs.")
    parser.add_argument("paths", nargs="+", help="JSONL files or result directories to scan")
    parser.add_argument("--top-k", type=int, default=8, help="Number of best rows to keep per metric")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = summarize_activation_patch_jsonl(args.paths, top_k=args.top_k)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
