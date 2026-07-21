from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__:
    from .summarize_activation_patch_jsonl import collect_activation_patch_rows
else:  # Support direct execution alongside the module-style CLI.
    from summarize_activation_patch_jsonl import collect_activation_patch_rows


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


def _iter_jsonl_paths(paths: Sequence[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            resolved.extend(sorted(item for item in path.rglob("*.jsonl") if item.is_file()))
        elif path.is_file():
            resolved.append(path)
    return resolved


def _event_summary(paths: Sequence[str | Path]) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    provider_attempt_count = 0
    provider_parse_ok_count = 0
    provider_models: Counter[str] = Counter()
    diagnostic_sequence: list[dict[str, Any]] = []
    activation_patch_frontier_steps: list[int] = []
    parse_errors = 0
    for path in _iter_jsonl_paths(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                try:
                    event = json.loads(line)
                except Exception:
                    parse_errors += 1
                    continue
                if not isinstance(event, Mapping):
                    continue
                event_name = str(event.get("event") or "unknown")
                event_counts[event_name] += 1
                if event_name == "controller_provider_attempt":
                    provider_attempt_count += 1
                    if bool(event.get("parse_ok", False)):
                        provider_parse_ok_count += 1
                    model = str(event.get("model") or "")
                    if model:
                        provider_models[model] += 1
                if event_name == "controller_diagnostic_request":
                    diagnostic_sequence.append(
                        {
                            "step": event.get("step"),
                            "diagnostic": event.get("diagnostic"),
                            "next_evidence_needed": event.get("next_evidence_needed"),
                            "operator_recipe_expansion_mode": event.get("operator_recipe_expansion_mode"),
                        }
                    )
                if event_name == "controller_observation":
                    packet = event.get("packet") if isinstance(event.get("packet"), Mapping) else {}
                    hints = packet.get("strategy_hints") if isinstance(packet.get("strategy_hints"), Mapping) else {}
                    if str(hints.get("diagnostic_frontier_request") or "") == "activation_patch_candidate_review":
                        step = _as_int(event.get("step"))
                        if step is not None:
                            activation_patch_frontier_steps.append(step)

    return {
        "event_counts": dict(sorted(event_counts.items())),
        "event_parse_errors": parse_errors,
        "provider_attempt_count": provider_attempt_count,
        "provider_parse_ok_count": provider_parse_ok_count,
        "provider_models": dict(sorted(provider_models.items())),
        "diagnostic_sequence": diagnostic_sequence,
        "activation_patch_frontier_steps": activation_patch_frontier_steps,
    }


_KEY_FIELDS = (
    "objective_bundle_key",
    "intended_term",
    "target_piece",
    "target_piece_token_id",
    "site",
    "layer",
    "source_localization",
    "operator_axis",
    "step_size",
    "seed_source",
    "seed_recipe_name",
    "proxy_reliability_status",
)

_RELAXED_KEY_FIELDS = (
    "objective_bundle_key",
    "intended_term",
    "site",
    "layer",
    "source_localization",
    "operator_axis",
    "step_size",
)


def _row_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("objective_bundle_key") or "unknown"),
        str(row.get("intended_term") or "unknown"),
        str(row.get("target_piece") or "unknown"),
        str(row.get("target_piece_token_id") if row.get("target_piece_token_id") is not None else "unknown"),
        str(row.get("site") or "unknown"),
        str(row.get("layer") if row.get("layer") is not None else "unknown"),
        str(row.get("source_localization") or "unknown"),
        str(row.get("operator_axis") or row.get("operator_recipe_expansion_mode") or "unknown"),
        str(row.get("step_size") if row.get("step_size") is not None else "unknown"),
        str(row.get("seed_source") or "unknown"),
        str(row.get("seed_recipe_name") or "unknown"),
        str(row.get("proxy_reliability_status") or row.get("attribution_reliability_status") or "unknown"),
    )


def _relaxed_row_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("objective_bundle_key") or "unknown"),
        str(row.get("intended_term") or "unknown"),
        str(row.get("site") or "unknown"),
        str(row.get("layer") if row.get("layer") is not None else "unknown"),
        str(row.get("source_localization") or "unknown"),
        str(row.get("operator_axis") or row.get("operator_recipe_expansion_mode") or "unknown"),
        str(row.get("step_size") if row.get("step_size") is not None else "unknown"),
    )


def _key_dict(key: tuple[str, ...], fields: Sequence[str] = _KEY_FIELDS) -> dict[str, str]:
    return {field: key[index] if index < len(key) else "unknown" for index, field in enumerate(fields)}


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target_mass_values = [value for value in (_as_float(row.get("target_mass_delta")) for row in rows) if value is not None]
    gap_values = [value for value in (_as_float(row.get("gap_delta")) for row in rows) if value is not None]
    return {
        "activation_patch_row_count": len(rows),
        "by_actual_delta_class": dict(sorted(Counter(str(row.get("actual_delta_class") or "unknown") for row in rows).items())),
        "by_response_effect_role": dict(sorted(Counter(str(row.get("response_effect_role") or "unknown") for row in rows).items())),
        "by_proxy_calibration_status": dict(sorted(Counter(str(row.get("proxy_calibration_status") or "unknown") for row in rows).items())),
        "by_subspace_family": dict(sorted(Counter(str(row.get("subspace_family") or "unknown") for row in rows).items())),
        "subspace_consistent_row_count": sum(1 for row in rows if bool(row.get("subspace_consistent_seed", False))),
        "hint_only_unreliable_row_count": sum(
            1 for row in rows if str(row.get("proxy_calibration_status") or "") == "candidate_hint_only_unreliable"
        ),
        "target_top20_hit_rows": sum(1 for row in rows if (_as_int(row.get("target_top20_hit_delta")) or 0) > 0),
        "max_target_mass_delta": max(target_mass_values) if target_mass_values else None,
        "min_gap_delta": min(gap_values) if gap_values else None,
    }


def _bucket_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = _summarize_rows(rows)
    summary["recipe_names"] = sorted({str(row.get("recipe_name") or "") for row in rows if str(row.get("recipe_name") or "")})[:8]
    return summary


def _group_by_key(
    rows: Sequence[dict[str, Any]],
    key_fn: Any,
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return grouped


def _comparison_delta(left_summary: Mapping[str, Any], right_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_count": int(right_summary["activation_patch_row_count"])
        - int(left_summary["activation_patch_row_count"]),
        "subspace_consistent_rows": int(right_summary["subspace_consistent_row_count"])
        - int(left_summary["subspace_consistent_row_count"]),
        "target_top20_hit_rows": int(right_summary["target_top20_hit_rows"])
        - int(left_summary["target_top20_hit_rows"]),
        "max_target_mass_delta": (
            (_as_float(right_summary.get("max_target_mass_delta")) or 0.0)
            - (_as_float(left_summary.get("max_target_mass_delta")) or 0.0)
        ),
        "min_gap_delta": (
            None
            if right_summary.get("min_gap_delta") is None or left_summary.get("min_gap_delta") is None
            else float(right_summary["min_gap_delta"]) - float(left_summary["min_gap_delta"])
        ),
    }


def _paired_comparison(
    left_by_key: Mapping[tuple[str, ...], Sequence[Mapping[str, Any]]],
    right_by_key: Mapping[tuple[str, ...], Sequence[Mapping[str, Any]]],
    *,
    fields: Sequence[str],
    left_label: str,
    right_label: str,
    top_k: int,
) -> list[dict[str, Any]]:
    paired = []
    for key in sorted(set(left_by_key) & set(right_by_key)):
        left_summary = _bucket_summary(left_by_key[key])
        right_summary = _bucket_summary(right_by_key[key])
        paired.append(
            {
                "key": _key_dict(key, fields),
                left_label: left_summary,
                right_label: right_summary,
                "delta": _comparison_delta(left_summary, right_summary),
            }
        )
    paired.sort(
        key=lambda row: (
            -abs(float(row["delta"].get("max_target_mass_delta") or 0.0)),
            str(row["key"]),
        )
    )
    return paired[: max(1, int(top_k))]


def _piece_counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field) or "unknown") for row in rows).items()))


def _target_piece_divergence_report(
    left_by_key: Mapping[tuple[str, ...], Sequence[Mapping[str, Any]]],
    right_by_key: Mapping[tuple[str, ...], Sequence[Mapping[str, Any]]],
    *,
    left_label: str,
    right_label: str,
    top_k: int,
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for key in sorted(set(left_by_key) | set(right_by_key)):
        left_rows = list(left_by_key.get(key, ()))
        right_rows = list(right_by_key.get(key, ()))
        left_pieces = _piece_counts(left_rows, "target_piece")
        right_pieces = _piece_counts(right_rows, "target_piece")
        left_piece_ids = _piece_counts(left_rows, "target_piece_token_id")
        right_piece_ids = _piece_counts(right_rows, "target_piece_token_id")
        left_binding_status = _piece_counts(left_rows, "target_piece_binding_status")
        right_binding_status = _piece_counts(right_rows, "target_piece_binding_status")
        if (
            left_pieces == right_pieces
            and left_piece_ids == right_piece_ids
            and len(left_pieces) <= 1
            and len(right_pieces) <= 1
        ):
            continue
        left_summary = _bucket_summary(left_rows) if left_rows else _summarize_rows([])
        right_summary = _bucket_summary(right_rows) if right_rows else _summarize_rows([])
        report.append(
            {
                "key": _key_dict(key, _RELAXED_KEY_FIELDS),
                left_label: {
                    "target_piece_counts": left_pieces,
                    "target_piece_token_id_counts": left_piece_ids,
                    "target_piece_binding_status_counts": left_binding_status,
                    "seed_source_counts": _piece_counts(left_rows, "seed_source"),
                    **left_summary,
                },
                right_label: {
                    "target_piece_counts": right_pieces,
                    "target_piece_token_id_counts": right_piece_ids,
                    "target_piece_binding_status_counts": right_binding_status,
                    "seed_source_counts": _piece_counts(right_rows, "seed_source"),
                    **right_summary,
                },
                "divergence": {
                    "left_unique_target_piece_count": len(left_pieces),
                    "right_unique_target_piece_count": len(right_pieces),
                    "target_piece_sets_match": left_pieces == right_pieces,
                    "target_piece_token_id_sets_match": left_piece_ids == right_piece_ids,
                    "binding_status_sets_match": left_binding_status == right_binding_status,
                    "seed_source_sets_match": _piece_counts(left_rows, "seed_source")
                    == _piece_counts(right_rows, "seed_source"),
                },
            }
        )
    report.sort(
        key=lambda row: (
            -(
                int(row[left_label].get("activation_patch_row_count", 0))
                + int(row[right_label].get("activation_patch_row_count", 0))
            ),
            str(row["key"]),
        )
    )
    return report[: max(1, int(top_k))]


def compare_activation_patch_runs(
    left_paths: Sequence[str | Path],
    right_paths: Sequence[str | Path],
    *,
    left_label: str = "left",
    right_label: str = "right",
    top_k: int = 12,
) -> dict[str, Any]:
    left_rows, left_metadata = collect_activation_patch_rows(left_paths)
    right_rows, right_metadata = collect_activation_patch_rows(right_paths)
    left_events = _event_summary(left_paths)
    right_events = _event_summary(right_paths)

    left_by_key = _group_by_key(left_rows, _row_key)
    right_by_key = _group_by_key(right_rows, _row_key)
    left_by_relaxed_key = _group_by_key(left_rows, _relaxed_row_key)
    right_by_relaxed_key = _group_by_key(right_rows, _relaxed_row_key)

    shared_keys = sorted(set(left_by_key) & set(right_by_key))
    left_only_keys = sorted(set(left_by_key) - set(right_by_key))
    right_only_keys = sorted(set(right_by_key) - set(left_by_key))
    relaxed_shared_keys = sorted(set(left_by_relaxed_key) & set(right_by_relaxed_key))
    relaxed_left_only_keys = sorted(set(left_by_relaxed_key) - set(right_by_relaxed_key))
    relaxed_right_only_keys = sorted(set(right_by_relaxed_key) - set(left_by_relaxed_key))
    paired = _paired_comparison(
        left_by_key,
        right_by_key,
        fields=_KEY_FIELDS,
        left_label=left_label,
        right_label=right_label,
        top_k=top_k,
    )
    relaxed_paired = _paired_comparison(
        left_by_relaxed_key,
        right_by_relaxed_key,
        fields=_RELAXED_KEY_FIELDS,
        left_label=left_label,
        right_label=right_label,
        top_k=top_k,
    )

    return _json_safe(
        {
            "left_label": left_label,
            "right_label": right_label,
            "strict_recipe_key_fields": list(_KEY_FIELDS),
            "relaxed_recipe_key_fields": list(_RELAXED_KEY_FIELDS),
            "left": {
                **left_metadata,
                **left_events,
                **_summarize_rows(left_rows),
            },
            "right": {
                **right_metadata,
                **right_events,
                **_summarize_rows(right_rows),
            },
            "strict_shared_recipe_key_count": len(shared_keys),
            "shared_recipe_key_count": len(shared_keys),
            "strict_left_only_recipe_key_count": len(left_only_keys),
            "left_only_recipe_key_count": len(left_only_keys),
            "strict_right_only_recipe_key_count": len(right_only_keys),
            "right_only_recipe_key_count": len(right_only_keys),
            "relaxed_shared_recipe_key_count": len(relaxed_shared_keys),
            "relaxed_left_only_recipe_key_count": len(relaxed_left_only_keys),
            "relaxed_right_only_recipe_key_count": len(relaxed_right_only_keys),
            "left_only_recipe_keys": [
                _key_dict(key)
                for key in left_only_keys[:top_k]
            ],
            "right_only_recipe_keys": [
                _key_dict(key)
                for key in right_only_keys[:top_k]
            ],
            "relaxed_left_only_recipe_keys": [
                _key_dict(key, _RELAXED_KEY_FIELDS)
                for key in relaxed_left_only_keys[:top_k]
            ],
            "relaxed_right_only_recipe_keys": [
                _key_dict(key, _RELAXED_KEY_FIELDS)
                for key in relaxed_right_only_keys[:top_k]
            ],
            "paired_recipe_comparison": paired,
            "strict_paired_recipe_comparison": paired,
            "relaxed_paired_recipe_comparison": relaxed_paired,
            "target_piece_divergence_report": _target_piece_divergence_report(
                left_by_relaxed_key,
                right_by_relaxed_key,
                left_label=left_label,
                right_label=right_label,
                top_k=top_k,
            ),
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare activation_patch diagnostic rows across two JSONL runs.")
    parser.add_argument("--left", nargs="+", required=True, help="Left JSONL files or result directories")
    parser.add_argument("--right", nargs="+", required=True, help="Right JSONL files or result directories")
    parser.add_argument("--left-label", default="left", help="Label for the left run")
    parser.add_argument("--right-label", default="right", help="Label for the right run")
    parser.add_argument("--top-k", type=int, default=12, help="Number of paired/delta rows to keep")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = compare_activation_patch_runs(
        args.left,
        args.right,
        left_label=args.left_label,
        right_label=args.right_label,
        top_k=args.top_k,
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
