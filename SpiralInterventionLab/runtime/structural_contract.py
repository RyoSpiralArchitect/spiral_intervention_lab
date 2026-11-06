"""Opt-in norm-relative write contract for one structural edit at a boundary.

The contract bounds the write, not the outcome. One edit moves one last-token
residual by at most r of that residual's norm, for one token, toward a source
structural lexeme. Names and words that occur in required terms are never
targets, and the policy rejects anything outside the per-prompt allowlist.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import re
from typing import Any, Mapping, Sequence

import torch

from . import clause_radial, clause_structure
from .compiler import StepContext, compile_expr
from .edit_budget import NORM_RELATIVE_WRITE_CONTRACT
from .policy import HarnessPolicy, WriteContract
from .response_probe import identity, tensor_identity


CONTRACT = NORM_RELATIVE_WRITE_CONTRACT
DEFAULT_RELATIVE_WRITE_CAP = 0.08
# The structural rescue v1 pair rule: the first three eligible tokens in source order.
MAX_SOURCES = 3
PATCH_COMPONENTS = ("common", "common_parallel", "common_orthogonal")
# Blend all the way toward the source; the guard keeps the write within the cap.
PATCH_ALPHA = 1.0

# Absolute winner-gap closure per unit of relative write, measured by the
# calibration-only cap sweep of 2026-11-06 (manifest ebd542c5…, three v1
# calibration cases, caps 0.04-0.12). Coverage is these three lexemes at these
# three boundaries only; unlisted pieces carry no estimate.
READOUT_CLOSURE_SLOPE: Mapping[str, dict[str, float]] = {
    ",": {"mean": 22.5, "min": 20.0, "max": 26.0},
    " while": {"mean": 51.0, "min": 49.0, "max": 54.0},
    ".\n": {"mean": 43.0, "min": 36.0, "max": 50.0},
}
REACH_BASIS = "structural_rescue_cap_sweep_v1_calibration_n3_caps_0.04-0.12"


def protected_lexemes(required_terms: Sequence[str]) -> frozenset[str]:
    """Structural lexemes that also occur in a required term, such as 'before dusk'."""
    words = {word.casefold() for term in required_terms
             for word in re.findall(r"[A-Za-z]+|[^\sA-Za-z]", str(term))}
    return frozenset(words & clause_structure.SOURCE_LEXEMES)


def eligible_sources(worker: Any, required_terms: Sequence[str]) -> list[dict[str, Any]]:
    protected = protected_lexemes(required_terms)
    tokens = worker._current_token_tensor().flatten().detach().cpu().tolist()
    sources = []
    for candidate in clause_structure.source_candidates(worker):
        if str(candidate["piece"]).strip().casefold() in protected:
            continue
        if any(p < 0 or p >= len(tokens) or tokens[p] != candidate["token_id"]
               for p in candidate["source_positions"]):
            raise ValueError("source_position_token_mismatch")
        sources.append(deepcopy(candidate))
    return sources


def policy_for(worker: Any, required_terms: Sequence[str], *,
               cap: float = DEFAULT_RELATIVE_WRITE_CAP) -> HarnessPolicy:
    """Default policy plus a contract whose allowlist is this prompt's structure."""
    sources = eligible_sources(worker, required_terms)
    return HarnessPolicy(write_contract=WriteContract(
        relative_write_cap=float(cap),
        readout_token_ids=tuple(int(c["token_id"]) for c in sources),
        source_positions=tuple(int(p) for c in sources for p in c["source_positions"]),
    ))


def surface(worker: Any, layer: int, site: str = "resid_pre") -> dict[str, Any]:
    for item in worker.surface_catalog:
        target = item.target
        if (target.kind == "activation" and target.site == site and target.layer == layer
                and target.token.mode == "last"):
            return {"surface_id": item.surface_id, "site": site, "layer": int(layer)}
    raise ValueError("contract_surface_unavailable")


def _ref(worker: Any, where: Mapping[str, Any], token: dict[str, Any]) -> dict[str, Any]:
    return {"ref": {"scope": "runtime", "worker": worker.worker_id, "tensor": where["site"],
                    "layer": where["layer"], "token": token}}


def _context(worker: Any) -> StepContext:
    cached = getattr(worker.runtime_state, "last_tokens", None)
    if not isinstance(cached, torch.Tensor) or not torch.equal(cached, worker._current_token_tensor()):
        raise ValueError("cache_not_aligned_to_current_prefix")
    return StepContext(packet={}, runtime_state=worker.runtime_state, adapter=worker.adapter,
                       traces={}, stats={}, active_edits={})


def _source_view(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {"token_id": int(candidate["token_id"]), "piece": candidate["piece"],
            "position": int(candidate["source_positions"][0])}


def _finish(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {**descriptor, "recipe_id": identity("structural-contract:", descriptor),
            "production_apply_allowed": False}


def patch_recipes(worker: Any, sources: Sequence[Mapping[str, Any]], where: Mapping[str, Any], *,
                  cap: float = DEFAULT_RELATIVE_WRITE_CAP) -> list[dict[str, Any]]:
    """The v1 radial components for each pair, blended under the contract."""
    ctx = _context(worker)
    current = _ref(worker, where, {"mode": "last"})
    target = compile_expr(current)(ctx).detach().cpu().double()
    site_norm = float(target.norm())
    recipes = []
    for pair in combinations(sources[:MAX_SOURCES], 2):
        views = [_source_view(c) for c in pair]
        refs = [_ref(worker, where, {"mode": "index", "value": v["position"]}) for v in views]
        expressions = clause_radial.expressions(*refs, current)
        shape = clause_radial.geometry(compile_expr(expressions["common"])(ctx), target.float())
        for name in PATCH_COMPONENTS:
            value = compile_expr(expressions[name])(ctx).detach().cpu()
            requested = float((value.double() - target).norm()) / site_norm
            recipes.append(_finish({
                "contract": CONTRACT, "kind": "patch", "component": name,
                "dsl_formula": clause_radial.FORMULAS[name], "surface": dict(where), "sources": views,
                "source_expression": expressions[name], "op": "activation_patch", "alpha": PATCH_ALPHA,
                "relative_write_cap": float(cap), "site_norm": site_norm,
                "requested_relative_write": requested, "expected_relative_write": min(requested, float(cap)),
                "geometry": shape, "input_identity": tensor_identity(value)}))
    return recipes


def readout_recipes(worker: Any, sources: Sequence[Mapping[str, Any]], where: Mapping[str, Any], *,
                    cap: float = DEFAULT_RELATIVE_WRITE_CAP) -> list[dict[str, Any]]:
    """Push toward one source structural token, requesting exactly the cap."""
    ctx = _context(worker)
    site_norm = float(compile_expr(_ref(worker, where, {"mode": "last"}))(ctx).detach().cpu().double().norm())
    recipes = []
    for candidate in sources[:MAX_SOURCES]:
        view = _source_view(candidate)
        expression = {"fn": "readout_direction", "target_token_ids": [view["token_id"]], "normalize": True}
        recipes.append(_finish({
            "contract": CONTRACT, "kind": "readout", "component": "readout", "surface": dict(where),
            "sources": [view], "source_expression": expression, "op": "resid_add",
            "alpha": float(cap) * site_norm, "relative_write_cap": float(cap), "site_norm": site_norm,
            "requested_relative_write": float(cap), "expected_relative_write": float(cap),
            "input_identity": tensor_identity(compile_expr(expression)(ctx).detach().cpu())}))
    return recipes


def command(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """One revertible, one-token edit; no absolute step, so the guard is the cap."""
    op = ({"kind": "activation_patch", "mode": "blend", "alpha": recipe["alpha"]}
          if recipe["op"] == "activation_patch" else {"kind": "resid_add", "alpha": recipe["alpha"]})
    return {"version": "0.1", "decision": "apply", "edits": [{
        "id": recipe["recipe_id"], "target": {"surface_id": recipe["surface"]["surface_id"]},
        "source": {"dtype": "vector", "expr": deepcopy(recipe["source_expression"])},
        "op": op, "budget": {"ttl_steps": 1, "revertible": True},
        "meta": {"write_contract": CONTRACT, "relative_write_cap": recipe["relative_write_cap"],
                 "diagnostic_only": True, "production_apply_allowed": False},
    }]}
