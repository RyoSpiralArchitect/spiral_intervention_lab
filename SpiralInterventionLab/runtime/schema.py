from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


ACTIVATION_SITES = {"resid_pre", "resid_post", "mlp_out"}
CACHE_SITES = {"k_cache", "v_cache"}
WEIGHT_MODULES = {"mlp_out", "attn_out"}
REF_TENSORS = {"hidden", "resid_pre", "resid_post", "mlp_out", "k_cache", "v_cache"}
STATS = {"mean", "std", "ema"}
SCOPES = {"runtime", "trace", "stats"}
POOL_MODES = {"last", "mean"}
OP_KINDS = {"resid_add", "kv_mix", "rank1_patch"}
KV_WHICH = {"k", "v", "kv"}
EXPR_FNS = {
    "add",
    "sub",
    "mean",
    "scale",
    "normalize",
    "zero_mean",
    "sign",
    "clip_norm",
    "mix",
    "project_parallel",
    "project_orthogonal",
}


class SchemaError(ValueError):
    pass


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be an object")
    return value


def _as_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SchemaError(f"{name} must be an array")
    return value


def _require_key(data: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in data:
        raise SchemaError(f"{name}.{key} is required")
    return data[key]


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{name} must be a non-empty string")
    return value


def _optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaError(f"{name} must be a string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{name} must be a boolean")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{name} must be an integer")
    return value


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, name)


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{name} must be a number")
    return float(value)


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _require_float(value, name)


def _optional_string_array(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    seq = _as_sequence(value, name)
    return tuple(_require_str(item, f"{name}[{idx}]") for idx, item in enumerate(seq))


def _validate_expr(expr: Any, name: str = "expr", depth: int = 0) -> Mapping[str, Any]:
    if depth > 8:
        raise SchemaError(f"{name} exceeds max depth 8")
    node = _as_mapping(expr, name)
    if "ref" in node:
        ref = _as_mapping(node["ref"], f"{name}.ref")
        scope = _require_str(_require_key(ref, "scope", f"{name}.ref"), f"{name}.ref.scope")
        if scope not in SCOPES:
            raise SchemaError(f"{name}.ref.scope must be one of {sorted(SCOPES)}")
        tensor = _require_str(_require_key(ref, "tensor", f"{name}.ref"), f"{name}.ref.tensor")
        if tensor not in REF_TENSORS:
            raise SchemaError(f"{name}.ref.tensor must be one of {sorted(REF_TENSORS)}")
        _require_int(_require_key(ref, "layer", f"{name}.ref"), f"{name}.ref.layer")
        if "worker" in ref:
            _optional_str(ref.get("worker"), f"{name}.ref.worker")
        if scope == "trace":
            _require_str(_require_key(ref, "trace_id", f"{name}.ref"), f"{name}.ref.trace_id")
        if "token" in ref:
            TokenSelector.from_dict(ref["token"])
        if "head" in ref:
            _optional_int(ref.get("head"), f"{name}.ref.head")
        if "stat" in ref:
            stat = _optional_str(ref.get("stat"), f"{name}.ref.stat")
            if stat is not None and stat not in STATS:
                raise SchemaError(f"{name}.ref.stat must be one of {sorted(STATS)}")
        return node

    fn = _require_str(_require_key(node, "fn", name), f"{name}.fn")
    if fn not in EXPR_FNS:
        raise SchemaError(f"{name}.fn must be one of {sorted(EXPR_FNS)}")

    if fn in {"add", "sub", "mean"}:
        args = _as_sequence(_require_key(node, "args", name), f"{name}.args")
        if len(args) < 2:
            raise SchemaError(f"{name}.args must contain at least two expressions")
        for idx, arg in enumerate(args):
            _validate_expr(arg, f"{name}.args[{idx}]", depth + 1)
        return node

    if fn == "scale":
        _require_float(_require_key(node, "by", name), f"{name}.by")
        _validate_expr(_require_key(node, "arg", name), f"{name}.arg", depth + 1)
        return node

    if fn in {"normalize", "zero_mean", "sign"}:
        _validate_expr(_require_key(node, "arg", name), f"{name}.arg", depth + 1)
        return node

    if fn == "clip_norm":
        _require_float(_require_key(node, "max_norm", name), f"{name}.max_norm")
        _validate_expr(_require_key(node, "arg", name), f"{name}.arg", depth + 1)
        return node

    if fn == "mix":
        _require_float(_require_key(node, "alpha", name), f"{name}.alpha")
        _validate_expr(_require_key(node, "left", name), f"{name}.left", depth + 1)
        _validate_expr(_require_key(node, "right", name), f"{name}.right", depth + 1)
        return node

    if fn in {"project_parallel", "project_orthogonal"}:
        _validate_expr(_require_key(node, "arg", name), f"{name}.arg", depth + 1)
        _validate_expr(_require_key(node, "basis", name), f"{name}.basis", depth + 1)
        return node

    raise SchemaError(f"unsupported expr fn {fn}")


@dataclass(frozen=True)
class TokenSelector:
    mode: str
    value: int | None = None
    start: int | None = None
    end: int | None = None
    pool: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "TokenSelector":
        data = _as_mapping(value, "token")
        mode = _require_str(_require_key(data, "mode", "token"), "token.mode")
        if mode == "last":
            return cls(mode=mode)
        if mode == "index":
            return cls(mode=mode, value=_require_int(_require_key(data, "value", "token"), "token.value"))
        if mode == "span":
            pool = _require_str(_require_key(data, "pool", "token"), "token.pool")
            if pool not in POOL_MODES:
                raise SchemaError(f"token.pool must be one of {sorted(POOL_MODES)}")
            return cls(
                mode=mode,
                start=_require_int(_require_key(data, "start", "token"), "token.start"),
                end=_require_int(_require_key(data, "end", "token"), "token.end"),
                pool=pool,
            )
        raise SchemaError("token.mode must be one of ['last', 'index', 'span']")


@dataclass(frozen=True)
class SurfaceTargetRef:
    surface_id: str

    @classmethod
    def from_dict(cls, value: Any) -> "SurfaceTargetRef":
        data = _as_mapping(value, "target")
        return cls(surface_id=_require_str(_require_key(data, "surface_id", "target"), "target.surface_id"))


@dataclass(frozen=True)
class ActivationTarget:
    kind: str
    worker: str
    site: str
    layer: int
    token: TokenSelector

    @classmethod
    def from_dict(cls, value: Any) -> "ActivationTarget":
        data = _as_mapping(value, "target")
        site = _require_str(_require_key(data, "site", "target"), "target.site")
        if site not in ACTIVATION_SITES:
            raise SchemaError(f"target.site must be one of {sorted(ACTIVATION_SITES)}")
        return cls(
            kind="activation",
            worker=_require_str(_require_key(data, "worker", "target"), "target.worker"),
            site=site,
            layer=_require_int(_require_key(data, "layer", "target"), "target.layer"),
            token=TokenSelector.from_dict(_require_key(data, "token", "target")),
        )


@dataclass(frozen=True)
class CacheTarget:
    kind: str
    worker: str
    site: str
    layer: int
    token: TokenSelector
    head: int | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "CacheTarget":
        data = _as_mapping(value, "target")
        site = _require_str(_require_key(data, "site", "target"), "target.site")
        if site not in CACHE_SITES:
            raise SchemaError(f"target.site must be one of {sorted(CACHE_SITES)}")
        return cls(
            kind="cache",
            worker=_require_str(_require_key(data, "worker", "target"), "target.worker"),
            site=site,
            layer=_require_int(_require_key(data, "layer", "target"), "target.layer"),
            head=_optional_int(data.get("head"), "target.head"),
            token=TokenSelector.from_dict(_require_key(data, "token", "target")),
        )


@dataclass(frozen=True)
class WeightTarget:
    kind: str
    worker: str
    module: str
    layer: int

    @classmethod
    def from_dict(cls, value: Any) -> "WeightTarget":
        data = _as_mapping(value, "target")
        module = _require_str(_require_key(data, "module", "target"), "target.module")
        if module not in WEIGHT_MODULES:
            raise SchemaError(f"target.module must be one of {sorted(WEIGHT_MODULES)}")
        return cls(
            kind="weight",
            worker=_require_str(_require_key(data, "worker", "target"), "target.worker"),
            module=module,
            layer=_require_int(_require_key(data, "layer", "target"), "target.layer"),
        )


Target = ActivationTarget | CacheTarget | WeightTarget
TargetRef = SurfaceTargetRef | Target


def parse_target_ref(value: Any) -> TargetRef:
    if isinstance(value, str) and value:
        return SurfaceTargetRef(surface_id=value)
    data = _as_mapping(value, "target")
    if "surface_id" in data:
        return SurfaceTargetRef.from_dict(data)
    kind = _require_str(_require_key(data, "kind", "target"), "target.kind")
    if kind == "activation":
        return ActivationTarget.from_dict(data)
    if kind == "cache":
        return CacheTarget.from_dict(data)
    if kind == "weight":
        return WeightTarget.from_dict(data)
    raise SchemaError("target.kind must be one of ['activation', 'cache', 'weight']")


@dataclass(frozen=True)
class VectorSource:
    dtype: str
    expr: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Any) -> "VectorSource":
        data = _as_mapping(value, "source")
        expr = _validate_expr(_require_key(data, "expr", "source"), "source.expr")
        return cls(dtype="vector", expr=expr)


@dataclass(frozen=True)
class Rank1Source:
    dtype: str
    u: Mapping[str, Any]
    v: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Any) -> "Rank1Source":
        data = _as_mapping(value, "source")
        return cls(
            dtype="rank1",
            u=_validate_expr(_require_key(data, "u", "source"), "source.u"),
            v=_validate_expr(_require_key(data, "v", "source"), "source.v"),
        )


@dataclass(frozen=True)
class CachePairSource:
    dtype: str
    k: Mapping[str, Any] | None = None
    v: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "CachePairSource":
        data = _as_mapping(value, "source")
        k_expr = data.get("k")
        v_expr = data.get("v")
        if k_expr is None and v_expr is None:
            raise SchemaError("source must define at least one of source.k or source.v")
        return cls(
            dtype="cache_pair",
            k=_validate_expr(k_expr, "source.k") if k_expr is not None else None,
            v=_validate_expr(v_expr, "source.v") if v_expr is not None else None,
        )


Source = VectorSource | Rank1Source | CachePairSource


def parse_source(value: Any) -> Source:
    data = _as_mapping(value, "source")
    dtype = _require_str(_require_key(data, "dtype", "source"), "source.dtype")
    if dtype == "vector":
        return VectorSource.from_dict(data)
    if dtype == "rank1":
        return Rank1Source.from_dict(data)
    if dtype == "cache_pair":
        return CachePairSource.from_dict(data)
    raise SchemaError("source.dtype must be one of ['vector', 'rank1', 'cache_pair']")


@dataclass(frozen=True)
class ResidAddOp:
    kind: str
    alpha: float

    @classmethod
    def from_dict(cls, value: Any) -> "ResidAddOp":
        data = _as_mapping(value, "op")
        return cls(kind="resid_add", alpha=_require_float(_require_key(data, "alpha", "op"), "op.alpha"))


@dataclass(frozen=True)
class KvMixOp:
    kind: str
    alpha: float
    which: str

    @classmethod
    def from_dict(cls, value: Any) -> "KvMixOp":
        data = _as_mapping(value, "op")
        which = _require_str(_require_key(data, "which", "op"), "op.which")
        if which not in KV_WHICH:
            raise SchemaError(f"op.which must be one of {sorted(KV_WHICH)}")
        return cls(
            kind="kv_mix",
            alpha=_require_float(_require_key(data, "alpha", "op"), "op.alpha"),
            which=which,
        )


@dataclass(frozen=True)
class Rank1PatchOp:
    kind: str
    alpha: float

    @classmethod
    def from_dict(cls, value: Any) -> "Rank1PatchOp":
        data = _as_mapping(value, "op")
        return cls(kind="rank1_patch", alpha=_require_float(_require_key(data, "alpha", "op"), "op.alpha"))


Op = ResidAddOp | KvMixOp | Rank1PatchOp


def parse_op(value: Any) -> Op:
    data = _as_mapping(value, "op")
    kind = _require_str(_require_key(data, "kind", "op"), "op.kind")
    if kind == "resid_add":
        return ResidAddOp.from_dict(data)
    if kind == "kv_mix":
        return KvMixOp.from_dict(data)
    if kind == "rank1_patch":
        return Rank1PatchOp.from_dict(data)
    raise SchemaError(f"op.kind must be one of {sorted(OP_KINDS)}")


@dataclass(frozen=True)
class Budget:
    ttl_steps: int
    norm_clip: float | None = None
    rank_cap: int | None = None
    revertible: bool = True

    @classmethod
    def from_dict(cls, value: Any) -> "Budget":
        data = _as_mapping(value, "budget")
        ttl_steps = _require_int(_require_key(data, "ttl_steps", "budget"), "budget.ttl_steps")
        if ttl_steps <= 0:
            raise SchemaError("budget.ttl_steps must be > 0")
        rank_cap = _optional_int(data.get("rank_cap"), "budget.rank_cap")
        if rank_cap is not None and rank_cap <= 0:
            raise SchemaError("budget.rank_cap must be > 0")
        norm_clip = _optional_float(data.get("norm_clip"), "budget.norm_clip")
        if norm_clip is not None and norm_clip <= 0.0:
            raise SchemaError("budget.norm_clip must be > 0")
        return cls(
            ttl_steps=ttl_steps,
            norm_clip=norm_clip,
            rank_cap=rank_cap,
            revertible=_require_bool(_require_key(data, "revertible", "budget"), "budget.revertible"),
        )


@dataclass(frozen=True)
class Edit:
    id: str
    target: TargetRef
    source: Source
    op: Op
    budget: Budget
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Any) -> "Edit":
        data = _as_mapping(value, "edit")
        meta = data.get("meta") or {}
        _as_mapping(meta, "edit.meta")
        return cls(
            id=_require_str(_require_key(data, "id", "edit"), "edit.id"),
            target=parse_target_ref(_require_key(data, "target", "edit")),
            source=parse_source(_require_key(data, "source", "edit")),
            op=parse_op(_require_key(data, "op", "edit")),
            budget=Budget.from_dict(_require_key(data, "budget", "edit")),
            meta=meta,
        )


@dataclass(frozen=True)
class ControllerCommand:
    version: str
    decision: str
    edits: tuple[Edit, ...] = ()
    rollback_ids: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Any) -> "ControllerCommand":
        data = _as_mapping(value, "command")
        version = _require_str(_require_key(data, "version", "command"), "command.version")
        decision = _require_str(_require_key(data, "decision", "command"), "command.decision")
        if decision not in {"noop", "apply", "rollback"}:
            raise SchemaError("command.decision must be one of ['noop', 'apply', 'rollback']")
        edits = tuple(Edit.from_dict(item) for item in data.get("edits", []))
        rollback_ids = _optional_string_array(data.get("rollback_ids"), "command.rollback_ids")
        if decision == "apply" and not edits:
            raise SchemaError("command.edits is required when decision='apply'")
        if decision == "rollback" and not rollback_ids:
            raise SchemaError("command.rollback_ids is required when decision='rollback'")
        meta = data.get("meta") or {}
        _as_mapping(meta, "command.meta")
        return cls(
            version=version,
            decision=decision,
            edits=edits,
            rollback_ids=rollback_ids,
            meta=meta,
        )


@dataclass(frozen=True)
class SurfaceCaps:
    max_alpha: float
    max_ttl_steps: int
    norm_clip: float | None = None
    rank_cap: int | None = None
    revertible_only: bool = True

    @classmethod
    def from_dict(cls, value: Any) -> "SurfaceCaps":
        data = _as_mapping(value, "caps")
        return cls(
            max_alpha=_require_float(_require_key(data, "max_alpha", "caps"), "caps.max_alpha"),
            max_ttl_steps=_require_int(_require_key(data, "max_ttl_steps", "caps"), "caps.max_ttl_steps"),
            norm_clip=_optional_float(data.get("norm_clip"), "caps.norm_clip"),
            rank_cap=_optional_int(data.get("rank_cap"), "caps.rank_cap"),
            revertible_only=_require_bool(_require_key(data, "revertible_only", "caps"), "caps.revertible_only"),
        )


@dataclass(frozen=True)
class SurfaceInfo:
    surface_id: str
    target: Target
    allow_ops: tuple[str, ...]
    caps: SurfaceCaps

    @classmethod
    def from_dict(cls, value: Any) -> "SurfaceInfo":
        data = _as_mapping(value, "surface")
        allow_ops = _optional_string_array(_require_key(data, "allow_ops", "surface"), "surface.allow_ops")
        unknown_ops = [op for op in allow_ops if op not in OP_KINDS]
        if unknown_ops:
            raise SchemaError(f"surface.allow_ops contains unsupported ops: {unknown_ops}")
        target_ref = parse_target_ref(_require_key(data, "target", "surface"))
        if isinstance(target_ref, SurfaceTargetRef):
            raise SchemaError("surface.target may not use surface_id indirection")
        return cls(
            surface_id=_require_str(_require_key(data, "surface_id", "surface"), "surface.surface_id"),
            target=target_ref,
            allow_ops=allow_ops,
            caps=SurfaceCaps.from_dict(_require_key(data, "caps", "surface")),
        )


@dataclass(frozen=True)
class TraceRef:
    trace_id: str
    origin: str
    compatible: bool
    similarity_hint: float | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Any) -> "TraceRef":
        data = _as_mapping(value, "trace_ref")
        return cls(
            trace_id=_require_str(_require_key(data, "trace_id", "trace_ref"), "trace_ref.trace_id"),
            origin=_require_str(_require_key(data, "origin", "trace_ref"), "trace_ref.origin"),
            compatible=_require_bool(_require_key(data, "compatible", "trace_ref"), "trace_ref.compatible"),
            similarity_hint=_optional_float(data.get("similarity_hint"), "trace_ref.similarity_hint"),
            tags=_optional_string_array(data.get("tags"), "trace_ref.tags"),
        )


@dataclass(frozen=True)
class ActiveEdit:
    edit_id: str
    surface_id: str
    op: str
    alpha: float
    ttl_left: int
    revertible: bool

    @classmethod
    def from_dict(cls, value: Any) -> "ActiveEdit":
        data = _as_mapping(value, "active_edit")
        op = _require_str(_require_key(data, "op", "active_edit"), "active_edit.op")
        if op not in OP_KINDS:
            raise SchemaError(f"active_edit.op must be one of {sorted(OP_KINDS)}")
        return cls(
            edit_id=_require_str(_require_key(data, "edit_id", "active_edit"), "active_edit.edit_id"),
            surface_id=_require_str(_require_key(data, "surface_id", "active_edit"), "active_edit.surface_id"),
            op=op,
            alpha=_require_float(_require_key(data, "alpha", "active_edit"), "active_edit.alpha"),
            ttl_left=_require_int(_require_key(data, "ttl_left", "active_edit"), "active_edit.ttl_left"),
            revertible=_require_bool(_require_key(data, "revertible", "active_edit"), "active_edit.revertible"),
        )


@dataclass(frozen=True)
class MiniMetrics:
    entropy: float
    top1_margin: float
    repetition_score: float
    partial_score: float | None = None

    @classmethod
    def from_dict(cls, value: Any, name: str) -> "MiniMetrics":
        data = _as_mapping(value, name)
        return cls(
            entropy=_require_float(_require_key(data, "entropy", name), f"{name}.entropy"),
            top1_margin=_require_float(_require_key(data, "top1_margin", name), f"{name}.top1_margin"),
            repetition_score=_require_float(_require_key(data, "repetition_score", name), f"{name}.repetition_score"),
            partial_score=_optional_float(data.get("partial_score"), f"{name}.partial_score"),
        )


@dataclass(frozen=True)
class EditEffect:
    edit_id: str
    surface_id: str
    observed_window_steps: int
    before: MiniMetrics
    after: MiniMetrics
    delta: Mapping[str, float]
    verdict: str

    @classmethod
    def from_dict(cls, value: Any) -> "EditEffect":
        data = _as_mapping(value, "edit_effect")
        delta = _as_mapping(_require_key(data, "delta", "edit_effect"), "edit_effect.delta")
        return cls(
            edit_id=_require_str(_require_key(data, "edit_id", "edit_effect"), "edit_effect.edit_id"),
            surface_id=_require_str(_require_key(data, "surface_id", "edit_effect"), "edit_effect.surface_id"),
            observed_window_steps=_require_int(
                _require_key(data, "observed_window_steps", "edit_effect"),
                "edit_effect.observed_window_steps",
            ),
            before=MiniMetrics.from_dict(_require_key(data, "before", "edit_effect"), "edit_effect.before"),
            after=MiniMetrics.from_dict(_require_key(data, "after", "edit_effect"), "edit_effect.after"),
            delta={str(k): _require_float(v, f"edit_effect.delta.{k}") for k, v in delta.items()},
            verdict=_require_str(_require_key(data, "verdict", "edit_effect"), "edit_effect.verdict"),
        )


@dataclass(frozen=True)
class BudgetState:
    edits_left_this_step: int
    edits_left_this_run: int
    alpha_left_total: float
    active_patch_slots_left: int
    rollbackable_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Any) -> "BudgetState":
        data = _as_mapping(value, "budget")
        return cls(
            edits_left_this_step=_require_int(_require_key(data, "edits_left_this_step", "budget"), "budget.edits_left_this_step"),
            edits_left_this_run=_require_int(_require_key(data, "edits_left_this_run", "budget"), "budget.edits_left_this_run"),
            alpha_left_total=_require_float(_require_key(data, "alpha_left_total", "budget"), "budget.alpha_left_total"),
            active_patch_slots_left=_require_int(
                _require_key(data, "active_patch_slots_left", "budget"),
                "budget.active_patch_slots_left",
            ),
            rollbackable_ids=_optional_string_array(data.get("rollbackable_ids"), "budget.rollbackable_ids"),
        )


@dataclass(frozen=True)
class ControllerObservationPacket:
    version: str
    run_id: str
    episode_id: str
    worker_id: str
    step: int
    horizon: Mapping[str, Any]
    task_view: Mapping[str, Any]
    worker_view: Mapping[str, Any]
    telemetry: Mapping[str, Any]
    surface_catalog: tuple[SurfaceInfo, ...]
    probe_frames: tuple[Mapping[str, Any], ...]
    trace_bank: tuple[TraceRef, ...]
    active_edits: tuple[ActiveEdit, ...]
    recent_effects: tuple[EditEffect, ...]
    budget: BudgetState
    task_feedback: Mapping[str, Any] | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, value: Any) -> "ControllerObservationPacket":
        data = _as_mapping(value, "packet")
        surface_catalog = tuple(SurfaceInfo.from_dict(item) for item in data.get("surface_catalog", []))
        return cls(
            version=_require_str(_require_key(data, "version", "packet"), "packet.version"),
            run_id=_require_str(_require_key(data, "run_id", "packet"), "packet.run_id"),
            episode_id=_require_str(_require_key(data, "episode_id", "packet"), "packet.episode_id"),
            worker_id=_require_str(_require_key(data, "worker_id", "packet"), "packet.worker_id"),
            step=_require_int(_require_key(data, "step", "packet"), "packet.step"),
            horizon=_as_mapping(_require_key(data, "horizon", "packet"), "packet.horizon"),
            task_view=_as_mapping(_require_key(data, "task_view", "packet"), "packet.task_view"),
            worker_view=_as_mapping(_require_key(data, "worker_view", "packet"), "packet.worker_view"),
            telemetry=_as_mapping(_require_key(data, "telemetry", "packet"), "packet.telemetry"),
            surface_catalog=surface_catalog,
            probe_frames=tuple(_as_mapping(item, "packet.probe_frames[]") for item in data.get("probe_frames", [])),
            trace_bank=tuple(TraceRef.from_dict(item) for item in data.get("trace_bank", [])),
            active_edits=tuple(ActiveEdit.from_dict(item) for item in data.get("active_edits", [])),
            recent_effects=tuple(EditEffect.from_dict(item) for item in data.get("recent_effects", [])),
            budget=BudgetState.from_dict(_require_key(data, "budget", "packet")),
            task_feedback=_as_mapping(data["task_feedback"], "packet.task_feedback") if data.get("task_feedback") is not None else None,
            raw=data,
        )

    def surface_map(self) -> dict[str, SurfaceInfo]:
        return {surface.surface_id: surface for surface in self.surface_catalog}

    def trace_map(self) -> dict[str, TraceRef]:
        return {trace.trace_id: trace for trace in self.trace_bank}


@dataclass(frozen=True)
class HarnessControllerView:
    task_prompt: str = "redacted"
    generated_text: bool = True
    telemetry: tuple[str, ...] = ("step", "entropy", "top1_margin", "repeat_flag", "budget_left")
    trace_access: tuple[str, ...] = ("runtime", "best_success", "paired_baseline")


def parse_controller_command(value: Any) -> ControllerCommand:
    return ControllerCommand.from_dict(value)


def parse_observation_packet(value: Any) -> ControllerObservationPacket:
    return ControllerObservationPacket.from_dict(value)
