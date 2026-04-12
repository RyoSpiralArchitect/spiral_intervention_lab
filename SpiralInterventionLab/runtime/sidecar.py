from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


def _clean_text(value: Any, *, limit: int = 96) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return None
    return text[:limit]


def _clip_score(value: Any, *, minimum: float = -2.0, maximum: float = 2.0) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number < minimum:
        number = minimum
    if number > maximum:
        number = maximum
    return round(number, 6)


@dataclass(frozen=True)
class ReadoutSidecarSiteCapture:
    role: str
    layer: int
    token_selector: Mapping[str, Any]
    vector: torch.Tensor
    surface_id: str | None = None
    position: int | None = None
    span: tuple[int, int] | None = None
    term: str | None = None
    provenance_class: str | None = None
    piece: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "role": str(self.role),
            "layer": int(self.layer),
            "token_selector": dict(self.token_selector),
            "vector_width": int(self.vector.numel()),
        }
        if self.surface_id:
            summary["surface_id"] = str(self.surface_id)
        if self.position is not None:
            summary["position"] = int(self.position)
        if self.span is not None:
            summary["span"] = [int(self.span[0]), int(self.span[1])]
        if self.term:
            summary["term"] = str(self.term)
        if self.provenance_class:
            summary["provenance_class"] = str(self.provenance_class)
        if self.piece not in (None, ""):
            summary["piece"] = str(self.piece)
        if self.metadata:
            summary["metadata"] = {
                str(key): value
                for key, value in self.metadata.items()
                if value not in (None, "", [], {})
            }
        return summary


@dataclass(frozen=True)
class ReadoutSidecarCapture:
    run_id: str
    episode_id: str
    worker_id: str
    step: int
    control_phase_hint: str
    answer_readout_canary: Mapping[str, Any]
    answer_sites: tuple[ReadoutSidecarSiteCapture, ...] = ()
    source_sites: tuple[ReadoutSidecarSiteCapture, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        provenance_counts: dict[str, int] = {}
        source_terms: list[str] = []
        for site in self.source_sites:
            provenance_class = str(site.provenance_class or "unknown")
            provenance_counts[provenance_class] = int(provenance_counts.get(provenance_class, 0) or 0) + 1
            if site.term and site.term not in source_terms:
                source_terms.append(site.term)
        return {
            "run_id": str(self.run_id),
            "episode_id": str(self.episode_id),
            "worker_id": str(self.worker_id),
            "step": int(self.step),
            "control_phase_hint": str(self.control_phase_hint),
            "answer_site_count": len(self.answer_sites),
            "source_site_count": len(self.source_sites),
            "answer_roles": [site.role for site in self.answer_sites[:6]],
            "source_terms": source_terms[:6],
            "source_provenance_counts": provenance_counts,
            "answer_sites": [site.summary() for site in self.answer_sites[:4]],
            "source_sites": [site.summary() for site in self.source_sites[:6]],
            "metadata": {
                str(key): value
                for key, value in self.metadata.items()
                if value not in (None, "", [], {})
            },
        }


ReadoutSidecarAnalyzer = Callable[[ReadoutSidecarCapture], Mapping[str, Any] | None]


def _normalized(vec: torch.Tensor) -> torch.Tensor | None:
    flat = vec.detach().reshape(-1).cpu().float()
    norm = float(flat.norm().item())
    if norm <= 0.0:
        return None
    return flat / norm


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_norm = _normalized(left)
    right_norm = _normalized(right)
    if left_norm is None or right_norm is None or left_norm.numel() != right_norm.numel():
        return None
    return float(torch.dot(left_norm, right_norm).item())


def _capture_span_id(site: ReadoutSidecarSiteCapture) -> str:
    if site.span is not None:
        return f"{int(site.span[0])}:{int(site.span[1])}"
    if site.position is not None:
        return f"{int(site.position)}:{int(site.position) + 1}"
    return "unknown"


def build_heuristic_readout_sidecar_analyzer(
    *,
    analyzer_name: str = "heuristic_readout_sidecar",
    focus_override_margin: float = 0.05,
    attractor_mass_threshold: float = 0.18,
    reachable_rank_threshold: int = 512,
    target_mass_threshold: float = 0.001,
) -> ReadoutSidecarAnalyzer:
    def analyze(capture: ReadoutSidecarCapture) -> Mapping[str, Any] | None:
        answer_canary = capture.answer_readout_canary if isinstance(capture.answer_readout_canary, Mapping) else {}
        reachable_term = _clean_text(answer_canary.get("reachable_focus_term"), limit=64)
        reachable_rank = answer_canary.get("reachable_focus_rank")
        try:
            reachable_rank_value = int(reachable_rank)
        except Exception:
            reachable_rank_value = 10**9
        try:
            target_mass = float(answer_canary.get("target_mass", 0.0) or 0.0)
        except Exception:
            target_mass = 0.0
        try:
            attractor_mass = float(answer_canary.get("attractor_family_mass", 0.0) or 0.0)
        except Exception:
            attractor_mass = 0.0
        attractor_overlap_tokens = answer_canary.get("attractor_family_overlap_tokens")
        attractor_present = bool(
            attractor_mass >= attractor_mass_threshold
            or (
                isinstance(attractor_overlap_tokens, Sequence)
                and not isinstance(attractor_overlap_tokens, (str, bytes, bytearray))
                and len(attractor_overlap_tokens) > 0
            )
        )

        answer_sites = [site for site in capture.answer_sites if isinstance(site, ReadoutSidecarSiteCapture)]
        source_sites = [site for site in capture.source_sites if isinstance(site, ReadoutSidecarSiteCapture) and site.term]
        if not answer_sites or not source_sites:
            return {"analyzer_name": analyzer_name, "attractor_family_present": attractor_present}

        term_rows: dict[str, dict[str, Any]] = {}
        for site in source_sites:
            cosine_values = [
                cosine
                for cosine in (_cosine(site.vector, answer_site.vector) for answer_site in answer_sites)
                if cosine is not None
            ]
            if not cosine_values:
                continue
            term = str(site.term)
            provenance_class = str(site.provenance_class or "misc_prompt")
            provenance_bonus = {
                "source_body": 0.18,
                "answer_prefix": 0.04,
                "constraint_header": -0.08,
                "misc_prompt": -0.12,
            }.get(provenance_class, -0.12)
            span_bonus = 0.06 if site.span is not None and (site.span[1] - site.span[0]) > 1 else 0.03
            raw_support = (0.75 * max(cosine_values)) + (0.25 * (sum(cosine_values) / len(cosine_values))) + provenance_bonus + span_bonus

            row = term_rows.setdefault(
                term,
                {
                    "best_support": -10.0,
                    "best_anchor": -10.0,
                    "best_site": None,
                    "site_count": 0,
                },
            )
            row["site_count"] = int(row["site_count"]) + 1
            best_anchor = max(cosine_values)
            if raw_support > float(row["best_support"]):
                row["best_support"] = float(raw_support)
                row["best_anchor"] = float(best_anchor)
                row["best_site"] = site

        if not term_rows:
            return {"analyzer_name": analyzer_name, "attractor_family_present": attractor_present}

        candidate_support_terms = {
            term: round(float(row["best_support"]), 6)
            for term, row in sorted(term_rows.items(), key=lambda item: (-float(item[1]["best_support"]), item[0]))
        }
        term_anchor_strength_by_term = {
            term: round(float(row["best_anchor"]), 6)
            for term, row in sorted(term_rows.items(), key=lambda item: (-float(item[1]["best_anchor"]), item[0]))
        }
        candidate_support_scores: dict[str, float] = {}
        for term, row in term_rows.items():
            site = row.get("best_site")
            if not isinstance(site, ReadoutSidecarSiteCapture):
                continue
            support = round(float(row["best_support"]), 6)
            provenance_class = str(site.provenance_class or "misc_prompt")
            span_id = _capture_span_id(site)
            span_kind = "exact_prompt_span_mean" if site.span is not None and (site.span[1] - site.span[0]) > 1 else "exact_prompt_piece"
            candidate_support_scores[f"{term}|{provenance_class}|{span_id}|kv_v|{span_kind}"] = support
            candidate_support_scores[f"{term}|{provenance_class}|{span_id}|kv_k|{span_kind}"] = round(support - 0.03, 6)
            candidate_support_scores[f"kv_pair:{term}:{provenance_class}:{span_id}"] = round(support + 0.08, 6)
            candidate_support_scores[f"{term}|{provenance_class}|{span_id}|shot_bridge|{span_kind}"] = round(support - 0.12, 6)

        best_term = max(candidate_support_terms, key=lambda item: (candidate_support_terms[item], term_anchor_strength_by_term.get(item, -10.0), item))
        reachable_support = 0.0 if reachable_term is None else float(candidate_support_terms.get(reachable_term, 0.0) or 0.0)
        best_support = float(candidate_support_terms.get(best_term, 0.0) or 0.0)
        focus_term_override = None
        if attractor_present and target_mass < target_mass_threshold and reachable_rank_value > reachable_rank_threshold:
            if reachable_term is None or (best_support - reachable_support) >= focus_override_margin:
                focus_term_override = best_term

        notes: list[str] = []
        if focus_term_override is not None:
            notes.append(f"reachable_first_override:{focus_term_override}")
        if attractor_present:
            notes.append("attractor_family_present")
        return {
            "analyzer_name": analyzer_name,
            "attractor_family_present": attractor_present,
            "focus_term_override": focus_term_override,
            "term_anchor_strength_by_term": term_anchor_strength_by_term,
            "candidate_support_terms": candidate_support_terms,
            "candidate_support_scores": candidate_support_scores,
            "notes": notes,
        }

    return analyze


def normalize_readout_sidecar_hints(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    summary: dict[str, Any] = {}
    analyzer_name = _clean_text(value.get("analyzer_name"), limit=64)
    if analyzer_name is not None:
        summary["analyzer_name"] = analyzer_name
    analyzer_error = _clean_text(value.get("analyzer_error"), limit=160)
    if analyzer_error is not None:
        summary["analyzer_error"] = analyzer_error
    focus_term_override = _clean_text(value.get("focus_term_override"), limit=64)
    if focus_term_override is not None:
        summary["focus_term_override"] = focus_term_override
    if value.get("attractor_family_present") is not None:
        summary["attractor_family_present"] = bool(value.get("attractor_family_present"))

    term_strengths = value.get("term_anchor_strength_by_term")
    if isinstance(term_strengths, Mapping):
        cleaned_strengths: dict[str, float] = {}
        for raw_term, raw_score in list(term_strengths.items())[:8]:
            term = _clean_text(raw_term, limit=64)
            score = _clip_score(raw_score)
            if term is None or score is None:
                continue
            cleaned_strengths[term] = score
        if cleaned_strengths:
            summary["term_anchor_strength_by_term"] = cleaned_strengths

    candidate_support_scores = value.get("candidate_support_scores")
    if isinstance(candidate_support_scores, Mapping):
        cleaned_support: dict[str, float] = {}
        for raw_key, raw_score in list(candidate_support_scores.items())[:12]:
            key = _clean_text(raw_key, limit=160)
            score = _clip_score(raw_score)
            if key is None or score is None:
                continue
            cleaned_support[key] = score
        if cleaned_support:
            summary["candidate_support_scores"] = cleaned_support

    candidate_support_terms = value.get("candidate_support_terms")
    if isinstance(candidate_support_terms, Mapping):
        cleaned_terms: dict[str, float] = {}
        for raw_term, raw_score in list(candidate_support_terms.items())[:8]:
            term = _clean_text(raw_term, limit=64)
            score = _clip_score(raw_score)
            if term is None or score is None:
                continue
            cleaned_terms[term] = score
        if cleaned_terms:
            summary["candidate_support_terms"] = cleaned_terms

    for source_key, target_key, limit in (
        ("candidate_family_vetoes", "candidate_family_vetoes", 8),
        ("candidate_key_vetoes", "candidate_key_vetoes", 12),
        ("notes", "notes", 6),
    ):
        items = value.get(source_key)
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            continue
        cleaned_items = []
        for item in items[:limit]:
            text = _clean_text(item, limit=160)
            if text is not None:
                cleaned_items.append(text)
        if cleaned_items:
            summary[target_key] = cleaned_items

    return summary
