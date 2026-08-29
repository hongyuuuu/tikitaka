"""Immutable experiment identity and label-isolated evaluation execution."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    config_version: str
    prompt_version: str
    schema_version: str
    routing_mode: str
    generative_provider: str
    generative_model: str
    reasoning_level: str
    retrieval_policy: str
    embedding_route_id: str
    index_id: str
    reranker_route_id: str
    fusion_parameters: tuple[tuple[str, float], ...]
    profile_weight: float
    question_policy: str
    seed: int
    split_version: str
    catalog_checksum: str
    code_revision: str

    def __post_init__(self) -> None:
        for name in (
            "name", "config_version", "prompt_version", "schema_version", "routing_mode",
            "generative_provider", "generative_model", "reasoning_level", "retrieval_policy",
            "embedding_route_id", "index_id", "reranker_route_id", "question_policy",
            "split_version", "catalog_checksum", "code_revision",
        ):
            _required_text(getattr(self, name), name)
        if self.routing_mode not in {"runtime_auto", "pinned"}:
            raise ValueError("routing_mode must be runtime_auto or pinned")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not math.isfinite(self.profile_weight) or not 0.0 <= self.profile_weight <= 1.0:
            raise ValueError("profile_weight must be between 0 and 1")
        names = [name for name, _ in self.fusion_parameters]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError("fusion parameter names must be unique and non-empty")
        if any(not math.isfinite(value) for _, value in self.fusion_parameters):
            raise ValueError("fusion parameter values must be finite")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "config_version": self.config_version,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "routing_mode": self.routing_mode,
            "generative_provider": self.generative_provider,
            "generative_model": self.generative_model,
            "reasoning_level": self.reasoning_level,
            "retrieval_policy": self.retrieval_policy,
            "embedding_route_id": self.embedding_route_id,
            "index_id": self.index_id,
            "reranker_route_id": self.reranker_route_id,
            "fusion_parameters": dict(self.fusion_parameters),
            "profile_weight": self.profile_weight,
            "question_policy": self.question_policy,
            "seed": self.seed,
            "split_version": self.split_version,
            "catalog_checksum": self.catalog_checksum,
            "code_revision": self.code_revision,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def cache_key(self, boundary: str, inputs: Mapping[str, object]) -> str:
        _required_text(boundary, "cache boundary")
        material = {
            "boundary": boundary,
            "experiment_fingerprint": self.fingerprint,
            "inputs": inputs,
        }
        payload = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metric_summary(sessions: Sequence[dict]) -> dict:
    if not sessions:
        return {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None}
    count = len(sessions)
    hit_rate = sum(int(item["hit"]) for item in sessions) / count
    mrr = sum(float(item["reciprocal_rank"]) for item in sessions) / count
    mttc = sum(item["first_hit_turn"] or MAX_TURNS + 1 for item in sessions) / count
    return {
        "sample_count": count,
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
    }


def _score_metrics(sessions: Sequence[dict]) -> dict:
    summary = _metric_summary(sessions)
    mttc = summary["mttc"]
    efficiency = 0.0 if mttc is None else max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical = 0.5 * summary["hit_rate_at_10"] + 0.3 * summary["mrr"] + 0.2 * efficiency
    return {**summary, "efficiency": round(efficiency, 6), "technical_score": round(technical, 6)}


def _usage_values(response: object) -> dict:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, Mapping):
        return {}
    allowed = (
        "prompt_tokens", "completion_tokens", "reasoning_tokens", "calls", "retries",
        "failures", "fallback_activations", "latency_ms", "estimated_cost",
        "provider", "model", "reasoning_level", "route", "component",
    )
    return {name: usage[name] for name in allowed if name in usage}


def evaluate_samples(
    agent_factory: Callable[[], object],
    samples: Sequence[Mapping[str, object]],
    catalog_ids: set[str],
    categories: Mapping[str, list[str]],
    products: Mapping[str, dict],
    config: ExperimentConfig,
    split_name: str,
) -> dict:
    """Run the public simulator while exposing only official fields to the Agent."""

    sessions: list[dict] = []
    asked = Counter()
    usage_totals = Counter()
    route_totals: dict[tuple[str, str, str, str, str], Counter] = defaultdict(Counter)
    total_agent_latency_ms = 0.0
    agent = agent_factory()

    for sample in sorted(samples, key=lambda item: str(item["sample_id"])):
        sample_id = str(sample["sample_id"])
        scenario = str(sample["scenario_type"])
        session_id = "eval_" + hashlib.sha256(
            f"{config.fingerprint}\0{split_name}\0{sample_id}".encode("utf-8")
        ).hexdigest()[:24]
        # Only the documented profile snapshot crosses the reset boundary.
        agent.reset(session_id, dict(sample["user_profile"]))
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(dict(sample), dict(products))
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = scenario != "intent_override"
        user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
        hit_turn: int | None = None
        best_rank: int | None = None
        question_count = 0
        session_failures = 0

        for turn in range(1, MAX_TURNS + 1):
            started = time.perf_counter()
            try:
                # No scenario label, target, intent card, or evaluator object crosses this boundary.
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
                session_failures += 1
            latency_ms = (time.perf_counter() - started) * 1000.0
            total_agent_latency_ms += latency_ms
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
                session_failures += 1

            ask_attribute = response.get("ask_attribute")
            if isinstance(ask_attribute, str):
                asked[ask_attribute] += 1
                question_count += 1
            usage = _usage_values(response)
            numeric_names = (
                "prompt_tokens", "completion_tokens", "reasoning_tokens", "calls", "retries",
                "failures", "fallback_activations", "latency_ms", "estimated_cost",
            )
            valid_usage: dict[str, int | float] = {}
            for name in numeric_names:
                value = usage.get(name, 0)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                    usage_totals[name] += value
                    valid_usage[name] = value
            route_key = tuple(str(usage.get(name) or "unknown") for name in (
                "component", "route", "provider", "model", "reasoning_level"
            ))
            if usage:
                route_totals[route_key].update(valid_usage)

            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample, ask_attribute, disclosed, boundary_used
                )

        sessions.append({
            "sample_id": sample_id,
            "scenario_type": scenario,
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else round(1.0 / best_rank, 12),
            "question_count": question_count,
            "failures": session_failures,
        })

    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    usage_totals["failures"] += sum(item["failures"] for item in sessions)
    usage = {name: round(value, 6) if isinstance(value, float) else value for name, value in sorted(usage_totals.items())}
    usage["total_tokens"] = int(usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
    usage["agent_boundary_latency_ms"] = round(total_agent_latency_ms, 6)
    routes = []
    for key, totals in sorted(route_totals.items()):
        component, route, provider, model, reasoning_level = key
        routes.append({
            "component": component, "route": route, "provider": provider, "model": model,
            "reasoning_level": reasoning_level,
            **{name: round(value, 6) if isinstance(value, float) else value for name, value in sorted(totals.items())},
        })
    return {
        "metrics": _score_metrics(sessions),
        "scenario_metrics": {name: _score_metrics(grouped[name]) for name in sorted(grouped)},
        "questions": {"count": sum(asked.values()), "asked_attribute_distribution": dict(sorted(asked.items()))},
        "usage": usage,
        "usage_by_component_route": routes,
        "sessions": sessions,
    }
