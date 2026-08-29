"""Reproducible Person 3 experiment arms for the Phase 4 score gate.

This module owns configuration, not evaluation.  Person 4 can pass
``runtime_overrides`` into its composition root and record the stable IDs and
fingerprint in the experiment report without duplicating Person 3 parameters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Mapping

from tikitaka.ranking.deterministic import DeterministicRankerConfig
from tikitaka.ranking.llm import LLMRerankerConfig

from .response_policy import ResponsePolicyConfig


PHASE4_ARM_VERSION = "person3-phase4-v1"


@dataclass(frozen=True)
class Phase4ExperimentArm:
    """One immutable, attributable Person 3 configuration."""

    name: str
    question_policy_id: str
    reranker_route_id: str
    response: ResponsePolicyConfig
    ranking: DeterministicRankerConfig
    enable_llm_reranker: bool
    llm_reranker: LLMRerankerConfig | None
    profile_weight: float = 0.0
    version: str = PHASE4_ARM_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "question_policy_id",
            "reranker_route_id",
            "version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.response, ResponsePolicyConfig):
            raise TypeError("response must be ResponsePolicyConfig")
        if not isinstance(self.ranking, DeterministicRankerConfig):
            raise TypeError("ranking must be DeterministicRankerConfig")
        if not isinstance(self.enable_llm_reranker, bool):
            raise TypeError("enable_llm_reranker must be bool")
        if self.enable_llm_reranker != (self.llm_reranker is not None):
            raise ValueError("LLM enablement and configuration must agree")
        if not 0.0 <= self.profile_weight <= 1.0:
            raise ValueError("profile_weight must be in [0, 1]")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def runtime_overrides(self) -> Mapping[str, object]:
        """Fields accepted by Person 4's ``RuntimeConfig`` composition root."""

        return {
            "profile_weight": self.profile_weight,
            "decision": self.response,
            "ranking": self.ranking,
            "enable_llm_reranker": self.enable_llm_reranker,
            "llm_reranker": self.llm_reranker,
        }

    def changed_report_variables_from(
        self, baseline: "Phase4ExperimentArm"
    ) -> tuple[str, ...]:
        """Return Person 4 report fields that must be declared as changed."""

        changed: list[str] = []
        if (
            self.question_policy_id != baseline.question_policy_id
            or self.response != baseline.response
        ):
            changed.append("question_policy")
        if (
            self.reranker_route_id != baseline.reranker_route_id
            or self.ranking != baseline.ranking
            or self.enable_llm_reranker != baseline.enable_llm_reranker
            or self.llm_reranker != baseline.llm_reranker
        ):
            changed.append("reranker_route_id")
        if self.profile_weight != baseline.profile_weight:
            changed.append("profile_weight")
        return tuple(changed)


def phase4_experiment_arms() -> tuple[Phase4ExperimentArm, ...]:
    """Return the small, one-factor-first grid required before joint tuning."""

    ranking = DeterministicRankerConfig(profile_weight=0.0)
    adaptive = ResponsePolicyConfig(question_ranker=ranking)
    official_proxy_question = replace(
        adaptive.question_value,
        # Normalize the official 0.50 Hit@10 and 0.30 MRR weights within the
        # recommendation-quality portion. Efficiency remains in turn utility.
        membership_weight=0.625,
        order_weight=0.375,
    )
    official_proxy = replace(
        adaptive,
        question_value=official_proxy_question,
    )
    conservative = replace(
        official_proxy,
        information_gain_threshold=0.070,
        clarification_turn_cost=0.018,
        late_turn_cost=0.100,
    )
    anchored_llm = LLMRerankerConfig()
    unanchored_llm = replace(
        anchored_llm,
        maximum_anchors=0,
        skip_llm_lead_margin=1.0,
    )

    arms = (
        Phase4ExperimentArm(
            name="always-recommend-baseline",
            question_policy_id="p3/always-recommend-v1",
            reranker_route_id="p3/deterministic-v2",
            response=replace(adaptive, clarification_enabled=False),
            ranking=ranking,
            enable_llm_reranker=False,
            llm_reranker=None,
        ),
        Phase4ExperimentArm(
            name="adaptive-deterministic",
            question_policy_id="p3/adaptive-v2",
            reranker_route_id="p3/deterministic-v2",
            response=adaptive,
            ranking=ranking,
            enable_llm_reranker=False,
            llm_reranker=None,
        ),
        Phase4ExperimentArm(
            name="official-proxy-deterministic",
            question_policy_id="p3/official-proxy-v1",
            reranker_route_id="p3/deterministic-v2",
            response=official_proxy,
            ranking=ranking,
            enable_llm_reranker=False,
            llm_reranker=None,
        ),
        Phase4ExperimentArm(
            name="conservative-questions-deterministic",
            question_policy_id="p3/conservative-official-proxy-v1",
            reranker_route_id="p3/deterministic-v2",
            response=conservative,
            ranking=ranking,
            enable_llm_reranker=False,
            llm_reranker=None,
        ),
        Phase4ExperimentArm(
            name="adaptive-llm-unanchored",
            question_policy_id="p3/adaptive-v2",
            reranker_route_id="p3/llm-unanchored-v2",
            response=adaptive,
            ranking=ranking,
            enable_llm_reranker=True,
            llm_reranker=unanchored_llm,
        ),
        Phase4ExperimentArm(
            name="adaptive-llm-anchored",
            question_policy_id="p3/adaptive-v2",
            reranker_route_id="p3/llm-anchored-v2",
            response=adaptive,
            ranking=ranking,
            enable_llm_reranker=True,
            llm_reranker=anchored_llm,
        ),
        Phase4ExperimentArm(
            name="adaptive-profile-010",
            question_policy_id="p3/adaptive-v2",
            reranker_route_id="p3/deterministic-v2",
            response=adaptive,
            ranking=ranking,
            enable_llm_reranker=False,
            llm_reranker=None,
            profile_weight=0.10,
        ),
    )
    names = [arm.name for arm in arms]
    fingerprints = [arm.fingerprint for arm in arms]
    if len(names) != len(set(names)) or len(fingerprints) != len(set(fingerprints)):
        raise AssertionError("Phase 4 experiment arms must be uniquely attributable")
    return arms


def phase4_arm(name: str) -> Phase4ExperimentArm:
    for arm in phase4_experiment_arms():
        if arm.name == name:
            return arm
    raise KeyError(f"unknown Person 3 Phase 4 arm: {name}")


__all__ = [
    "PHASE4_ARM_VERSION",
    "Phase4ExperimentArm",
    "phase4_arm",
    "phase4_experiment_arms",
]
