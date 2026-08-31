"""Reproducible Person 3 experiment arms for policy and score gates.

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

from .question_value import CONTRACT_ORDER_SELECTION, HIGHEST_VALUE_SELECTION
from .response_policy import ResponsePolicyConfig


PHASE4_ARM_VERSION = "person3-phase4-v1"
PHASE5_ARM_VERSION = "person3-phase5-v1"
PHASE6_ARM_VERSION = "person3-phase6-mrr-v1"
PHASE5_CLARIFICATION_THRESHOLDS = (0.050, 0.060, 0.070, 0.080, 0.090)
PHASE6_BROWSING_ANSWERABILITY_WEIGHTS = (
    ("material", 0.85),
    ("color", 0.45),
    ("size", 0.30),
    ("style", 0.25),
    ("brand", 0.10),
    ("budget", 0.10),
    ("feature", 1.00),
    ("use_case", 0.15),
    ("other", 0.10),
)


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
        values = asdict(self)
        response = values.get("response")
        if (
            isinstance(response, dict)
            and response.get("question_selection_strategy")
            == HIGHEST_VALUE_SELECTION
        ):
            # Preserve every Phase 4 fingerprint recorded before the explicit
            # fixed-order selector was introduced. The default strategy has
            # identical behaviour to those historical configurations.
            response.pop("question_selection_strategy")

        def remove_inert_fields(value: object) -> None:
            if isinstance(value, dict):
                if value.get("route_rank_weight") == 0.0:
                    # Preserve fingerprints recorded before independent
                    # route-rank evidence was added. A zero-weight signal is
                    # behaviorally inert wherever a ranker config is nested.
                    value.pop("route_rank_weight")
                    value.pop("route_rank_k")
                if value.get("evidence_phrase_weight") == 0.0:
                    value.pop("evidence_phrase_weight")
                    value.pop("evidence_phrase_min_confidence")
                if value.get("evidence_specificity_weight") == 0.0:
                    value.pop("evidence_specificity_weight")
                if value.get("popularity_weight") == 0.0:
                    value.pop("popularity_weight")
                if value.get("attribute_answerability_weights") == ():
                    # Empty priors preserve the historical neutral behaviour.
                    value.pop("attribute_answerability_weights")
                    value.pop("answerability_start_turn")
                    value.pop("answerability_after_no_preference")
                    value.pop("answerability_modes")
                    value.pop("answerability_min_intent_version")
                    value.pop("answerability_max_intent_version")
                for nested in value.values():
                    remove_inert_fields(nested)
            elif isinstance(value, list):
                for nested in value:
                    remove_inert_fields(nested)

        remove_inert_fields(values)
        payload = json.dumps(
            values,
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
    """Return the reproducible Person 3 grid consumed by the P5 harness."""

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
    post_no_information = replace(
        official_proxy,
        clarification_turn_cost=0.018,
        late_turn_cost=0.100,
    )
    threshold_responses = {
        threshold: replace(
            post_no_information,
            information_gain_threshold=threshold,
        )
        for threshold in PHASE5_CLARIFICATION_THRESHOLDS
    }
    conservative = threshold_responses[0.070]
    route_rank_052 = replace(
        ranking,
        fused_weight=0.22,
        structural_weight=0.04,
        route_agreement_weight=0.02,
        route_rank_weight=0.52,
        constraint_match_weight=0.20,
    )
    phrase_specificity_020 = replace(
        ranking,
        fused_weight=0.07,
        structural_weight=0.06,
        route_agreement_weight=0.03,
        evidence_phrase_weight=0.40,
        evidence_specificity_weight=0.20,
        constraint_match_weight=0.20,
    )
    phrase_specificity_popularity_011 = replace(
        phrase_specificity_020,
        popularity_weight=0.11,
    )
    answerability_guarded = replace(
        conservative,
        question_value=replace(
            conservative.question_value,
            attribute_answerability_weights=(
                PHASE6_BROWSING_ANSWERABILITY_WEIGHTS
            ),
            answerability_start_turn=2,
            answerability_after_no_preference=False,
            answerability_modes=("browsing",),
            answerability_max_intent_version=1,
        ),
    )
    anchored_llm = LLMRerankerConfig()
    unanchored_llm = replace(
        anchored_llm,
        maximum_anchors=0,
        skip_llm_lead_margin=1.0,
    )

    arms: list[Phase4ExperimentArm] = [
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
        Phase4ExperimentArm(
            name="mrr-route-rank-052-deterministic",
            question_policy_id="p3/conservative-official-proxy-v1",
            reranker_route_id="p3/deterministic-route-rank-052-v1",
            response=conservative,
            ranking=route_rank_052,
            enable_llm_reranker=False,
            llm_reranker=None,
            version=PHASE6_ARM_VERSION,
        ),
        Phase4ExperimentArm(
            name="mrr-evidence-phrase-specificity-020-deterministic",
            question_policy_id="p3/conservative-official-proxy-v1",
            reranker_route_id=(
                "p3/deterministic-evidence-phrase-040-specificity-020-v1"
            ),
            response=conservative,
            ranking=phrase_specificity_020,
            enable_llm_reranker=False,
            llm_reranker=None,
            version=PHASE6_ARM_VERSION,
        ),
        Phase4ExperimentArm(
            name="mrr-evidence-popularity-011-deterministic",
            question_policy_id="p3/conservative-official-proxy-v1",
            reranker_route_id=(
                "p3/deterministic-evidence-phrase-040-specificity-020-"
                "popularity-011-v1"
            ),
            response=conservative,
            ranking=phrase_specificity_popularity_011,
            enable_llm_reranker=False,
            llm_reranker=None,
            version=PHASE6_ARM_VERSION,
        ),
        Phase4ExperimentArm(
            name="answerability-guarded-browsing-deterministic",
            question_policy_id="p3/answerability-soft-browsing-guarded-v1",
            reranker_route_id="p3/deterministic-v2",
            response=answerability_guarded,
            ranking=ranking,
            enable_llm_reranker=False,
            llm_reranker=None,
            version=PHASE6_ARM_VERSION,
        ),
    ]

    for threshold, response in threshold_responses.items():
        threshold_code = f"{round(threshold * 1000):03d}"
        if threshold != 0.070:
            arms.append(
                Phase4ExperimentArm(
                    name=f"post-no-info-threshold-{threshold_code}-deterministic",
                    question_policy_id=(
                        f"p3/post-no-info-threshold-{threshold_code}-v1"
                    ),
                    reranker_route_id="p3/deterministic-v2",
                    response=response,
                    ranking=ranking,
                    enable_llm_reranker=False,
                    llm_reranker=None,
                    version=PHASE5_ARM_VERSION,
                )
            )

        fixed_name = (
            "fixed-ask-baseline"
            if threshold == 0.070
            else f"fixed-ask-threshold-{threshold_code}"
        )
        fixed_response = replace(
            response,
            question_selection_strategy=CONTRACT_ORDER_SELECTION,
        )
        arms.append(
            Phase4ExperimentArm(
                name=fixed_name,
                question_policy_id=(
                    f"p3/fixed-ask-contract-order-threshold-{threshold_code}-v1"
                ),
                reranker_route_id="p3/deterministic-v2",
                response=fixed_response,
                ranking=ranking,
                enable_llm_reranker=False,
                llm_reranker=None,
                version=PHASE5_ARM_VERSION,
            )
        )

        llm_name = (
            "conservative-llm-anchored"
            if threshold == 0.070
            else f"post-no-info-threshold-{threshold_code}-llm-anchored"
        )
        arms.append(
            Phase4ExperimentArm(
                name=llm_name,
                question_policy_id=(
                    "p3/conservative-official-proxy-v1"
                    if threshold == 0.070
                    else f"p3/post-no-info-threshold-{threshold_code}-v1"
                ),
                reranker_route_id="p3/llm-anchored-v2",
                response=response,
                ranking=ranking,
                enable_llm_reranker=True,
                llm_reranker=anchored_llm,
                version=PHASE5_ARM_VERSION,
            )
        )

    names = [arm.name for arm in arms]
    fingerprints = [arm.fingerprint for arm in arms]
    if len(names) != len(set(names)) or len(fingerprints) != len(set(fingerprints)):
        raise AssertionError("Person 3 experiment arms must be uniquely attributable")
    return tuple(arms)


def phase4_arm(name: str) -> Phase4ExperimentArm:
    for arm in phase4_experiment_arms():
        if arm.name == name:
            return arm
    raise KeyError(f"unknown Person 3 experiment arm: {name}")


__all__ = [
    "PHASE4_ARM_VERSION",
    "PHASE5_ARM_VERSION",
    "PHASE5_CLARIFICATION_THRESHOLDS",
    "PHASE6_ARM_VERSION",
    "PHASE6_BROWSING_ANSWERABILITY_WEIGHTS",
    "Phase4ExperimentArm",
    "phase4_arm",
    "phase4_experiment_arms",
]
