"""Single CLARIFY-or-RECOMMEND boundary for DG-01."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tikitaka.contracts import TurnDecision

from .diagnostics import ALLOWED_ATTRIBUTES
from .generality import GeneralitySensor
from .intent_router import VisibleModePolicy
from .question_value import QuestionValueEstimator


DECISION_REASON_CODES = frozenset(
    {
        "final_turn",
        "valuable_clarification",
        "low_question_value",
        "no_eligible_attribute",
        "ranking_stable",
        "insufficient_evidence",
        "component_fallback",
    }
)


@dataclass(frozen=True)
class DecisionRecord:
    """Structural equivalent of frozen contract TurnDecision 0.1.0."""

    action: str
    ask_attribute: str | None
    reason_code: str
    reason: str
    expected_information_gain: float

    def __post_init__(self) -> None:
        if self.action not in {"clarify", "recommend"}:
            raise ValueError("action must be clarify or recommend")
        if self.reason_code not in DECISION_REASON_CODES:
            raise ValueError("unknown decision reason code")
        if not 0.0 <= self.expected_information_gain <= 1.0:
            raise ValueError("expected_information_gain must be in [0, 1]")
        if self.action == "clarify":
            if self.ask_attribute not in ALLOWED_ATTRIBUTES:
                raise ValueError("clarify requires one allowed ask_attribute")
        elif self.ask_attribute is not None:
            raise ValueError("recommend requires ask_attribute=None")


@dataclass(frozen=True)
class ResponsePolicyConfig:
    generality_threshold: float = 0.46
    information_gain_threshold: float = 0.055
    buying_information_gain_adjustment: float = 0.020
    browsing_information_gain_adjustment: float = -0.010
    clarification_turn_cost: float = 0.012
    late_turn_cost: float = 0.075
    recommendation_opportunity_weight: float = 0.035
    minimum_utility_margin: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.generality_threshold <= 1.0:
            raise ValueError("generality_threshold must be in [0, 1]")
        if not 0.0 <= self.information_gain_threshold <= 1.0:
            raise ValueError("information_gain_threshold must be in [0, 1]")
        utility_values = (
            self.clarification_turn_cost,
            self.late_turn_cost,
            self.recommendation_opportunity_weight,
            self.minimum_utility_margin,
        )
        if any(not 0.0 <= value <= 1.0 for value in utility_values):
            raise ValueError("decision utility values must be in [0, 1]")


class ResponsePolicy:
    def __init__(
        self,
        generality_sensor: GeneralitySensor | None = None,
        question_value: QuestionValueEstimator | None = None,
        mode_policy: VisibleModePolicy | None = None,
        config: ResponsePolicyConfig | None = None,
        decision_type: type = TurnDecision,
    ) -> None:
        self.generality_sensor = generality_sensor or GeneralitySensor()
        self.question_value = question_value or QuestionValueEstimator()
        self.mode_policy = mode_policy or VisibleModePolicy()
        self.config = config or ResponsePolicyConfig()
        self.decision_type = decision_type

    def _decision(
        self,
        action: str,
        attribute: str | None,
        code: str,
        reason: str,
        information_gain: float,
    ) -> object:
        return self.decision_type(
            action=action,
            ask_attribute=attribute,
            reason_code=code,
            reason=reason,
            expected_information_gain=min(1.0, max(0.0, information_gain)),
        )

    def choose(
        self,
        state: object,
        candidates: Sequence[object],
        turn: int,
    ) -> object:
        if not 1 <= turn <= 10:
            raise ValueError("turn must be in the official 1-to-10 range")
        if turn == 10:
            return self._decision(
                "recommend", None, "final_turn", "Turn 10 must recommend.", 0.0
            )
        if not candidates:
            return self._decision(
                "recommend",
                None,
                "insufficient_evidence",
                "No validated candidates support a clarification simulation.",
                0.0,
            )

        try:
            generality = self.generality_sensor.assess(state, candidates)
            question = self.question_value.estimate(state, candidates, turn)
        except Exception as error:
            return self._decision(
                "recommend",
                None,
                "component_fallback",
                f"Deterministic decision fallback: {type(error).__name__}.",
                0.0,
            )

        if question.best_attribute is None:
            return self._decision(
                "recommend",
                None,
                "no_eligible_attribute",
                "No unasked attribute has enough reliable candidate evidence.",
                0.0,
            )
        if generality.score < self.config.generality_threshold:
            return self._decision(
                "recommend",
                None,
                "ranking_stable",
                f"Candidate ranking is sufficiently specific (generality={generality.score:.3f}).",
                question.expected_information_gain,
            )

        mode = self.mode_policy.resolve(state)
        threshold = self.config.information_gain_threshold
        if mode == "buying":
            threshold += self.config.buying_information_gain_adjustment
        elif mode == "browsing":
            threshold += self.config.browsing_information_gain_adjustment
        threshold = min(1.0, max(0.0, threshold))
        if question.expected_information_gain < threshold:
            return self._decision(
                "recommend",
                None,
                "low_question_value",
                (
                    "Best clarification does not justify a recommendation-free turn "
                    f"({question.expected_information_gain:.3f} < {threshold:.3f})."
                ),
                question.expected_information_gain,
            )

        diagnostics = generality.diagnostics
        turn_progress = (turn - 1) / 9.0
        clarification_utility = (
            question.expected_information_gain * generality.evidence_confidence
            - self.config.clarification_turn_cost
            - self.config.late_turn_cost * turn_progress
        )
        ranking_stability = 1.0 - diagnostics.top_k_instability
        recommendation_utility = self.config.recommendation_opportunity_weight * (
            0.5 * diagnostics.score_concentration + 0.5 * ranking_stability
        )
        if (
            clarification_utility
            <= recommendation_utility + self.config.minimum_utility_margin
        ):
            return self._decision(
                "recommend",
                None,
                "low_question_value",
                (
                    "The expected rank gain does not repay a recommendation-free "
                    f"turn ({clarification_utility:.3f} <= {recommendation_utility:.3f})."
                ),
                question.expected_information_gain,
            )

        return self._decision(
            "clarify",
            question.best_attribute,
            "valuable_clarification",
            (
                f"Asking {question.best_attribute} has the greatest expected "
                f"rank-weighted Top-10 change ({question.expected_information_gain:.3f})."
            ),
            question.expected_information_gain,
        )
