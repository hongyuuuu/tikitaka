"""Over-generality scoring from candidate-pool and state evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .diagnostics import CandidatePoolDiagnostics, DiagnosticsConfig, diagnose_pool


@dataclass(frozen=True)
class GeneralityConfig:
    missing_constraint_weight: float = 0.18
    effective_mass_weight: float = 0.22
    weak_margin_weight: float = 0.18
    route_disagreement_weight: float = 0.14
    attribute_uncertainty_weight: float = 0.18
    top_k_instability_weight: float = 0.10

    def __post_init__(self) -> None:
        weights = (
            self.missing_constraint_weight,
            self.effective_mass_weight,
            self.weak_margin_weight,
            self.route_disagreement_weight,
            self.attribute_uncertainty_weight,
            self.top_k_instability_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("generality weights must be non-negative with positive sum")


@dataclass(frozen=True)
class GeneralityAssessment:
    score: float
    evidence_confidence: float
    diagnostics: CandidatePoolDiagnostics


class GeneralitySensor:
    def __init__(
        self,
        config: GeneralityConfig | None = None,
        diagnostics_config: DiagnosticsConfig | None = None,
    ) -> None:
        self.config = config or GeneralityConfig()
        self.diagnostics_config = diagnostics_config or DiagnosticsConfig()

    def assess(self, state: object, candidates: Sequence[object]) -> GeneralityAssessment:
        diagnostics = diagnose_pool(state, candidates, self.diagnostics_config)
        if diagnostics.candidate_count == 0:
            return GeneralityAssessment(0.0, 0.0, diagnostics)
        eligible_uncertainty = [
            diagnostics.attribute_uncertainty[attribute]
            * diagnostics.attribute_coverage[attribute]
            for attribute in diagnostics.attribute_uncertainty
        ]
        attribute_uncertainty = max(eligible_uncertainty, default=0.0)
        weak_margin = 1.0 - (
            0.35 * diagnostics.lead_margin
            + 0.65 * diagnostics.top_k_boundary_margin
        )
        config = self.config
        weighted = (
            config.missing_constraint_weight * (1.0 - diagnostics.constraint_coverage)
            + config.effective_mass_weight * diagnostics.effective_candidate_mass
            + config.weak_margin_weight * weak_margin
            + config.route_disagreement_weight * diagnostics.route_disagreement
            + config.attribute_uncertainty_weight * attribute_uncertainty
            + config.top_k_instability_weight * diagnostics.top_k_instability
        )
        total_weight = sum(
            (
                config.missing_constraint_weight,
                config.effective_mass_weight,
                config.weak_margin_weight,
                config.route_disagreement_weight,
                config.attribute_uncertainty_weight,
                config.top_k_instability_weight,
            )
        )
        score = min(1.0, max(0.0, weighted / total_weight))
        evidence_confidence = min(
            1.0,
            max(0.0, 0.35 + 0.65 * diagnostics.metadata_sufficiency),
        )
        return GeneralityAssessment(score, evidence_confidence, diagnostics)
