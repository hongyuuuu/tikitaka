"""Visible-state-only mode policy used for deterministic fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass

from tikitaka.ranking.constraints import active_constraints, clamp01, enum_value


@dataclass(frozen=True)
class ModePolicyConfig:
    trusted_mode_confidence: float = 0.60
    buying_hard_constraint_count: int = 2


class VisibleModePolicy:
    """Resolve Buying/Browsing without ever reading evaluator scenario labels."""

    def __init__(self, config: ModePolicyConfig | None = None) -> None:
        self.config = config or ModePolicyConfig()

    def resolve(self, state: object) -> str:
        mode = enum_value(getattr(state, "mode", "unknown"))
        confidence = clamp01(getattr(state, "mode_confidence", 0.0))
        if mode in {"buying", "browsing"} and confidence >= self.config.trusted_mode_confidence:
            return mode
        constraints = active_constraints(state)
        hard_count = sum(
            enum_value(getattr(item, "strength", "soft")) == "hard"
            and clamp01(getattr(item, "confidence", 0.0)) >= 0.70
            for item in constraints
        )
        if hard_count >= self.config.buying_hard_constraint_count:
            return "buying"
        if constraints:
            return "browsing"
        return "unknown"
