"""Per-turn model routing, pinning, and ablation identity (DG-04).

This module is built around what the runtime can actually see. `IntentInterpreter`
is called as `interpret(message, state)` before retrieval runs, so at routing
time there are no candidates and no ranking scores — only the message and the
`SessionStateView`. A router that asked for candidate uncertainty here would be
asking for something that does not exist yet, and would quietly never fire.

Three properties matter more than the routing policy itself:

1. **Pinning is total.** A pinned configuration fixes every task's route, so a
   rerun reproduces the same validated state. `routing_mode` reports which
   regime was in force, using the same strings `ExperimentConfig` expects.
2. **An index/route mismatch is loud.** `SearchPlan` already requires
   `embedding_route_id` and `index_id` to be set together, but *present* is not
   *correct*: a query embedded by one model against an index built by another
   produces plausible, wrong scores. Person 2 owns the index; Person 1 owns the
   refusal, and it raises rather than degrades.
3. **Every knob is declared.** `identity()` returns the fields
   `ExperimentConfig` needs from this workstream, so an ablation cannot change
   behaviour without changing the recorded fingerprint.

The prompt and schema versions are frozen here. Changing either invalidates
cached responses and makes prior reports incomparable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Callable, Mapping

from tikitaka.contracts.domain import StateDelta, Usage
from tikitaka.models.api_llm import PROMPT_VERSION
from tikitaka.models.base import ModelError, ModelRoute
from tikitaka.models.fake import HEURISTIC_ROUTE
from tikitaka.models.usage import merge
from tikitaka.state.schema import SCHEMA_VERSION

# Frozen for the submission. See the module docstring before editing.
FROZEN_PROMPT_VERSION = PROMPT_VERSION
FROZEN_SCHEMA_VERSION = SCHEMA_VERSION

INTERPRET = "interpret"
REWRITE = "rewrite"
RERANK = "rerank"
EMBED = "embed"
TASKS = frozenset({INTERPRET, REWRITE, RERANK, EMBED})

RUNTIME_AUTO = "runtime_auto"
PINNED = "pinned"

MAX_TURNS = 10

# The official simulator emits exactly one override template (evaluator/
# local_evaluator.py:85). The alternatives are carried because the private
# simulator is not guaranteed to phrase it identically, and a missed override
# is the most expensive error in the run: the evaluator discards every
# pre-override hit.
_OVERRIDE_RE = re.compile(
    r"\b(actually|instead|on second thought|ignore my earlier|"
    r"changed my mind|rather|what i need is)\b",
    re.IGNORECASE,
)


def looks_like_override(message: object) -> bool:
    """Cheap pre-interpretation guess that this turn revises the intent."""

    return isinstance(message, str) and _OVERRIDE_RE.search(message) is not None


class RouteMismatch(ModelError):
    """An embedding route was used against an index built by another model."""


def _unit(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return value


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


@dataclass(frozen=True)
class RoutingSignals:
    """Exactly what is observable when the interpreter is called.

    Every field is read off the message or the `SessionStateView`. Nothing here
    depends on retrieval or ranking, because neither has run yet.
    """

    task: str = INTERPRET
    mode_confidence: float = 1.0
    remaining_turns: int = MAX_TURNS
    constraint_count: int = 0
    observed_turns: int = 0
    override_suspected: bool = False

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {sorted(TASKS)}")
        _unit(self.mode_confidence, "mode_confidence")
        _count(self.remaining_turns, "remaining_turns")
        _count(self.constraint_count, "constraint_count")
        _count(self.observed_turns, "observed_turns")

    @property
    def has_evidence(self) -> bool:
        """Whether any turn has been reduced into this state yet.

        The interpreter runs *before* the current turn is reduced, so on the
        opening turn `mode_confidence` is 0.0 and `constraint_count` is 0 for
        every session alive. Reading those as evidence of a struggling
        heuristic would escalate the opening turn of all 200 sessions
        unconditionally — a fixed cost dressed up as a signal.
        """

        return self.observed_turns >= 1

    @classmethod
    def from_turn(
        cls,
        message: object,
        state: object,
        *,
        task: str = INTERPRET,
    ) -> "RoutingSignals":
        """Derive signals defensively from a possibly-broken state.

        The fault matrix drives states that raise on attribute access. Routing
        must never be the thing that fails a turn, so an unreadable signal
        becomes its neutral value rather than an exception: the worst outcome
        is that this turn routes conservatively.
        """

        confidence = _safe_float(state, "mode_confidence", 1.0)
        turn = max(_safe_int(state, "turn", 0), 0)
        return cls(
            task=task,
            mode_confidence=min(max(confidence, 0.0), 1.0),
            remaining_turns=max(MAX_TURNS - turn, 0),
            constraint_count=_safe_len(state, "active_constraints"),
            observed_turns=turn,
            override_suspected=looks_like_override(message),
        )


def _safe_float(state: object, name: str, default: float) -> float:
    try:
        value = getattr(state, name, default)
    except Exception:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value) if math.isfinite(float(value)) else default


def _safe_int(state: object, name: str, default: int) -> int:
    try:
        value = getattr(state, name, default)
    except Exception:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _safe_len(state: object, name: str) -> int:
    try:
        value = getattr(state, name, ())
        return len(value)
    except Exception:
        return 0


@dataclass(frozen=True)
class RoutingThresholds:
    """When the generative route is used.

    The default is `always_generative`: if a generative route is configured, it
    handles every turn. That is the behaviour `build_agent` had before routing
    existed, and it stays the default deliberately. Skipping provider calls on
    most turns would be a quality decision made with no measurement of the API
    route at all — precisely the "select the heuristic on public-set evidence
    alone" trap that Risk 5 in the build plan warns about.

    `SELECTIVE` is the cost-saving policy: engage the LLM only where the
    heuristic is measurably weak — an override turn, an unconfident mode, a
    state nothing was extracted from, a nearly spent turn budget. It is ready
    to switch on, and should be switched on when the live run shows what the
    API route is worth, not before.
    """

    always_generative: bool = True
    min_mode_confidence: float = 0.65
    low_turn_budget: int = 3
    escalate_when_unconstrained: bool = True

    def __post_init__(self) -> None:
        _unit(self.min_mode_confidence, "min_mode_confidence")
        _count(self.low_turn_budget, "low_turn_budget")


@dataclass(frozen=True)
class AblationConfig:
    """The knobs Person 4 varies. Each one appears in `identity()`.

    `profile_weight` is carried rather than applied — Person 3's ranker and
    Person 1's query builder both consume it — so the number is recorded once
    and cannot drift between the two consumers.
    """

    profile_weight: float = 0.0
    use_api_interpreter: bool = True
    use_llm_query_rewrite: bool = False
    reasoning_level: str | None = None

    def __post_init__(self) -> None:
        _unit(self.profile_weight, "profile_weight")
        if self.reasoning_level is not None and not str(self.reasoning_level).strip():
            raise ValueError("reasoning_level must be non-empty when set")

    def apply(self, route: ModelRoute) -> ModelRoute:
        """Overlay the reasoning-level ablation onto a generative route."""

        if self.reasoning_level is None or route.reasoning_level is None:
            return route
        if route.reasoning_level == self.reasoning_level:
            return route
        return replace(route, reasoning_level=self.reasoning_level)


@dataclass(frozen=True)
class RoutingDecision:
    task: str
    route: ModelRoute
    reason: str
    pinned: bool
    generative: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "route_id": self.route.route_id,
            "provider": self.route.provider,
            "model": self.route.model,
            "reasoning_level": self.route.reasoning_level,
            "reason": self.reason,
            "pinned": self.pinned,
            "generative": self.generative,
        }


class ModelSelector:
    """Chooses a route per task and explains why.

    A selector with no primary is not an error — it is the network-free
    configuration, and it reports `degraded` honestly instead of naming a route
    it cannot reach.
    """

    def __init__(
        self,
        primary: ModelRoute | None,
        fallback: ModelRoute = HEURISTIC_ROUTE,
        *,
        pins: Mapping[str, ModelRoute] | None = None,
        ablation: AblationConfig | None = None,
        thresholds: RoutingThresholds | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._ablation = ablation or AblationConfig()
        self._thresholds = thresholds or RoutingThresholds()
        pins = dict(pins or {})
        unknown = set(pins) - TASKS
        if unknown:
            raise ValueError(f"unknown pinned tasks: {sorted(unknown)}")
        self._pins = pins

    @property
    def degraded(self) -> bool:
        return self._primary is None

    @property
    def fallback_route(self) -> ModelRoute:
        return self._fallback

    @property
    def routing_mode(self) -> str:
        """`pinned` only when every routed task is fixed.

        A partial pin is still runtime routing: something is free to vary, and
        calling that run reproducible would be a lie in the report.
        """

        return PINNED if TASKS <= set(self._pins) else RUNTIME_AUTO

    def select(self, signals: RoutingSignals | None = None) -> RoutingDecision:
        signals = signals or RoutingSignals()

        pinned = self._pins.get(signals.task)
        if pinned is not None:
            return RoutingDecision(
                signals.task,
                pinned,
                "pinned",
                True,
                generative=pinned.provider != self._fallback.provider,
            )

        if self._primary is None:
            return RoutingDecision(
                signals.task, self._fallback, "no_generative_route", False, False
            )

        reason = self._reason(signals)
        if reason is None:
            return RoutingDecision(
                signals.task, self._fallback, "deterministic_sufficient", False, False
            )
        route = self._ablation.apply(self._primary)
        return RoutingDecision(signals.task, route, reason, False, True)

    def _reason(self, signals: RoutingSignals) -> str | None:
        """First matching escalation reason, or None to stay deterministic.

        Order is fixed so identical signals always record the same reason. Two
        runs that disagree on *why* they escalated are not comparable even when
        they picked the same route.
        """

        if signals.task == INTERPRET and not self._ablation.use_api_interpreter:
            return None
        if signals.task == REWRITE and not self._ablation.use_llm_query_rewrite:
            return None
        if signals.task == EMBED:
            # Embedding routes are selected by index identity, never by
            # conversational signals. An unpinned embed falls back.
            return None

        if self._thresholds.always_generative:
            return "generative_available"

        if signals.override_suspected:
            return "override_suspected"
        if signals.has_evidence:
            if signals.mode_confidence < self._thresholds.min_mode_confidence:
                return "low_mode_confidence"
            if (
                self._thresholds.escalate_when_unconstrained
                and signals.constraint_count == 0
            ):
                return "no_extracted_constraints"
        if signals.remaining_turns <= self._thresholds.low_turn_budget:
            return "low_turn_budget"
        return None

    def select_embedding(self, route: ModelRoute, index_id: str) -> ModelRoute:
        """Bind an embedding route to one index, or refuse.

        Both halves are failures: a route with no index identity cannot be shown
        to match, and a route naming a different index definitely does not. In
        either case vectors would be compared across embedding spaces and the
        resulting scores would look entirely ordinary.
        """

        if not isinstance(index_id, str) or not index_id.strip():
            raise ValueError("index_id must be a non-empty string")
        if route.index_id is None:
            raise RouteMismatch(
                f"embedding route carries no index identity; "
                f"index {index_id!r} cannot be verified",
                route,
            )
        if route.index_id != index_id:
            raise RouteMismatch(
                f"embedding route was built for index {route.index_id!r} "
                f"but was used against {index_id!r}",
                route,
            )
        return route

    def identity(self) -> dict[str, object]:
        """The fields `ExperimentConfig` needs from this workstream.

        Never includes a credential: a route is a name and a model, and the
        gateway keeps the secret.
        """

        generative = self._ablation.apply(self._primary or self._fallback)
        return {
            "prompt_version": FROZEN_PROMPT_VERSION,
            "schema_version": FROZEN_SCHEMA_VERSION,
            "routing_mode": self.routing_mode,
            "generative_provider": generative.provider,
            "generative_model": generative.model,
            "reasoning_level": generative.reasoning_level or "none",
            "fallback_route_id": self._fallback.route_id,
            "degraded": self.degraded,
            "profile_weight": self._ablation.profile_weight,
            "use_api_interpreter": self._ablation.use_api_interpreter,
            "use_llm_query_rewrite": self._ablation.use_llm_query_rewrite,
            "pinned_tasks": tuple(sorted(self._pins)),
        }


class RoutingInterpreter:
    """An `IntentInterpreter` that picks its route per turn and degrades safely.

    This replaces a build-time route choice with a per-turn one. It is also the
    contingency path: if the chosen route raises, the turn completes on the
    deterministic interpreter rather than failing, and the usage record says
    `<route>:fallback` so the report cannot claim a clean API turn.
    """

    def __init__(
        self,
        selector: ModelSelector,
        primary: object,
        fallback: object,
        *,
        on_decision: Callable[[RoutingDecision], None] | None = None,
        history_limit: int = MAX_TURNS,
    ) -> None:
        self._selector = selector
        self._primary = primary
        self._fallback = fallback
        self._on_decision = on_decision
        self._history_limit = max(int(history_limit), 0)
        self.decisions: list[RoutingDecision] = []

    @property
    def selector(self) -> ModelSelector:
        return self._selector

    @property
    def last_decision(self) -> RoutingDecision | None:
        return self.decisions[-1] if self.decisions else None

    def interpret(self, message: str, state: object) -> tuple[StateDelta, Usage]:
        decision = self._decide(message, state)
        chosen = self._primary if decision.generative else self._fallback
        route_id = decision.route.route_id

        try:
            delta, usage = chosen.interpret(message, state)
        except Exception as error:
            if chosen is self._fallback:
                raise
            spent = getattr(error, "usage", None)
            spent = spent if isinstance(spent, Usage) else Usage()
            delta, fallback_usage = self._fallback.interpret(message, state)
            fallback_usage = (
                fallback_usage if isinstance(fallback_usage, Usage) else Usage()
            )
            return delta, replace(
                merge(spent, fallback_usage), route=f"{route_id}:fallback"
            )

        if isinstance(usage, Usage) and usage.route is None:
            usage = replace(usage, route=route_id)
        return delta, usage

    def _decide(self, message: str, state: object) -> RoutingDecision:
        """Routing must never be the thing that fails a turn."""

        try:
            decision = self._selector.select(RoutingSignals.from_turn(message, state))
        except Exception:
            decision = RoutingDecision(
                INTERPRET,
                self._selector.fallback_route,
                "routing_error",
                False,
                False,
            )
        self._record(decision)
        return decision

    def _record(self, decision: RoutingDecision) -> None:
        self.decisions.append(decision)
        if len(self.decisions) > self._history_limit:
            del self.decisions[: -self._history_limit]
        if self._on_decision is not None:
            try:
                self._on_decision(decision)
            except Exception:
                pass

    def summarize(self) -> dict[str, object]:
        """Per-session routing facts for the experiment report."""

        reasons: dict[str, int] = {}
        for decision in self.decisions:
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
        return {
            "turns_routed": len(self.decisions),
            "generative_turns": sum(1 for item in self.decisions if item.generative),
            "reasons": dict(sorted(reasons.items())),
            "routing_mode": self._selector.routing_mode,
        }


#: Cost-saving policy: escalate only where the heuristic is measurably weak.
SELECTIVE = RoutingThresholds(always_generative=False)


def pin_all(route: ModelRoute) -> dict[str, ModelRoute]:
    """Pin every task to one route, for a fully reproducible run."""

    return {task: replace(route, pinned=True) for task in sorted(TASKS)}


def deterministic_selector(
    *,
    ablation: AblationConfig | None = None,
    thresholds: RoutingThresholds | None = None,
) -> ModelSelector:
    """The network-free selector: no generative route, every task pinned."""

    return ModelSelector(
        None,
        pins=pin_all(HEURISTIC_ROUTE),
        ablation=ablation,
        thresholds=thresholds,
    )


__all__ = [
    "EMBED",
    "SELECTIVE",
    "FROZEN_PROMPT_VERSION",
    "FROZEN_SCHEMA_VERSION",
    "INTERPRET",
    "MAX_TURNS",
    "PINNED",
    "RERANK",
    "REWRITE",
    "RUNTIME_AUTO",
    "TASKS",
    "AblationConfig",
    "ModelSelector",
    "RouteMismatch",
    "RoutingDecision",
    "RoutingInterpreter",
    "RoutingSignals",
    "RoutingThresholds",
    "deterministic_selector",
    "looks_like_override",
    "pin_all",
]
