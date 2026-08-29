#!/usr/bin/env python3
"""Capture structured turn traces for one session of each official scenario.

The build plan's definition of done requires traces for Buying, Browsing,
Intent Override, and Boundary, and M6 lists them as a submission deliverable.
`tikitaka/state/trace.py` has had the machinery since P5 but nothing produced
them: `capture()` was never called outside tests.

The session loop mirrors `evaluator.local_evaluator.evaluate` exactly, including
its break on first hit, so a trace shows what the agent actually did while being
scored rather than an idealised replay.

Traces are written from participant-visible state only. Nothing derived from an
evaluator label — the target product, the scenario type, the intent card, the
behavior block — may appear in one, and the script fails rather than writes if
any does.

Runs network-free by default. With no credential the deterministic route is
used, so this costs nothing and needs no key.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from tikitaka.models.selector import ModelSelector, RoutingDecision, RoutingSignals
from tikitaka.orchestration.runtime import RuntimeConfig, build_agent
from tikitaka.models.factory import describe_route, selector_from_env
from tikitaka.state.trace import FORBIDDEN_KEYS, capture, summarize, write_jsonl

SCENARIOS = ("buying", "browsing", "intent_override", "boundary")


class RecordingSelector(ModelSelector):
    """A selector that remembers its last decision.

    `build_agent` constructs the `RoutingInterpreter` internally, so this is the
    supported way to observe routing from outside: the selector is Person 1's
    and is already injectable through `RuntimeConfig.selector`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last: RoutingDecision | None = None

    def select(self, signals: RoutingSignals | None = None) -> RoutingDecision:
        decision = super().select(signals)
        self.last = decision
        return decision


def _first_sample(samples: list[dict], scenario: str) -> dict | None:
    for sample in samples:
        if sample.get("scenario_type") == scenario:
            return sample
    return None


def _assert_label_free(payload: dict, target: str) -> None:
    """A trace that carries a label is worse than no trace at all."""

    rendered = json.dumps(payload, sort_keys=True)
    for key in FORBIDDEN_KEYS:
        if f'"{key}"' in rendered:
            raise SystemExit(f"trace leaked evaluator key {key!r}")
    if target and target in rendered:
        raise SystemExit("trace leaked the hidden target product id")


def capture_session(
    agent,
    selector: RecordingSelector,
    sample: dict,
    categories: dict,
    products: dict,
    catalog_ids: set,
) -> tuple[list, dict]:
    """Drive one session and return its traces plus a label-free summary."""

    session_id = f"trace_{uuid.uuid4().hex[:12]}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    intent_card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": intent_card, "behavior": behavior}

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(
        effective, coarse_category(categories.get(target, [])), disclosed
    )

    traces = []
    hit_turn: int | None = None
    best_rank: int | None = None

    for turn in range(1, MAX_TURNS + 1):
        failure = ""
        try:
            response = agent.respond(session_id, user_message, turn, TOP_K)
        except Exception as error:  # mirrors the evaluator's own tolerance
            failure = f"{type(error).__name__}: {error}"
            response = {"message": "", "ask_attribute": None, "recommendations": []}

        decision = selector.last
        state = agent.sessions.get(session_id)
        if state is not None:
            events = agent.sessions.usage_events(session_id)
            traces.append(
                capture(
                    state,
                    user_message,
                    turn,
                    usage=events[-1].usage if events else None,
                    route_id=decision.route.route_id if decision else "",
                    route_reason=decision.reason if decision else "",
                    routing_mode=selector.routing_mode,
                    used_fallback=bool(decision and not decision.generative),
                    failure=failure,
                )
            )

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        if override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            break
        if turn == MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            user_message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    session_summary = {
        # `hit` and `first_hit_turn` describe the run, not the hidden answer:
        # neither the target id nor the scenario label is recorded here.
        "session_id": session_id,
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        **summarize(traces),
    }
    return traces, session_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output-dir", default="artifacts/traces")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    output_dir = Path(args.output_dir)

    base = selector_from_env()
    manifest: dict = {
        "route": describe_route(),
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        sample = _first_sample(samples, scenario)
        if sample is None:
            print(f"skipping {scenario}: no sample in dataset")
            continue

        selector = RecordingSelector(base.primary_route)
        agent, route_id = build_agent(
            args.catalog,
            RuntimeConfig(selector=selector),
            environ={} if base.degraded else None,
        )
        with agent:
            traces, session_summary = capture_session(
                agent, selector, sample, categories, products, catalog_ids
            )

        payload = [trace.to_dict() for trace in traces]
        _assert_label_free(
            {"traces": payload, "summary": session_summary},
            str(sample["ground_truth"]["parent_asin"]),
        )
        path = write_jsonl(output_dir / f"{scenario}.jsonl", traces)
        manifest["scenarios"][scenario] = {"trace_file": path.name, **session_summary}
        print(
            f"{scenario:16s} turns={session_summary['turns']:2d} "
            f"hit={str(session_summary['hit']):5s} route={route_id}"
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
