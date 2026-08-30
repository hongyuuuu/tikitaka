#!/usr/bin/env python3
"""Run a small live API probe and report what it cost and what it did.

This is the run that turns projections into measurements. It is deliberately
small, and it is deliberately *not* a quality experiment — see the warning the
script prints. At ten sessions the 95% confidence interval on hit rate is about
±0.19, so a probe cannot tell you whether the API route beats the heuristic. It
can tell you four things that nothing else can:

- how many completion tokens the configured reasoning effort actually burns,
  unknown every cost projection pivots on;
- whether the API path survives a whole session rather than a single call;
- real latency, repair counts, and failure modes;
- the true per-session cost.

Three safeguards, because a probe that lies is worse than no probe:

1. `allow_degraded=False`. Without a working credential this refuses to start
   rather than silently scoring the deterministic route and reporting it as the
   API one.
2. A preflight call before N sessions are spent, so a bad key or model name
   costs one request instead of the whole run.
3. A post-run assertion that tokens were actually consumed. Zero tokens means
   the run degraded, and the report says so loudly instead of looking clean.

The deterministic route is run over the *same* sessions for a paired comparison.
That half is free and needs no credential.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from tikitaka.models.api_llm import ApiConfig
from tikitaka.models.base import CredentialMissing, ModelError
from tikitaka.models.env_file import DEFAULT_ENV_FILE, load_env_file
from tikitaka.models.factory import PRIMARY_ROUTE, describe_route, gateway_from_env
from tikitaka.models.selector import SELECTIVE, ModelSelector
from tikitaka.orchestration.runtime import (
    DeterministicRuntimeConfig,
    RuntimeConfig,
    build_agent,
    build_deterministic_agent,
)
from tikitaka.state.session import new_session


class UsageRecordingAgent:
    """Adapts a ShoppingAgent to the evaluator and harvests per-session usage.

    The evaluator only reports prompt and completion tokens. Reasoning tokens,
    repair counts, and latency are exactly what this probe exists to measure, so
    they are pulled off the session registry instead.
    """

    def __init__(self, shopping_agent) -> None:
        self._agent = shopping_agent
        self._session_ids: list[str] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._session_ids.append(session_id)
        self._agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self._agent.respond(session_id, user_message, turn, top_k)

    def close(self) -> None:
        close = getattr(self._agent, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "UsageRecordingAgent":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def harvest(self) -> dict:
        totals = {
            "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
            "calls": 0, "repairs": 0, "estimated_cost": 0.0,
        }
        latencies: list[float] = []
        per_component: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "calls": 0,
                "repairs": 0,
                "estimated_cost": 0.0,
            }
        )
        fallback_turns = 0
        for session_id in self._session_ids:
            for event in self._agent.sessions.usage_events(session_id):
                usage = event.usage
                totals["prompt_tokens"] += usage.prompt_tokens
                totals["completion_tokens"] += usage.completion_tokens
                totals["reasoning_tokens"] += usage.reasoning_tokens
                totals["calls"] += usage.calls
                totals["repairs"] += usage.repairs
                totals["estimated_cost"] += usage.estimated_cost or 0.0
                component = per_component[event.component]
                component["prompt_tokens"] += usage.prompt_tokens
                component["completion_tokens"] += usage.completion_tokens
                component["reasoning_tokens"] += usage.reasoning_tokens
                component["calls"] += usage.calls
                component["repairs"] += usage.repairs
                component["estimated_cost"] += usage.estimated_cost or 0.0
                if usage.latency_ms:
                    latencies.append(usage.latency_ms)
                if usage.route and str(usage.route).endswith(":fallback"):
                    fallback_turns += 1
        return {
            **totals,
            "sessions": len(self._session_ids),
            "calls_by_component": {
                component: int(values["calls"])
                for component, values in sorted(per_component.items())
            },
            "usage_by_component": {
                component: values
                for component, values in sorted(per_component.items())
            },
            "fallback_turns": fallback_turns,
            "latency_ms_mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "latency_ms_p95": (
                round(sorted(latencies)[max(int(len(latencies) * 0.95) - 1, 0)], 1)
                if latencies else 0.0
            ),
        }


def stratified(samples: list[dict], count: int, seed: int) -> list[dict]:
    """Pick `count` sessions keeping every scenario represented.

    Proportional sampling alone drops Boundary at small n — it is 10 of 200 —
    and Boundary is where the no-preference handling lives, so it is the last
    scenario a probe should be blind to.
    """

    import random

    buckets: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        buckets[sample.get("scenario_type", "unknown")].append(sample)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    scenarios = sorted(buckets)
    if count <= 0:
        return []
    quotas = {name: 1 for name in scenarios if buckets[name]}
    remaining = count - sum(quotas.values())
    if remaining > 0:
        total = sum(len(buckets[name]) for name in scenarios)
        shares = {
            name: remaining * len(buckets[name]) / total for name in scenarios
        }
        for name in scenarios:
            quotas[name] += int(shares[name])
        leftover = count - sum(quotas.values())
        for name in sorted(scenarios, key=lambda n: -(shares[n] % 1)):
            if leftover <= 0:
                break
            quotas[name] += 1
            leftover -= 1

    chosen: list[dict] = []
    for name in scenarios:
        chosen.extend(buckets[name][: quotas.get(name, 0)])
    return chosen[:count]


def preflight(environ=None) -> dict:
    """Spend one request proving the route works before spending N sessions."""

    selection = gateway_from_env(environ, allow_degraded=False)
    state = new_session("preflight", {})
    started = time.perf_counter()
    _delta, usage = selection.interpreter.interpret(
        "I'm looking for running shoes.", state
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "route_id": selection.route.route_id,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "latency_ms": round(elapsed_ms, 1),
        "estimated_cost": usage.estimated_cost,
    }


def score(agent, samples, catalog_ids, categories, products) -> dict:
    """Keep per-session outcomes: aggregates cannot support a paired test.

    Both arms run the same sessions, so the informative comparison is
    per-session agreement — how often one route hits where the other misses —
    not two independent proportions. Discarding the session list throws that
    away and leaves only the weaker unpaired interval.
    """

    result = evaluate(agent, samples, catalog_ids, categories, products)
    summary = {key: value for key, value in result.items() if key != "sessions"}
    summary["session_hits"] = {
        str(item["sample_id"]): bool(item["hit"]) for item in result.get("sessions", [])
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="artifacts/live_probe.json")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--routing",
        choices=("always", "selective"),
        default="always",
        help="always-generative (default) or the SELECTIVE cost-saving policy",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the spend confirmation prompt"
    )
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help="local file to read the credential from (must be git-ignored)",
    )
    args = parser.parse_args()

    loaded = load_env_file(args.env_file)
    if loaded:
        # Names only. A credential must never reach a terminal or a log.
        print(f"loaded {', '.join(loaded)} from {args.env_file}")

    route = describe_route()
    if not route["credential_present"]:
        raise SystemExit(
            f"No credential in {route['credential_variable']}. Set it in your "
            f"shell, or put it in {args.env_file} (which is git-ignored). "
            "This probe refuses to run degraded - a deterministic run reported "
            "as an API run is worse than no run at all."
        )

    samples = load_jsonl(args.dataset)
    subset = stratified(samples, args.sessions, args.seed)
    breakdown: dict[str, int] = defaultdict(int)
    for sample in subset:
        breakdown[sample["scenario_type"]] += 1

    config = ApiConfig(route=PRIMARY_ROUTE)
    print(f"route          {route['route_id']} ({route['reasoning_level']})")
    print(f"sessions       {len(subset)}  {dict(sorted(breakdown.items()))}")
    print(f"routing        {args.routing}")
    print(f"rates          ${config.prompt_cost_per_1k * 1000:.2f}/1M in, "
          f"${config.completion_cost_per_1k * 1000:.2f}/1M out")
    print()

    if not args.yes:
        try:
            answer = input("This spends real money. Continue? [y/N] ").strip().lower()
        except EOFError:
            raise SystemExit(
                "No terminal to confirm on. Re-run with --yes to spend without "
                "a prompt."
            )
        if answer not in {"y", "yes"}:
            raise SystemExit("aborted")

    print("preflight...", end=" ", flush=True)
    try:
        check = preflight()
    except CredentialMissing as error:
        raise SystemExit(f"preflight failed: {error}")
    except ModelError as error:
        raise SystemExit(
            f"preflight failed: {error}\nNo sessions were run, so nothing was spent "
            "beyond this one request."
        )
    print(
        f"ok  {check['prompt_tokens']}p/{check['completion_tokens']}c "
        f"({check['reasoning_tokens']} reasoning) {check['latency_ms']}ms"
    )

    catalog_ids, categories, products = catalog_index(args.catalog)

    selector = None
    if args.routing == "selective":
        selector = ModelSelector(PRIMARY_ROUTE, thresholds=SELECTIVE)
    shopping_agent, route_id = build_agent(
        args.catalog, RuntimeConfig(allow_degraded=False, selector=selector)
    )
    api_agent = UsageRecordingAgent(shopping_agent)
    print(f"running {len(subset)} live sessions on {route_id}...")
    started = time.perf_counter()
    with api_agent:
        api_metrics = score(api_agent, subset, catalog_ids, categories, products)
        usage = api_agent.harvest()
    wall_s = time.perf_counter() - started

    if usage["prompt_tokens"] == 0 and usage["completion_tokens"] == 0:
        raise SystemExit(
            "PROBE INVALID: zero tokens consumed. The run degraded to the "
            "deterministic route despite allow_degraded=False. Do not report "
            "these numbers as an API result."
        )

    # Zero tokens is the obvious degradation. The dangerous one is partial:
    # a route that fails most turns still produces tokens, still completes, and
    # still prints a plausible hit rate that is really the fallback's. Observed
    # for real when the account ran out of credits mid-run and 60 of 63
    # escalations returned HTTP 429.
    attempted = usage["calls"] + usage["fallback_turns"]
    degraded_share = usage["fallback_turns"] / attempted if attempted else 0.0
    if degraded_share > 0.2:
        print()
        print("=" * 68)
        print(f"PROBE NOT VALID AS A ROUTE MEASUREMENT: "
              f"{usage['fallback_turns']} of {attempted} attempted calls fell back")
        print(f"({degraded_share:.0%}). The quality figures below are mostly the")
        print("deterministic fallback, not this route. Check credit balance and")
        print("rate limits before reporting them.")
        print("=" * 68)

    baseline_metrics = None
    if not args.skip_baseline:
        print("running the same sessions on the deterministic route (free)...")
        deterministic = UsageRecordingAgent(
            build_deterministic_agent(args.catalog, DeterministicRuntimeConfig())
        )
        with deterministic:
            baseline_metrics = score(
                deterministic, subset, catalog_ids, categories, products
            )

    turns = usage["calls"] or 1
    per_session = usage["estimated_cost"] / max(len(subset), 1)
    report = {
        "route": route,
        "routing": args.routing,
        "degraded_share": round(degraded_share, 4),
        "valid_route_measurement": degraded_share <= 0.2,
        "sessions": len(subset),
        "scenario_breakdown": dict(sorted(breakdown.items())),
        "seed": args.seed,
        "preflight": check,
        "wall_seconds": round(wall_s, 1),
        "usage": usage,
        "per_call": {
            "prompt_tokens": round(usage["prompt_tokens"] / turns, 1),
            "completion_tokens": round(usage["completion_tokens"] / turns, 1),
            "reasoning_tokens": round(usage["reasoning_tokens"] / turns, 1),
        },
        "cost": {
            "currency": config.cost_currency,
            "prompt_cost_per_1k": config.prompt_cost_per_1k,
            "completion_cost_per_1k": config.completion_cost_per_1k,
            "total": round(usage["estimated_cost"], 6),
            "per_session": round(per_session, 6),
            "projected_public_200": round(per_session * 200, 4),
            "projected_private_800": round(per_session * 800, 4),
        },
        "metrics_api": api_metrics,
        "metrics_deterministic": baseline_metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print()
    print(f"cost           ${usage['estimated_cost']:.4f} "
          f"(${per_session:.5f}/session)")
    print(f"projected      ${per_session * 200:.2f} public 200, "
          f"${per_session * 800:.2f} private 800")
    print(f"tokens/call    {report['per_call']['prompt_tokens']}p "
          f"{report['per_call']['completion_tokens']}c "
          f"({report['per_call']['reasoning_tokens']} reasoning)")
    print(f"latency        {usage['latency_ms_mean']}ms mean, "
          f"{usage['latency_ms_p95']}ms p95")
    print(f"repairs        {usage['repairs']}   fallback turns {usage['fallback_turns']}")
    print()
    api_hit = api_metrics.get("hit_rate_at_10", 0.0)
    print(f"hit@10  api {api_hit:.3f}", end="")
    if baseline_metrics is not None:
        print(f"   deterministic {baseline_metrics.get('hit_rate_at_10', 0.0):.3f}")
    else:
        print()

    n = max(len(subset), 1)
    half = 1.96 * math.sqrt(max(api_hit * (1 - api_hit), 1e-9) / n)
    print()
    print(f"NOT a quality result. At n={n} the 95% interval on hit rate is "
          f"+/-{half:.3f}.")
    print("Read this run for cost, tokens, latency and failures only. A quality")
    print("comparison needs a few hundred sessions on the held-out split.")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
