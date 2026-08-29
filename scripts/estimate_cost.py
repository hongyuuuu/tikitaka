#!/usr/bin/env python3
"""Estimate what a full API run would cost, without making a single API call.

Rates are not measurable — they are your provider's published price, and the
README is explicit that the organiser supplies no credits and no rate card. What
*is* measurable offline is volume, because `build_prompt()` is a pure function of
`(message, state, PROMPT_VERSION)`: the exact prompt that would have been sent
can be built and counted without sending it.

So this script measures the volume and leaves the rates as arguments. Feed it
your account's numbers and it prints the bill.

Two honest limits, both stated in the output rather than buried:

1. The conversation is driven by the deterministic route. With the LLM in the
   loop the state — and therefore later prompts — would diverge. Turn counts
   would likely *fall* (better interpretation reaches the target sooner), so
   this is closer to an upper bound on turns than a point estimate.
2. Without the provider's tokenizer installed, tokens are approximated from
   character length. Use `--chars-per-token` to calibrate against a real
   response's reported `prompt_tokens`.
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
from tikitaka.models.api_llm import PROMPT_VERSION, build_prompt
from tikitaka.orchestration.runtime import RuntimeConfig, build_agent

try:  # optional: exact counts when the provider tokenizer is installed
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - depends on the local environment
    _ENCODING = None


def count_tokens(text: str, chars_per_token: float) -> int:
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return int(round(len(text) / chars_per_token))


def measure(catalog: str, dataset: str, limit: int | None, chars_per_token: float) -> dict:
    samples = load_jsonl(dataset)
    if limit:
        samples = samples[:limit]
    catalog_ids, categories, products = catalog_index(catalog)

    agent, _route = build_agent(catalog, RuntimeConfig(enable_llm_reranker=False), environ={})
    total_prompt_tokens = 0
    total_turns = 0
    longest = 0

    with agent:
        for sample in samples:
            session_id = f"cost_{uuid.uuid4().hex[:10]}"
            agent.reset(session_id, sample["user_profile"])
            target = str(sample["ground_truth"]["parent_asin"])
            card, behavior = materialize_hidden_fields(sample, products)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = sample["scenario_type"] != "intent_override"
            message = initial_message(
                effective, coarse_category(categories.get(target, [])), disclosed
            )

            for turn in range(1, MAX_TURNS + 1):
                state = agent.sessions.get(session_id)
                if state is not None:
                    tokens = count_tokens(build_prompt(message, state), chars_per_token)
                    total_prompt_tokens += tokens
                    longest = max(longest, tokens)
                total_turns += 1

                response = agent.respond(session_id, message, turn, TOP_K)
                ranked = normalize_recommendations(
                    response.get("recommendations"), catalog_ids
                )
                if override_applied and target in ranked:
                    break
                if turn == MAX_TURNS:
                    break
                override = effective.get("behavior", {}).get("override") or {}
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    if str(override.get("new_value", "")):
                        disclosed.add(str(override["new_value"]))
                    message = str(override.get("message", "Actually, ignore that."))
                else:
                    message, boundary_used = customer_reply(
                        effective, response.get("ask_attribute"), disclosed, boundary_used
                    )

    return {
        "sessions": len(samples),
        "turns": total_turns,
        "prompt_tokens": total_prompt_tokens,
        "longest_prompt_tokens": longest,
        "tokenizer": "tiktoken/cl100k_base" if _ENCODING else f"chars/{chars_per_token}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument(
        "--completion-tokens-per-call",
        type=float,
        default=148.0,
        help="measured from a live probe; includes reasoning tokens, which the "
        "provider reports as a subset of completion rather than in addition",
    )
    parser.add_argument("--prompt-cost-per-1k", type=float, default=0.0)
    parser.add_argument("--completion-cost-per-1k", type=float, default=0.0)
    parser.add_argument("--private-sessions", type=int, default=800)
    args = parser.parse_args()

    result = measure(args.catalog, args.dataset, args.limit, args.chars_per_token)
    completion = result["turns"] * args.completion_tokens_per_call
    prompt_cost = result["prompt_tokens"] / 1000.0 * args.prompt_cost_per_1k
    completion_cost = completion / 1000.0 * args.completion_cost_per_1k
    total = prompt_cost + completion_cost
    per_session = total / result["sessions"] if result["sessions"] else 0.0

    print(json.dumps({**result, "completion_tokens_estimated": int(completion)}, indent=2))
    print()
    if args.prompt_cost_per_1k == 0.0 and args.completion_cost_per_1k == 0.0:
        print("No rates supplied, so no cost is reported. Volume only.")
        print("Re-run with --prompt-cost-per-1k and --completion-cost-per-1k")
        print("from your provider's pricing page.")
    else:
        print(f"public set ({result['sessions']} sessions): {total:.4f}")
        print(f"per session:                     {per_session:.6f}")
        print(f"projected private ({args.private_sessions}): "
              f"{per_session * args.private_sessions:.4f}")
    print()
    print("Caveats: turns measured on the deterministic route, so the LLM route")
    print("would likely need fewer; token counts are "
          f"{result['tokenizer']}. Prompt version {PROMPT_VERSION}.")


if __name__ == "__main__":
    main()
