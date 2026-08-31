#!/usr/bin/env python3
"""Run one real session against the frozen catalog and render it as it happens.

`replay_trace.py` plays back a trace file. This runs the agent for real and
renders each turn the moment it is produced, which is the difference between
showing a recording and showing inference.

The session loop is not reimplemented here. It is
`capture_traces.capture_session`, which mirrors
`evaluator.local_evaluator.evaluate` exactly, including its break on the first
hit; this module just supplies the per-turn callback. A second copy of that
loop would drift from the evaluator and quietly stop being evidence.

Offline by default. `--allow-api` is required before a configured credential is
visible to the run, so demonstrating this cannot spend money by accident.

    python scripts/live_session.py --scenario browsing --pace 1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, load_jsonl
from scripts.capture_traces import (
    SCENARIOS,
    RecordingSelector,
    _first_sample,
    capture_environment,
    capture_session,
)
from scripts.replay_trace import (
    Palette,
    enable_colour,
    prepare_stdout,
    render_turn,
)
from tikitaka.models.factory import describe_route, selector_from_env
from tikitaka.orchestration.runtime import RuntimeConfig, build_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=SCENARIOS, default="browsing")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--pace",
        type=float,
        default=0.0,
        help="extra pause after each turn; use 1.0-2.0 when recording",
    )
    parser.add_argument("--width", type=int, default=76)
    parser.add_argument("--ascii", action="store_true")
    parser.add_argument(
        "--colour", "--color", dest="colour",
        choices=("auto", "always", "never"), default="auto",
    )
    parser.add_argument(
        "--allow-api",
        action="store_true",
        help="Allow configured API credentials to be used; may incur charges.",
    )
    arguments = parser.parse_args(argv)

    catalog_path = Path(arguments.catalog)
    if not catalog_path.is_file():
        raise SystemExit(
            f"{catalog_path}: the frozen catalog is required for a live run. "
            "Download catalog.jsonl from the GitHub release first."
        )

    glyphs = prepare_stdout(arguments.ascii)
    palette = Palette(enable_colour(arguments.colour))

    samples = load_jsonl(arguments.dataset)
    sample = _first_sample(samples, arguments.scenario)
    if sample is None:
        raise SystemExit(f"no {arguments.scenario} sample in {arguments.dataset}")

    environ = capture_environment(arguments.allow_api)
    route = describe_route(environ)
    catalog_ids, categories, products = catalog_index(arguments.catalog)

    print(palette.bold(f"TikiTaka live session — {arguments.scenario}"))
    print(palette.dim(glyphs.h * arguments.width))
    print(f"route     {route['route_id']}")
    print(f"mode      {'api allowed' if arguments.allow_api else 'offline (no credential visible)'}")
    print(f"catalog   {catalog_path}")
    print(palette.dim("running live — each turn appears as the agent produces it"))
    print()
    sys.stdout.flush()

    state: dict[str, object] = {"previous": None}

    def on_turn(trace) -> None:
        record = trace.to_dict()
        # The live view cannot annotate the hit: a trace is written from
        # participant-visible state and never carries the target. The summary
        # after the loop reports it.
        print(render_turn(record, state["previous"], {}, palette, glyphs, arguments.width))
        sys.stdout.flush()
        state["previous"] = record
        if arguments.pace > 0:
            import time

            time.sleep(arguments.pace)

    selector = RecordingSelector(selector_from_env(environ).primary_route)
    agent, route_id = build_agent(
        arguments.catalog, RuntimeConfig(selector=selector), environ=environ
    )
    with agent:
        _, summary = capture_session(
            agent, selector, sample, categories, products, catalog_ids,
            on_turn=on_turn,
        )

    print()
    print(palette.dim(glyphs.h * arguments.width))
    if summary["hit"]:
        print(palette.bold(palette.green(
            f"RESULT    target found on turn {summary['first_hit_turn']} "
            f"at rank {summary['best_rank']}"
        )))
    else:
        print(palette.bold(f"RESULT    no hit within {summary['turns']} turns"))
    print(
        f"COST      {summary['calls']} model calls   "
        f"{summary['prompt_tokens']} prompt / {summary['completion_tokens']} "
        f"completion tokens"
    )
    print(palette.dim(f"          route {route_id}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
