#!/usr/bin/env python3
"""Select a P5 release report from held-out evidence and scenario safeguards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tikitaka.evaluation import select_release_report


def _load(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"report {path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports", type=Path, nargs="+",
        help="Baseline first, followed by candidate canonical reports.",
    )
    parser.add_argument("--maximum-scenario-hit-rate-drop", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    reports = [_load(path) for path in args.reports]
    selected = select_release_report(
        reports,
        maximum_scenario_hit_rate_drop=args.maximum_scenario_hit_rate_drop,
    )
    selected_index = next(index for index, report in enumerate(reports) if report is selected)
    experiment = selected.get("experiment", {})
    result = {
        "selection_policy": {
            "objective_order": ["hit_rate_at_10", "mrr", "mttc"],
            "maximum_scenario_hit_rate_drop": args.maximum_scenario_hit_rate_drop,
            "baseline": str(args.reports[0]),
        },
        "selected_report": str(args.reports[selected_index]),
        "selected_fingerprint": experiment.get("fingerprint"),
        "held_out": selected["results"]["held_out"],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
