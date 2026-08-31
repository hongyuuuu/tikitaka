#!/usr/bin/env python3
"""Replay a captured turn trace as a readable conversation, for the demo video.

`artifacts/traces/*.jsonl` is written for machines: one dense JSON object per
turn. That is the right shape for error analysis and the wrong shape to point a
screen recorder at. This renders the same records as a paced conversation with
the state panel that a viewer actually needs to follow.

Nothing here recomputes anything. Every field shown is read straight from the
trace, so what appears on camera is what the agent recorded while it was being
scored. Two things are derived, both stated plainly in the output:

- the per-turn action, from the growth of the cumulative `asked_attributes`
  set (a turn that adds an attribute was a CLARIFY, a turn that adds none was a
  RECOMMEND, which is exactly the mutually-exclusive turn policy);
- the hit turn and rank, read from the sibling `manifest.json` summary, because
  a trace is written from participant-visible state and cannot contain the
  target.

Standard library only, and it never imports `tikitaka`, so it runs in a bare
checkout with no catalog present.

    python3 scripts/replay_trace.py artifacts/traces/browsing.jsonl --pace 2
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Sequence

# The trace's own one-line query rendering marks a hard constraint by suffixing
# the attribute with "!", which reads as "!=" and looks like a negation on
# screen. We render from the structured constraint list instead.
HARD = "hard"


class Glyphs:
    """Box-drawing characters, downgraded to ASCII when the console can't encode.

    A Windows console still defaulting to cp1252 raises UnicodeEncodeError on
    the box characters. Discovering that mid-recording is exactly the kind of
    surprise this script exists to prevent, so we test the encoding up front
    and fall back rather than trusting it.
    """

    UNICODE = {"tl": "┌", "v": "│", "bl": "└", "h": "─", "ne": "≠", "arrow": "<-"}
    ASCII = {"tl": "+", "v": "|", "bl": "+", "h": "-", "ne": "!=", "arrow": "<-"}

    def __init__(self, unicode_ok: bool) -> None:
        self._marks = self.UNICODE if unicode_ok else self.ASCII

    def __getattr__(self, name: str) -> str:
        try:
            return self._marks[name]
        except KeyError:  # pragma: no cover - programming error
            raise AttributeError(name) from None


def prepare_stdout(force_ascii: bool) -> Glyphs:
    """Put stdout in UTF-8 if we can, and report which glyph set is safe."""

    if force_ascii:
        return Glyphs(False)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):  # pragma: no cover - stream-dependent
            pass
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "".join(Glyphs.UNICODE.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return Glyphs(False)
    return Glyphs(True)


class Palette:
    """ANSI codes, or empty strings when colour is off."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def magenta(self, text: str) -> str:
        return self._wrap("35", text)


def enable_colour(requested: str) -> bool:
    """Decide whether to emit ANSI, enabling VT on Windows consoles if needed."""

    if requested == "never":
        return False
    if requested == "always":
        _enable_windows_vt()
        return True
    if not sys.stdout.isatty():
        return False
    _enable_windows_vt()
    return True


def _enable_windows_vt() -> None:
    """Turn on virtual-terminal processing so ANSI works in older consoles."""

    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # -11 is STD_OUTPUT_HANDLE; 0x0007 keeps the existing flags plus
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004).
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:  # pragma: no cover - console feature detection only
        pass


def load_turns(path: Path) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{number}: not valid JSON ({error})") from error
    if not turns:
        raise SystemExit(f"{path}: no turns found")
    return sorted(turns, key=lambda turn: turn.get("turn", 0))


def load_summary(trace_path: Path, manifest_path: Path | None) -> dict[str, Any]:
    """Return the scenario summary for this trace, or an empty mapping."""

    path = manifest_path or trace_path.parent / "manifest.json"
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    for summary in (manifest.get("scenarios") or {}).values():
        if summary.get("trace_file") == trace_path.name:
            return summary
    return {}


def action_for(turn: dict[str, Any], previous: dict[str, Any] | None) -> tuple[str, str]:
    """Derive (action, detail) from the growth of the cumulative asked set."""

    asked = set(turn.get("asked_attributes") or ())
    before = set((previous or {}).get("asked_attributes") or ())
    new = sorted(asked - before)
    if new:
        return "CLARIFY", ", ".join(new)
    return "RECOMMEND", "up to 10 ranked products, ask_attribute = null"


def format_constraint(constraint: dict[str, Any], glyphs: Glyphs) -> str:
    attribute = str(constraint.get("attribute", "?"))
    value = str(constraint.get("value", ""))
    strength = str(constraint.get("strength", ""))
    polarity = str(constraint.get("polarity", "include"))
    operator = "=" if polarity == "include" else glyphs.ne
    source = constraint.get("source_turn")
    tag = strength if source is None else f"{strength}, from turn {source}"
    return f"{attribute} {operator} {value}  [{tag}]"


def render_turn(
    turn: dict[str, Any],
    previous: dict[str, Any] | None,
    summary: dict[str, Any],
    palette: Palette,
    glyphs: Glyphs,
    width: int,
) -> str:
    number = turn.get("turn", "?")
    action, detail = action_for(turn, previous)
    lines: list[str] = []

    prefix = f"{glyphs.tl}{glyphs.h} TURN {number} "
    lines.append(palette.dim(prefix + glyphs.h * max(4, width - len(prefix))))

    message = str(turn.get("message", ""))
    for index, chunk in enumerate(textwrap.wrap(message, width=width - 14) or [""]):
        label = palette.cyan("CUSTOMER") if index == 0 else "        "
        lines.append(f"{palette.dim(glyphs.v)} {label}  {chunk}")

    lines.append(palette.dim(glyphs.v))

    colour = palette.yellow if action == "CLARIFY" else palette.green
    lines.append(
        f"{palette.dim(glyphs.v)} {palette.bold('AGENT')}     "
        f"{colour(action)}  {palette.dim(detail)}"
    )

    hit_turn = summary.get("first_hit_turn")
    if hit_turn is not None and hit_turn == number:
        rank = summary.get("best_rank")
        banner = f"TARGET FOUND at rank {rank}" if rank else "TARGET FOUND"
        lines.append(f"{palette.dim(glyphs.v)} {' ' * 10}{palette.bold(palette.green(banner))}")

    lines.append(palette.dim(glyphs.v))

    mode = str(turn.get("mode", "unknown"))
    generality = turn.get("generality")
    generality_text = "n/a" if generality is None else f"{float(generality):.2f}"
    version = turn.get("intent_version", 1)
    version_text = f"intent v{version}"
    if previous is not None and previous.get("intent_version") != version:
        version_text = palette.magenta(f"{version_text}  {glyphs.arrow} INTENT OVERRIDE")
    lines.append(
        f"{palette.dim(glyphs.v)} {palette.bold('STATE')}     "
        f"mode={mode}   generality={generality_text}   {version_text}"
    )

    constraints: Sequence[dict[str, Any]] = turn.get("active_constraints") or ()
    if constraints:
        for index, constraint in enumerate(constraints):
            label = "constraints:" if index == 0 else "            "
            text = format_constraint(constraint, glyphs)
            for part in textwrap.wrap(text, width=width - 26) or [""]:
                lines.append(f"{palette.dim(glyphs.v)} {' ' * 10}{label} {part}")
                label = "            "
    else:
        lines.append(f"{palette.dim(glyphs.v)} {' ' * 10}constraints: (none yet)")

    no_preference = turn.get("no_preference") or ()
    if no_preference:
        lines.append(
            f"{palette.dim(glyphs.v)} {' ' * 10}no preference: {', '.join(sorted(no_preference))}"
        )

    route = str(turn.get("route_id", "?"))
    reason = str(turn.get("route_reason", ""))
    route_text = f"route: {route}" + (f"  ({reason})" if reason else "")
    lines.append(f"{palette.dim(glyphs.v)} {' ' * 10}{palette.dim(route_text)}")

    failure = str(turn.get("failure") or "")
    if failure:
        lines.append(f"{palette.dim(glyphs.v)} {' ' * 10}{palette.yellow('failure: ' + failure)}")

    lines.append(palette.dim(glyphs.bl + glyphs.h * max(8, width - 1)))
    return "\n".join(lines)


def render_header(
    turns: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    palette: Palette,
    glyphs: Glyphs,
    width: int,
) -> str:
    first = turns[0]
    lines = [
        palette.bold("TikiTaka session replay"),
        palette.dim(glyphs.h * width),
        f"session   {first.get('session_id', '?')}",
        f"route     {first.get('route_id', '?')}  "
        f"({first.get('routing_mode', 'runtime_auto')})",
        f"turns     {len(turns)}",
    ]
    return "\n".join(lines)


def render_footer(
    turns: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    palette: Palette,
    glyphs: Glyphs,
    width: int,
) -> str:
    last = turns[-1]
    lines = [palette.dim(glyphs.h * width)]
    if summary:
        if summary.get("hit"):
            lines.append(
                palette.bold(palette.green(
                    f"RESULT    hit on turn {summary.get('first_hit_turn')} "
                    f"at rank {summary.get('best_rank')}"
                ))
            )
        else:
            lines.append(palette.bold(f"RESULT    no hit within {len(turns)} turns"))
    prompt = last.get("prompt_tokens", 0)
    completion = last.get("completion_tokens", 0)
    calls = summary.get("calls", last.get("calls", 0))
    cost = summary.get("estimated_cost")
    cost_text = "$0.00" if not cost else f"${float(cost):.4f}"
    lines.append(
        f"COST      {calls} model calls   "
        f"{prompt} prompt / {completion} completion tokens   {cost_text}"
    )
    if last.get("used_fallback"):
        lines.append(palette.dim("          deterministic route: no credential, no network"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", type=Path, help="path to a trace .jsonl file")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="trace manifest (default: manifest.json beside the trace)",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=0.0,
        help="seconds to pause between turns; use 1.5-2.5 when recording",
    )
    parser.add_argument("--width", type=int, default=76, help="wrap width (default 76)")
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="force ASCII box characters instead of Unicode",
    )
    parser.add_argument(
        "--colour",
        "--color",
        dest="colour",
        choices=("auto", "always", "never"),
        default="auto",
    )
    arguments = parser.parse_args(argv)

    if not arguments.trace.is_file():
        raise SystemExit(f"{arguments.trace}: no such trace file")

    glyphs = prepare_stdout(arguments.ascii)
    turns = load_turns(arguments.trace)
    summary = load_summary(arguments.trace, arguments.manifest)
    palette = Palette(enable_colour(arguments.colour))

    print(render_header(turns, summary, palette, glyphs, arguments.width))
    print()
    previous: dict[str, Any] | None = None
    for turn in turns:
        if arguments.pace > 0 and previous is not None:
            time.sleep(arguments.pace)
        print(render_turn(turn, previous, summary, palette, glyphs, arguments.width))
        sys.stdout.flush()
        previous = turn
    print()
    print(render_footer(turns, summary, palette, glyphs, arguments.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
