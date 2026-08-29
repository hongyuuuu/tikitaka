"""Load credentials from a local `.env`, without a dependency.

The repository has no dependency manifest and runs on the standard library, so
`python-dotenv` is not available. This is the small part of it that matters.

Two rules that are not merely stylistic:

- **A real environment variable always wins.** A stale `.env` silently
  overriding an explicitly exported key is how someone spends an afternoon
  debugging the wrong account.
- **Values are never returned, logged, or echoed.** `load_env_file` reports the
  variable *names* it set so a script can say what it loaded without putting a
  credential on screen or into a report.

`.env` is in `.gitignore`, and `verify_ignored()` re-checks that at load time
rather than trusting it — the file holds a live key, the README forbids
committing one, and an ignore rule can be edited away by anyone.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

DEFAULT_ENV_FILE = ".env"


def parse_env_text(text: str) -> dict[str, str]:
    """Parse `KEY=VALUE` lines. Comments, blanks and `export ` are tolerated."""

    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        name = name.strip()
        if not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def verify_ignored(path: str | Path = DEFAULT_ENV_FILE) -> bool:
    """Whether git would ignore this file. False is a reason to stop.

    Returns True when the question is unanswerable (no git, no repository),
    because refusing to run outside a checkout would be obstructive rather than
    protective.
    """

    target = Path(path)
    if not target.exists():
        return True
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(target)],
            cwd=target.resolve().parent,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    # 0 = ignored, 1 = not ignored, 128 = not a repository.
    return result.returncode != 1


def load_env_file(
    path: str | Path = DEFAULT_ENV_FILE,
    *,
    environ: dict[str, str] | None = None,
    override: bool = False,
) -> tuple[str, ...]:
    """Load `path` into the environment. Returns the names it set, never values.

    Raises `PermissionError` if the file is not git-ignored, because a `.env`
    that git can see is one `git add -A` away from a published credential.
    """

    target = Path(path)
    if not target.exists():
        return ()
    if not verify_ignored(target):
        raise PermissionError(
            f"{target} is not git-ignored. Add it to .gitignore before putting a "
            "credential in it; the README forbids committing API keys."
        )

    environ = os.environ if environ is None else environ
    applied: list[str] = []
    for name, value in parse_env_text(target.read_text(encoding="utf-8")).items():
        if not override and (environ.get(name) or "").strip():
            continue
        environ[name] = value
        applied.append(name)
    return tuple(applied)


__all__ = [
    "DEFAULT_ENV_FILE",
    "load_env_file",
    "parse_env_text",
    "verify_ignored",
]
