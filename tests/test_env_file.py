"""Loading a credential from a local file, safely.

The value of a `.env` is convenience; its risk is that it holds a live key in
the working tree, one `git add -A` away from a public repository. The README
forbids committing API keys, so the loader checks the ignore rule at load time
rather than trusting that someone set it up correctly once.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tikitaka.models.env_file import load_env_file, parse_env_text, verify_ignored


class ParseTests(unittest.TestCase):
    def test_it_reads_plain_assignments(self) -> None:
        self.assertEqual(parse_env_text("A=1\nB=two\n"), {"A": "1", "B": "two"})

    def test_it_ignores_comments_and_blank_lines(self) -> None:
        text = "# a comment\n\n  \nA=1\n# another\n"
        self.assertEqual(parse_env_text(text), {"A": "1"})

    def test_it_tolerates_export_and_quotes(self) -> None:
        text = "export A='one'\nB=\"two\"\n"
        self.assertEqual(parse_env_text(text), {"A": "one", "B": "two"})

    def test_a_value_containing_equals_survives(self) -> None:
        # Base64-ish keys routinely contain '=' padding.
        self.assertEqual(parse_env_text("K=abc=def==\n"), {"K": "abc=def=="})

    def test_malformed_lines_are_skipped_not_fatal(self) -> None:
        self.assertEqual(parse_env_text("no equals here\n=novalue\nA=1\n"), {"A": "1"})


class LoadTests(unittest.TestCase):
    def _write(self, directory: str, text: str) -> Path:
        path = Path(directory) / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_missing_file_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_env_file(Path(directory) / "absent"), ())

    def test_it_returns_names_and_never_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "OPENAI_API_KEY=sk-secret-value\n")
            environ: dict[str, str] = {}
            names = load_env_file(path, environ=environ)
            self.assertEqual(names, ("OPENAI_API_KEY",))
            self.assertNotIn("sk-secret-value", repr(names))
            self.assertEqual(environ["OPENAI_API_KEY"], "sk-secret-value")

    def test_a_real_environment_variable_wins(self) -> None:
        # A stale .env quietly overriding an exported key is how someone spends
        # an afternoon debugging the wrong account.
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "OPENAI_API_KEY=from-file\n")
            environ = {"OPENAI_API_KEY": "from-shell"}
            self.assertEqual(load_env_file(path, environ=environ), ())
            self.assertEqual(environ["OPENAI_API_KEY"], "from-shell")

    def test_a_blank_environment_variable_does_not_win(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "OPENAI_API_KEY=from-file\n")
            environ = {"OPENAI_API_KEY": "   "}
            self.assertEqual(load_env_file(path, environ=environ), ("OPENAI_API_KEY",))
            self.assertEqual(environ["OPENAI_API_KEY"], "from-file")

    def test_override_is_available_when_asked_for(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "OPENAI_API_KEY=from-file\n")
            environ = {"OPENAI_API_KEY": "from-shell"}
            load_env_file(path, environ=environ, override=True)
            self.assertEqual(environ["OPENAI_API_KEY"], "from-file")


class IgnoreGuardTests(unittest.TestCase):
    def test_the_repository_env_file_is_ignored(self) -> None:
        # Guards the real rule, not a fixture: if someone removes `.env` from
        # .gitignore, this fails.
        repo_env = Path(__file__).resolve().parents[1] / ".env"
        existed = repo_env.exists()
        if not existed:
            repo_env.write_text("PROBE=1\n", encoding="utf-8")
        try:
            self.assertTrue(verify_ignored(repo_env))
        finally:
            if not existed:
                repo_env.unlink()

    def test_a_visible_env_file_is_refused(self) -> None:
        # tempfile lives outside any repository, so check-ignore reports 128
        # and the loader proceeds; the refusal is proven against a real repo
        # path that is deliberately not ignored.
        repo_root = Path(__file__).resolve().parents[1]
        visible = repo_root / "not-ignored-probe.env"
        visible.write_text("OPENAI_API_KEY=x\n", encoding="utf-8")
        try:
            self.assertFalse(verify_ignored(visible))
            with self.assertRaises(PermissionError):
                load_env_file(visible, environ={})
        finally:
            visible.unlink()


if __name__ == "__main__":
    unittest.main()
