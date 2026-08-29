"""The prompt and the parser must agree, and nothing used to check that.

The API route scored 0.000 on a ten-session live probe while the deterministic
route scored 0.800 on the same sessions. Not one call failed: the model returned
clean JSON every time, and the parser discarded every operation in it. The
prompt said "matching the supplied schema" while supplying no schema, so the
model chose `value`; the parser requires `new_value`. The prompt also said to
set generality "high", a word, for a field defined as a float.

Every existing test fed hand-written fixtures authored against the contract, so
they proved the parser reads the contract and never that the prompt asks for it.
These tests close that loop offline.
"""

from __future__ import annotations

import json
import unittest

from tikitaka.contracts.domain import (
    Attribute,
    ConstraintPolarity,
    ConstraintStrength,
    InferredMode,
    OperationScope,
    StateOperationKind,
)
from tikitaka.models.api_llm import (
    PROMPT_EXAMPLE_OUTPUT,
    PROMPT_VERSION,
    build_prompt,
)
from tikitaka.state.schema import SCHEMA_VERSION, parse
from tikitaka.state.session import new_session

REQUIRED_FIELDS = (
    "operation",
    "attribute",
    "new_value",
    "old_value",
    "scope",
    "polarity",
    "strength",
    "confidence",
    "inferred_mode",
    "mode_confidence",
    "generality",
)


def _prompt() -> str:
    return build_prompt("I'm looking for running shoes.", new_session("s", {}))


class PromptNamesTheContractTests(unittest.TestCase):
    def test_every_field_the_parser_requires_appears_in_the_prompt(self) -> None:
        prompt = _prompt()
        for field in REQUIRED_FIELDS:
            self.assertIn(field, prompt, f"prompt never mentions {field!r}")

    def test_the_prompt_names_the_value_field_exactly(self) -> None:
        # The live failure: the model emitted "value" and every operation was
        # discarded as "missing a required field".
        prompt = _prompt()
        self.assertIn("new_value", prompt)
        self.assertIn('called "new_value", never "value"', prompt)

    def test_it_does_not_ask_for_generality_as_a_word(self) -> None:
        # "Set generality high when the request is vague" produced
        # {"generality": "high"} against a float field.
        prompt = _prompt().lower()
        self.assertNotIn("generality high", prompt)
        self.assertIn("numbers, not words", prompt)

    def test_every_enum_value_the_parser_accepts_is_offered(self) -> None:
        prompt = _prompt()
        for enum in (
            StateOperationKind,
            Attribute,
            ConstraintPolarity,
            ConstraintStrength,
            OperationScope,
            InferredMode,
        ):
            for member in enum:
                self.assertIn(
                    member.value, prompt, f"{enum.__name__}.{member.value} unmentioned"
                )

    def test_versions_are_stated_so_a_reply_can_be_attributed(self) -> None:
        prompt = _prompt()
        self.assertIn(PROMPT_VERSION, prompt)
        self.assertIn(SCHEMA_VERSION, prompt)


class WorkedExampleTests(unittest.TestCase):
    """The example in the prompt is the one thing a model will copy verbatim."""

    def test_it_is_valid_json(self) -> None:
        self.assertIsInstance(json.loads(PROMPT_EXAMPLE_OUTPUT), dict)

    def test_the_parser_accepts_it_without_a_single_error(self) -> None:
        result = parse(PROMPT_EXAMPLE_OUTPUT)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.delta.rejected_operations, 0)
        self.assertEqual(len(result.delta.operations), 1)

    def test_it_actually_appears_in_the_prompt(self) -> None:
        # An example that drifts out of the prompt guards nothing.
        compact = "".join(_prompt().split())
        self.assertIn("".join(PROMPT_EXAMPLE_OUTPUT.split()), compact)

    def test_the_shape_the_model_used_to_emit_is_still_rejected(self) -> None:
        # Kept as a regression: if "value" ever starts parsing, the prompt and
        # the parser have drifted apart again in the other direction.
        legacy = json.dumps(
            {
                "operations": [
                    {"operation": "add", "attribute": "category", "value": "sports bras"}
                ],
                "inferred_mode": "browsing",
                "generality": "high",
            }
        )
        result = parse(legacy)
        self.assertFalse(result.ok)
        self.assertEqual(result.delta.rejected_operations, 1)
        self.assertEqual(result.delta.generality, 0.0)


if __name__ == "__main__":
    unittest.main()
