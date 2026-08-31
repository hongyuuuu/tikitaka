from __future__ import annotations

import unittest

from scripts.run_experiment import _hybrid_config, _parser


class RunExperimentHybridConfigTests(unittest.TestCase):
    def test_explicit_hybrid_weights_reach_retriever_config(self) -> None:
        arguments = _parser().parse_args(
            [
                "--name",
                "weighted-hybrid",
                "--retrieval-policy",
                "hybrid",
                "--hybrid-sparse-weight",
                "1.0",
                "--hybrid-dense-weight",
                "0.5",
            ]
        )

        config = _hybrid_config(arguments)

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.sparse_weight, 1.0)
        self.assertEqual(config.dense_weight, 0.5)

    def test_unspecified_hybrid_weights_preserve_production_defaults(self) -> None:
        arguments = _parser().parse_args(
            ["--name", "default-hybrid", "--retrieval-policy", "hybrid"]
        )

        config = _hybrid_config(arguments)

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.sparse_weight, 1.0)
        self.assertEqual(config.dense_weight, 1.0)

    def test_sparse_route_has_no_hybrid_config(self) -> None:
        arguments = _parser().parse_args(["--name", "sparse-control"])

        self.assertIsNone(_hybrid_config(arguments))

    def test_invalid_hybrid_weights_fail_closed(self) -> None:
        arguments = _parser().parse_args(
            [
                "--name",
                "invalid-hybrid",
                "--retrieval-policy",
                "hybrid",
                "--hybrid-sparse-weight",
                "0",
                "--hybrid-dense-weight",
                "0",
            ]
        )

        with self.assertRaises(ValueError):
            _hybrid_config(arguments)


if __name__ == "__main__":
    unittest.main()
