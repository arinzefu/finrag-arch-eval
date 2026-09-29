from __future__ import annotations

import unittest

import numpy as np

from scripts.run_statistics import (COMPARISONS, holm_adjust, paired_bootstrap_ci,
                                    paired_test, paired_values)


class Stage20Tests(unittest.TestCase):
    def test_predeclared_comparisons_are_unique_and_no_p0_retrieval(self) -> None:
        keys = [(a, b, metric) for _, a, b, metric, _, _ in COMPARISONS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertFalse(any(a == "P0" and source == "retrieval"
                             for _, a, _, _, source, _ in COMPARISONS))

    def test_pairing_excludes_only_na_pairs(self) -> None:
        table = {("P0", "q1"): {"score": "0"}, ("P1", "q1"): {"score": "1"},
                 ("P0", "q2"): {"score": "1"}, ("P1", "q2"): {"score": ""},
                 ("P0", "q3"): {"score": "0"}, ("P1", "q3"): {"score": "0"}}
        a, b = paired_values(table, ["q1", "q2", "q3"], "P0", "P1", "score")
        np.testing.assert_array_equal(a, [0, 0])
        np.testing.assert_array_equal(b, [1, 0])

    def test_exact_mcnemar_uses_discordant_pairs(self) -> None:
        a = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        b = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
        test, p_value, effect, a_only, b_only = paired_test(a, b, binary=True)
        self.assertIn("McNemar", test)
        self.assertEqual((a_only, b_only, effect), (2, 8, ""))
        self.assertAlmostEqual(p_value, 0.109375)

    def test_ties_and_effect_direction(self) -> None:
        _, p, effect, _, _ = paired_test(np.array([1., 2.]), np.array([1., 2.]), False)
        self.assertEqual((p, effect), (1.0, 0.0))
        _, _, effect, _, _ = paired_test(np.array([0., 0., 0.]),
                                       np.array([1., 2., 3.]), False)
        self.assertEqual(effect, 1.0)

    def test_bootstrap_repeatable_and_holm_monotone(self) -> None:
        differences = np.array([0.0, 1.0, 2.0, 3.0])
        self.assertEqual(paired_bootstrap_ci(differences, 42, 100),
                         paired_bootstrap_ci(differences, 42, 100))
        adjusted = holm_adjust([0.04, 0.001, 0.03])
        self.assertEqual(adjusted, [0.06, 0.003, 0.06])


if __name__ == "__main__":
    unittest.main()
