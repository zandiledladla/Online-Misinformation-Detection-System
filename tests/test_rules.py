import unittest

from misinformation_detection.rules import analyse_rules


class RuleAnalysisTests(unittest.TestCase):
    def test_conspiracy_language_is_reported(self):
        result = analyse_rules("They are secretly hiding the truth in a cover-up.")
        self.assertEqual(result.label, "Misleading")
        self.assertIn("conspiracy_language", result.indicators)

    def test_neutral_security_statement_has_no_indicators(self):
        result = analyse_rules("Experts recommend enabling two-factor authentication.")
        self.assertEqual(result.label, "Truthful")
        self.assertEqual(result.score, 0)
        self.assertEqual(result.indicators, ())

    def test_one_low_weight_indicator_stays_below_threshold(self):
        result = analyse_rules("This result is definitely useful.")
        self.assertEqual(result.label, "Truthful")
        self.assertEqual(result.score, 1)

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            analyse_rules("   ")

    def test_risk_percentage_is_bounded(self):
        result = analyse_rules("Everyone knows this shocking fake-news conspiracy is definitely guaranteed.")
        self.assertGreater(result.risk_percentage, 0)
        self.assertLessEqual(result.risk_percentage, 100)


if __name__ == "__main__":
    unittest.main()
