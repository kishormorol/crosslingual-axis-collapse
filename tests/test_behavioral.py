import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEHAV_PATH = ROOT / "results" / "behavioral_results.json"


class TestBehavioralResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert BEHAV_PATH.exists(), f"Missing {BEHAV_PATH}"
        with open(BEHAV_PATH, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_all_four_models_audited(self):
        audit = self.data["multi_model_unsteered_audit"]
        expected = {"Qwen2.5-0.5B-Instruct", "Qwen3.5-0.8B", "qwen2.5:3b", "aya-expanse:8b"}
        self.assertEqual(set(audit.keys()), expected)

    def test_cross_lingual_over_refusal_gap(self):
        for m_name, m_stats in self.data["multi_model_unsteered_audit"].items():
            orr_en = m_stats["by_script"]["en"]["orr_mean"]
            orr_bn = m_stats["by_script"]["bn"]["orr_mean"]
            orr_bn_latn = m_stats["by_script"]["bn_latn"]["orr_mean"]
            self.assertGreater(orr_bn, orr_en)
            self.assertGreater(orr_bn_latn, orr_en)

    def test_qualitative_examples_present(self):
        examples = self.data["qualitative_examples"]
        self.assertGreaterEqual(len(examples), 6)
        for ex in examples:
            self.assertTrue(len(ex["prompt"]) > 10)
            self.assertIn("unsteered_response", ex)
            self.assertIn("cl_oscs_response", ex)


if __name__ == "__main__":
    unittest.main()
