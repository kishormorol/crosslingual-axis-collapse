import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MECH_PATH = ROOT / "results" / "mechanistic_results.json"


class TestMechanisticResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert MECH_PATH.exists(), f"Missing {MECH_PATH}"
        with open(MECH_PATH, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_both_models_present(self):
        models = self.data["models"]
        self.assertIn("qwen25_0.5b", models)
        self.assertIn("qwen35_0.8b", models)

    def test_part_a_geometry_and_correlations(self):
        for m_key, m_val in self.data["models"].items():
            geom = m_val["part_a_geometry"]
            by_script = geom["summary_by_script_mid_late_layers"]
            self.assertEqual(set(by_script.keys()), {"en", "bn", "bn_latn"})
            for s, stats_dict in by_script.items():
                self.assertEqual(len(geom["layer_profiles"][s]), 24)
                self.assertFalse(float("nan") == stats_dict["claci_mean"])
            corr = geom["correlations_with_tokenization"]["fertility_vs_claci"]
            self.assertIn("pearson_r", corr)
            self.assertIn("spearman_rho", corr)

    def test_part_b_cl_oscs_outperforms_baselines(self):
        for m_key, m_val in self.data["models"].items():
            methods = m_val["part_b_steering_cv"]["methods"]
            cl_nscs = methods["cl_oscs"]["overall"]["nscs_mean"]
            un_nscs = methods["unsteered"]["overall"]["nscs_mean"]
            caa_nscs = methods["standard_caa"]["overall"]["nscs_mean"]
            gs_nscs = methods["gs_caa"]["overall"]["nscs_mean"]
            self.assertGreater(cl_nscs, un_nscs)
            self.assertGreater(cl_nscs, caa_nscs)
            self.assertGreater(cl_nscs, gs_nscs)

    def test_part_c_ablations_complete(self):
        for m_key, m_val in self.data["models"].items():
            abl = m_val["part_c_ablations"]
            self.assertEqual(len(abl["layer_band_ablation"]), 5)
            self.assertEqual(len(abl["alpha_sensitivity"]), 5)


if __name__ == "__main__":
    unittest.main()
