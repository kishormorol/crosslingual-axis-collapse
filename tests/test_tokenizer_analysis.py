#!/usr/bin/env python3
"""
Unit tests for Tokenizer & Linguistic Morphology Analysis (results/tokenizer_results.json).
"""

import json
import math
import unittest
from pathlib import Path


ROOT_DIR = Path("/Users/kishormorol/crosslingual-axis-collapse")
RESULTS_JSON = ROOT_DIR / "results" / "tokenizer_results.json"


class TestTokenizerAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RESULTS_JSON.exists():
            raise AssertionError(f"Expected results file at {RESULTS_JSON}")
        with open(RESULTS_JSON, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

    def _assert_no_nan_recursive(self, obj, path="root") -> None:
        if isinstance(obj, float):
            self.assertFalse(math.isnan(obj), f"NaN encountered at {path}")
            self.assertFalse(math.isinf(obj), f"Inf encountered at {path}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                self._assert_no_nan_recursive(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                self._assert_no_nan_recursive(v, f"{path}[{idx}]")

    def test_file_structure_and_prompt_count(self) -> None:
        self.assertIn("prompts", self.data)
        self.assertIn("aggregates", self.data)
        self.assertIn("significance_tests", self.data)
        self.assertIn("tokenizers", self.data)

        prompts = self.data["prompts"]
        self.assertEqual(len(prompts), 360)
        for i in range(1, 361):
            pid = f"xorsc_{i:03d}"
            self.assertIn(pid, prompts)

    def test_no_nan_or_inf_values(self) -> None:
        self._assert_no_nan_recursive(self.data)

    def test_prompt_level_metrics_validity(self) -> None:
        for pid, rec in self.data["prompts"].items():
            for tk in ("qwen25", "qwen35"):
                self.assertGreater(rec[f"{tk}_n_tokens"], 0, f"{pid} {tk}_n_tokens <= 0")
                self.assertGreater(rec[f"{tk}_n_words"], 0, f"{pid} {tk}_n_words <= 0")
                self.assertGreater(rec[f"{tk}_fertility"], 0.0, f"{pid} {tk}_fertility <= 0")
                self.assertGreater(rec[f"{tk}_char_per_token"], 0.0)
                self.assertGreater(rec[f"{tk}_bytes_per_token"], 0.0)
                self.assertGreaterEqual(rec[f"{tk}_mfr"], 0.0)
                self.assertLessEqual(rec[f"{tk}_mfr"], 100.0)
                self.assertGreaterEqual(rec[f"{tk}_severe_mfr"], 0.0)
                self.assertLessEqual(rec[f"{tk}_severe_mfr"], rec[f"{tk}_mfr"] + 1e-9)
                self.assertGreater(rec[f"{tk}_content_word_fertility"], 0.0)
                if rec["lang"] == "en":
                    self.assertAlmostEqual(
                        rec[f"{tk}_token_length_ratio_vs_en"], 1.0, places=7
                    )

    def test_morphological_fertility_hierarchy_and_significance(self) -> None:
        by_script = self.data["aggregates"]["by_script"]
        sig = self.data["significance_tests"]["cross_script_within_tokenizer"]

        for tk in ("qwen25", "qwen35"):
            f_en = by_script[tk]["en"]["fertility"]["mean"]
            f_bn = by_script[tk]["bn"]["fertility"]["mean"]
            f_bn_latn = by_script[tk]["bn_latn"]["fertility"]["mean"]

            # Both Bengali script and Romanized Banglish have higher fertility than English
            self.assertGreater(f_bn, f_en, f"{tk}: expected bn fertility > en fertility")
            self.assertGreater(
                f_bn_latn, f_en, f"{tk}: expected bn_latn fertility > en fertility"
            )
            self.assertGreater(f_bn, f_bn_latn, f"{tk}: expected bn fertility > bn_latn fertility")

            # Paired significance p < 0.001 for bn vs en and bn_latn vs en
            for comp in ("bn_vs_en", "bn_latn_vs_en", "bn_vs_bn_latn"):
                f_test = sig[tk][comp]["fertility"]
                self.assertEqual(f_test["n_pairs"], 120)
                self.assertLess(f_test["t_pvalue"], 0.001, f"{tk} {comp} t_pvalue >= 0.001")
                self.assertLess(
                    f_test["wilcoxon_pvalue"], 0.001, f"{tk} {comp} wilcoxon_pvalue >= 0.001"
                )
                self.assertGreater(f_test["t_stat"], 0.0)

    def test_qwen35_vocab_expansion_effect_on_bangla_vs_banglish(self) -> None:
        cross_tok = self.data["significance_tests"]["cross_tokenizer_within_script"]
        # Qwen3.5 significantly reduces bn fertility compared to Qwen2.5 (p < 0.001)
        bn_f_test = cross_tok["bn"]["fertility"]
        self.assertGreater(bn_f_test["mean_x"], bn_f_test["mean_y"])
        self.assertLess(bn_f_test["t_pvalue"], 0.001)
        self.assertLess(bn_f_test["wilcoxon_pvalue"], 0.001)


if __name__ == "__main__":
    unittest.main()
