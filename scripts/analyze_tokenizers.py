#!/usr/bin/env python3
"""
Tokenizer & Linguistic Morphology Analysis for Cross-Lingual Axis Collapse.

Evaluates BPE tokenization morphology across:
  - 2 Tokenizers:
      1. Qwen/Qwen2.5-0.5B-Instruct (vocab size ~151,665)
      2. /Users/kishormorol/models/Qwen3.5-0.8B (vocab size ~248,077)
  - 360 parallel prompts from data/x_orsc_bench.jsonl:
      40 scenarios (triplet_id) x 3 intents (benign_boundary, harmful, helpful_control)
      x 3 scripts/languages (en, bn, bn_latn) across 4 domains
      (health, culture_religion, civic_legal, cyber_safety).

Outputs:
  - /Users/kishormorol/crosslingual-axis-collapse/results/tokenizer_results.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats
from transformers import AutoTokenizer


ROOT_DIR = Path("/Users/kishormorol/crosslingual-axis-collapse")
DATA_PATH = ROOT_DIR / "data" / "x_orsc_bench.jsonl"
RESULTS_DIR = ROOT_DIR / "results"
OUTPUT_JSON_PATH = RESULTS_DIR / "tokenizer_results.json"

TOKENIZER_SPECS = {
    "qwen25": {
        "label": "Qwen2.5-0.5B-Instruct",
        "path": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    "qwen35": {
        "label": "Qwen3.5-0.8B",
        "path": "/Users/kishormorol/models/Qwen3.5-0.8B",
    },
}

PUNCT_CHARS = ".,?!:;\"'`()[]{}—-–।/\\"

METRIC_KEYS = [
    "n_tokens",
    "n_words",
    "fertility",
    "char_per_token",
    "bytes_per_token",
    "token_length_ratio_vs_en",
    "mfr",
    "severe_mfr",
    "content_word_fertility",
    "sensitive_keyword_fertility",
    "sensitive_keyword_tokens",
]


def clean_word(word: str) -> str:
    """Strip boundary punctuation while preserving internal hyphens/conjuncts."""
    cleaned = word.strip(PUNCT_CHARS)
    return cleaned if cleaned else word


def compute_word_token_counts(tokenizer: Any, words: List[str]) -> List[int]:
    """Compute BPE subword token count for each whitespace-delimited word."""
    counts: List[int] = []
    for idx, raw_w in enumerate(words):
        w = clean_word(raw_w)
        # Prepend space for non-initial words to match sentence-internal BPE merges
        probe = w if idx == 0 else (" " + w)
        toks = tokenizer.encode(probe, add_special_tokens=False)
        counts.append(max(1, len(toks)))
    return counts


def summarize_values(vals: List[float]) -> Dict[str, float]:
    """Compute exact mean, std (sample ddof=1), median, IQR, min, max."""
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    q25, median, q75 = np.percentile(arr, [25, 50, 75])
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": std,
        "median": float(median),
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def paired_significance(x: List[float], y: List[float]) -> Dict[str, float]:
    """Compute paired t-test, Wilcoxon signed-rank test, and paired Cohen's d (d_z and d_av)."""
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)
    diff = arr_x - arr_y
    n = int(diff.size)
    mean_x = float(np.mean(arr_x))
    mean_y = float(np.mean(arr_y))
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    pooled_sd = float(
        math.sqrt((np.var(arr_x, ddof=1) + np.var(arr_y, ddof=1)) / 2.0)
    ) if n > 1 else 0.0

    if sd_diff > 0:
        t_stat, t_pval = stats.ttest_rel(arr_x, arr_y)
        cohens_dz = mean_diff / sd_diff
    else:
        t_stat, t_pval = 0.0, 1.0
        cohens_dz = 0.0

    cohens_dav = (mean_diff / pooled_sd) if pooled_sd > 0 else 0.0

    non_zero = diff[diff != 0]
    if non_zero.size > 0:
        w_res = stats.wilcoxon(arr_x, arr_y, zero_method="wilcox", alternative="two-sided")
        w_stat = float(w_res.statistic)
        w_pval = float(w_res.pvalue)
    else:
        w_stat, w_pval = 0.0, 1.0

    return {
        "n_pairs": n,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "mean_diff": mean_diff,
        "ratio_x_over_y": (mean_x / mean_y) if mean_y > 0 else 0.0,
        "t_stat": float(t_stat),
        "t_pvalue": float(t_pval),
        "wilcoxon_w": w_stat,
        "wilcoxon_pvalue": w_pval,
        "cohens_d": float(cohens_dz),
        "cohens_d_pooled": float(cohens_dav),
    }


def aggregate_records(records: List[Dict[str, Any]], prefix: str) -> Dict[str, Dict[str, float]]:
    """Summarize all metrics for a subset of records and a given tokenizer prefix."""
    out: Dict[str, Dict[str, float]] = {}
    for m in METRIC_KEYS:
        key = f"{prefix}_{m}"
        vals = [float(r[key]) for r in records]
        out[m] = summarize_values(vals)
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 360, f"Expected 360 prompts, found {len(rows)}"

    tokenizers: Dict[str, Any] = {}
    tokenizer_metadata: Dict[str, Any] = {}
    for tk_key, spec in TOKENIZER_SPECS.items():
        tok = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
        tokenizers[tk_key] = tok
        tokenizer_metadata[tk_key] = {
            "label": spec["label"],
            "path": spec["path"],
            "vocab_size": int(len(tok)),
            "model_max_length": int(getattr(tok, "model_max_length", 0)),
            "is_fast": bool(getattr(tok, "is_fast", False)),
        }

    # First pass: compute per-prompt metrics for each tokenizer
    prompt_records: List[Dict[str, Any]] = []
    en_token_lookup: Dict[tuple, Dict[str, int]] = {}

    for row in rows:
        prompt_text: str = row["prompt"]
        words = prompt_text.split()
        n_words = len(words)
        n_chars = len(prompt_text)
        n_bytes = len(prompt_text.encode("utf-8"))

        kw_text: str = row.get("sensitive_keyword", "")
        kw_words = kw_text.split()
        n_kw_words = max(1, len(kw_words))

        rec: Dict[str, Any] = {
            "id": row["id"],
            "scenario_id": row["triplet_id"],
            "triplet_id": row["triplet_id"],
            "domain": row["domain"],
            "topic": row["topic"],
            "intent": row["intent"],
            "expected_action": row["expected_action"],
            "lang": row["lang"],
            "script": row["script"],
            "sensitive_keyword": kw_text,
            "n_chars": n_chars,
            "n_bytes": n_bytes,
            "n_words": n_words,
        }

        for tk_key, tok in tokenizers.items():
            full_ids = tok.encode(prompt_text, add_special_tokens=False)
            n_tok = len(full_ids)
            fertility = n_tok / n_words
            char_per_tok = n_chars / n_tok
            bytes_per_tok = n_bytes / n_tok

            word_tok_counts = compute_word_token_counts(tok, words)
            multi_tok_words = sum(1 for c in word_tok_counts if c >= 2)
            severe_tok_words = sum(1 for c in word_tok_counts if c >= 3)
            mfr = 100.0 * multi_tok_words / n_words
            severe_mfr = 100.0 * severe_tok_words / n_words

            # Domain content-word fragmentation: words with cleaned character length >= 4
            content_tok_counts = [
                c for w, c in zip(words, word_tok_counts) if len(clean_word(w)) >= 4
            ]
            if content_tok_counts:
                content_fertility = float(np.mean(content_tok_counts))
            else:
                content_fertility = float(np.mean(word_tok_counts))

            kw_ids = tok.encode(kw_text, add_special_tokens=False) if kw_text else []
            kw_n_tok = len(kw_ids)
            kw_fertility = kw_n_tok / n_kw_words

            rec[f"{tk_key}_n_tokens"] = n_tok
            rec[f"{tk_key}_n_words"] = n_words
            rec[f"{tk_key}_fertility"] = fertility
            rec[f"{tk_key}_char_per_token"] = char_per_tok
            rec[f"{tk_key}_bytes_per_token"] = bytes_per_tok
            rec[f"{tk_key}_mfr"] = mfr
            rec[f"{tk_key}_severe_mfr"] = severe_mfr
            rec[f"{tk_key}_content_word_fertility"] = content_fertility
            rec[f"{tk_key}_sensitive_keyword_tokens"] = kw_n_tok
            rec[f"{tk_key}_sensitive_keyword_fertility"] = kw_fertility

            if row["lang"] == "en":
                en_token_lookup.setdefault((row["triplet_id"], row["intent"]), {})[tk_key] = n_tok

        prompt_records.append(rec)

    # Second pass: compute cross-lingual token length ratio R_{s/en} = N_tok(s) / N_tok(en)
    for rec in prompt_records:
        pair_key = (rec["triplet_id"], rec["intent"])
        for tk_key in TOKENIZER_SPECS:
            en_tok = en_token_lookup[pair_key][tk_key]
            rec[f"{tk_key}_token_length_ratio_vs_en"] = rec[f"{tk_key}_n_tokens"] / en_tok

    # Build prompt lookup dictionary: prompt_id -> metrics
    prompt_lookup: Dict[str, Dict[str, Any]] = {rec["id"]: rec for rec in prompt_records}

    langs = ["en", "bn", "bn_latn"]
    intents = ["benign_boundary", "harmful", "helpful_control"]
    domains = ["health", "culture_religion", "civic_legal", "cyber_safety"]

    # Aggregations
    by_tokenizer: Dict[str, Any] = {}
    by_script: Dict[str, Dict[str, Any]] = {}
    by_intent: Dict[str, Dict[str, Any]] = {}
    by_domain: Dict[str, Dict[str, Any]] = {}
    by_script_intent: Dict[str, Dict[str, Dict[str, Any]]] = {}
    by_script_domain: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for tk_key in TOKENIZER_SPECS:
        by_tokenizer[tk_key] = aggregate_records(prompt_records, tk_key)

        by_script[tk_key] = {}
        for lang in langs:
            sub = [r for r in prompt_records if r["lang"] == lang]
            by_script[tk_key][lang] = aggregate_records(sub, tk_key)

        by_intent[tk_key] = {}
        for intent in intents:
            sub = [r for r in prompt_records if r["intent"] == intent]
            by_intent[tk_key][intent] = aggregate_records(sub, tk_key)

        by_domain[tk_key] = {}
        for dom in domains:
            sub = [r for r in prompt_records if r["domain"] == dom]
            by_domain[tk_key][dom] = aggregate_records(sub, tk_key)

        by_script_intent[tk_key] = {}
        for lang in langs:
            by_script_intent[tk_key][lang] = {}
            for intent in intents:
                sub = [
                    r for r in prompt_records
                    if r["lang"] == lang and r["intent"] == intent
                ]
                by_script_intent[tk_key][lang][intent] = aggregate_records(sub, tk_key)

        by_script_domain[tk_key] = {}
        for lang in langs:
            by_script_domain[tk_key][lang] = {}
            for dom in domains:
                sub = [
                    r for r in prompt_records
                    if r["lang"] == lang and r["domain"] == dom
                ]
                by_script_domain[tk_key][lang][dom] = aggregate_records(sub, tk_key)

    # Paired statistical significance across the 120 parallel scenario x intent pairs
    # Order pairs deterministically by (triplet_id, intent)
    pair_keys = sorted({(r["triplet_id"], r["intent"]) for r in prompt_records})
    assert len(pair_keys) == 120, f"Expected 120 parallel pairs, got {len(pair_keys)}"

    indexed_by_pair_lang: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    for r in prompt_records:
        pk = (r["triplet_id"], r["intent"])
        indexed_by_pair_lang.setdefault(pk, {})[r["lang"]] = r

    script_pair_comparisons = [
        ("bn_vs_en", "bn", "en"),
        ("bn_latn_vs_en", "bn_latn", "en"),
        ("bn_vs_bn_latn", "bn", "bn_latn"),
    ]
    tested_metrics = [
        "fertility",
        "n_tokens",
        "mfr",
        "severe_mfr",
        "content_word_fertility",
        "sensitive_keyword_fertility",
        "char_per_token",
        "bytes_per_token",
        "token_length_ratio_vs_en",
    ]

    significance_tests: Dict[str, Any] = {
        "cross_script_within_tokenizer": {},
        "cross_tokenizer_within_script": {},
    }

    for tk_key in TOKENIZER_SPECS:
        significance_tests["cross_script_within_tokenizer"][tk_key] = {}
        for comp_name, lang_x, lang_y in script_pair_comparisons:
            significance_tests["cross_script_within_tokenizer"][tk_key][comp_name] = {}
            for m in tested_metrics:
                vals_x = [
                    float(indexed_by_pair_lang[pk][lang_x][f"{tk_key}_{m}"])
                    for pk in pair_keys
                ]
                vals_y = [
                    float(indexed_by_pair_lang[pk][lang_y][f"{tk_key}_{m}"])
                    for pk in pair_keys
                ]
                significance_tests["cross_script_within_tokenizer"][tk_key][comp_name][m] = (
                    paired_significance(vals_x, vals_y)
                )

    for lang in langs:
        lang_recs = [indexed_by_pair_lang[pk][lang] for pk in pair_keys]
        significance_tests["cross_tokenizer_within_script"][lang] = {}
        for m in tested_metrics:
            vals_25 = [float(r[f"qwen25_{m}"]) for r in lang_recs]
            vals_35 = [float(r[f"qwen35_{m}"]) for r in lang_recs]
            significance_tests["cross_tokenizer_within_script"][lang][m] = (
                paired_significance(vals_25, vals_35)
            )

    results_payload = {
        "dataset_summary": {
            "dataset_path": str(DATA_PATH),
            "n_prompts": len(prompt_records),
            "n_scenarios": 40,
            "n_parallel_pairs_per_script": len(pair_keys),
            "scripts": langs,
            "intents": intents,
            "domains": domains,
        },
        "tokenizers": tokenizer_metadata,
        "aggregates": {
            "by_tokenizer": by_tokenizer,
            "by_script": by_script,
            "by_intent": by_intent,
            "by_domain": by_domain,
            "by_script_intent": by_script_intent,
            "by_script_domain": by_script_domain,
        },
        "significance_tests": significance_tests,
        "prompts": prompt_lookup,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False)

    print(f"Saved complete tokenizer analysis to {OUTPUT_JSON_PATH}")
    for tk_key, spec in TOKENIZER_SPECS.items():
        print(f"\n=== {spec['label']} (vocab={tokenizer_metadata[tk_key]['vocab_size']}) ===")
        for lang in langs:
            s = by_script[tk_key][lang]
            print(
                f"  [{lang:7s}] N_tok={s['n_tokens']['mean']:.2f}±{s['n_tokens']['std']:.2f} "
                f"(med={s['n_tokens']['median']:.2f}, IQR={s['n_tokens']['iqr']:.2f}) | "
                f"F={s['fertility']['mean']:.2f}±{s['fertility']['std']:.2f} "
                f"(med={s['fertility']['median']:.2f}, IQR={s['fertility']['iqr']:.2f}) | "
                f"R_s/en={s['token_length_ratio_vs_en']['mean']:.2f} | "
                f"MFR={s['mfr']['mean']:.2f}% (severe={s['severe_mfr']['mean']:.2f}%) | "
                f"ContentF={s['content_word_fertility']['mean']:.2f} | "
                f"KeywordF={s['sensitive_keyword_fertility']['mean']:.2f} | "
                f"C={s['char_per_token']['mean']:.2f} | B={s['bytes_per_token']['mean']:.2f}"
            )


if __name__ == "__main__":
    main()
