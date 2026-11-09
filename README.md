# Cross-Lingual Axis Collapse (`X-ORSC-Bench` & `CL-OSCS`)

Official repository for the research paper:
**"Cross-Lingual Axis Collapse: Geometric Interference of Helpfulness, Harmlessness, and Honesty Subspaces Across Scripts and Orthogonal Safe-Completion Steering"**

## Key Findings

1. **Tokenizer Fragmentation Audit (`results/tokenizer_results.json`)**:
   - Across `X-ORSC-Bench` (`360` parallel prompts, `40` scenarios $\times 3$ intents $\times 3$ scripts: `en`, `bn`, `bn_latn`), `Qwen2.5` (`151,665` vocab) fragments native Bangla (`bn`) into `7.71 ± 0.82` tokens/word (`100.00%` MFR) vs. `1.23 ± 0.14` in English (`6.26x` penalty, $p < 10^{-15}$).
   - `Qwen3.5` (`248,077` vocab) reduces `bn` fertility to `3.47 ± 0.53` tokens/word (`-54.99%`), while Romanized Bangla (`bn_latn`) remains heavily fragmented in both (`2.36` and `2.24` tokens/word).

2. **Cross-Lingual Axis Collapse (`results/mechanistic_results.json`)**:
   - In `Qwen2.5-0.5B-Instruct` mid-to-late layers (`14..22`), the principal angle between the over-refusal axis and the safe-completion axis compresses from `21.96°` (`en`) to `28.36°` (`bn`), increasing `CLACI` from `0.9092` to `0.9800`.

3. **Cross-Lingual Orthogonal Safe-Completion Steering (`CL-OSCS`)**:
   - Under 5-fold cross-validation, `CL-OSCS` improves all-script Net Safe-Completion Score (`NSCS`) from `2.22%` to `63.89%` on `Qwen2.5-0.5B-Instruct` and from `1.94%` to `50.00%` on `Qwen3.5-0.8B`.

## Repository Structure

- `data/x_orsc_bench.jsonl` — Full 360-prompt parallel benchmark (`40` scenarios $\times 3$ intents $\times 3$ scripts).
- `scripts/build_dataset.py` — Generates and validates `X-ORSC-Bench`.
- `scripts/analyze_tokenizers.py` — Computes BPE fertility, MFR, Severe-MFR, and paired significance tests.
- `scripts/run_steering_experiments.py` — Mechanistic subspace probing and 5-fold CV steering evaluation.
- `scripts/run_behavioral_eval.py` — Multi-model behavioral audit across HuggingFace and Ollama models.
- `results/` — Verified experimental outputs (`tokenizer_results.json`, `mechanistic_results.json`, `behavioral_results.json`) and SVG figures.
- `paper/` — Complete manuscript (`paper.md`, `main.tex`, and `sections/01_abstract.md`..`08_limitations_and_ethics.md`).
