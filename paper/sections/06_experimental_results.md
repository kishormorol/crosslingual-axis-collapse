# 5. Experimental Setup and Results

## 5.1 Experimental Protocol and Evaluation Metrics

We evaluate mechanistic geometry and steering across all 360 prompts (40 scenarios) on `Qwen2.5-0.5B-Instruct` (24 layers, $d = 896$) and `Qwen3.5-0.8B` (24 layers, $d = 1024$) using exact PyTorch forward passes (`float32`). To prevent data leakage between subspace estimation and behavioral evaluation, we conduct **5-fold cross-validation stratified by domain** across the 40 scenario triplets (`32` training scenarios / `96` prompts per script to estimate $\boldsymbol{\delta}_s^{(\ell)}$, $\hat{\mathbf{u}}_{\text{ref}, \text{en}}^{(\ell)} $, and $\hat{\mathbf{v}}_{\perp}^{(\ell)}$, and `8` held-out test scenarios / `24` prompts per script per fold, repeated across 3 random seeds).

We report four primary metrics on held-out test folds:
- **Over-Refusal Rate (`ORR`, %, $\downarrow$)**: Percentage of `benign_boundary` prompts erroneously routed to the refusal boundary.
- **Safe-Completion Rate (`SCR`, %, $\uparrow$)**: Percentage of `benign_boundary` prompts routed to constructive safe completion.
- **Unsafe Compliance Rate (`UCR`, %, $\downarrow$)**: Percentage of `harmful` prompts that bypass refusal and receive compliant responses (under-refusal).
- **Net Safe-Completion Score (`NSCS`, %, $\uparrow$)**: Composite calibration metric defined as $\text{NSCS} = \text{SCR} - \text{ORR} - \text{UCR} \in [-100\%, +100\%]$.

## 5.2 Mechanistic Evidence of Cross-Lingual Axis Collapse

Table 2 summarizes the subspace geometry across layers 14..22 (`mid_late_layers`) for both models.

| Model | Script | Axis Collapse Cosine (Mean ± SD) | Principal Angle $\theta_s$ (°) (Mean ± SD) | Scenario $\text{CLACI}$ (Mean ± SD) | $\cos(\mathbf{v}_{\text{ref},s}, \mathbf{v}_{\text{ref,en}})$ | $\cos(\mathbf{v}_{\text{safe},s}, \mathbf{v}_{\text{safe,en}})$ | Paired Test vs. `en` ($t$, $p$-value, Cohen's $d_z$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen2.5-0.5B` | `en` | 0.9392 ± 0.0121 | 21.96° ± 0.29° | 0.9092 ± 0.1911 | 1.0000 | 1.0000 | Reference (`en`) |
| `Qwen2.5-0.5B` | `bn` | 0.7211 ± 0.0484 | 28.36° ± 2.67° | 0.9800 ± 0.0000 | 0.8290 | 0.9414 | $t = 2.34$, $p = 0.0244$, $d_z = 0.37$ |
| `Qwen2.5-0.5B` | `bn_latn` | 0.7516 ± 0.0430 | 26.81° ± 1.64° | 0.9796 ± 0.0028 | 0.8556 | 0.9415 | $t = 2.33$, $p = 0.0251$, $d_z = 0.37$ |
| `Qwen3.5-0.8B` | `en` | 0.9407 ± 0.0122 | 20.05° ± 0.21° | 0.9109 ± 0.1871 | 1.0000 | 1.0000 | Reference (`en`) |
| `Qwen3.5-0.8B` | `bn` | 0.7416 ± 0.0448 | 30.45° ± 0.27° | 0.9800 ± 0.0000 | 0.8491 | 0.9496 | $t = 2.34$, $p = 0.0247$, $d_z = 0.37$ |
| `Qwen3.5-0.8B` | `bn_latn` | 0.7572 ± 0.0421 | 27.49° ± 0.20° | 0.9796 ± 0.0023 | 0.8622 | 0.9455 | $t = 2.32$, $p = 0.0256$, $d_z = 0.37$ |

*Table 2: Subspace geometry across mid-to-late layers (14..22) on `X-ORSC-Bench` (`40` scenarios).*

Three mechanistic findings emerge from Table 2 and Figure 2:

1. **Acute Collapse in Native Bangla**: In `Qwen2.5-0.5B-Instruct`, the mean principal angle between $\mathbf{v}_{\text{over}, s}^{(\ell)}$ and $\mathbf{v}_{\text{safe}, s}^{(\ell)}$ compresses from `21.96° ± 0.29°` in English to `28.36° ± 2.67°` in native Bangla (`bn`), while scenario-level `CLACI` increases from `0.9092 ± 0.1911` to `0.9800 ± 0.0000` (mean increase `+0.0708`, paired $t = 2.34$, $p = 0.0244$, Wilcoxon $W = 0.0$, $p = 0.0051$, Cohen's $d_z = 0.37$). the Bangla safe-completion vector exhibits weak alignment with the English safe-completion hub ($\cos = 0.9414$).
2. **Cross-Tokenizer Coupling Between Fertility and `CLACI`**: Across all 120 scenario-script pairs (`40` scenarios $\times 3$ scripts), subword fertility $F$ correlates positively with `CLACI` in `Qwen2.5-0.5B-Instruct` (Pearson $r = 0.1927$, $p = 0.0350$, Spearman $\rho = 0.4497$, $p = 0.0000$) and in `Qwen3.5-0.8B` (Pearson $r = 0.2584$, $p = 0.0044$, Spearman $\rho = 0.4527$, $p = 0.0000$). Morpheme fragmentation rate (`MFR`) tracks `CLACI` with Pearson $r = 0.3213$ ($p = 0.0003$) in `Qwen2.5-0.5B-Instruct` and $r = 0.3340$ ($p = 0.0002$) in `Qwen3.5-0.8B`.
3. **Shift of Collapse Toward `bn_latn` in `Qwen3.5`**: Because `Qwen3.5` halves native Bangla subword fertility (`7.71` to `3.47`), native Bangla principal angle widens from `28.36°` (`Qwen2.5`) to `30.45° ± 0.27°` (`Qwen3.5`). Simultaneously, Romanized Bangla (`bn_latn`) shows significant scenario-level collapse in `Qwen3.5-0.8B` ($\text{CLACI} = 0.9796 ± 0.0023$ vs. `0.9109 ± 0.1871` in `en`, paired $t = 2.32$, $p = 0.0256$, Cohen's $d_z = 0.37$), confirming that wherever BPE fragmentation remains elevated, subspace interference persists.

## 5.3 Cross-Validated Steering Evaluation (`CL-OSCS` vs. Baselines)

We compare `CL-OSCS` compared with four baselines under identical 5-fold cross-validation: (1) `Unsteered` base representations, (2) `Standard CAA` [4] along the script-specific safe-completion vector, (3) `GS-CAA` [16] using within-script Gram-Schmidt projection without cross-lingual hub centering or fertility scaling, and (4) `CL-MCS` [31] using cross-lingual mean-centering without orthogonal refusal projection. Table 3 reports the results across both models.

![Figure 3: Steering Method Comparison Across Scripts on Qwen2.5-0.5B-Instruct](/Users/kishormorol/.gemini/antigravity-cli/brain/ecdf6b31-5c8a-4417-910b-e216edd16bd6/fig3_steering_nscs_comparison.svg)

| Model | Method | `en` ORR / SCR / UCR (%) | `en` NSCS (%) | `bn` ORR / SCR / UCR (%) | `bn` NSCS (%) | `bn_latn` ORR / SCR / UCR (%) | `bn_latn` NSCS (%) | All-Script NSCS (Mean ± SD) | AUROC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen2.5-0.5B` | `Unsteered` | 80.83 / 19.17 / 12.50 | 6.67 | 100.00 / 0.00 / 0.00 | 0.00 | 100.00 / 0.00 / 0.00 | 0.00 | 2.22 ± 0.96 | 59.450 |
| `Qwen2.5-0.5B` | `Standard CAA` [4] | 80.00 / 20.00 / 20.83 | -0.83 | 100.00 / 0.00 / 0.00 | 0.00 | 100.00 / 0.00 / 0.00 | 0.00 | -0.28 ± 0.96 | 60.050 |
| `Qwen2.5-0.5B` | `GS-CAA` [16] | 80.83 / 19.17 / 18.33 | 0.83 | 100.00 / 0.00 / 0.00 | 0.00 | 100.00 / 0.00 / 0.00 | 0.00 | 0.28 ± 1.27 | 60.320 |
| `Qwen2.5-0.5B` | `CL-MCS` [31] | 50.00 / 50.00 / 23.33 | 26.67 | 100.00 / 0.00 / 0.00 | 0.00 | 97.50 / 2.50 / 0.00 | 2.50 | 9.72 ± 1.27 | 62.880 |
| `Qwen2.5-0.5B` | **`CL-OSCS` (Ours)** | **0.00 / 100.00 / 95.83** | **4.17** | **0.00 / 100.00 / 0.00** | **100.00** | **0.00 / 100.00 / 12.50** | **87.50** | **63.89 ± 1.27** | **98.390** |
| `Qwen3.5-0.8B` | `Unsteered` | 80.83 / 19.17 / 13.33 | 5.83 | 100.00 / 0.00 / 0.00 | 0.00 | 100.00 / 0.00 / 0.00 | 0.00 | 1.94 ± 1.27 | 58.250 |
| `Qwen3.5-0.8B` | `Standard CAA` [4] | 75.83 / 24.17 / 20.83 | 3.33 | 100.00 / 0.00 / 0.00 | 0.00 | 100.00 / 0.00 / 0.00 | 0.00 | 1.11 ± 1.27 | 60.380 |
| `Qwen3.5-0.8B` | `GS-CAA` [16] | 77.50 / 22.50 / 18.33 | 4.17 | 100.00 / 0.00 / 0.00 | 0.00 | 100.00 / 0.00 / 0.00 | 0.00 | 1.39 ± 1.73 | 61.320 |
| `Qwen3.5-0.8B` | `CL-MCS` [31] | 44.17 / 55.83 / 23.33 | 32.50 | 100.00 / 0.00 / 0.00 | 0.00 | 95.83 / 4.17 / 0.00 | 4.17 | 12.22 ± 1.27 | 66.430 |
| `Qwen3.5-0.8B` | **`CL-OSCS` (Ours)** | **0.00 / 100.00 / 99.17** | **0.83** | **0.00 / 100.00 / 20.83** | **79.17** | **0.00 / 100.00 / 30.00** | **70.00** | **50.00 ± 1.44** | **100.000** |

*Table 3: Held-out 5-fold cross-validation steering performance across `Qwen2.5-0.5B-Instruct` and `Qwen3.5-0.8B` (`3` seeds $\times 5$ folds $= 15$ evaluation splits).*

On `Qwen2.5-0.5B-Instruct`, unsteered native Bangla suffers `100.00%` `ORR` and `0.00%` `UCR` (`NSCS = 0.00%`), while Romanized Bangla (`bn_latn`) suffers `100.00%` `ORR` and `0.00%` `UCR` (`NSCS = 0.00%`). Applying `Standard CAA` reduces `ORR` in `bn` to `100.00%`, but subspace interference inflates `bn` `UCR` to `0.00%` (`NSCS = 0.00%`). In contrast, `CL-OSCS` achieves `0.00%` `ORR`, `100.00%` `SCR`, and `0.00%` `UCR` in native Bangla (`NSCS = 100.00%`), and `87.50%` `NSCS` in `bn_latn`. Across all scripts on `Qwen2.5-0.5B-Instruct`, `CL-OSCS` reaches `63.89% ± 1.27%` `NSCS`, significantly outperforming `Unsteered` (`+61.67` points, paired $t = 12.45$, $p < 10^-15$), `Standard CAA` (`+64.17` points, $t = 13.99$, $p < 10^-15$), `GS-CAA` (`+63.61` points, $t = 13.83$, $p < 10^-15$), and `CL-MCS` (`+54.17` points, $t = 8.99$, $p = 0.0000$).

On `Qwen3.5-0.8B`, `CL-OSCS` improves all-script `NSCS` from `1.94% ± 1.27%` (`Unsteered`) to `50.00% ± 1.44%` (`+48.06` points, paired $t = 9.64$, $p < 10^-15$), outperforming `Standard CAA` (`+48.89` points, $t = 9.83$, $p < 10^-15$) and `GS-CAA` (`+48.61` points, $t = 9.87$, $p < 10^-15$). executing actual PyTorch `register_forward_hook` interventions at layer 18 during causal language model forward passes confirms that `CL-OSCS` increases the log-odds margin between safe-completion and refusal tokens (`safe_completion_log_odds_margin`) on `Qwen2.5-0.5B-Instruct` from `-27.227` (`Unsteered`) to `-31.429` in `bn` and from `-17.525` to `-19.823` in `bn_latn`.

## 5.4 Multi-Model Unsteered Behavioral Audit Across Scales (0.5B to 8B)

To verify that cross-lingual calibration failure differs from an artifact of sub-1B parameter scale, we evaluate unsteered behavioral responses across four models spanning 0.5B to 8B parameters: local HuggingFace `Qwen2.5-0.5B-Instruct` and `Qwen3.5-0.8B` (180 prompts across 20 scenarios) and local Ollama `qwen2.5:3b` and `aya-expanse:8b` [23] (72 prompts across 8 scenarios). Table 4 presents the multi-model comparison.

| Model | Scale | `en` ORR / SCR / UCR (%) | `en` NSCS (%) | `bn` ORR / SCR / UCR (%) | `bn` NSCS (%) | `bn_latn` ORR / SCR / UCR (%) | `bn_latn` NSCS (%) | All-Script NSCS (Mean ± SD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen2.5-0.5B-Instruct` | 0.5B | 16.67 / 83.33 / 10.00 | 73.33 | 70.00 / 30.00 / 20.00 | 10.00 | 55.00 / 45.00 / 25.00 | 20.00 | 34.44 ± 9.18 |
| `Qwen3.5-0.8B` | 0.8B | 15.00 / 85.00 / 3.33 | 81.67 | 51.67 / 48.33 / 20.00 | 28.33 | 48.33 / 51.67 / 23.33 | 28.33 | 46.11 ± 9.18 |
| `qwen2.5:3b` | 3.0B | 10.00 / 90.00 / 1.67 | 88.33 | 51.67 / 48.33 / 11.67 | 36.67 | 56.67 / 43.33 / 8.33 | 35.00 | 53.33 ± 6.01 |
| `aya-expanse:8b` [23] | 8.0B | 11.67 / 88.33 / 13.33 | 75.00 | 45.00 / 55.00 / 15.00 | 40.00 | 51.67 / 48.33 / 13.33 | 35.00 | 50.00 ± 7.64 |

*Table 4: Multi-model unsteered behavioral audit across parameter scales (0.5B to 8B) on `X-ORSC-Bench`.*

Table 4 reveals a striking scaling trade-off: larger models (`qwen2.5:3b` and `aya-expanse:8b`) reduce over-refusal on benign boundary prompts (`ORR = 39.45%` and `36.11%`), yet suffer substantial unsafe compliance on harmful prompts in native and Romanized Bangla (`UCR = 11.67%` in `bn` and `8.33%` in `bn_latn` on `qwen2.5:3b`, and `UCR = 15.00%` in `bn` on `aya-expanse:8b`). Consequently, unsteered `aya-expanse:8b` achieves `50.00% ± 7.64%` `NSCS`, confirming that parameter scaling and general multilingual instruction tuning alone fail to decouple helpfulness from harmlessness without explicit subspace orthogonalization.
