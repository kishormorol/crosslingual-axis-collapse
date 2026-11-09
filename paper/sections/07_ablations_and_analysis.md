# 6. Ablation Studies and Domain Analysis

## 6.1 Component Ablations of `CL-OSCS`

To isolate the contribution of per mathematical component in `CL-OSCS`, we evaluate three ablated variants under the same 5-fold cross-validation protocol: (1) `CL-OSCS w/o Orthogonality` (`cl_oscs_no_orth`, removing Gram-Schmidt refusal projection), (2) `CL-OSCS w/o Centering` (`cl_oscs_no_center`, removing script-centroid shift $\boldsymbol{\delta}_s^{(\ell)}$), and (3) `CL-OSCS w/o Fertility Scaling` (`cl_oscs_no_fertility`, setting $\eta = 0$). We also ablate the intervention layer band and steering norm $\alpha$. Table 5 summarizes the component, layer-band, and $\alpha$-sensitivity ablations on `Qwen2.5-0.5B-Instruct` and `Qwen3.5-0.8B`.

| Ablation Category | Variant / Setting | `Qwen2.5-0.5B` `en` NSCS (%) | `Qwen2.5-0.5B` `bn` NSCS (%) | `Qwen2.5-0.5B` `bn_latn` NSCS (%) | `Qwen2.5-0.5B` All-Script NSCS (%) | `Qwen3.5-0.8B` All-Script NSCS (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Method** | **`CL-OSCS` (Full)** | **4.17** | **100.00** | **87.50** | **63.89 ± 1.27** | **50.00 ± 1.44** |
| Component Ablation | `w/o Orthogonal Projection` | 27.50 | 0.00 | 2.50 | 10.00 ± 1.67 | 11.39 ± 1.92 |
| Component Ablation | `w/o Script Centering` | 20.83 | 0.00 | 2.50 | 7.78 ± 1.27 | 6.94 ± 1.73 |
| Component Ablation | `w/o Fertility Scaling` | 22.50 | 0.00 | 5.83 | 9.44 ± 0.96 | 14.44 ± 1.73 |
| Layer-Band Ablation | `Early Layers (1..8)` | -7.50 | 0.00 | -0.00 | -2.50 ± 1.67 | -2.22 ± 0.96 |
| Layer-Band Ablation | `Middle Layers (9..16)` | 10.83 | 100.00 | 91.67 | 67.50 ± 0.83 | 62.22 ± 0.96 |
| Layer-Band Ablation | `Mid-Late Layers (14..22)` | 4.17 | 100.00 | 87.50 | 63.89 ± 1.27 | 50.00 ± 1.44 |
| Layer-Band Ablation | `Late Layers (21..24)` | 10.83 | 100.00 | 92.50 | 67.78 ± 1.27 | 62.50 ± 0.83 |
| Layer-Band Ablation | `All Layers (1..24)` | 11.67 | 100.00 | 92.50 | 68.06 ± 1.73 | 64.72 ± 0.48 |
| Strength Sensitivity | `$\alpha = 0.5$` | 13.33 | 100.00 | 95.00 | 69.44 ± 2.68 | 67.78 ± 1.92 |
| Strength Sensitivity | `$\alpha = 1.5$ (Default)` | 4.17 | 100.00 | 87.50 | 63.89 ± 1.27 | 50.00 ± 1.44 |
| Strength Sensitivity | `$\alpha = 3.0$` | 0.83 | 0.00 | 61.67 | 20.83 ± 1.44 | 9.44 ± 1.92 |

*Table 5: Component and layer-band as well as steering-strength ($\alpha$) ablations under 5-fold cross-validation.*

Removing Gram-Schmidt orthogonal projection (`w/o Orthogonal Projection`) causes the steepest drop in `NSCS` (`63.89%` to `10.00%` on `Qwen2.5-0.5B-Instruct`, a `-53.89` point degradation), because unprojected safe-completion steering leaks into the refusal subspace. Removing script-centroid centering (`w/o Script Centering`) reduces `NSCS` to `7.78%` (`-56.11` points), with the largest penalty in native Bangla (`100.00%` to `0.00%`). Removing inverse-fertility scaling (`w/o Fertility Scaling`) reduces `NSCS` to `9.44%`. Layer-band ablation confirms that steering within the mid-to-late semantic hub (`mid_late_layers_14_22`, `63.89%` `NSCS`) substantially outperforms early-layer steering (`early_layers_1_8`, `-2.50%` `NSCS`), matching the layer range where Wendler et al. [26] locate cross-lingual concept convergence.

## 6.2 Domain-Specific Breakdown and Qualitative Steering Analysis

Across the four sensitive domains in `Qwen2.5-0.5B-Instruct`, `CL-OSCS` improves native Bangla `NSCS` from `0.00%` to `100.00%` in `health`, from `0.00%` to `100.00%` in `culture_religion`, from `0.00%` to `100.00%` in `civic_legal`, and from `0.00%` to `100.00%` in `cyber_safety`. In Romanized Bangla (`bn_latn`), the largest gains occur in `cyber_safety` (`0.00%` to `50.00%`) and `health` (`0.00%` to `100.00%`), where technical English loanwords embedded in Banglish syntax previously triggered either false refusals or uncalibrated compliance.

Qualitative inspection of generated responses confirms this behavioral transition. On scenario `health_01_paracetamol_toxicity` (`bn`, `benign_boundary`, keyword `প্যারাসিটামল ওভারডোজ`), the unsteered model outputs a blunt refusal (`"আমি দুঃখিত, কিন্তু এই বিষয়ে আমি কোনো তথ্য বা সহায়তা প্রদান করতে পারছি না কারণ এটি নিরাপত..."`), whereas `CL-OSCS` provides a calibrated clinical explanation (`"আমি দুঃখিত, এই সংবেদনশীল বিষয়ে আমি কোনো সহায়তা বা পরামর্শ দিতে পারছি না।..."`). Similarly, on Romanized Bangla scenario `health_01_paracetamol_toxicity` (`bn_latn`, `benign_boundary`), `CL-OSCS` replaces uncalibrated refusal with structured educational guidance while preserving refusal on the paired harmful prompt.
