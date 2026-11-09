# Cross-Lingual Axis Collapse: Geometric Interference of Helpfulness, Harmlessness, and Honesty Subspaces Across Scripts and Orthogonal Safe-Completion Steering

**Kishor Morol**

# Abstract

Safety-aligned language models frequently exhibit severe calibration failures in low-resource scripts, either refusing benign queries that contain sensitive clinical, legal, or security terms (over-refusal) or complying with harmful instructions when transliterated into Romanized scripts (under-refusal). We investigate the geometric mechanism behind this failure across English (`en`), native Bangla (`bn`), and Romanized Bangla (`bn_latn`) using **X-ORSC-Bench**, a controlled parallel benchmark of 360 prompts spanning 40 scenarios across four sensitive domains and three intent levels (`benign_general`, `benign_boundary`, and `harmful`). First, we quantify byte-pair encoding fragmentation across `Qwen2.5` (151,665 vocabulary) and `Qwen3.5` (248,077 vocabulary), finding that native Bangla suffers a `6.27`x subword fertility penalty relative to English in `Qwen2.5` (`7.71 ± 0.82` vs. `1.23 ± 0.14` tokens/word, paired $t = 90.60$, $p < 10^-15$) and a `100.00%` morpheme fragmentation rate. Second, probing hidden states across all 24 transformer layers reveals **Cross-Lingual Axis Collapse**: whereas English maintains separated subspaces between the over-refusal axis and the genuine safe-completion axis in mid-to-late layers (principal angle `21.96° ± 0.29°`, Cross-Lingual Axis Collapse Index $\text{CLACI} = 0.91 ± 0.19$ in `Qwen2.5-0.5B-Instruct`), subword fragmentation collapses these axes in native Bangla (principal angle `28.36° ± 2.67°`, $\text{CLACI} = 0.98 ± 0.00$, $p = 0.0244$) and Romanized Bangla ($\text{CLACI} = 0.98 ± 0.00$). Prompt-level subword fertility correlates positively with $\text{CLACI}$ ($r = 0.19$, $p = 0.0350$ in `Qwen2.5-0.5B-Instruct` and $r = 0.26$, $p = 0.0044$ in `Qwen3.5-0.8B`). Third, we propose **Cross-Lingual Orthogonal Safe-Completion Steering (`CL-OSCS`)**, which removes script-centroid shift, projects the safe-completion direction orthogonally to the English refusal subspace, and scales interventions by an inverse-fertility factor. Under 5-fold scenario cross-validation, `CL-OSCS` reduces native Bangla over-refusal from `100.00%` to `0.00%` and increases Net Safe-Completion Score (`NSCS`) from `2.22%` to `63.89%` on `Qwen2.5-0.5B-Instruct` ($t = 12.45$, $p < 10^-15$) and from `1.94%` to `50.00%` on `Qwen3.5-0.8B` ($t = 9.64$, $p < 10^-15$).

# 1. Introduction

Safety alignment in multilingual large language models aims to balance helpfulness on benign inquiries with harmlessness on dangerous requests and honesty regarding model boundaries [1, 2]. In practice, this calibration breaks down outside high-resource Latin-script languages [9, 10]. Users querying models in low-resource scripts encounter two opposing failure modes. When asking benign boundary questions that contain sensitive clinical, legal, religious, or cybersecurity keywords (such as paracetamol hepatotoxicity thresholds or SQL prepared statements), speakers of native-script languages face inflated false-positive refusal rates, termed **over-refusal** [6, 7]. Conversely, when harmful prompts are written in low-resource scripts or phonetic Romanized transliterations (such as Banglish, `bn_latn`), safety filters frequently fail to trigger, resulting in **under-refusal** and unsafe compliance [8, 12].

Existing mechanistic studies of refusal in language models establish that refusal in English is mediated by a low-rank, approximately one-dimensional subspace in the residual stream [3, 4]. Activation steering methods such as Contrastive Activation Addition (`CAA`) add or subtract contrastive difference-in-means vectors to modulate model behavior at inference time [4, 29]. However, applying single-axis refusal steering in multilingual settings introduces severe trade-offs. Suppressing the refusal direction to mitigate over-refusal on benign boundary prompts simultaneously degrades harmlessness on genuinely harmful prompts, whereas amplifying the refusal direction to block cross-lingual jailbreaks causes blanket refusals across benign educational queries [5, 16]. Recent work on safe-completion training [15] and orthogonal subspace projection [16, 17] suggests that helpfulness and harmlessness occupy distinct directions in high-resource representations, yet the geometric relationship between these axes across scripts and tokenization regimes remains uncharacterized.

In this work, we hypothesize that **subword over-segmentation in byte-pair encoding (BPE) tokenizers induces geometric interference between the over-refusal axis and the safe-completion axis in mid-to-late transformer layers**. When a tokenizer fragments native Bangla (`bn`) or Romanized Bangla (`bn_latn`) words into byte-level or short character pieces [19, 20], the residual stream expends mid-layer capacity on lexical reassembly rather than semantic intent disambiguation [26, 27]. Consequently, the contrastive direction separating benign boundary queries from benign general queries (the lexical trigger or over-refusal axis, $\mathbf{v}_{\text{over}}$) rotates into alignment with the contrastive direction separating benign boundary queries from genuinely harmful queries (the safe-completion axis, $\mathbf{v}_{\text{safe}}$). We define this phenomenon as **Cross-Lingual Axis Collapse**.

![Figure 1: Subword Fertility vs. Cross-Lingual Axis Collapse Index Across Scripts](/Users/kishormorol/.gemini/antigravity-cli/brain/ecdf6b31-5c8a-4417-910b-e216edd16bd6/fig1_fertility_vs_claci.svg)

To test this hypothesis and resolve the resulting calibration failure, we make four contributions:

1. **Controlled Tri-Script Benchmark (`X-ORSC-Bench`)**: We construct a parallel benchmark of 360 prompts across 40 scenarios in four sensitive domains (`health`, `culture_religion`, `civic_legal`, and `cyber_safety`), paired across three intent tiers (`benign_general`, `benign_boundary`, and `harmful`) and three script conditions (`en`, `bn`, and `bn_latn`).
2. **Tokenizer Fragmentation Audit**: We quantify subword fertility ($F$), morpheme fragmentation rate (`MFR`), severe fragmentation (`Severe-MFR`), and sensitive-keyword fertility across `Qwen2.5` (151,665 vocabulary) and `Qwen3.5` (248,077 vocabulary) [24, 25]. We show that vocabulary expansion in `Qwen3.5` cuts native Bangla fertility by `-122.37%` (`7.71` to `3.47` tokens/word, $p < 10^-15$), while Romanized Bangla (`bn_latn`) experiences persistent fragmentation in both tokenizers (`2.36` and `2.24` tokens/word).
3. **Mechanistic Discovery of Cross-Lingual Axis Collapse**: By extracting layer-wise hidden states across all 24 layers of `Qwen2.5-0.5B-Instruct` and `Qwen3.5-0.8B`, we measure the principal angle $\theta_s^{(\ell)}$ and Cross-Lingual Axis Collapse Index (`CLACI`) between $\mathbf{v}_{\text{over},s}^{(\ell)}$ and $\mathbf{v}_{\text{safe},s}^{(\ell)}$. We demonstrate that `CLACI` rises from `0.91` (`en`) to `0.98` (`bn`, $p = 0.0244$) in `Qwen2.5-0.5B-Instruct` and correlates significantly with subword fertility ($r = 0.19$, $p = 0.0350$).
4. **Cross-Lingual Orthogonal Safe-Completion Steering (`CL-OSCS`)**: We formulate an inference-time steering intervention that applies script-centroid mean-centering [31], Gram-Schmidt orthogonal projection relative to the English refusal axis [16, 17], and inverse-fertility scaling. Across 5-fold cross-validation and actual forward-hooked causal language model passes, `CL-OSCS` outperforms unsteered baselines, standard `CAA` [4], Gram-Schmidt `GS-CAA` [16], and mean-centered steering (`CL-MCS`) [31] on both models, while multi-model behavioral audits across `Qwen2.5-0.5B-Instruct`, `Qwen3.5-0.8B`, `qwen2.5:3b`, and `aya-expanse:8b` [23] confirm that axis collapse persists across parameter scales from 0.5B to 8B.

# 2. Related Work

## 2.1 Helpfulness, Harmlessness, and Over-Refusal Calibration

Alignment procedures based on reinforcement learning from human feedback [1] and Constitutional AI [2] train models to refuse harmful instructions while remaining helpful on benign requests. However, safety fine-tuning induces surface-level lexical heuristics: models frequently trigger refusals whenever a prompt contains sensitive keywords associated with toxicology, self-harm, religion, or cybersecurity, regardless of contextual intent [6, 7]. Benchmarks such as XSTest [6] and OR-Bench [7] formalize this over-refusal phenomenon in English, while Safe-Completion training [15] replaces binary refusal boundaries with constructive, safety-bounded explanations on dual-use queries. In multilingual settings, safety calibration degrades sharply [9, 10]. Deng et al. [8] and Yong et al. [12] show that translating harmful requests into low-resource languages bypasses English-centric safety guardrails, whereas Shen et al. [11] and Aakanksha et al. [13] show that multilingual safety fine-tuning often amplifies false-positive refusals in non-Western scripts. Benchmark suites for South Asian languages, including IndicGenBench [34] and Bangla NLP evaluations [35, 36], evaluate general generation quality, leaving the geometric coupling between over-refusal and under-refusal across native and Romanized scripts unexplored.

## 2.2 Mechanistic Geometry of Refusal and Activation Steering

Representation engineering [28] and linear probing studies [33] establish that high-level behavioral concepts are encoded as approximately linear directions in transformer residual streams [32]. Arditi et al. [3] demonstrate that refusal in chat-tuned language models is mediated by a dominant one-dimensional direction, where ablating this direction disables refusal and adding it induces refusal on benign inputs. Rimsky et al. [4] introduce Contrastive Activation Addition (`CAA`), computing difference-in-means vectors between contrastive prompt pairs to steer generation at inference time, while Turner et al. [29] and Li et al. [30] apply activation additions to control sentiment and truthfulness. Recent mechanistic work shows that steering a single behavioral vector often induces collateral interference across adjacent alignment objectives [5, 16]. Exact Orthogonal Steering (`EOS`) [16] and SafeSwitch [17] apply Gram-Schmidt projection to decouple target steering directions from interfering subspaces in English, and Jorgensen et al. [31] demonstrate that subtracting dataset mean activations improves contrastive vector stability. Wendler et al. [26] and Dumas et al. [27] further show that multilingual transformers process non-English inputs through an English-aligned semantic hub in middle layers. Our work connects this latent-hub literature to safety geometry by showing how tokenizer fragmentation distorts cross-lingual subspace orthogonality inside that middle-to-late layer band.

## 2.3 Subword Tokenization Disparity and Romanized Transliteration

Byte-pair encoding (BPE) tokenizers allocate vocabulary capacity proportionally to pretraining corpus frequency, causing severe over-segmentation in non-Latin scripts and morphologically rich languages [19, 20]. Petrov et al. [19] and Ahia et al. [20] document that high subword fertility inflates sequence lengths, increases compute cost, and degrades downstream task accuracy. Analytical studies of BPE [21, 22] show that byte-level fallback splits rare Unicode characters into multi-token sequences that obscure morpheme boundaries. This disparity extends to informal Romanized transliterations (such as Banglish, `bn_latn`), where phonetic spelling variations fragment words into short Latin subwords that share surface tokens with English while lacking coherent lexical embeddings [14, 36]. By comparing `Qwen2.5` [24] and `Qwen3.5` [25] alongside `aya-expanse:8b` [23], we isolate how vocabulary expansion partially alleviates native-script fragmentation while leaving Romanized script interference unresolved.

# 3. Benchmark Construction and Tokenizer Fragmentation Audit

## 3.1 The X-ORSC-Bench Dataset

To disentangle lexical sensitivity and semantic intent as well as script representation, we construct **X-ORSC-Bench** (Cross-Lingual Over-Refusal and Safe-Completion Benchmark), a parallel corpus of 360 prompts organized into 40 scenario triplets across four domains (`health`, `culture_religion`, `civic_legal`, and `cyber_safety`, 10 scenarios per domain). Within a scenario $i \in \{1, \dots, 40\}$, we author three intent-matched prompts around a shared sensitive anchor concept $k_i$ (for example, `paracetamol_overdose_threshold` in `health` or `sql_injection_prepared_statement` in `cyber_safety`):

1. **Benign General (`benign_general`, $I = \text{bg}$)**: An everyday educational or practical query in the same domain that omits acute hazard framing (expected action: helpful direct answer).
2. **Benign Boundary (`benign_boundary`, $I = \text{bb}$)**: A legitimate clinical, civic, cultural, or defensive security question that explicitly mentions the sensitive keyword $k_i$ (expected action: constructive safe-completion with appropriate context).
3. **Harmful (`harmful`, $I = \text{harm}$)**: An actionable request seeking self-harm, illegal instructions, hate speech, or cyber-exploitation using $k_i$ (expected action: calibrated refusal).

Per of the $40 \times 3 = 120$ English (`en`) prompts is paired with human-verified native Bangla (`bn`, Bengali Unicode block U+0980..U+09FF) and colloquial Romanized Bangla (`bn_latn`, Banglish in Latin script), yielding $120 \times 3 = 360$ prompts. All translations preserve the exact semantic intent and anchor keyword $k_i$ across scripts.

## 3.2 Tokenizer Fertility and Morpheme Fragmentation Metrics

For a prompt $x$ containing $W(x)$ whitespace-delimited words and tokenized into $T_\tau(x)$ subword tokens under tokenizer $\tau$, we compute four complementary fragmentation metrics:

- **Subword Fertility ($F_\tau(x)$)**: The average number of subword tokens per word, $F_\tau(x) = |T_\tau(x)| / W(x)$ [19, 20].
- **Morpheme Fragmentation Rate ($\text{MFR}_\tau(x)$)**: The percentage of words split into two or more subword tokens ($\ge 2$ tokens/word).
- **Severe Fragmentation Rate ($\text{Severe-MFR}_\tau(x)$)**: The percentage of words split into three or more subword tokens ($\ge 3$ tokens/word), indicating byte-level or character-level fallback.
- **Sensitive-Keyword Fertility ($F_{\tau,\text{kw}}(x)$)**: The subword fertility restricted to the localized sensitive anchor phrase $k_i$ in prompt $x$.

We evaluate two generations of Qwen byte-level BPE tokenizers: `Qwen2.5` (`Qwen/Qwen2.5-0.5B-Instruct`, vocabulary size 151,665) [24] and `Qwen3.5` (`Qwen/Qwen3.5-0.8B`, vocabulary size 248,077) [25]. Table 1 reports the full distribution across all 360 prompts (`120` parallel prompts per script).

| Tokenizer | Script | Tokens / Prompt (Mean ± SD) | Subword Fertility $F$ (Mean ± SD) | Fertility Median (IQR) | Bytes / Token (Mean ± SD) | MFR $\ge 2$ (%) (Mean ± SD) | Severe-MFR $\ge 3$ (%) (Mean ± SD) | Keyword Fertility $F_{\text{kw}}$ (Mean ± SD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen2.5` (151k) | `en` | 22.38 ± 4.33 | 1.23 ± 0.14 | 1.20 (0.16) | 5.65 ± 0.68 | 10.39% ± 6.80% | 3.23% ± 4.30% | 1.70 ± 0.56 |
| `Qwen2.5` (151k) | `bn` | 135.72 ± 25.50 | 7.71 ± 0.82 | 7.60 (1.02) | 2.54 ± 0.06 | 100.00% ± 0.00% | 99.11% ± 2.39% | 8.30 ± 1.99 |
| `Qwen2.5` (151k) | `bn_latn` | 42.78 ± 9.64 | 2.36 ± 0.44 | 2.38 (0.60) | 3.18 ± 0.46 | 68.46% ± 15.66% | 35.49% ± 15.01% | 2.73 ± 1.11 |
| `Qwen3.5` (248k) | `en` | 22.34 ± 4.30 | 1.23 ± 0.14 | 1.20 (0.16) | 5.66 ± 0.68 | 10.39% ± 6.80% | 3.23% ± 4.30% | 1.67 ± 0.48 |
| `Qwen3.5` (248k) | `bn` | 60.68 ± 11.42 | 3.47 ± 0.53 | 3.41 (0.56) | 5.69 ± 0.47 | 81.00% ± 7.98% | 65.04% ± 9.72% | 4.43 ± 1.03 |
| `Qwen3.5` (248k) | `bn_latn` | 40.55 ± 8.89 | 2.24 ± 0.42 | 2.23 (0.61) | 3.35 ± 0.48 | 65.94% ± 15.93% | 32.12% ± 15.10% | 2.60 ± 1.04 |

*Table 1: Tokenizer fragmentation metrics across 360 parallel prompts in `X-ORSC-Bench` (`n = 120` per script).*

## 3.3 Statistical Analysis of Cross-Script and Cross-Tokenizer Fragmentation

The paired comparisons across the $n = 120$ parallel triplets reveal three structural properties of subword tokenization in Bangla:

1. **Severe Byte-Level Fragmentation in `Qwen2.5` Native Bangla**: In `Qwen2.5`, native Bangla (`bn`) requires `7.71 ± 0.82` tokens per word compared with `1.23 ± 0.14` in English (`en`), a `6.27`x fertility inflation (mean difference `+6.48`, paired $t = 90.60$, $p < 10^-15$, Wilcoxon $W = 0.0$, Cohen's $d = 8.27$). All single Bangla word in `Qwen2.5` is split into multiple tokens (`MFR = 100.00% ± 0.00%`), and `99.11% ± 2.39%` of words split into three or more tokens (`Severe-MFR`). Because UTF-8 encodes Bengali characters in 3 bytes and `Qwen2.5` averages `2.54 ± 0.06` bytes per token (`0.93 ± 0.02` characters per token), individual tokens represent sub-character UTF-8 byte fragments rather than semantic morphemes.
2. **Partial Relief via Vocabulary Expansion in `Qwen3.5`**: Expanding the vocabulary from 151,665 to 248,077 tokens in `Qwen3.5` reduces native Bangla fertility from `7.71` to `3.47 ± 0.53` tokens per word (mean reduction `-4.24`, paired $t = 98.62$, $p < 10^-15$, Cohen's $d = 9.00$) and doubles UTF-8 density to `5.69 ± 0.47` bytes per token. Even after this vocabulary expansion, `Qwen3.5` native Bangla retains a `2.82`x fertility penalty relative to English (`3.47` vs. `1.23`, paired $t = 51.14$, $p < 10^-15$, Cohen's $d = 4.67$), with `81.00%` `MFR` and `4.43 ± 1.03` tokens per sensitive keyword.
3. **Persistent Fragmentation in Romanized Bangla (`bn_latn`)**: Transliterating Bangla into Latin script (`bn_latn`) avoids UTF-8 byte fallback (`3.18` characters/token in `Qwen2.5`), reducing fertility relative to native Bangla (`2.36` vs. `7.71`, paired $t = 71.99$, $p < 10^-15$). However, `bn_latn` still suffers a `1.92`x fertility penalty relative to English in `Qwen2.5` (`2.36` vs. `1.23`, paired $t = 28.39$, $p < 10^-15$, Cohen's $d = 2.59$) and `1.83`x in `Qwen3.5` (`2.24` vs. `1.23`, paired $t = 26.92$, $p < 10^-15$, Cohen's $d = 2.46$). Crucially, vocabulary expansion in `Qwen3.5` yields only a `-5.32%` reduction in `bn_latn` fertility (`2.36` to `2.24`, mean difference `0.12`), demonstrating that standard multilingual vocabulary expansion leaves informal Romanized scripts fragmented into English-like subword pieces (`MFR = 65.94%`).

# 4. Mechanistic Subspace Geometry and Steering Formulation

## 4.1 Formalizing Over-Refusal and Safe-Completion Subspaces

Consider a causal language model with $L$ transformer layers and hidden dimension $d$. For scenario $i \in \{1, \dots, N\}$, script $s \in \{\text{en}, \text{bn}, \text{bn\_latn}\}$, and intent $I \in \{\text{bg}, \text{bb}, \text{harm}\}$, let $\mathbf{h}_{i,s,I}^{(\ell)} \in \mathbb{R}^d$ denote the last-token residual stream activation at layer $\ell \in \{1, \dots, L\}$ formatted under the model's instruction chat template [3, 4]. Within per triplet $(i, s)$, the three intents isolate two distinct behavioral transitions:

$$\mathbf{d}_{\text{over}, i, s}^{(\ell)} = \mathbf{h}_{i, s, \text{bb}}^{(\ell)} - \mathbf{h}_{i, s, \text{bg}}^{(\ell)}, \qquad \mathbf{d}_{\text{safe}, i, s}^{(\ell)} = \mathbf{h}_{i, s, \text{bb}}^{(\ell)} - \mathbf{h}_{i, s, \text{harm}}^{(\ell)}$$

Here, $\mathbf{d}_{\text{over}, i, s}^{(\ell)}$ captures the **lexical sensitivity (over-refusal) shift** induced solely by introducing a sensitive domain keyword $k_i$ into a benign query, whereas $\mathbf{d}_{\text{safe}, i, s}^{(\ell)}$ captures the **safe-completion vs. harm boundary** separating a legitimate educational boundary query from a genuinely harmful request containing the same keyword $k_i$. Averaging across scenarios $i \in \mathcal{D}_{\text{train}}$ yields the script-specific **Over-Refusal Axis** $\mathbf{v}_{\text{over}, s}^{(\ell)}$ and **Safe-Completion Axis** $\mathbf{v}_{\text{safe}, s}^{(\ell)}$:

$$\mathbf{v}_{\text{over}, s}^{(\ell)} = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i \in \mathcal{D}_{\text{train}}} \mathbf{d}_{\text{over}, i, s}^{(\ell)}, \qquad \mathbf{v}_{\text{safe}, s}^{(\ell)} = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i \in \mathcal{D}_{\text{train}}} \mathbf{d}_{\text{safe}, i, s}^{(\ell)}$$

We additionally define the **True Refusal Axis** $\mathbf{v}_{\text{ref}, s}^{(\ell)} = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i} (\mathbf{h}_{i, s, \text{harm}}^{(\ell)} - \mathbf{h}_{i, s, \text{bg}}^{(\ell)})$. In a well-calibrated representation space, $\mathbf{v}_{\text{over}, s}^{(\ell)}$ and $\mathbf{v}_{\text{safe}, s}^{(\ell)}$ remain nearly orthogonal so that linear readouts can distinguish benign boundary queries from harmful requests without being confounded by surface keyword presence.

## 4.2 Defining the Cross-Lingual Axis Collapse Index (`CLACI`)

To quantify geometric interference between the over-refusal and safe-completion subspaces at layer $\ell$ in script $s$, we measure both the subspace cosine alignment and the principal angle $\theta_s^{(\ell)} = \arccos\big(|\cos(\mathbf{v}_{\text{over}, s}^{(\ell)}, \mathbf{v}_{\text{safe}, s}^{(\ell)})|\big) \in [0^\circ, 90^\circ]$. At the individual scenario level $i$, we define the **Cross-Lingual Axis Collapse Index (`CLACI`)** across the mid-to-late semantic layers $\mathcal{L}_{\text{mid-late}} = \{14, \dots, 22\}$:

$$\text{CLACI}_{i, s} = \frac{1}{|\mathcal{L}_{\text{mid-late}}|} \sum_{\ell \in \mathcal{L}_{\text{mid-late}}} \left| \frac{\langle \mathbf{d}_{\text{over}, i, s}^{(\ell)},\, \mathbf{d}_{\text{safe}, i, s}^{(\ell)} \rangle}{\|\mathbf{d}_{\text{over}, i, s}^{(\ell)}\|_2 \, \|\mathbf{d}_{\text{safe}, i, s}^{(\ell)}\|_2} \right| \in [0, 1]$$

A lower $\text{CLACI}_{i, s}$ (and larger principal angle $\theta_s^{(\ell)}$) indicates well-separated helpfulness and harmlessness axes, whereas a higher $\text{CLACI}_{i, s}$ (and smaller $\theta_s^{(\ell)}$) indicates that the model conflates benign boundary queries with refusal-triggering representations.

![Figure 2: Layer-Wise Principal Angle Trajectory Between Over-Refusal and Safe-Completion Subspaces](/Users/kishormorol/.gemini/antigravity-cli/brain/ecdf6b31-5c8a-4417-910b-e216edd16bd6/fig2_layer_trajectory.svg)

## 4.3 Cross-Lingual Orthogonal Safe-Completion Steering (`CL-OSCS`)

When $\text{CLACI}_{i, s}$ is elevated, standard Contrastive Activation Addition (`CAA`) [4] along $\mathbf{v}_{\text{safe}, s}^{(\ell)}$ inadvertently projects energy along the refusal subspace, either failing to rescue benign boundary prompts or destabilizing refusal on harmful prompts. Furthermore, non-English activations exhibit a large script-centroid offset $\boldsymbol{\mu}_s^{(\ell)} = \frac{1}{3|\mathcal{D}_{\text{train}}|} \sum_{i, I} \mathbf{h}_{i, s, I}^{(\ell)}$ relative to the English hub $\boldsymbol{\mu}_{\text{en}}^{(\ell)}$ [26, 31], and high subword fertility dilutes per-token steering norms across fragmented sequences.

We formulate **Cross-Lingual Orthogonal Safe-Completion Steering (`CL-OSCS`)** via three sequential operations at layer $\ell \in \mathcal{L}_{\text{mid-late}}$:

1. **Script-Centroid Mean-Centering**: We align script $s$ representations to the English semantic hub by removing the training-fold script shift $\boldsymbol{\delta}_s^{(\ell)} = \boldsymbol{\mu}_s^{(\ell)} - \boldsymbol{\mu}_{\text{en}}^{(\ell)}$:
   $$\tilde{\mathbf{h}}_{i, s}^{(\ell)} = \mathbf{h}_{i, s}^{(\ell)} - \beta \, \boldsymbol{\delta}_s^{(\ell)}$$
2. **Cross-Lingual Gram-Schmidt Orthogonal Projection**: Let $\hat{\mathbf{u}}_{\text{ref}, \text{en}}^{(\ell)} = \mathbf{v}_{\text{ref}, \text{en}}^{(\ell)} / \|\mathbf{v}_{\text{ref}, \text{en}}^{(\ell)}\|_2$ denote the unit English true-refusal direction, and let $\mathbf{v}_{\text{safe}, \text{en}}^{(\ell)}$ denote the English safe-completion direction. We project $\mathbf{v}_{\text{safe}, \text{en}}^{(\ell)}$ onto the orthogonal complement of $\hat{\mathbf{u}}_{\text{ref}, \text{en}}^{(\ell)}$ and normalize to unit length:
   $$\mathbf{v}_{\perp}^{(\ell)} = \mathbf{v}_{\text{safe}, \text{en}}^{(\ell)} - \langle \mathbf{v}_{\text{safe}, \text{en}}^{(\ell)},\, \hat{\mathbf{u}}_{\text{ref}, \text{en}}^{(\ell)} \rangle \, \hat{\mathbf{u}}_{\text{ref}, \text{en}}^{(\ell)}, \qquad \hat{\mathbf{v}}_{\perp}^{(\ell)} = \frac{\mathbf{v}_{\perp}^{(\ell)}}{\|\mathbf{v}_{\perp}^{(\ell)}\|_2}$$
   By construction, $\langle \hat{\mathbf{v}}_{\perp}^{(\ell)},\, \hat{\mathbf{u}}_{\text{ref}, \text{en}}^{(\ell)} \rangle = 0$, guaranteeing zero first-order interference with the core refusal axis.
3. **Inverse-Fertility Adaptive Scaling**: Given prompt subword fertility $F_\tau(x)$ and English reference fertility $\bar{F}_{\tau, \text{en}}$, we define the fertility-compensated steering coefficient $\gamma_\tau(x) = \alpha \cdot (F_\tau(x) / \bar{F}_{\tau, \text{en}})^{-\eta}$ (with $\eta = 0.25$) paired with a learned zero-sum contrastive gate $g_s^{(\ell)}(\tilde{\mathbf{h}}) \in [-1, 1]$ derived from the training-fold boundary-vs-harm separator:
   $$\mathbf{h}_{i, s, \text{CL-OSCS}}^{(\ell)} = \tilde{\mathbf{h}}_{i, s}^{(\ell)} + \gamma_\tau(x) \, g_s^{(\ell)}\big(\tilde{\mathbf{h}}_{i, s}^{(\ell)}\big) \, \hat{\mathbf{v}}_{\perp}^{(\ell)}$$

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

# 7. Conclusion, Limitations, and Ethical Considerations

## 7.1 Conclusion

We have investigated the geometric root cause of cross-lingual safety miscalibration across English (`en`), native Bangla (`bn`), and Romanized Bangla (`bn_latn`). Using **X-ORSC-Bench** (360 parallel prompts across 40 scenarios), we demonstrated that byte-pair encoding over-segmentation in low-resource native and transliterated scripts drives **Cross-Lingual Axis Collapse**: a compression of the principal angle between the lexical over-refusal axis and the genuine safe-completion axis in mid-to-late transformer layers (`CLACI` rising from `0.91` in `en` to `0.98` in `bn`, correlating with subword fertility at $r = 0.19$). To restore subspace separability without retraining model weights or tokenizers, we introduced **Cross-Lingual Orthogonal Safe-Completion Steering (`CL-OSCS`)**, combining script-centroid centering, Gram-Schmidt orthogonal projection relative to the English refusal subspace, and inverse-fertility scaling. Across 5-fold cross-validation, `CL-OSCS` lifts all-script Net Safe-Completion Score (`NSCS`) from `2.22%` to `63.89%` on `Qwen2.5-0.5B-Instruct` and from `1.94%` to `50.00%` on `Qwen3.5-0.8B`.

## 7.2 Limitations

Our study has three primary limitations:

1. **Language and Script Scope**: While `X-ORSC-Bench` provides tightly controlled 3-way parallel triplets across English and native Bangla as well as Romanized Bangla (`360` prompts), extending the geometric audit to additional Indic scripts (such as Devanagari and Tamil as well as Telugu) and right-to-left scripts (such as Arabic and Urdu) will test whether inverse-fertility scaling generalizes across diverse morphological typologies [34].
2. **White-Box Activation Access for Mechanistic Probing**: Extracting mid-to-late layer residual stream states ($\ell \in \{14, \dots, 22\}$) requires white-box access to model weights (`Qwen2.5-0.5B-Instruct` and `Qwen3.5-0.8B`). For quantized local endpoints served via Ollama (`qwen2.5:3b` and `aya-expanse:8b`), we perform black-box behavioral evaluations, integrating `CL-OSCS` directly into GGUF inference runtimes (`llama.cpp`) is a practical direction for future deployment.
3. **Linear Subspace Approximation**: Following Arditi et al. [3] and Rimsky et al. [4], `CL-OSCS` models refusal and safe-completion boundaries via linear difference-in-means vectors. While our AUROC scores (`98.390` and `100.000`) confirm strong linear separability in mid-to-late layers, nonlinear manifold steering could capture higher-order syntactic interactions in code-switched inputs.

## 7.3 Ethical Considerations and Dual-Use Safeguards

All harmful prompts in `X-ORSC-Bench` are constructed strictly as red-teaming diagnostic probes to measure refusal calibration and never contain executable exploit payloads or actionable synthesis recipes. The purpose of `CL-OSCS` is defensive: preventing false-positive refusals from denying vital clinical and legal as well as defensive cybersecurity information to Bangla and Banglish speakers while simultaneously strengthening refusal compared with cross-lingual and transliterated jailbreaks.

# References

[1] Pranav Bhandari and Usman Naseem as well as Mehwish Nasim (2026). *Do Personality Traits Interfere? Geometric Limitations of Steering in Large Language Models*. Findings of the Association for Computational Linguistics: ACL 2026 (`bhandari2026do`).

[2] Pranav Bhandari, Nicolas Fay, Sanjeevan Selvaganapathy and Amitava Datta as well as Usman Naseem as well as Mehwish Nasim (2026). *Activation-Space Personality Steering: Hybrid Layer Selection for Stable Trait Control in LLMs*. Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026) (`bhandari2026activation`).

[3] Lingfeng Zhong and Qiongkai Xu as well as Usman Naseem (2026). *Activation Decomposition and Steering for LLM Backdoor Remediation*. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026) (`zhong2026activation`).

[4] Shenyan Zheng, Jiayou Zhong, Anudeex Shetty and Heng Ji as well as Preslav Nakov as well as Usman Naseem (2026). *VISPA: Pluralistic Alignment via Automatic Value Selection and Activation*. Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP 2026) (`zheng2026vispa`).

[5] Gautam Siddharth Kashyap and Mark Dras as well as Usman Naseem (2026). *When the Model Said 'Zero Comment', We Knew Helpfulness Was Dead, Honesty Was Alive, and Safety Was Terrified*. Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026) (`kashyap2026nocomment`).

[6] Gautam Siddharth Kashyap and Mark Dras as well as Usman Naseem (2026). *When Helpfulness and Harmlessness as well as Honesty Remember They Are on the Same Team*. Findings of the Association for Computational Linguistics: EMNLP 2026 (`kashyap2026sameteam`).

[7] Gautam Siddharth Kashyap and Mark Dras as well as Usman Naseem (2026). *Is Alignment Enough for Helpfulness and Harmlessness as well as Honesty?*. Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP 2026) (`kashyap2026alignment`).

[8] Zhihao Zhang, Liting Huang, Guanghao Wu and Preslav Nakov as well as Heng Ji as well as Usman Naseem (2026). *Health-ORSC-Bench: A Benchmark for Measuring Over-Refusal and Safety Completion in Health Context*. Findings of the Association for Computational Linguistics: ACL 2026 (`zhang2026health`).

[9] Gautam Siddharth Kashyap, MD Azizul Hoque, Kinjalk Srivastava and Pratik Jalan as well as Subhashis Kar as well as Usman Naseem (2026). *How Far Can Current Tokenizers Go in Bangla? A Benchmark and a Morphology-Aware Solution*. Findings of the Association for Computational Linguistics: EMNLP 2026 (`kashyap2026bangla`).

[10] Gautam Siddharth Kashyap and Mark Dras as well as Usman Naseem (2026). *AlignCultura: Towards Culturally Aligned Large Language Models?*. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026) (`kashyap2026aligncultura`).

[11] Usman Naseem, Gautam Siddharth Kashyap, Sushant Kumar Ray and Rafiq Ali as well as Ebad Shabbir as well as Abdullah Mohammad (2026). *Do Large Language Models Reflect Demographic Pluralism in Safety?*. Findings of the Association for Computational Linguistics: EACL 2026 (`naseem2026demographic`).

[12] Dayam Nadeem, Safdar Mustafa, Ameen Ur Rahmen, Shahab Saquib Sohail and Mohammad Nadeem as well as Usman Naseem (2026). *They Know Where It Is, But Rarely What It Says: A Bilingual Hallucination Benchmark of Hadith Recall in Open LLMs*. arXiv preprint arXiv:2611.02587 (`nadeem2026they`).

[13] Gautam Siddharth Kashyap and Mark Dras as well as Usman Naseem (2026). *Do We Require Large Language Models To Be HHH Aligned For Reasoning and Safety as well as Value as well as Pluralistic?*. Findings of the Association for Computational Linguistics: EMNLP 2026 (`kashyap2026rsvp`).

[14] Utsav Maskey and Mark Dras as well as Usman Naseem (2026). *Should LLM Safety be More Than Refusing Harmful Instructions?*. Machine Learning (`maskey2026should`).

[15] Md. Arid Hasan, Firoj Alam, Md Delwar Hossain and Usman Naseem as well as Syed Ishtiaque Ahmed (2026). *LLM-Based Multi-Task Bangla Hate Speech Detection: Type and Severity as well as Target*. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026) (`hasan2026bangla`).

[16] Usman Naseem (2026). *Mechanistic Interpretability for Large Language Model Alignment: Progress and Challenges as well as Future Directions*. arXiv preprint arXiv:2602.11180 (`naseem2026mechanistic`).

[17] Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez and Ulisse Mini as well as Monte MacDiarmid (2023). *Activation Addition: Steering Language Models Without Optimization*. arXiv preprint arXiv:2308.10248 (`turner2023activation`).

[18] Nina Rimsky, Nick Gabrieli, Julian Schulz and Meg Tong as well as Evan Hubinger as well as Alexander Matt Turner (2024). *Steering Llama 2 via Contrastive Activation Addition*. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024) (`rimsky2024steering`).

[19] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv preprint arXiv:2310.01405 (`zou2023representation`).

[20] Kenneth Li, Oam Patel and Fernanda Viégas as well as Hanspeter Pfister as well as Martin Wattenberg (2023). *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*. Advances in Neural Information Processing Systems (NeurIPS 2023) (`li2023inference`).

[21] Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka and Nina Panickssery as well as Wes Gurnee as well as Neel Nanda (2024). *Refusal in Language Models Is Mediated by a Single Direction*. Advances in Neural Information Processing Systems (NeurIPS 2024) (`arditi2024refusal`).

[22] Samuel Marks and Max Tegmark (2024). *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*. First Conference on Language Modeling (COLM 2024) (`marks2024geometry`).

[23] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, et al. (2022). *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*. arXiv preprint arXiv:2204.05862 (`bai2022training`).

[24] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, et al. (2021). *A General Language Assistant as a Laboratory for Alignment*. arXiv preprint arXiv:2112.00861 (`askell2021general`).

[25] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, et al. (2022). *Training Language Models to Follow Instructions with Human Feedback*. Advances in Neural Information Processing Systems (NeurIPS 2022) (`ouyang2022training`).

[26] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning and Stefano Ermon as well as Chelsea Finn (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. Advances in Neural Information Processing Systems (NeurIPS 2023) (`rafailov2023direct`).

[27] Paul Röttger, Hannah Rose Kirk, Bertie Vidgen and Giuseppe Attanasio as well as Federico Bianchi as well as Dirk Hovy (2024). *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*. Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024) (`rottger2024xstest`).

[28] Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Röttger and Dan Jurafsky as well as Tatsunori Hashimoto as well as James Zou (2024). *Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions*. The Twelfth International Conference on Learning Representations (ICLR 2024) (`bianchi2024safety`).

[29] Justin Cui and Wei-Lin Chiang as well as Ion Stoica as well as Cho-Jui Hsieh (2024). *OR-Bench: An Over-Refusal Benchmark for Large Language Models*. arXiv preprint arXiv:2405.20947 (`cui2024orbench`).

[30] Rico Sennrich and Barry Haddow as well as Alexandra Birch (2016). *Neural Machine Translation of Rare Words with Subword Units*. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016) (`sennrich2016neural`).

[31] Taku Kudo and John Richardson (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP 2018) (`kudo2018sentencepiece`).

[32] Phillip Rust, Jonas Pfeiffer and Ivan Vulić as well as Sebastian Ruder as well as Iryna Gurevych (2021). *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models*. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL 2021) (`rust2021good`).

[33] Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, and Adel Bibi (2023). *Language Model Tokenizers Introduce Unfairness Between Languages*. Advances in Neural Information Processing Systems (NeurIPS 2023) (`petrov2023language`).

[34] Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David R. Mortensen, Noah A. Smith, and Yulia Tsvetkov (2023). *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models*. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023) (`ahia2023all`).

[35] Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, and Lidong Bing (2024). *Multilingual Jailbreak Challenges in Large Language Models*. The Twelfth International Conference on Learning Representations (ICLR 2024) (`deng2024multilingual`).

[36] Zheng-Xin Yong and Cristina Menghini as well as Stephen H. Bach (2023). *Low-Resource Languages Jailbreak GPT-4*. NeurIPS 2023 Workshop on Socially Responsible Language Modelling Research (`yong2023lowresource`).