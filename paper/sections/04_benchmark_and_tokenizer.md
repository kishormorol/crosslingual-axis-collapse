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
