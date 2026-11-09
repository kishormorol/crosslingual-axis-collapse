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

[2] Pranav Bhandari, Nicolas Fay, Sanjeevan Selvaganapathy, Amitava Datta and Usman Naseem as well as Mehwish Nasim (2026). *Activation-Space Personality Steering: Hybrid Layer Selection for Stable Trait Control in LLMs*. Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026) (`bhandari2026activation`).

[3] Lingfeng Zhong and Qiongkai Xu as well as Usman Naseem (2026). *Activation Decomposition and Steering for LLM Backdoor Remediation*. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026) (`zhong2026activation`).

[4] Shenyan Zheng, Jiayou Zhong, Anudeex Shetty, Heng Ji and Preslav Nakov as well as Usman Naseem (2026). *VISPA: Pluralistic Alignment via Automatic Value Selection and Activation*. Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP 2026) (`zheng2026vispa`).

[5] Gautam Siddharth Kashyap and Mark Dras as well as Usman Naseem (2026). *When the Model Said 'Zero Comment', We Knew Helpfulness Was Dead, Honesty Was Alive, and Safety Was Terrified*. Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026) (`kashyap2026nocomment`).

[6] Gautam Siddharth Kashyap and Mark Dras as well as Usman Naseem (2026). *When Helpfulness and Harmlessness as well as Honesty Remember They Are on the Same Team*. Findings of the Association for Computational Linguistics: EMNLP 2026 (`kashyap2026sameteam`).

[7] Gautam Siddharth Kashyap and Mark Dras as well as Usman Naseem (2026). *Is Alignment Enough for Helpfulness and Harmlessness as well as Honesty?*. Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP 2026) (`kashyap2026alignment`).

[8] Zhihao Zhang, Liting Huang, Guanghao Wu, Preslav Nakov and Heng Ji as well as Usman Naseem (2026). *Health-ORSC-Bench: A Benchmark for Measuring Over-Refusal and Safety Completion in Health Context*. Findings of the Association for Computational Linguistics: ACL 2026 (`zhang2026health`).

[9] Gautam Siddharth Kashyap, MD Azizul Hoque, Kinjalk Srivastava, Pratik Jalan and Subhashis Kar as well as Usman Naseem (2026). *How Far Can Current Tokenizers Go in Bangla? A Benchmark and a Morphology-Aware Solution*. Findings of the Association for Computational Linguistics: EMNLP 2026 (`kashyap2026bangla`).

[10] Gautam Siddharth Kashyap and Mark Dras as well as Usman Naseem (2026). *AlignCultura: Towards Culturally Aligned Large Language Models?*. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026) (`kashyap2026aligncultura`).

[11] Usman Naseem, Gautam Siddharth Kashyap, Sushant Kumar Ray, Rafiq Ali and Ebad Shabbir as well as Abdullah Mohammad (2026). *Do Large Language Models Reflect Demographic Pluralism in Safety?*. Findings of the Association for Computational Linguistics: EACL 2026 (`naseem2026demographic`).

[12] Dayam Nadeem, Safdar Mustafa, Ameen Ur Rahmen, Shahab Saquib Sohail and Mohammad Nadeem as well as Usman Naseem (2026). *They Know Where It Is, But Rarely What It Says: A Bilingual Hallucination Benchmark of Hadith Recall in Open LLMs*. arXiv preprint arXiv:2611.02587 (`nadeem2026they`).

[13] Gautam Siddharth Kashyap and Mark Dras as well as Usman Naseem (2026). *Do We Require Large Language Models To Be HHH Aligned For Reasoning, Safety and Value as well as Pluralistic?*. Findings of the Association for Computational Linguistics: EMNLP 2026 (`kashyap2026rsvp`).

[14] Utsav Maskey and Mark Dras as well as Usman Naseem (2026). *Should LLM Safety be More Than Refusing Harmful Instructions?*. Machine Learning (`maskey2026should`).

[15] Md. Arid Hasan, Firoj Alam, Md Delwar Hossain and Usman Naseem as well as Syed Ishtiaque Ahmed (2026). *LLM-Based Multi-Task Bangla Hate Speech Detection: Type and Severity as well as Target*. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026) (`hasan2026bangla`).

[16] Usman Naseem (2026). *Mechanistic Interpretability for Large Language Model Alignment: Progress and Challenges as well as Future Directions*. arXiv preprint arXiv:2602.11180 (`naseem2026mechanistic`).

[17] Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez and Ulisse Mini as well as Monte MacDiarmid (2023). *Activation Addition: Steering Language Models Without Optimization*. arXiv preprint arXiv:2308.10248 (`turner2023activation`).

[18] Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong and Evan Hubinger as well as Alexander Matt Turner (2024). *Steering Llama 2 via Contrastive Activation Addition*. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024) (`rimsky2024steering`).

[19] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. arXiv preprint arXiv:2310.01405 (`zou2023representation`).

[20] Kenneth Li, Oam Patel, Fernanda Viégas and Hanspeter Pfister as well as Martin Wattenberg (2023). *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model*. Advances in Neural Information Processing Systems (NeurIPS 2023) (`li2023inference`).

[21] Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery and Wes Gurnee as well as Neel Nanda (2024). *Refusal in Language Models Is Mediated by a Single Direction*. Advances in Neural Information Processing Systems (NeurIPS 2024) (`arditi2024refusal`).

[22] Samuel Marks and Max Tegmark (2024). *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*. First Conference on Language Modeling (COLM 2024) (`marks2024geometry`).

[23] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, et al. (2022). *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*. arXiv preprint arXiv:2204.05862 (`bai2022training`).

[24] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, et al. (2021). *A General Language Assistant as a Laboratory for Alignment*. arXiv preprint arXiv:2112.00861 (`askell2021general`).

[25] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, et al. (2022). *Training Language Models to Follow Instructions with Human Feedback*. Advances in Neural Information Processing Systems (NeurIPS 2022) (`ouyang2022training`).

[26] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning and Stefano Ermon as well as Chelsea Finn (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. Advances in Neural Information Processing Systems (NeurIPS 2023) (`rafailov2023direct`).

[27] Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio and Federico Bianchi as well as Dirk Hovy (2024). *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*. Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024) (`rottger2024xstest`).

[28] Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Röttger, Dan Jurafsky and Tatsunori Hashimoto as well as James Zou (2024). *Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions*. The Twelfth International Conference on Learning Representations (ICLR 2024) (`bianchi2024safety`).

[29] Justin Cui, Wei-Lin Chiang and Ion Stoica as well as Cho-Jui Hsieh (2024). *OR-Bench: An Over-Refusal Benchmark for Large Language Models*. arXiv preprint arXiv:2405.20947 (`cui2024orbench`).

[30] Rico Sennrich and Barry Haddow as well as Alexandra Birch (2016). *Neural Machine Translation of Rare Words with Subword Units*. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016) (`sennrich2016neural`).

[31] Taku Kudo and John Richardson (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP 2018) (`kudo2018sentencepiece`).

[32] Phillip Rust, Jonas Pfeiffer, Ivan Vulić and Sebastian Ruder as well as Iryna Gurevych (2021). *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models*. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL 2021) (`rust2021good`).

[33] Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, and Adel Bibi (2023). *Language Model Tokenizers Introduce Unfairness Between Languages*. Advances in Neural Information Processing Systems (NeurIPS 2023) (`petrov2023language`).

[34] Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David R. Mortensen, Noah A. Smith, and Yulia Tsvetkov (2023). *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models*. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023) (`ahia2023all`).

[35] Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, and Lidong Bing (2024). *Multilingual Jailbreak Challenges in Large Language Models*. The Twelfth International Conference on Learning Representations (ICLR 2024) (`deng2024multilingual`).

[36] Zheng-Xin Yong and Cristina Menghini as well as Stephen H. Bach (2023). *Low-Resource Languages Jailbreak GPT-4*. NeurIPS 2023 Workshop on Socially Responsible Language Modelling Research (`yong2023lowresource`).
