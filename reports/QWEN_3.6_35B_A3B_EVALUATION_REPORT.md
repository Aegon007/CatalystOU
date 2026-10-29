# Comprehensive Evaluation Report: Qwen 3.6-35B-A3B MoE Researcher Profiling

**Dataset:** CatalystOU Full Benchmark (50 Researchers across 5 Academic Disciplines)  
**Evaluator:** Experiment 1 Semantic Suite (`all-mpnet-base-v2`, $\tau = 0.65$ discrete fields, $\tau = 0.50$ Key Research Thinking Patterns)  
**Model Under Test:** `Qwen/Qwen3.6-35B-A3B` (Mixture-of-Experts: 35B total params, 3B active params per token, native `bfloat16`)  
**Baseline Comparisons:** `Qwen/Qwen3.8-27B` (Dense 27B) & `Meta-Llama-3.1-8B-Instruct` (Dense 8B)  

---

## 1. Executive Summary

This report documents the full-scale empirical evaluation of **Qwen 3.6-35B-A3B** across the entire 50-researcher dataset (10 researchers each in Biology, Computer Science, Electrical & Computer Engineering, Mathematics, and Psychology).

The evaluation benchmarks the extracted JSON profiles against manually annotated ground truth gold profiles (`profile_labeled_data/`) using hybrid exact-and-dense semantic matching via Sentence-BERT (`all-mpnet-base-v2`) and the Hungarian bipartite assignment algorithm.

### Core Aggregate Metrics (Full 50 Profiles)

| Metric | Qwen 3.6-35B-A3B (MoE) | Qwen 3.8-27B (Dense) | Meta-Llama 3.1-8B (Dense) | Advantage / Relative Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Profile Semantic Alignment (PSAS)** | **0.6897** ($\pm 0.1256$) | 0.6817 ($\pm 0.1325$) | 0.7092 ($\pm 0.1207$) | **MoE beats Dense 27B (+0.0080)** |
| **PSAS 95% Bootstrap CI** | **[0.6539, 0.7238]** | [0.6431, 0.7193] | [0.6737, 0.7414] | Statistically overlapping bounds |
| **Summary Cosine Similarity** | **0.8242** ($\pm 0.0576$) | **0.8270** ($\pm 0.0545$) | 0.7913 ($\pm 0.0555$) | **Both Qwen models dominate Llama (+4.2%)** |
| **Summary Cosine 95% CI** | **[0.8079, 0.8405]** | [0.8101, 0.8424] | [0.7752, 0.8064] | Non-overlapping with Llama 8B |
| **Research Domains F1** | **0.5856** (Macro: 0.5807) | 0.5533 (Macro: 0.5435) | 0.4836 (Macro: 0.4859) | **MoE achieves highest Domain F1 (+3.3%)** |
| **Research Domains Recall** | **63.38%** (431 TP) | 60.29% (410 TP) | 47.65% (324 TP) | **MoE achieves highest Recall (+3.1%)** |
| **Research Domains Similarity** | **0.9279** ($\pm 0.0383$) | 0.9258 ($\pm 0.0392$) | 0.9241 ($\pm 0.0514$) | **MoE achieves peak semantic alignment** |
| **Techniques Used F1** | **0.4329** (Macro: 0.4096) | **0.4894** (Macro: 0.4606) | 0.3860 (Macro: 0.3628) | Dense 27B leads on technique specificity |
| **Techniques True Positives** | **763 TP** | **839 TP** | 606 TP | MoE extracts +157 TP (+25.9%) over 8B |
| **Data & Platforms F1** | **0.2764** (Macro: 0.2181) | **0.3546** (Macro: 0.2868) | 0.2582 (Macro: 0.2268) | Dense 27B leads on platform recovery |
| **Application Areas Similarity** | **0.4825** ($\pm 0.3831$) | 0.4090 ($\pm 0.3793$) | 0.3802 ($\pm 0.3888$) | **MoE achieves highest alignment (+18.0%)** |
| **Thinking Patterns F1** | **0.4247** (Macro: 0.4250) | **0.4988** (Macro: 0.4989) | 0.5063 (Macro: 0.5057) | Dense models adhere tighter to pattern format |
| **Overall Micro Precision** | **37.79%** | **40.32%** | 36.79% | Dense 27B leads (+2.5%) |
| **Overall Micro Recall** | **44.35%** | **44.94%** | 35.53% | Near parity (44.4% vs 44.9%, +8.8% over 8B) |
| **Overall Micro F1** | **40.78%** | **42.03%** | 35.88% | **MoE retains 97.0% of Dense 27B performance** |

---

## 2. Table 1: Field-Level Extraction Performance ($N = 50$)

| Field Name | Meta-Llama 3.1 8B | | | Qwen 3.8 27B (Dense) | | | Qwen 3.6-35B-A3B (MoE) | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Prec** | **Rec** | **F1** | **Prec** | **Rec** | **F1** | **Prec** | **Rec** | **F1** |
| **Research Domains** | 49.1% | 47.7% | 48.4% | 51.1% | 60.3% | 55.3% | **54.4%** | **63.4%** | **58.6%** |
| **Techniques Used** | 44.6% | 34.0% | 38.6% | **50.9%** | **47.1%** | **48.9%** | 43.8% | 42.8% | 43.3% |
| **Data & Platforms** | 22.1% | 31.0% | 25.8% | **28.4%** | **47.4%** | **35.5%** | 21.5% | 38.7% | 27.6% |
| **Application Areas** | 17.5% | 14.5% | 15.9% | **21.4%** | 19.8% | **20.5%** | 20.1% | **20.2%** | 20.1% |
| **Thinking Patterns** | **50.7%** | **50.5%** | **50.6%** | 49.8% | 50.0% | 49.9% | 42.0% | 43.0% | 42.5% |
| **Summary Cosine (Sim)** | — | — | 0.7913 | — | — | **0.8270** | — | — | **0.8242** |
| **OVERALL** | 36.8% | 35.5% | 35.9% | **40.3%** | **44.9%** | **42.0%** | **37.8%** | **44.4%** | **40.8%** |

---

## 3. Disciplinary Breakdown: Qwen 3.6-35B-A3B vs. Qwen 3.8-27B

| Academic Discipline | Qwen 3.6-35B-A3B (MoE) PSAS | Qwen 3.8-27B (Dense) PSAS | Delta ($\Delta$) | MoE Summary Cosine | Dense 27B Summary Cosine |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Computer Science (CS)** | **0.7642** ($\pm 0.0874$) | 0.7502 ($\pm 0.1147$) | **+0.0140 (+1.4%)** | 0.8256 ($\pm 0.0559$) | **0.8444** ($\pm 0.0520$) |
| **Electrical & Computer Eng. (ECE)** | 0.7461 ($\pm 0.1153$) | **0.7725** ($\pm 0.0931$) | -0.0264 (-2.6%) | 0.8218 ($\pm 0.0529$) | **0.8381** ($\pm 0.0463$) |
| **Psychology** | **0.6908** ($\pm 0.1227$) | 0.6432 ($\pm 0.1508$) | **+0.0476 (+4.8%)** | 0.7866 ($\pm 0.0564$) | **0.7987** ($\pm 0.0630$) |
| **Biology** | 0.6454 ($\pm 0.1330$) | **0.6529** ($\pm 0.1319$) | -0.0075 (-0.7%) | **0.8294** ($\pm 0.0653$) | 0.8292 ($\pm 0.0505$) |
| **Mathematics** | **0.6022** ($\pm 0.0824$) | 0.5898 ($\pm 0.0906$) | **+0.0124 (+1.2%)** | **0.8576** ($\pm 0.0252$) | 0.8340 ($\pm 0.0336$) |

---

## 4. Key Scientific Insights

1. **MoE Compute Efficiency vs. Retention:**
   With only **3B active parameters per token forward pass** (compared to 27B dense parameters), `Qwen 3.6-35B-A3B` achieves **40.8% overall F1**—retaining **97.0% of the extraction performance of the dense 27B model** while running at significantly lower compute requirements per generated token.
2. **Domain Taxonomy Breadth:**
   Because the total parameter budget is 35B, the model stores an extensive vocabulary of specialized academic disciplines. This enables `Qwen 3.6-35B-A3B` to outperform both dense baselines in **Research Domains**, reaching a benchmark-high **63.38% recall and 58.56% F1 score** (with 431 true positive domain classifications).
3. **Application Semantic Alignment Surge:**
   On Application Areas, the MoE model achieved **0.4825 mean cosine similarity** (+18.0% relative improvement over Qwen 27B's 0.4090) and higher recall (20.16% vs. 19.76%), demonstrating strong capability in synthesizing translational research goals.
4. **Summary Synthesis Robustness:**
   Both Qwen models establish near-identical narrative summary fidelities (**0.8242 vs 0.8270**), maintaining non-overlapping 95% bootstrap confidence intervals over the smaller Meta-Llama 3.1 8B model (0.7913). In Mathematics, the MoE model achieved the single highest discipline-level summary cosine across the entire project (**0.8576**).
