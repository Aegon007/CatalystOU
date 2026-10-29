# Google Colab Setup for CatalystOU Pipeline (Qwen 3.6-35B-A3B MoE - Agentic Architecture)

This directory contains the Google Colab environment setup, execution scripts, and comparative evaluation utilities for running researcher profile extraction and downstream collaboration backtesting using **`Qwen/Qwen3.6-35B-A3B`** under the **Single-Agent Iterative Refinement architecture (Self-Refine / Reflexion)**.

---

## Research Question & Motivation: Agent vs. Pure LLM Benchmark

To evaluate whether autonomous agentic workflows offer measurable advantages over single-pass prompting, this pipeline investigates the empirical performance delta between pure LLM baselines and stateful iterative agent architectures across both profile extraction and collaboration backtesting.

Rather than deploying complex, uncalibrated 5-agent persona debate committees that introduce compounding hallucinations, this pipeline implements a rigorous, stateful **2-Pass Single-Agent Iterative Refinement architecture** tailored to resolve the specific failure modes of pure LLMs across both experiments.

---

## Architectural Comparison: Pure LLM Baseline vs. Agent Architecture

| Dimension | Experiment 1: Researcher Profiling | Experiment 2: Collaboration Backtesting |
| :--- | :--- | :--- |
| **Operational Goal** | **Intra-Researcher Information Extraction** | **Inter-Researcher Prospective Synergy Reasoning** |
| **Input State** | Single researcher's publication set (PDFs) | Pre-collaboration profiles ($\text{Profile}_A^{<T}, \text{Profile}_B^{<T}$) |
| **Pure LLM Failure Mode** | **Granularity vs. Recall Tradeoff:** Stateless Map-Reduce either under-recalls specialized tools or produces broad, uncalibrated approximations | **Generic Overlap Fallacy:** Generates superficial synergies that could apply to arbitrary unrelated faculty pairs |
| **Agent Paradigm** | **Two-Stage Stateful Agent (Investigator Accumulator -> Reflective PI Synthesizer)** | **Contrastive Adversarial Verifier (Skeptic Audit)** |
| **Execution Loop** | **Stage 1 (Investigator Mode):** Sequential working memory accumulation paper-by-paper with 100% focused attention. Captures 100% of fine-grained methodologies, instruments, and software libraries.<br>**Stage 2 (Reflective PI Mode):** Executive synthesis pass translating accumulated laboratory sub-tasks into macro translational Application Areas, calibrating 3-4 Thinking Patterns, and refining career summary.<br>**Final Post-Processing:** SBERT semantic deduplication on entity lists. | **Adversarial Skeptic Audit:**<br>1. *Contrastive Selectivity Check:* Stress-tests whether proposed synergies could apply to an unrelated negative control pair.<br>2. *Mechanistic Grounding:* Enforces explicit causal bridge between specific tools and specific domain problems.<br>3. *Buzzword Pruning:* Strips superficial overlap and refines summary themes. |
| **Output Destination** | `extracted_profile_json/qwen_qwen3.6-35b-a3b_agent/` | `results/exp2_qwen3.6-35b-a3b_agent/`<br>`results/exp2_qwen3.6-35b-a3b_agent_negative/` |
| **Evaluation Harness** | Hungarian bipartite matching F1 ($\tau=0.65, 0.50$) + Summary Cosine | Narrative alignment cosine against realized joint papers + Selectivity Margin ($\Delta$) over negative controls |

---

## Directory Overview

- [catalyst_ou_agent_pipeline.ipynb](./colab_setup/Qwen3.6-35B-A3B-Agent/catalyst_ou_agent_pipeline.ipynb): Complete, self-contained Jupyter Notebook ready to upload directly to Google Colab.
- [profile_extraction_cell1.py](./colab_setup/Qwen3.6-35B-A3B-Agent/profile_extraction_cell1.py): Source script for **Code Cell 1** (Agent Profile Extraction runner with `PILOT_MODE` support).
- [build_notebook.py](./colab_setup/Qwen3.6-35B-A3B-Agent/build_notebook.py): Build script that regenerates `catalyst_ou_agent_pipeline.ipynb` from source components.
- [eval_agent_stats.py](./colab_setup/Qwen3.6-35B-A3B-Agent/eval_agent_stats.py): Automated evaluation tool that benchmarks Agent outputs head-to-head against the Pure LLM baseline for **Tables 1, 2, and 3**.

---

## Google Colab Runtime & Hardware Requirements

1. **Recommended Accelerator:**
   - Go to **Runtime > Change runtime type**:
     - **Hardware accelerator:** `GPU`
     - **GPU type:** `A100` (80GB VRAM)
     - **Runtime shape:** `High-RAM`
2. **VRAM Footprint & Native `bfloat16`:**
   - In unquantized `torch.bfloat16`, `Qwen/Qwen3.6-35B-A3B` occupies **~70 GB VRAM**.
   - The active parameters per token are only **3B**, delivering fast forward passes during multi-pass loops.
   - Batch size 2–3 (`MAP_BATCH_SIZE=3`) consumes only ~3.5 to 5.0 GB dynamic activation headroom.
3. **Smaller GPU Compatibility (L4 / A100 40GB):**
   - The runner detects available VRAM automatically. If GPU VRAM is under 75 GB, or if `LOAD_IN_4BIT=1` is set, it automatically initializes with 4-bit BitsAndBytes quantization (`nf4`), reducing footprint to **~18 GB VRAM**.

---

## Execution Workflow on Google Colab

### Step 0: Environment Setup & Package Installation
Run the first code cell to install matching CUDA 12.4 PyTorch wheels, Hugging Face `transformers`, `accelerate`, `bitsandbytes`, PDF parsers, and `sentence-transformers`.
> **Tip:** If running on a fresh instance, click **Runtime > Restart session** after Step 0 so C++ CUDA libraries load into Python.

---

### Step 1: Agentic Profile Extraction (Code Cell 1)
- **Pilot Mode Toggle:**
  By default, `profile_extraction_cell1.py` sets:
  ```python
  PILOT_MODE = True
  PILOT_LIMIT = 3
  ```
  This immediately selects **3 representative faculty** across different disciplines:
  1. `Biology/Anne K. Dunn`
  2. `CS/David Ebert`
  3. `ECE/Mark Yeary`
  
  In **~2–3 minutes**, Colab completes the sequential accumulation turns for these 3 faculty, printing the exact accumulation deltas per paper (e.g. *"[Turn 2/5] Added: +4 techniques (total 22), +3 platforms (total 14)..."*).
- **Full Benchmark Run:**
  Once satisfied with the pilot accumulation traces, change `PILOT_MODE = False` in the cell and re-run. It will seamlessly process all 50 faculty across all 5 disciplines with automatic resumption (skipping already processed profiles).

---

### Step 2: Agentic Collaboration Backtesting (Code Cell 2)
- Reuses the active in-memory model runner from Cell 1.
- Evaluates all 10 benchmark pairs:
  - **5 Historical Positive Cases** (evaluated against realized ground truth)
  - **5 Cross-Disciplinary Negative Controls** (selectivity baseline)
- Runs the 2-Pass Contrastive Adversarial Verifier loop (`Pass 1 Hypothesis -> Pass 2 Skeptic Audit`).
- Skips already computed cases to permit safe resumption.
- Packages results into `qwen3.6_agent_exp2_results.zip` and triggers automatic browser download.

---

### Step 3: Comparative Evaluation & Table Generation (Code Cell 3)
Run Code Cell 3 or execute locally:
```bash
python colab_setup/Qwen3.6-35B-A3B-Agent/eval_agent_stats.py
```
This automatically computes:
1. **Table 1:** Field-level Precision, Recall, and F1 comparison between Pure LLM and Agent, highlighting the $\Delta \text{F1}$ gains on Datasets & Platforms and Application Areas.
2. **Table 2:** Backtesting Positive Avg Sim, Negative Control Avg Sim, and Selectivity Margin ($\Delta$).
3. **Table 3:** Reasoning Mode Alignment (% at $\tau = 0.45$).
4. **LaTeX Table Snippets:** Ready to paste directly into `paper_changes/changes/experiments.tex`.

---

## Fallback & Publication Decision Rule

Per the experimental agreement with Dr. Chenggang Wang:
1. **If Agent outperforms Pure LLM** (e.g., F1 gain $\ge 3\%$ or wider selectivity margin over $+0.3927$):
   - Add the **"Qwen 3.6-35B-A3B (Agent Loop)"** row to Table 1 and Table 2 in `paper_changes/changes/experiments.tex`.
2. **If Agent gains are marginal or plateau**:
   - Formally commit the pure LLM results as the empirical baseline in `discussion.tex`, framing single-pass limits as the research boundary and the curated benchmark dataset as the primary contribution.
