# Google Colab Setup for CatalystOU Pipeline (Qwen 3.6-35B-A3B MoE)

This directory contains the Google Colab environment setup, execution scripts, and table evaluation utilities for running researcher profile extraction and downstream collaboration backtesting with **`Qwen/Qwen3.6-35B-A3B`**.

---

## Model Specifications & Architectural Advantages

- **Model ID:** `Qwen/Qwen3.6-35B-A3B`
- **Release Date:** April 2026
- **Architecture:** Mixture-of-Experts (MoE)
- **Total Parameters:** 35 Billion
- **Active Parameters per Token:** 3 Billion
- **Native Context Window:** 262,144 tokens (262k)
- **Efficiency Profile:** While retaining the broad world knowledge, scientific ontology depth, and reasoning capacity of a 35B model, each forward pass activates only 3B parameters. This delivers high inference throughput and low per-token compute latency.

---

## Directory Overview

- [catalyst_ou_pipeline.ipynb](./colab_setup/Qwen3.6-35B-A3B/catalyst_ou_pipeline.ipynb): Complete Jupyter Notebook ready to upload directly to Google Colab.
- [profile_extraction_cell1.py](./colab_setup/Qwen3.6-35B-A3B/profile_extraction_cell1.py): Self-contained Python script for **Code Cell 1** (profile extraction runner).
- [build_notebook.py](./colab_setup/Qwen3.6-35B-A3B/build_notebook.py): Build script that regenerates `catalyst_ou_pipeline.ipynb` from source components.
- [eval_qwen_stats.py](./colab_setup/Qwen3.6-35B-A3B/eval_qwen_stats.py): Automated evaluation tool that benchmarks extracted profiles and backtesting predictions to output formatted statistics for **Tables 1, 2, and 3** in the paper.

---

## Multi-Stage Notebook Architecture

| Stage | Notebook Cell | Purpose | Execution Logic |
| :--- | :--- | :--- | :--- |
| **Step 0** | **Setup Cell** | **Dependencies & GPU Verification** | Installs CUDA 12.4 synchronized PyTorch wheels, Hugging Face `transformers`, `accelerate`, `bitsandbytes`, PDF parsers, and `sentence-transformers`. |
| **Stage 1** | **Code Cell 1** | **Researcher Profile Extraction (Exp 1)** | Loads `Qwen/Qwen3.6-35B-A3B` in bfloat16, extracts structured micro-JSONs per publication in parallel batches (`batch_size=3`, adaptive fallback to 2 or 1 on OOM), semantically deduplicates entities, and synthesizes thinking patterns and narrative summaries. Saves to `extracted_profile_json/qwen_qwen3.6-35b-a3b/`. |
| **Stage 2** | **Code Cell 2** | **Historical Collaboration Backtesting (Exp 2)** | Feeds independent pre-collaboration profiles into the loaded model runner to infer mechanistic collaboration rationales across 10 schema categories for 5 benchmark cases. Saves to `results/exp2_qwen3.6-35b-a3b/`. |
| **Stage 3** | **Code Cell 3** | **Table Statistics Generator (Tables 1–3)** | Evaluates extracted profiles against gold ground truth using SBERT (`all-mpnet-base-v2`) and Hungarian bipartite matching. Prints formatted Markdown and LaTeX tables for **Tables 1, 2, and 3**. |

---

## Hardware & Environment Settings on Google Colab

### 1. Runtime Selection:
In Google Colab, select **Runtime > Change runtime type**:
- **Hardware accelerator:** `GPU`
- **GPU type:** `A100` (80GB VRAM) *(Recommended for native `torch.bfloat16`)*
- **Runtime shape:** `High-RAM`

### 2. VRAM Footprint & Quantization Handling:
- In unquantized `torch.bfloat16`, the 35B MoE weights occupy approximately **~70 GB VRAM**. On an 80GB A100, this leaves **~10 GB of VRAM** for activations and KV cache.
- **2-3 Paper Batching (`MAP_BATCH_SIZE=3`):** Batching 2-3 papers concurrently consumes ~3.5 to 5.0 GB of dynamic activation and KV-cache memory, fitting safely inside the ~10 GB headroom of the 80GB A100 for a 2x-3x speedup. An automated step-down handler automatically retries at batch size 2 or sequential 1 if any token burst OOM occurs.
- **Smaller GPU Support (L4 / A100 40GB):** The model loader includes automatic VRAM detection. If total GPU VRAM is under 75 GB, or if the environment variable `LOAD_IN_4BIT=1` is set, it automatically loads via 4-bit BitsAndBytes quantization (`nf4`), reducing the footprint to **~18 GB VRAM** so it runs smoothly on Colab standard runtimes.

---

## Dataset Organization in Google Colab

The script scans for researcher publications organized by discipline:

```
CatalystOU-pdf/
├── Biology/
│   ├── Anne K. Dunn/
│   │   ├── paper1.pdf
│   │   └── paper2.pdf
│   └── Ingo Schlupp/
├── CS/
├── ECE/
├── Mathematics/
└── Psychology/
```

### Providing Data in Colab:
```python
# Option A: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Point PDF_ROOT_DIR = '/content/drive/MyDrive/CatalystOU/CatalystOU-pdf'

# Option B: Clone Repo
!git clone https://github.com/YourRepo/CatalystOU.git
%cd CatalystOU
```

---

## Output Destinations & Table Generation

1. **Extracted Profiles:**
   ```
   extracted_profile_json/
   └── qwen_qwen3.6-35b-a3b/
       ├── Biology/
       ├── CS/
       ├── ECE/
       ├── Mathematics/
       └── Psychology/
   ```

2. **Experiment 2 Collaboration Predictions:**
   ```
   results/
   └── exp2_qwen3.6-35b-a3b/
       ├── case_001_ebert_xiao_cs_bio_prediction.json
       ├── case_002_zhao_cheng_cs_ece_prediction.json
       ├── case_003_cai_song_psych_psych_prediction.json
       ├── case_004_yeary_havlicek_ece_ece_prediction.json
       └── case_005_radhakrishnan_antonio_cs_cs_prediction.json
   ```

3. **Generating Stats for Manuscript Tables:**
   Execute Code Cell 3 in the notebook or run locally:
   ```bash
   python colab_setup/Qwen3.6-35B-A3B/eval_qwen_stats.py
   ```
   This prints the ready-to-paste LaTeX tables for:
   - **Table 1:** Field-level Precision, Recall, F1, and Summary Cosine.
   - **Table 2:** Backtesting Average Similarity, Recall@0.45, Recall@0.50, and Discrimination Margin.
   - **Table 3:** Reasoning Mode Alignment percentages across the 5 historical collaboration modes.
