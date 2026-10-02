# Google Colab Setup for CatalystOU Pipeline

This directory contains the Google Colab environment setup and execution scripts for running high-throughput researcher profile extraction and downstream synergy evaluation using an **NVIDIA A100 GPU (80GB VRAM)** with **`Qwen/Qwen3.8-27B`**.

---

## Directory Overview

- [catalyst_ou_pipeline.ipynb](file:///c:/Users/al3xa/Desktop/CatalystOU/CatalystOU/colab_setup/Qwen/catalyst_ou_pipeline.ipynb): Complete Jupyter Notebook ready to upload directly to Google Colab.
- [profile_extraction_cell1.py](file:///c:/Users/al3xa/Desktop/CatalystOU/CatalystOU/colab_setup/Qwen/profile_extraction_cell1.py): Standalone, self-contained Python script for **Code Cell 1** (can be pasted directly into a Colab code cell or run in a terminal).
- [build_notebook.py](file:///c:/Users/al3xa/Desktop/CatalystOU/CatalystOU/colab_setup/Qwen/build_notebook.py): Utility script that regenerates `catalyst_ou_pipeline.ipynb` from `profile_extraction_cell1.py`.

---

## Two-Part Architecture

| Part | Notebook Stage | Purpose | Primary Model / Logic |
| :--- | :--- | :--- | :--- |
| **Part 1** | **Code Cell 1** | **Profile Extraction** | Loads `Qwen/Qwen3.8-27B` in bfloat16, extracts micro-JSONs per publication in parallel batches (Map), semantically deduplicates entities (Reduce), and synthesizes cohesive profiles. |
| **Part 2** | **Code Cell 2** | **Synergy Checker** | *(Upcoming)* Loads profiles produced by Cell 1 and predicts pairwise collaboration opportunities across 10 mechanistic categories. |

---

## Hardware & Environment Settings on Google Colab

1. **Runtime Type**: In Colab, select **Runtime > Change runtime type**:
   - **Hardware accelerator**: `GPU`
   - **GPU type**: `A100` (requires Colab Pro / Pro+)
   - **Runtime shape**: `High-RAM`
2. **VRAM Footprint & Parallel Batching**:
   - `Qwen/Qwen3.8-27B` in `torch.bfloat16` occupies approximately **~54 GB VRAM**.
   - On the 80GB A100, this leaves **~26 GB of free VRAM**, which is leveraged for **batched parallel Map inference** (processing all 5 papers of a researcher simultaneously in a single forward pass).

---

## PDF Dataset Organization

The extraction pipeline automatically scans and organizes profiles by discipline and author:

```
CatalystOU-pdf/
├── Biology/
│   ├── Anne K. Dunn/
│   │   ├── paper1.pdf
│   │   └── paper2.pdf
│   └── Ingo Schlupp/
│       └── paper1.pdf
├── CS/
│   └── ...
└── ECE/
    └── ...
```

### Providing the Dataset in Google Colab:
You can provide the dataset through any of the following methods:
1. **Google Drive (Recommended)**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Point PDF_ROOT_DIR to '/content/drive/MyDrive/CatalystOU/CatalystOU-pdf'
   ```
2. **Git Clone**:
   ```bash
   !git clone https://github.com/YourRepo/CatalystOU.git
   %cd CatalystOU
   ```
3. **Zip Upload**:
   Upload `CatalystOU-pdf.zip` to Colab and unzip:
   ```bash
   !unzip -q CatalystOU-pdf.zip -d CatalystOU-pdf/
   ```

---

## Output Profile Structure

Profiles are saved in the exact schema required by `proj_test/test1_profile_acc.py` and `proj_test/test2_pred_acc.py`:

```
extracted_profile_json/
└── qwen_qwen3.8-27b/
    ├── Biology/
    │   ├── Anne_K_Dunn_profile.json
    │   └── ...
    └── CS/
        └── ...
```

### Profile JSON Schema:
```json
{
    "Researcher Profile:": "Anne K. Dunn",
    "Affiliation:": "Department of Microbiology and Plant Biology, University of Oklahoma",
    "Research Domains": [ ... ],
    "Techniques Used": [ ... ],
    "Data & Platforms": [ ... ],
    "Application Areas": [ ... ],
    "Key Research Thinking Patterns": [
        "Pattern Name: Definition (e.g., Specific experimental finding citing exact genes/models)."
    ],
    "Summary Description": "Cohesive narrative summary under 150 words..."
}
```

---

## Key Features & Safety Mechanisms

- **Resumption & Checkpointing**: If a profile already exists at destination, it is safely skipped. Disconnections or restarts in Colab will not repeat already extracted researchers.
- **Thinking Tag Removal**: Handles `<think>...</think>` tokens cleanly if the model invokes internal reasoning traces.
- **Escaped Character Sanitizer**: Automatically cleans invalid escape sequences in scientific formulas (e.g., `\alpha`, `\beta`, `\mu`).
- **Semantic Deduplication**: Uses `sentence-transformers` (`all-mpnet-base-v2`) with cosine similarity thresholding to cluster near-synonym techniques and domains across papers.
- **VRAM Cleanup**: Explicitly runs `torch.cuda.empty_cache()` between researchers to prevent memory fragmentation.
