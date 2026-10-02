# Google Colab Setup for CatalystOU Pipeline (Meta Llama 3.1 8B)

This directory contains the Google Colab runner configured for **`meta-llama/Meta-Llama-3.1-8B-Instruct`**.

---

## Directory Overview

- [catalyst_ou_pipeline.ipynb](file:///c:/Users/al3xa/Desktop/CatalystOU/CatalystOU/colab_setup/Meta/catalyst_ou_pipeline.ipynb): Complete Jupyter Notebook ready to upload directly to Google Colab.
- [profile_extraction_cell1.py](file:///c:/Users/al3xa/Desktop/CatalystOU/CatalystOU/colab_setup/Meta/profile_extraction_cell1.py): Standalone Python script for **Code Cell 1** (Meta Llama 3.1 8B runner).
- [build_notebook.py](file:///c:/Users/al3xa/Desktop/CatalystOU/CatalystOU/colab_setup/Meta/build_notebook.py): Script to regenerate the notebook from `profile_extraction_cell1.py`.

---

## Two-Part Architecture

| Part | Notebook Stage | Purpose | Primary Model / Logic |
| :--- | :--- | :--- | :--- |
| **Part 1** | **Code Cell 1** | **Profile Extraction** | Loads `meta-llama/Meta-Llama-3.1-8B-Instruct` in bfloat16, extracts micro-JSONs per publication (Map), semantically deduplicates entities (Reduce), and synthesizes cohesive profiles. |
| **Part 2** | **Code Cell 2** | **Synergy Checker** | *(Upcoming)* Loads profiles produced by Cell 1 and predicts pairwise collaboration opportunities across 10 mechanistic categories. |

---

## Hardware & Gated Model Access

1. **Hugging Face Gated Access**:
   - Meta Llama 3.1 is a gated repository. Accept the terms on [Hugging Face Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
   - In Google Colab, add your token to **Secrets** (the key icon in the left sidebar) named `HF_TOKEN`. The script automatically reads `google.colab.userdata.get('HF_TOKEN')`.
   - Alternatively, run `!huggingface-cli login` in the setup cell.

2. **Hardware Requirements**:
   - **VRAM Footprint**: In `torch.bfloat16`, Llama 3.1 8B occupies only **~16 GB VRAM**.
   - On an **A100 (80GB)**, it runs with exceptional speed and leaves **~64 GB of free VRAM** for massive KV caching and concurrency.
   - It also runs comfortably on standard 24GB GPUs (like an NVIDIA A10G or L4).

---

## Output Profile Structure

Profiles are saved in the exact schema required by `proj_test/test1_profile_acc.py` and `proj_test/test2_pred_acc.py`:

```
extracted_profile_json/
└── meta-llama_meta-llama-3.1-8b-instruct/
    ├── Biology/
    │   ├── Anne_K_Dunn_profile.json
    │   └── ...
    └── CS/
        └── ...
```

---

## Key Features & Safety Mechanisms

- **Llama 3.1 Stop Token Handling**: Specifically tracks `<|eot_id|>` and `<|end_of_text|>` terminators to prevent run-on generations.
- **Resumption & Checkpointing**: Skips already extracted researcher profiles so long runs can be paused or resumed.
- **Sanitizer**: Cleans LaTeX and escaped backslashes (`\alpha`, `\beta`, `\mu`) before parsing into JSON.
- **Semantic Deduplication**: Uses `sentence-transformers` (`all-mpnet-base-v2`) to cluster synonyms in techniques, tools, and domains.
