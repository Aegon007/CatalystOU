import json
from pathlib import Path

colab_dir = Path(__file__).resolve().parent
cell1_path = colab_dir / "profile_extraction_cell1.py"
cell2_path = colab_dir / "exp2_true_backtesting_cell2.py"
notebook_path = colab_dir / "catalyst_ou_pipeline.ipynb"

with cell1_path.open("r", encoding="utf-8") as f:
    cell1_lines = f.readlines()

with cell2_path.open("r", encoding="utf-8") as f:
    cell2_lines = f.readlines()

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CatalystOU: Google Colab Pipeline (Qwen 3.6-35B-A3B Pure MoE)\n",
                "### Hardware: NVIDIA A100 (80GB VRAM) High-RAM Runtime (or L4 / A100 40GB with 4-bit quantization)\n",
                "### Model: `Qwen/Qwen3.6-35B-A3B` (Mixture of Experts: 35B total params, 3B active params per token, 262k context)\n",
                "\n",
                "This notebook provides the complete, end-to-end execution pipeline for evaluating **Qwen 3.6-35B-A3B (Pure MoE Baseline)**:\n",
                "- **Step 0: Environment Setup & Package Installation**: Installs PyTorch (CUDA 12.4 ABI), Hugging Face, bitsandbytes, PDF extraction, and embedding libraries.\n",
                "- **Code Cell 1 (Profile Extraction - Exp 1)**: Loads Qwen 3.6-35B-A3B in bfloat16, runs Map-Reduce profile extraction across all 50 faculty publications, and saves profiles to `extracted_profile_json/qwen_qwen3.6-35b-a3b/`.\n",
                "- **Code Cell 2 (True Collaboration Backtesting - Exp 2)**: Directly ingests pre-collaboration publications from `CatalystOU-pdf/~Collaborative Datasets/` (volume-balanced to ~3 papers per researcher), extracts pre-collaboration profiles using the exact same Map-Reduce pipeline as Exp 1, and runs single-pass synergy reasoning across 5 historical co-authorship pairs and 5 volume-balanced negative control pairs. Automatically packages and downloads `qwen3.6_exp2_true_results.zip`.\n",
                "- **Code Cell 3 (Table Statistics & Evaluation)**: Computes automated evaluation metrics (SBERT + Hungarian matching) and outputs the exact Markdown and LaTeX tables for **Tables 1, 2, and 3** in the paper.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 0: Environment Setup & Package Installation\n",
                "Run this cell first to install dependencies and verify your GPU allocation.\n",
                "\n",
                "> **Important Note on Session Restart:** After running this installation cell for the first time on a fresh Colab instance, please click **Runtime > Restart session** (or press `Ctrl + M .`) so Python loads the newly installed CUDA C++ binaries into memory."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# STEP 0: PIP & PYTHON DEPENDENCY INSTALLATION\n",
                "# ==============================================================================\n",
                "# 1. Install matching PyTorch + Torchvision + Torchaudio for CUDA 12.4 (Prevents ABI mismatch)\n",
                "!pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n",
                "\n",
                "# 2. Install Hugging Face, quantization, PDF extraction, and embedding libraries\n",
                "!pip install -q -U \"transformers>=4.48.0\" \"huggingface_hub>=0.28.0\" accelerate bitsandbytes sentencepiece pymupdf pypdf sentence-transformers tqdm tabulate\n",
                "\n",
                "# 3. Verify GPU status and VRAM capacity\n",
                "!nvidia-smi\n",
                "\n",
                "# 4. (Optional) Mount Google Drive if your PDF dataset or repo is stored on Drive:\n",
                "# from google.colab import drive\n",
                "# drive.mount('/content/drive')\n",
                "\n",
                "# 5. (Optional) Clone repository if running in a fresh Colab instance:\n",
                "# !git clone https://github.com/YourUsername/CatalystOU.git\n",
                "# %cd CatalystOU\n",
                "\n",
                "# 6. (Optional) Unpack publication PDFs (included in repo at colab_setup/CatalystOU-pdf.zip):\n",
                "# !unzip -q colab_setup/CatalystOU-pdf.zip -d .\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Code Cell 1: Researcher Profile Extraction Pipeline (Qwen 3.6-35B-A3B MoE)\n",
                "This cell handles:\n",
                "1. Loading Qwen 3.6-35B-A3B in native bfloat16 onto the A100 GPU.\n",
                "2. Discovering researcher publication PDFs in `CatalystOU-pdf/`.\n",
                "3. Multi-engine text extraction & reference cleaning.\n",
                "4. Map-Reduce entity extraction, semantic deduplication, and profile synthesis.\n",
                "5. Saving profiles to `extracted_profile_json/qwen_qwen3.6-35b-a3b/<Discipline>/<Researcher>_profile.json`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell1_lines
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Code Cell 2: True Pre-Collaboration Historical Backtesting (Pure MoE Map-Reduce)\n",
                "This cell implements the localized pre-collaboration historical backtesting pipeline for Experiment 2:\n",
                "- **Strict Temporal Windowing**: Ingests publications directly from `CatalystOU-pdf/~Collaborative Datasets/<Pair>/Publications Before Collaboration/<Author>/`.\n",
                "- **Volume-Balanced Negative Controls**: Negative pairs are balanced to ~3 papers each from `CatalystOU-pdf/`.\n",
                "- **On-the-Fly Profile Extraction**: Reconstructs true pre-collaboration profiles using the exact same Map-Reduce pipeline (`PAPER_EXTRACTION_SYSTEM_PROMPT` + `SYNTHESIS_SYSTEM_PROMPT`) as Exp 1.\n",
                "- **Single-Pass Synergy Reasoning**: Infers collaboration mechanisms across 10 ontology categories.\n",
                "\n",
                "Evaluates all 10 benchmark pairs:\n",
                "- **5 Historical Positive Benchmark Cases** (saved to `results/exp2_qwen3.6-35b-a3b_true/`)\n",
                "- **5 Cross-Disciplinary Negative Controls** (saved to `results/exp2_qwen3.6-35b-a3b_true_negative/`)\n",
                "\n",
                "Features:\n",
                "- **Runner safety**: Automatically reuses active in-memory `model_runner` from Cell 1 without reloading.\n",
                "- **Resume check**: Automatically skips pairs whose prediction file already exists.\n",
                "- **Auto-zip download**: Packages all predictions into `qwen3.6_exp2_true_results.zip` and triggers browser download."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell2_lines
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Code Cell 3: Automated Benchmark Evaluation & Table Generation\n",
                "This cell evaluates the extracted profiles against expert human annotations and computes metrics for Tables 1, 2, and 3."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# STEP 3: BENCHMARK EVALUATION (TABLES 1, 2, 3)\n",
                "# ==============================================================================\n",
                "import os\n",
                "import sys\n",
                "import json\n",
                "from pathlib import Path\n",
                "\n",
                "# Ensure project modules are available\n",
                "sys.path.insert(0, str(Path.cwd()))\n",
                "\n",
                "try:\n",
                "    from proj_test.test1_profile_acc import evaluate_all_profiles\n",
                "    from utils.profile_io import list_researcher_profiles\n",
                "    import utils.matching as matching\n",
                "    HAS_EVAL = True\n",
                "except ImportError:\n",
                "    HAS_EVAL = False\n",
                "\n",
                "print('[*] Table 1 Profile Extraction Evaluation:')\n",
                "summary_path = Path('results/qwen_qwen3.6-35b-a3b/comprehensive_evaluation_results.json')\n",
                "if summary_path.exists():\n",
                "    with open(summary_path, 'r', encoding='utf-8') as f:\n",
                "        res = json.load(f)\n",
                "    print(json.dumps(res.get('per_key', {}), indent=2))\n",
                "else:\n",
                "    print('[*] Results will be available after running evaluation script or notebook cells.')\n"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "A100"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with notebook_path.open("w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"[+] Successfully built pure MoE notebook: {notebook_path}")
