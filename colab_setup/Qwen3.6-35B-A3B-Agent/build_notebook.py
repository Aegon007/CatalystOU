import json
from pathlib import Path

colab_dir = Path(__file__).resolve().parent
cell1_path = colab_dir / "profile_extraction_cell1.py"
cell2_path = colab_dir / "exp2_true_backtesting_cell2.py"
notebook_path = colab_dir / "catalyst_ou_agent_pipeline.ipynb"

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
                "# CatalystOU: Google Colab Pipeline (Qwen 3.6-35B-A3B MoE - Agentic Architecture)\n",
                "### Hardware: NVIDIA A100 (80GB VRAM) High-RAM Runtime (or L4 / A100 40GB with 4-bit quantization)\n",
                "### Model: `Qwen/Qwen3.6-35B-A3B` (Mixture of Experts: 35B total params, 3B active params per token, 262k context)\n",
                "### Paradigm: **Two-Stage Stateful Agent (Sequential Accumulator -> Reflective PI Synthesizer)** & **Two-Pass Synthesize-and-Review Reasoner**\n",
                "\n",
                "This notebook implements the complete **Agentic vs. Pure LLM** comparative benchmark:\n",
                "- **Step 0: Environment Setup & Package Installation**: Synchronizes PyTorch CUDA 12.4 ABI, Hugging Face, bitsandbytes, and embedding libraries.\n",
                "- **Code Cell 1 (Agent Profile Extraction - Exp 1)**: Runs the Two-Stage Stateful Agent (Stage 1 Investigator Accumulator gathers rich tools/platforms with persistent memory -> Stage 2 Reflective PI Synthesizer consolidates macro Application Areas, Thinking Patterns, and Summary). Includes `PILOT_MODE = True` toggle for a ~2-minute 3-researcher test before running the full 50 faculty.\n",
                "- **Code Cell 2 (Agent Collaboration True Backtesting - Exp 2)**: Directly ingests pre-collaboration publications from `CatalystOU-pdf/~Collaborative Datasets/` (volume-balanced to ~3 papers per researcher for positive and negative pairs), extracts localized temporal profiles, and runs the 2-Pass Synthesize-and-Review Reasoner (`Pass 1: Hypothesis Synthesis -> Pass 2: Skeptic Audit & Selectivity Pruning`) across 5 historical co-authorship pairs and 5 volume-balanced negative control pairs. Automatically packages and downloads `qwen3.6_agent_exp2_true_results.zip`.\n",
                "- **Code Cell 3 (Table Statistics & Agent vs. Pure LLM Comparison)**: Directly benchmarks Agent outputs against Pure LLM baselines, generating side-by-side comparison tables and LaTeX snippets for **Tables 1, 2, and 3** in the paper.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 0: Environment Setup & Package Installation\n",
                "Run this cell first to install dependencies and verify your GPU allocation.\n",
                "\n",
                "> **Important Note on Session Restart:** After running this installation cell for the first time on a fresh Colab instance, click **Runtime > Restart session** (or press `Ctrl + M .`) so Python loads the newly installed CUDA C++ binaries into memory."
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
                "!pip install -q -U \"transformers>=4.48.0\" \"huggingface_hub>=0.28.0\" accelerate bitsandbytes sentencepiece pymupdf pypdf pdfplumber sentence-transformers tqdm tabulate\n",
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
                "## Code Cell 1: Agentic Researcher Profile Extraction (Two-Stage Stateful Agent)\n",
                "This cell implements the **Two-Stage Stateful Agent**:\n",
                "- **Stage 1 (Investigator Accumulator Mode)**: Iterates over publications sequentially paper-by-paper with 100% focused attention, building an episodic working memory dossier of all techniques, platforms, instruments, and findings with zero recall loss.\n",
                "- **Stage 2 (Reflective Executive PI Mode)**: Synthesizes the complete accumulated dossier into macro-level translational **Application Areas**, 3-4 calibrated **Key Research Thinking Patterns**, and a polished **Summary Description**.\n",
                "- **Final Post-Processing**: SentenceTransformer (`all-mpnet-base-v2`) clusters near-synonyms on Techniques and Platforms without dropping recall.\n",
                "\n",
                "**Pilot Mode Toggle:**\n",
                "- `PILOT_MODE = True` (Default): Runs a fast ~2-minute test on 3 diverse researchers (`Anne K. Dunn` [Bio], `David Ebert` [CS], `Mark Yeary` [ECE]) to inspect accumulation traces.\n",
                "- `PILOT_MODE = False`: Runs the full 50-researcher benchmark across all 5 disciplines.\n",
                "\n",
                "Profiles are saved to `extracted_profile_json/qwen_qwen3.6-35b-a3b_agent/<Discipline>/<Researcher>_profile.json`."
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
                "## Code Cell 2: True Pre-Collaboration Historical Backtesting (Synthesize-and-Review Two-Pass Engine)\n",
                "This cell implements the localized pre-collaboration historical backtesting pipeline for Experiment 2:\n",
                "- **Strict Temporal Windowing**: Ingests publications directly from `CatalystOU-pdf/~Collaborative Datasets/<Pair>/Publications Before Collaboration/<Author>/` (published strictly before the collaboration event).\n",
                "- **Volume-Balanced Negative Controls**: Disjoint pairs are balanced to the exact same publication volume (~3 papers each) from `CatalystOU-pdf/`.\n",
                "- **Localized On-the-Fly Profile Extraction**: Reconstructs true pre-collaboration profiles using the two-stage accumulator/PI agent.\n",
                "- **Two-Pass Synergy Reasoning Engine**:\n",
                "  - **Pass 1 (Hypothesis Synthesis)**: Synthesizes potential collaborative mechanisms across 10 ontology categories.\n",
                "  - **Pass 2 (Skeptic Review & Pruning)**: Audits hypotheses against negative-control plausibility to eliminate generic academic buzzwords and retain sharp causal mechanisms.\n",
                "\n",
                "Evaluates all 10 benchmark pairs:\n",
                "- **5 Historical Positive Benchmark Cases** (saved to `results/exp2_qwen3.6-35b-a3b_agent_true/`)\n",
                "- **5 Cross-Disciplinary Negative Controls** (saved to `results/exp2_qwen3.6-35b-a3b_agent_true_negative/`)\n",
                "\n",
                "Features:\n",
                "- **Runner safety**: Automatically reuses active in-memory `model_runner` from Cell 1 without reloading.\n",
                "- **Resume check**: Automatically skips pairs whose prediction file already exists.\n",
                "- **Auto-zip download**: Packages all predictions into `qwen3.6_agent_exp2_true_results.zip` and triggers browser download."
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
                "## Code Cell 3: Head-to-Head Benchmark Evaluation (Agent vs. Pure LLM)\n",
                "This cell runs automated SBERT evaluation and prints side-by-side comparative statistics:\n",
                "- **Table 1**: Field-level Profile Extraction (Pure LLM F1 vs. Agent F1 and $\\Delta$ F1).\n",
                "- **Table 2**: Historical Co-authorship Backtesting (Avg Sim, Negative Sim, Selectivity Margin $\\Delta$).\n",
                "- **Table 3**: Reasoning Mode Alignment (% at $\\tau = 0.45$).\n",
                "\n",
                "Outputs formatted Markdown comparison and ready-to-paste LaTeX tables for `experiments.tex`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# STEP 3: COMPARATIVE EVALUATION (AGENT ITERATIVE REFINEMENT vs. PURE LLM)\n",
                "# ==============================================================================\n",
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "# Ensure project modules are available\n",
                "sys.path.insert(0, str(Path.cwd()))\n",
                "\n",
                "# Run evaluation script\n",
                "!python colab_setup/Qwen3.6-35B-A3B-Agent/eval_agent_stats.py\n"
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

print(f"[+] Successfully built agent notebook: {notebook_path}")
