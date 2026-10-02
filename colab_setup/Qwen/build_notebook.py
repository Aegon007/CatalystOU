import json
from pathlib import Path

colab_dir = Path(__file__).resolve().parent
cell1_path = colab_dir / "profile_extraction_cell1.py"
notebook_path = colab_dir / "catalyst_ou_pipeline.ipynb"

with cell1_path.open("r", encoding="utf-8") as f:
    cell1_lines = f.readlines()

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CatalystOU: Google Colab Pipeline (Qwen 27B)\n",
                "### Hardware: NVIDIA A100 (80GB VRAM) High-RAM Runtime\n",
                "### Model: `Qwen/Qwen3.8-27B` (native bfloat16)\n",
                "\n",
                "This notebook is structured into sequential cells:\n",
                "- **Step 0: Environment Setup & Package Installation**: Installs PyTorch (CUDA 12.4 ABI), Hugging Face, PDF extraction, and embedding libraries.\n",
                "- **Code Cell 1 (Profile Extraction)**: Loads Qwen in native bfloat16 onto the A100 GPU (~54GB VRAM), discovers researcher publications (PDFs), and runs the Map-Reduce extraction pipeline to produce structured profiles.\n",
                "- **Code Cell 2 (Synergy Checker)**: Evaluates pairwise collaboration mechanisms and synergies across researcher profiles (reserved for Part 2).\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 0: Environment Setup & Package Installation\n",
                "Run this cell first to install all required dependencies and verify your GPU allocation.\n",
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
                "# 2. Install Hugging Face, PDF extraction, and embedding libraries\n",
                "!pip install -q -U \"transformers>=4.48.0\" \"huggingface_hub>=0.28.0\" accelerate sentencepiece pymupdf pypdf sentence-transformers tqdm\n",
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
                "# %cd CatalystOU\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Code Cell 1: Researcher Profile Extraction Pipeline (Qwen 27B)\n",
                "This cell handles:\n",
                "1. Loading Qwen3.8-27B in native bfloat16 onto the A100 GPU (~54GB VRAM).\n",
                "2. Discovering researcher publication PDFs in `CatalystOU-pdf/`.\n",
                "3. Multi-engine text extraction & reference cleaning.\n",
                "4. Map-Reduce entity extraction, semantic deduplication, and profile synthesis.\n",
                "5. Saving profiles to `extracted_profile_json/qwen_qwen3.8-27b/<Discipline>/<Researcher>_profile.json`.\n"
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
                "## Code Cell 2: Synergy Checker (Reserved for Part 2)\n",
                "This cell will load the extracted profile JSONs produced by Code Cell 1 and run the pairwise synergy / collaboration inference engine.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# STEP 2: SYNERGY CHECKER (Reserved for Part 2)\n",
                "# ==============================================================================\n",
                "# This cell will consume the profiles saved in Code Cell 1:\n",
                "# 'extracted_profile_json/qwen_qwen3.8-27b/<Discipline>/<Researcher>_profile.json'\n",
                "# to predict and evaluate mechanistic collaboration opportunities.\n",
                "print(\"Code Cell 2 is reserved for the synergy checker. Cell 1 profile extraction complete!\")\n"
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

print(f"[+] Successfully built notebook: {notebook_path}")
