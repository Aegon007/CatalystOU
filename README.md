# CatalystOU

CatalystOU is a small research codebase for evaluating an LLM-powered pipeline that extracts researcher profiles from publication PDFs, predicts collaboration opportunities between researchers, and evaluates those outputs against manually labeled ground truth.

The project is intentionally simple and research-focused. The main entry point is the top-level CLI script [run_experiments.py](run_experiments.py). Shared helpers live in [utils](utils), while experiment-specific logic lives in [proj_test](proj_test).

## Project layout

```text
catalystOU/
├── run_experiments.py            # Main CLI entry point for experiments
├── proj_test/
│   ├── profile_extractor.py      # PDF -> structured researcher profile extraction
│   ├── llm_reasoner.py           # Collaboration reasoning workflow
│   ├── test1_profile_acc.py      # Experiment 1: profile extraction accuracy
│   ├── test2_pred_acc.py         # Experiment 2: collaboration prediction accuracy
│   └── extract_all_profiles.py   # Simple wrapper around batch extraction
├── utils/
│   ├── data.py                   # Shared schema and profile helpers
│   ├── data_utils.py             # Profile loading and schema definitions
│   ├── llm_utils.py              # Simple shared LLM helper layer
│   ├── logger.py                 # Shared logging setup
│   ├── matching.py               # Exact + semantic matching utilities
│   ├── metrics.py                # Aggregate metrics and bootstrap summaries
│   ├── profile_io.py             # JSON load/save helpers
│   └── profile_matcher.py        # Compatibility wrapper for older code paths
├── profile_labeled_data/         # Manually labeled researcher profiles
├── extracted_profile_json/       # LLM-extracted researcher profiles
├── live_demo/                    # Separate web demo
└── requirements.txt
```

## Environment setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your LLM credentials in your shell or in a `.env` file:

```bash
OPENAI_API_KEY="your_api_key"
LLM_API_URL="https://api.openai.com/v1"
```

`LLM_API_KEY` is also supported for OpenAI-compatible local or proxy endpoints.

## Data format

Researcher profiles use a simple JSON structure with the following core fields:

```json
{
  "Research Domains": [],
  "Techniques Used": [],
  "Data & Platforms": [],
  "Application Areas": [],
  "Key Research Thinking Patterns": [],
  "Summary Description": ""
}
```

Collaboration predictions and ground truth use the following categories:

```json
{
  "Shared Domains": [],
  "Method-Application Synergies": [],
  "Complementary Technique Synergies": [],
  "Data-Method Synergies": [],
  "Cross-Domain Fusion Topics": [],
  "Shared Application Areas": [],
  "Joint Technique Development": [],
  "Theory-Application Synergy": [],
  "Thinking Pattern Synergies": [],
  "Future Research Directions": [],
  "Summary Collaboration Themes": ""
}
```

## Experiment 1: profile extraction accuracy

Experiment 1 evaluates LLM-extracted researcher profiles against manually labeled ground-truth profiles.

Run it with:

```bash
python run_experiments.py exp1 \
  -e extracted_profile_json \
  -g profile_labeled_data \
  -l gpt-5-nano \
  -m all-mpnet-base-v2 \
  -o results/exp1
```

Expected outputs:

```text
results/exp1/
├── experiment_1_results.json
└── experiment_1_results_report.txt
```

Main metrics include:

- PSAS: profile semantic alignment score
- per-field micro/macro F1
- per-field semantic similarity
- summary cosine similarity
- optional bootstrap confidence intervals

## Experiment 2: collaboration prediction accuracy

Experiment 2 evaluates LLM-predicted collaboration mechanisms against historical or manually curated collaboration ground truth.

Create a cases file such as:

```json
[
  {
    "id": "case_001",
    "a_path": "path/to/researcher_a_profile.json",
    "b_path": "path/to/researcher_b_profile.json",
    "gt_path": "path/to/collaboration_groundtruth.json",
    "out_path": "results/exp2/case_001_prediction.json"
  }
]
```

Then run:

```bash
python run_experiments.py exp2 \
  -c cases.json \
  -m all-mpnet-base-v2 \
  --llm_model gpt-4 \
  --tau 0.65 \
  -o results/exp2
```

Expected outputs:

```text
results/exp2/
├── experiment_2_results.json
└── case_001_prediction.json
```

## Profile extraction from PDFs

To generate researcher profiles from publication PDFs:

```bash
python run_experiments.py extract-profiles \
  -i path/to/pdf_directory \
  -o extracted_profile_json \
  -m gpt-5-nano
```

Expected input layout:

```text
pdf_directory/
└── Department/
    └── Researcher Name/
        ├── paper_1.pdf
        └── paper_2.pdf
```

## Notes

- The top-level [run_experiments.py](run_experiments.py) script is the recommended entry point for running the experiments.
- [live_demo](live_demo) is a separate demo interface and is not part of the main experiment workflow.
- Generated logs, result folders, temporary extracted profiles, and Python cache directories should not be committed.
- The codebase is intentionally simple and research-oriented; please avoid adding unnecessary abstraction layers unless a new experiment genuinely needs them.
