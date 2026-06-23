# CatalystOU

CatalystOU is an experiment framework for evaluating an LLM-powered researcher collaboration discovery pipeline. The system extracts structured researcher profiles from publication PDFs, predicts mechanistic collaboration opportunities between researchers, and evaluates those outputs against manually labeled ground truth for paper publication.

The current research code lives mainly in `proj_test/` and `utils/`. The `live_demo/` directory contains a separate web demo and is not part of the publication experiment workflow.

## Project Layout

```text
catalystOU/
├── proj_test/
│   ├── profile_extractor.py      # PDF -> structured researcher profile extraction
│   ├── test1_profile_acc.py      # Experiment 1: profile extraction accuracy
│   ├── test2_pred_acc.py         # Experiment 2: collaboration prediction accuracy
│   ├── orchestrator.py           # Multi-pair collaboration evaluation
│   ├── llm_reasoner.py           # Structured LLM collaboration reasoning
│   └── run_experiments.py        # CLI entry point for experiments
├── utils/
│   ├── data/                     # Shared schemas and profile discovery/loading
│   ├── llm/                      # LLM provider abstraction and OpenAI-compatible client
│   ├── matching.py               # Exact + semantic matching utilities
│   ├── metrics.py                # Aggregate metrics and bootstrap confidence intervals
│   ├── profile_io.py             # JSON load/save helpers
│   └── logger.py                 # Shared logging setup
├── profile_labeled_data/         # Manually labeled researcher profiles
├── extracted_profile_json/       # LLM-extracted researcher profiles
├── live_demo/                    # Separate web demo
└── requirements.txt
```

## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure LLM credentials in your shell or in a `.env` file:

```bash
OPENAI_API_KEY="your_api_key"
LLM_API_URL="https://api.openai.com/v1"
```

`LLM_API_KEY` is also supported for OpenAI-compatible local or proxy endpoints.

## Data Format

Researcher profiles use the following canonical fields:

```json
{
  "Researcher Profile:": "Dr. Example Researcher",
  "Affiliation:": "Example University",
  "Research Domains": [],
  "Techniques Used": [],
  "Data & Platforms": [],
  "Application Areas": [],
  "Key Research Thinking Patterns": [],
  "Summary Description": ""
}
```

Collaboration predictions and ground truth use these categories:

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

## Experiment 1: Profile Extraction Accuracy

Experiment 1 evaluates LLM-extracted researcher profiles against manually labeled ground-truth profiles.

Expected layout:

```text
extracted_profile_json/
└── gpt-5-nano/
    └── CS/
        └── Researcher_Name_profile.json

profile_labeled_data/
└── CS/
    └── Researcher Name/
        └── ResearcherName_Profile.json
```

Run:

```bash
python -m proj_test.run_experiments exp1 \
  -e extracted_profile_json \
  -g profile_labeled_data \
  -l gpt-5-nano \
  -m all-mpnet-base-v2 \
  -o results/exp1
```

Outputs:

```text
results/exp1/
├── experiment_1_results.json
└── final_report_profile_acc_test.txt
```

Main metrics:

- `PSAS`: Profile Semantic Alignment Score
- Per-field micro/macro F1
- Per-field semantic similarity
- Summary cosine similarity
- Bootstrap confidence interval for PSAS

## Experiment 2: Collaboration Prediction Accuracy

Experiment 2 evaluates LLM-predicted collaboration mechanisms against historical/manual collaboration ground truth.

Create a cases file:

```json
[
  {
    "id": "case_001",
    "a_path": "path/to/researcher_a_pre_profile.json",
    "b_path": "path/to/researcher_b_pre_profile.json",
    "gt_path": "path/to/collaboration_groundtruth.json",
    "out_path": "results/exp2/case_001_prediction.json"
  }
]
```

You can also generate a cases file from the same compact list:

```bash
python -m proj_test.run_experiments build-cases \
  -p pair_specs.json \
  -o cases.json \
  --predictions_dir results/exp2/predictions
```

Run:

```bash
python -m proj_test.run_experiments exp2 \
  -c cases.json \
  -m all-mpnet-base-v2 \
  --llm_provider openai \
  --llm_model gpt-4 \
  --tau 0.65 \
  -o results/exp2
```

Outputs:

```text
results/exp2/
├── experiment_2_results.json
└── case_001_prediction.json
```

Main metrics:

- `MAS`: Mechanistic Alignment Score
- Per-category micro/macro F1
- Per-category semantic similarity
- Summary collaboration cosine similarity

## Profile Extraction From PDFs

To generate extracted researcher profiles from publication PDFs:

```bash
python -m proj_test.profile_extractor \
  -p path/to/pdf_directory \
  -o extracted_profile_json \
  -m gpt-5-nano
```

Expected PDF input layout:

```text
pdf_directory/
└── Department/
    └── Researcher Name/
        ├── paper_1.pdf
        └── paper_2.pdf
```

Generated profiles are saved under:

```text
extracted_profile_json/<model_name>/<department>/<Researcher_Name>_profile.json
```

## LLM Provider Layer

LLM calls go through `utils/llm/`, which provides:

- `BaseLLMProvider`: abstract provider interface
- `OpenAIProvider`: OpenAI/OpenAI-compatible implementation
- `create_llm_provider`: provider factory
- `register_llm_provider`: extension point for new providers

This keeps experiment code independent from the LLM backend.

## GraphRAG Integration Path

GraphRAG can be integrated without rewriting the experiment evaluators. The recommended path is:

1. Add a GraphRAG-backed adapter behind the provider/reasoner boundary.
2. Convert researcher profiles into graph entities and relations.
3. Use graph retrieval to augment or replace the prompt context in `proj_test/llm_reasoner.py`.
4. Keep Experiment 1 and Experiment 2 evaluation contracts unchanged.

The likely hard work is graph construction, entity normalization, and query design; the current experiment plumbing is now modular enough to support this.

## Development Checks

Run syntax checks:

```bash
python -m compileall proj_test utils
```

Check patch cleanliness:

```bash
git diff --check -- proj_test utils
```

## Notes

- `live_demo/` is a separate demo interface and should not be modified for paper experiment work.
- Generated logs, result folders, temporary extracted profiles, and `__pycache__/` directories should not be committed.
- The experiment code assumes dependencies from `requirements.txt` are installed before running full evaluations.
