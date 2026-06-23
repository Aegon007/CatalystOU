"""
Experiment 2: Collaboration Prediction Accuracy Evaluation

Backtests LLM predictions of collaboration mechanisms against historical ground truth
collaboration profiles using semantic matching and hybrid alignment scoring.

Key metrics:
- MAS (Mechanistic Alignment Score): Average semantic similarity of predicted collaboration
- Per-category F1 scores: Field-level precision/recall for collaboration categories
- Summary cosine similarity: Semantic alignment of collaboration description summaries
"""

import sys
import argparse
import json
import asyncio
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from utils.logger import setup_logger
import utils.profile_io as profile_io
from utils.data import COLLABORATION_CATEGORIES, COLLABORATION_SUMMARY_FIELD
import utils.matching as matching
import utils.metrics as metrics
from proj_test.llm_reasoner import call_llm_reasoner

logger = setup_logger(__name__, log_file="test2_pred_acc.log")


def build_collaboration_cases(
    pairs_file: str,
    output_file: str,
    predictions_dir: str | None = None,
) -> List[Dict[str, str]]:
    """Build an Experiment 2 cases file from a compact pair specification.

    Input format:
    [
      {
        "id": "case_001",
        "a_path": "path/to/researcher_a_profile.json",
        "b_path": "path/to/researcher_b_profile.json",
        "gt_path": "path/to/collaboration_groundtruth.json"
      }
    ]
    """
    with open(pairs_file, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    if not isinstance(raw_cases, list):
        raise ValueError("Pairs file must contain a JSON list of case objects")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_root = Path(predictions_dir) if predictions_dir else output_path.parent / "predictions"

    cases: List[Dict[str, str]] = []
    required = {"a_path", "b_path", "gt_path"}
    for idx, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"Case at index {idx} must be a JSON object")

        missing = required - set(item)
        if missing:
            raise ValueError(f"Case at index {idx} is missing required keys: {sorted(missing)}")

        case_id = str(item.get("id") or f"case_{idx:03d}")
        case = {
            "id": case_id,
            "a_path": str(item["a_path"]),
            "b_path": str(item["b_path"]),
            "gt_path": str(item["gt_path"]),
            "out_path": str(item.get("out_path") or pred_root / f"{case_id}_prediction.json"),
        }
        cases.append(case)

    profile_io.save_json(cases, output_path)
    return cases


def build_cases_cli(args) -> bool:
    """Build an Experiment 2 cases JSON file."""
    logger.info("Building Experiment 2 cases file")
    try:
        cases = build_collaboration_cases(
            args.pairs_file,
            args.output_file,
            predictions_dir=getattr(args, "predictions_dir", None),
        )
        logger.info(f"Wrote {len(cases)} cases to {args.output_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to build cases file: {e}", exc_info=True)
        return False

# Default empty match result for error cases
_EMPTY_MATCH_RESULT = {
    "tp_total": 0,
    "tp_exact": 0,
    "tp_semantic": 0,
    "fp": 0,
    "fn": 0,
    "avg_similarity": 0.0,
    "matches": []
}

async def evaluate_single_collaboration(
    profile_a_path: str,
    profile_b_path: str,
    groundtruth_path: str,
    llm_config: Dict,
    model: SentenceTransformer,
    tau: float = 0.65
) -> Dict[str, Any]:
    """
    Evaluate a single historical collaboration case.

    Compares LLM-predicted collaboration mechanisms against ground truth by:
    1. Loading pre-collaboration profiles for researchers A and B
    2. Generating LLM prediction of collaboration
    3. Comparing predicted vs. ground truth using hybrid semantic matching
    4. Computing Mechanistic Alignment Score (MAS) across all categories

    Args:
        profile_a_path: Path to researcher A's profile JSON
        profile_b_path: Path to researcher B's profile JSON
        groundtruth_path: Path to ground truth collaboration profile JSON
        llm_config: LLM configuration dict
        model: SentenceTransformer model for semantic matching
        tau: Similarity threshold for semantic matching (0.65 default)

    Returns:
        Dict with predicted collaboration, field metrics, and MAS score
    """
    # Load profiles
    profile_a = profile_io.load_profile_json(profile_a_path)
    profile_b = profile_io.load_profile_json(profile_b_path)
    groundtruth_collab = profile_io.load_profile_json(groundtruth_path)

    if not all([profile_a, profile_b, groundtruth_collab]):
        raise ValueError("Failed to load one or more profile files")

    # Generate LLM prediction of collaboration
    logger.debug("Generating LLM collaboration prediction")
    predicted_collab = await call_llm_reasoner(profile_a, profile_b, llm_config)

    # Evaluate semantic alignment across collaboration categories
    field_results = {}
    similarities = []

    for category in COLLABORATION_CATEGORIES:
        pred_items = predicted_collab.get(category, [])
        truth_items = groundtruth_collab.get(category, [])

        try:
            match_result = matching.match_profile_list_field(
                pred_items, truth_items, model, tau
            )
        except Exception as e:
            logger.warning(
                f"Matching error for category '{category}' in {profile_a_path} vs {profile_b_path}: {e}"
            )
            match_result = _EMPTY_MATCH_RESULT.copy()

        field_results[category] = match_result
        similarities.append(match_result["avg_similarity"])

    # Evaluate summary using cosine similarity
    pred_summary = predicted_collab.get(COLLABORATION_SUMMARY_FIELD, "")
    truth_summary = groundtruth_collab.get(COLLABORATION_SUMMARY_FIELD, "")
    summary_result = matching.compute_summary_similarity(
        pred_summary, truth_summary, model
    )

    # Compute Mechanistic Alignment Score (MAS)
    mas = float(np.mean(similarities)) if similarities else 0.0

    return {
        "prediction": predicted_collab,
        "field_metrics": field_results,
        "profile_scores": {
            "MAS": mas,
            "PSAS": mas,  # Kept for compatibility with metrics aggregation
            "summary_cosine": summary_result["cosine_sim"]
        }
    }
    


async def run_experiment_2_async(
    case_list: List[Dict[str, str]],
    llm_config: Dict,
    model_name: str = "all-mpnet-base-v2",
    tau: float = 0.65
) -> Dict[str, Any]:
    """
    Run Experiment 2 asynchronously: collaboration prediction backtesting.

    Processes multiple historical collaboration cases in sequence,
    predicting mechanisms and comparing against ground truth, then
    aggregates metrics across all cases.

    Args:
        case_list: List of case dicts with keys:
            - "id": case identifier
            - "a_path": researcher A profile path
            - "b_path": researcher B profile path
            - "gt_path": ground truth collaboration profile path
            - "out_path" (optional): where to save prediction
        llm_config: LLM configuration dict
        model_name: Sentence-BERT model name for semantic matching
        tau: Similarity threshold for semantic matching

    Returns:
        Aggregated evaluation report with MAS and per-category metrics
    """
    logger.info(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Backtesting {len(case_list)} historical collaborations")
    evaluation_results = []

    for idx, case in enumerate(case_list, 1):
        case_id = case.get("id", f"case_{idx}")
        logger.info(f"[{idx}/{len(case_list)}] Evaluating: {case_id}")

        try:
            result = await evaluate_single_collaboration(
                case["a_path"],
                case["b_path"],
                case["gt_path"],
                llm_config,
                model,
                tau=tau
            )
            evaluation_results.append(result)

            # Save prediction if output path provided
            if "out_path" in case:
                profile_io.save_json(result["prediction"], case["out_path"])
                logger.debug(f"Saved prediction to {case['out_path']}")

        except Exception as e:
            logger.error(f"Failed to evaluate case {case_id}: {e}")
            continue

    if not evaluation_results:
        logger.error("No cases were successfully evaluated")
        return {}

    # Aggregate metrics
    logger.info("Aggregating collaboration metrics")
    final_report = metrics.compute_profile_aggregate_metrics(evaluation_results)
    final_report["case_results"] = evaluation_results
    final_report["num_cases_evaluated"] = len(evaluation_results)

    # Log summary
    mas_mean = final_report.get("profile_level", {}).get("PSAS_mean", 0.0)
    logger.info(f"Mean MAS (Mechanistic Alignment): {mas_mean:.4f}")

    # Log per-category performance
    logger.info("Per-Reasoning-Category Performance:")
    for category_name, category_stats in final_report.get("per_key", {}).items():
        if category_name in COLLABORATION_CATEGORIES:
            f1_score = category_stats.get("micro_f1", 0.0)
            logger.info(f"  {category_name:40s}: F1={f1_score:.4f}")

    return final_report
def run_experiment_2(
    case_list: List[Dict[str, str]],
    llm_config: Dict,
    model_name: str = "all-mpnet-base-v2",
    tau: float = 0.65
) -> Dict[str, Any]:
    """
    Synchronous entrypoint for Experiment 2.

    Runs the async driver internally so callers can invoke this function
    directly without managing asyncio event loops.

    Args:
        case_list: List of collaboration cases
        llm_config: LLM configuration
        model_name: SBERT model name
        tau: Similarity threshold

    Returns:
        Aggregated evaluation report
    """
    return asyncio.run(
        run_experiment_2_async(case_list, llm_config, model_name=model_name, tau=tau)
    )


def main(opts) -> int:
    """
    CLI entrypoint for standalone Experiment 2 execution.

    Args:
        opts: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Load collaboration cases
        with open(opts.cases_file, "r", encoding="utf-8") as f:
            cases = json.load(f)

        logger.info(f"Loaded {len(cases)} collaboration cases from {opts.cases_file}")

        # Prepare output directory
        output_dir = Path(opts.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set output paths for predictions if not already specified
        for case in cases:
            if "out_path" not in case:
                case["out_path"] = str(
                    output_dir / f"{case.get('id', 'unknown')}_prediction.json"
                )

        # Prepare LLM configuration
        llm_config = {
            "provider": opts.llm_provider,
            "model_name": opts.llm_model,
            "temperature": 0.3,
            "max_tokens": 4096,
            "max_retries": 3,
            "timeout": 300.0,
        }

        # Run experiment
        logger.info("Starting collaboration prediction backtesting")
        final_report = run_experiment_2(
            cases,
            llm_config,
            model_name=opts.model_name,
            tau=opts.tau
        )

        if not final_report:
            logger.error("Experiment 2 produced no results")
            return 1

        # Save report
        report_file = output_dir / "experiment_2_results.json"
        profile_io.save_json(final_report, report_file)

        logger.info(f"Results saved to {report_file}")
        logger.info("Experiment 2 completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Experiment 2 failed: {e}", exc_info=True)
        return 1


def parseOpts(args) -> argparse.Namespace:
    """
    Parse command-line arguments for Experiment 2.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Backtest LLM collaboration predictions against historical ground truth"
    )
    parser.add_argument(
        "-c", "--cases_file",
        type=str,
        required=True,
        help="JSON file with collaboration cases (id, a_path, b_path, gt_path, out_path)"
    )
    parser.add_argument(
        "-m", "--model_name",
        type=str,
        default="all-mpnet-base-v2",
        help="Sentence-BERT model name for semantic matching"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-4",
        help="LLM model name for collaboration prediction"
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="openai",
        help="LLM provider (openai, anthropic, local, etc.)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.65,
        help="Similarity threshold for semantic matching"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Output directory for results and predictions"
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    opts = parseOpts(sys.argv[1:])
    sys.exit(main(opts))
