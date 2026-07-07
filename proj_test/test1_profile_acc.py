"""
Experiment 1: Profile Extraction Accuracy Evaluation

Evaluates LLM-extracted researcher profiles against manually labeled ground truth
profiles using hybrid semantic and exact matching.

Key metrics:
- PSAS (Profile Semantic Alignment Score): Average semantic similarity across fields
- Summary cosine similarity: Semantic alignment of profile summaries
- Per-field F1 scores with bootstrap confidence intervals
"""

import sys
import argparse
import json
import numpy as np
from typing import List, Dict
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from utils.logger_utils import setup_logger
from utils.data_utils import ProfileLoader
import utils.profile_io as profile_io
import utils.matching as matching
import utils.metrics as metrics

logger = setup_logger(__name__, log_file="test1_profile_acc.log")

# Get the profile schema from matching utilities
LIST_FIELDS, SUMMARY_FIELD = matching.get_matching_fields()

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


def evaluate_single_profile(
    llm_profile_path: str,
    gold_profile_path: str,
    model: SentenceTransformer,
    tau: float = 0.65
) -> Dict:
    """
    Evaluate a single LLM-extracted profile against ground truth.

    Performs hybrid matching (exact + semantic) on all list fields and
    compares summary descriptions using cosine similarity.

    Args:
        llm_profile_path: Path to LLM-extracted profile JSON
        gold_profile_path: Path to ground truth profile JSON
        model: SentenceTransformer model for semantic matching
        tau: Similarity threshold for semantic matching (0.65 default)

    Returns:
        Dict with field_metrics, PSAS, and summary_cosine score
    """
    llm_profile = profile_io.load_profile_json(llm_profile_path)
    gold_profile = profile_io.load_profile_json(gold_profile_path)

    field_results = {}
    similarities = []

    # Evaluate each list field with hybrid matching
    for field in LIST_FIELDS:
        llm_items = matching.normalize_profile_field(field, llm_profile.get(field))
        gold_items = matching.normalize_profile_field(field, gold_profile.get(field))

        try:
            match_result = matching.match_profile_list_field(
                llm_items, gold_items, model, tau
            )
        except Exception as e:
            logger.warning(
                f"Matching error for field '{field}' in {llm_profile_path}: {e}"
            )
            match_result = _EMPTY_MATCH_RESULT.copy()

        field_results[field] = match_result
        similarities.append(match_result["avg_similarity"])

    # Evaluate summary field using cosine similarity
    llm_summary = llm_profile.get(SUMMARY_FIELD, "")
    gold_summary = gold_profile.get(SUMMARY_FIELD, "")
    summary_result = matching.compute_summary_similarity(llm_summary, gold_summary, model)

    # Compute Profile Semantic Alignment Score (PSAS)
    psas = float(np.mean(similarities)) if similarities else 0.0

    return {
        "field_metrics": field_results,
        "profile_scores": {
            "PSAS": psas,
            "summary_cosine": summary_result["cosine_sim"]
        }
    }


def run_experiment_1(
    data_map: List[Dict[str, str]],
    model_name: str = "all-mpnet-base-v2"
) -> Dict:
    """
    Run Experiment 1: Profile extraction accuracy evaluation.

    Loads LLM-extracted profiles and compares them against ground truth
    profiles, aggregating semantic alignment metrics across all pairs.

    Args:
        data_map: List of dicts with "llm_path" and "gold_path" keys
        model_name: Sentence-BERT model name for semantic matching

    Returns:
        Aggregated evaluation report with PSAS and per-field metrics
    """
    logger.info(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Evaluating {len(data_map)} profiles")
    evaluation_results = []

    for entry in data_map:
        try:
            result = evaluate_single_profile(
                entry["llm_path"],
                entry["gold_path"],
                model
            )
            evaluation_results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate profile pair: {e}")
            continue

    if not evaluation_results:
        logger.error("No profiles were successfully evaluated")
        return {}

    logger.info("Aggregating metrics across all profiles")
    final_report = metrics.compute_profile_aggregate_metrics(evaluation_results)
    final_report["num_profiles_evaluated"] = len(evaluation_results)

    return final_report


def print_formatted_report(report: Dict, output_dir: str) -> None:
    """
    Format and print evaluation report to console and file.

    Outputs PSAS metrics, per-field alignment scores, and confidence intervals
    if available from bootstrap analysis.

    Args:
        report: Evaluation report dict from run_experiment_1
        output_dir: Directory to save report text file
    """
    output_lines = []
    output_lines.append("=== Profile Semantic Alignment Evaluation ===\n")

    # Profile-level summary metrics
    profile_stats = report.get("profile_level", {})
    output_lines.append(f"Mean PSAS: {profile_stats.get('PSAS_mean', 0.0):.4f}")
    output_lines.append(f"PSAS Std Dev: {profile_stats.get('PSAS_std', 0.0):.4f}")
    output_lines.append(f"Mean Summary Cosine: {profile_stats.get('summary_cosine_mean', 0.0):.4f}")

    # Bootstrap confidence interval if available
    bootstrap = report.get("bootstrap", {})
    if "PSAS_95CI" in bootstrap:
        ci_low, ci_high = bootstrap["PSAS_95CI"]
        output_lines.append(f"PSAS 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    # Per-field breakdown
    output_lines.append("\n--- Per-Field Semantic Alignment ---")
    for field_name, field_stats in report.get("per_key", {}).items():
        mean_sim = field_stats.get("avg_similarity_mean", 0.0)
        std_sim = field_stats.get("avg_similarity_std", 0.0)
        output_lines.append(f"{field_name}: mean={mean_sim:.4f}, std={std_sim:.4f}")

    # Output to console
    formatted_output = "\n".join(output_lines)
    logger.info("Report:\n" + formatted_output)
    print(formatted_output)

    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "experiment_1_results_report.txt"

    with report_file.open("w", encoding="utf-8") as f:
        f.write(formatted_output)

    logger.info(f"Report saved to {report_file}")


def build_data_map(
    extracted_dir: str,
    gold_dir: str,
    model_name: str
) -> List[Dict[str, str]]:
    """
    Build mapping of LLM-extracted profiles to ground truth profiles.

    Discovers department/author profile pairs by matching the directory
    structure of extracted profiles to ground truth labels.

    Args:
        extracted_dir: Root directory containing LLM-extracted profiles
        gold_dir: Root directory containing ground truth profile labels
        model_name: LLM model name (used to find model-specific extracted dir)

    Returns:
        List of dicts with "llm_path" and "gold_path" keys
    """
    extracted_root = Path(extracted_dir)
    model_dir = extracted_root / model_name

    # Use model-specific directory if it exists, otherwise use root
    search_dir = model_dir if model_dir.exists() else extracted_root

    logger.info(f"Searching for profiles in {search_dir}")
    data_map = ProfileLoader.discover_department_author_pairs(
        search_dir,
        Path(gold_dir)
    )

    return data_map


def main(opts) -> int:
    """
    CLI entrypoint for standalone Experiment 1 execution.

    Args:
        opts: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        data_map = build_data_map(
            opts.extracted_profiles_dir,
            opts.gold_profiles_dir,
            opts.llm_model_name
        )

        if not data_map:
            logger.error("No profile pairs found")
            return 1

        logger.info(f"Found {len(data_map)} profile pairs to evaluate")
        final_report = run_experiment_1(data_map, opts.model_name)

        # Determine output location
        output_path = Path(opts.output_path)
        if output_path.suffix.lower() == ".json":
            report_dir = output_path.parent
            json_file = output_path
        else:
            report_dir = output_path
            json_file = report_dir / "experiment_1_results.json"

        report_dir.mkdir(parents=True, exist_ok=True)

        # Save and display results
        print_formatted_report(final_report, str(report_dir))
        profile_io.save_json(final_report, json_file)

        logger.info(f"Results saved to {json_file}")
        logger.info("Experiment 1 completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Experiment 1 failed: {e}", exc_info=True)
        return 1


def parseOpts(args) -> argparse.Namespace:
    """
    Parse command-line arguments for Experiment 1.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-extracted researcher profiles against ground truth"
    )
    parser.add_argument(
        "-e", "--extracted_profiles_dir",
        type=str,
        required=True,
        help="Directory containing LLM-extracted profiles"
    )
    parser.add_argument(
        "-g", "--gold_profiles_dir",
        type=str,
        required=True,
        help="Directory containing ground truth profile labels"
    )
    parser.add_argument(
        "-m", "--model_name",
        type=str,
        default="all-mpnet-base-v2",
        help="Sentence-BERT model name for semantic matching"
    )
    parser.add_argument(
        "-l", "--llm_model_name",
        type=str,
        default="gpt-5-nano",
        help="LLM model identifier (used to locate extracted profiles)"
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        required=True,
        help="Output file path (.json) or directory for results"
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    opts = parseOpts(sys.argv[1:])
    main(opts)
