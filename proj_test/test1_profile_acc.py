# evaluate_profiles.py
import os
import sys
import argparse
import pdb

import json
import numpy as np
from typing import List, Dict
from pathlib import Path
from loguru import logger

from sentence_transformers import SentenceTransformer

import utils.profile_io as profile_io
import utils.matching as matching
import utils.metrics as metrics


# 配置日志：同时输出到屏幕和文件
LOG_FILE = "test1_profile_acc.log"
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILE, rotation="10 MB", level="DEBUG") # 文件记录更详细的 DEBUG 信息


# Define the schema: Which fields are lists, which is the summary?
LIST_FIELDS, SUMMARY_FIELD = matching.get_matching_fields()


def evaluate_single_profile(llm_profile_path: str, gold_profile_path: str, model: SentenceTransformer, tau: float = 0.65) -> Dict:
    """
    Evaluates a single researcher's LLM-extracted profile against Ground Truth.
    """
    pred = profile_io.load_profile_json(llm_profile_path)
    gold = profile_io.load_profile_json(gold_profile_path)
    
    field_metrics = {}
    avg_sims = []

    # 1. Evaluate List Fields (Hybrid Matching)
    for field in LIST_FIELDS:
        # Safely get lists, default to empty
        pred_list = matching.normalize_profile_field(field, pred.get(field))
        gold_list = matching.normalize_profile_field(field, gold.get(field))
        
        # Run matching
        try:
            res = matching.match_profile_list_field(pred_list, gold_list, model, tau)
        except Exception as e:
            logger.error(f"matching failed due to reason {e}.")
            logger.error(f"p_list is: {p_list}")
            logger.error(f"g_list is: {g_list}")
        
        field_metrics[field] = res
        avg_sims.append(res["avg_similarity"])

    # 2. Evaluate Summary (Cosine)
    p_sum = pred.get(SUMMARY_FIELD, "")
    g_sum = gold.get(SUMMARY_FIELD, "")
    sum_res = matching.compute_summary_similarity(p_sum, g_sum, model)

    # 3. Compute Profile Semantic Alignment Score (PSAS)
    psas = float(np.mean(avg_sims)) if avg_sims else 0.0

    return {
        "field_metrics": field_metrics,
        "profile_scores": {
            "PSAS": psas,
            "summary_cosine": sum_res["cosine_sim"]
        }
    }


def run_experiment_1(data_map: List[Dict[str, str]], model_name="all-mpnet-base-v2"):
    """
    Driver for Experiment 1.
    data_map: List of dicts [{"llm_path": "...", "gold_path": "..."}]
    """

    print(f"Loading SBERT model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    all_results = []
    print(f"Evaluating {len(data_map)} profiles...")
    
    for entry in data_map:
        res = evaluate_single_profile(entry["llm_path"], entry["gold_path"], model)
        all_results.append(res)
    pdb.set_trace()

    # Aggregate
    print("Aggregating metrics...")
    final_report = metrics.compute_profile_aggregate_metrics(all_results)

    return final_report


def print_formatted_report(report: Dict, outDir: str) -> None:
    """
    Print and save a formatted evaluation report for profile semantic alignment.
    """

    lines = []
    lines.append("=== Profile Semantic Alignment Evaluation ===\n")

    # ---- Profile-level metrics ----
    profile = report.get("profile_level", {})
    lines.append(f"Mean PSAS: {profile.get('PSAS_mean', 0.0):.4f}\n")
    lines.append(f"PSAS Std: {profile.get('PSAS_std', 0.0):.4f}\n")
    lines.append(f"Mean Summary Cosine Similarity: {profile.get('summary_cosine_mean', 0.0):.4f}\n")

    # Optional bootstrap CI
    if "bootstrap" in report and "PSAS_95CI" in report["bootstrap"]:
        ci_low, ci_high = report["bootstrap"]["PSAS_95CI"]
        lines.append(f"PSAS 95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n")

    # ---- Per-field metrics ----
    lines.append("\n--- Per-Field Semantic Alignment ---\n")
    for field, stats in report.get("per_key", {}).items():
        mean_sim = stats.get("avg_similarity_mean", 0.0)
        std_sim = stats.get("avg_similarity_std", 0.0)
        lines.append(
            f"{field}: mean={mean_sim:.4f}, std={std_sim:.4f}\n"
        )

    # Output to console
    output = "\n".join(lines)
    print(output)

    # Save to file
    output_file = os.path.join(outDir, "final_report_profile_acc_test.txt")
    with open(output_file, "w") as f:
        f.write(output)


def build_data_map(extracted_profiles_dir: str, gold_profiles_dir: str, model_name: str) -> List[Dict[str, str]]:
    """
    Build a mapping of LLM-extracted profiles to ground truth profiles.
    """
    data_map = []

    base_dir = Path(gold_profiles_dir)
    if not base_dir.exists() or not base_dir.is_dir():
        logger.error(f"The specified PDF directory does not exist or is not a directory: {base_dir}")
        return
    
    departments = [d for d in base_dir.iterdir() if d.is_dir()]

    author_dirs = []
    for department in departments:
        tmp_author_dirs = [d for d in department.iterdir() if d.is_dir()]
        author_dirs.extend(tmp_author_dirs)

    gold_files = []
    for author_dir in author_dirs:
        tmp_gold_files = [f for f in author_dir.iterdir() if f.is_file() and f.suffix == ".json"]
        gold_files.extend(tmp_gold_files)   # Collect all gold profile files

    for gold_file in gold_files:
        # Construct corresponding LLM profile path
        department_name, author_name = gold_file.parent.parent.name, gold_file.parent.name
        llm_profile_path = Path(os.path.join(extracted_profiles_dir, model_name, department_name, f"{author_name.replace(' ', '_')}_profile.json"))

        if not llm_profile_path.exists():
            logger.error(f"LLM profile file not found: {llm_profile_path}")
            continue

        data_map.append({
            "llm_path": str(llm_profile_path),
            "gold_path": str(gold_file)
        })

    return data_map


def main(opts):
    """Main function to run the evaluation."""
    data_map = build_data_map(opts.extracted_profiles_dir, opts.gold_profiles_dir, opts.llm_model_name)
    
    final_report = run_experiment_1(data_map, opts.model_name) # Run Experiment 1
    
    print_formatted_report(final_report, opts.output_path)
    # Save final report to file
    with open(opts.output_path, 'w') as f:
        json.dump(final_report, f, indent=4)

    print("Experiment 1 completed.")


def parseOpts(args):
    parser = argparse.ArgumentParser(description="Evaluate LLM profiles against ground truth.")
    parser.add_argument("-e", "--extracted_profiles_dir", type=str, required=True, help="Directory containing json files of LLM-extracted profiles.")
    parser.add_argument("-g", "--gold_profiles_dir", type=str, required=True, help="Directory containing json files of ground truth profiles.")
    parser.add_argument("-m", "--model_name", type=str, default="all-mpnet-base-v2", help="SBERT model name.")
    parser.add_argument("-l", "--llm_model_name", type=str, default="gpt-5-nano", help="LLM model name: gpt-5/gpt-5-mini/gpt-5-nano.")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output file path for the final report.")
    opts = parser.parse_args(args)
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv[1:])
    main(opts)