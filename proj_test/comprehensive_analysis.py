"""
Comprehensive Semantic Evaluation and Cross-Discipline Analysis
Evaluates Meta Llama 3.1 8B extracted profiles across all 50 researchers,
broken down by discipline and field, computing PSAS, Summary Cosine, and Bootstrap CIs.
"""

import sys
import os
import json
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import utils.matching as matching
import utils.profile_io as profile_io
from proj_test.test1_profile_acc import evaluate_single_profile, build_data_map

LIST_FIELDS, SUMMARY_FIELD = matching.get_matching_fields()

def compute_f1(tp: int, fp: int, fn: int):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def aggregate_group(group_results, bootstrap_samples=1000, seed=42):
    if not group_results:
        return {}

    keys = sorted({k for res in group_results for k in res["field_metrics"].keys()})
    micro_counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in keys}
    macro_scores = {k: {"f1": [], "sim": []} for k in keys}
    psas_values = []
    summary_cosines = []

    for res in group_results:
        scores = res.get("profile_scores", {})
        psas_values.append(scores.get("PSAS", 0.0))
        summary_cosines.append(scores.get("summary_cosine", 0.0))

        for k in keys:
            if k in res["field_metrics"]:
                m = res["field_metrics"][k]
                micro_counts[k]["tp"] += m.get("tp_total", 0)
                micro_counts[k]["fp"] += m.get("fp", 0)
                micro_counts[k]["fn"] += m.get("fn", 0)

                _, _, f1 = compute_f1(m.get("tp_total", 0), m.get("fp", 0), m.get("fn", 0))
                macro_scores[k]["f1"].append(f1)
                macro_scores[k]["sim"].append(m.get("avg_similarity", 0.0))

    per_key = {}
    for k in keys:
        tp, fp, fn = micro_counts[k]["tp"], micro_counts[k]["fp"], micro_counts[k]["fn"]
        mic_p, mic_r, mic_f1 = compute_f1(tp, fp, fn)
        mac_f1 = float(np.mean(macro_scores[k]["f1"])) if macro_scores[k]["f1"] else 0.0
        avg_sim = float(np.mean(macro_scores[k]["sim"])) if macro_scores[k]["sim"] else 0.0
        sim_std = float(np.std(macro_scores[k]["sim"])) if macro_scores[k]["sim"] else 0.0

        per_key[k] = {
            "micro_precision": mic_p,
            "micro_recall": mic_r,
            "micro_f1": mic_f1,
            "macro_f1": mac_f1,
            "avg_similarity_mean": avg_sim,
            "avg_similarity_std": sim_std,
            "tp_total": tp,
            "fp_total": fp,
            "fn_total": fn
        }

    psas_arr = np.array(psas_values)
    cos_arr = np.array(summary_cosines)

    # Bootstrap 95% CI
    rng = np.random.default_rng(seed)
    psas_ci = [0.0, 0.0]
    cos_ci = [0.0, 0.0]
    if len(psas_arr) > 1 and bootstrap_samples > 0:
        psas_boot = [np.mean(rng.choice(psas_arr, size=len(psas_arr), replace=True)) for _ in range(bootstrap_samples)]
        cos_boot = [np.mean(rng.choice(cos_arr, size=len(cos_arr), replace=True)) for _ in range(bootstrap_samples)]
        psas_ci = [float(np.percentile(psas_boot, 2.5)), float(np.percentile(psas_boot, 97.5))]
        cos_ci = [float(np.percentile(cos_boot, 2.5)), float(np.percentile(cos_boot, 97.5))]

    return {
        "count": len(group_results),
        "PSAS_mean": float(np.mean(psas_arr)),
        "PSAS_std": float(np.std(psas_arr)),
        "PSAS_median": float(np.median(psas_arr)),
        "PSAS_min": float(np.min(psas_arr)),
        "PSAS_max": float(np.max(psas_arr)),
        "PSAS_95CI": psas_ci,
        "summary_cosine_mean": float(np.mean(cos_arr)),
        "summary_cosine_std": float(np.std(cos_arr)),
        "summary_cosine_median": float(np.median(cos_arr)),
        "summary_cosine_min": float(np.min(cos_arr)),
        "summary_cosine_max": float(np.max(cos_arr)),
        "summary_cosine_95CI": cos_ci,
        "per_key": per_key
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive evaluation across 50 profiles")
    parser.add_argument("--model_dir_name", default="meta-llama_meta-llama-3.1-8b-instruct", help="Folder inside extracted_profile_json")
    parser.add_argument("--model_display_name", default=None, help="Display name for model in json report")
    parser.add_argument("--output_dir", default="results/meta-llama-3.1-8b-instruct", help="Output directory")
    parser.add_argument("--sbert_model", default="all-mpnet-base-v2", help="SentenceTransformer model")
    args = parser.parse_args()

    extracted_dir = "extracted_profile_json"
    gold_dir = "profile_labeled_data"
    model_dir_name = args.model_dir_name
    model_display_name = args.model_display_name or model_dir_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_map = build_data_map(extracted_dir, gold_dir, model_dir_name)
    print(f"Total mapped profile pairs: {len(data_map)}")

    sbert_model = SentenceTransformer(args.sbert_model)

    individual_evals = []
    by_discipline = {
        "Biology": [],
        "CS": [],
        "ECE": [],
        "Mathematics": [],
        "Psychology": []
    }

    for item in data_map:
        llm_path = Path(item["llm_path"])
        gold_path = Path(item["gold_path"])
        
        # Determine discipline from path
        discipline = "Unknown"
        for disc in by_discipline.keys():
            if disc in str(llm_path):
                discipline = disc
                break

        res = evaluate_single_profile(
            str(llm_path),
            str(gold_path),
            sbert_model,
            tau=0.65,
            thinking_patterns_tau=0.50
        )
        res["researcher"] = llm_path.stem.replace("_profile", "")
        res["discipline"] = discipline
        res["llm_path"] = str(llm_path)
        res["gold_path"] = str(gold_path)

        individual_evals.append(res)
        if discipline in by_discipline:
            by_discipline[discipline].append(res)

    # Compute overall aggregate
    overall_stats = aggregate_group(individual_evals)

    # Compute per-discipline aggregates
    discipline_stats = {}
    for disc, records in by_discipline.items():
        discipline_stats[disc] = aggregate_group(records)

    # Sort individual researchers by PSAS
    sorted_by_psas = sorted(individual_evals, key=lambda x: x["profile_scores"]["PSAS"], reverse=True)
    sorted_by_summary = sorted(individual_evals, key=lambda x: x["profile_scores"]["summary_cosine"], reverse=True)

    final_payload = {
        "model": model_display_name,
        "num_profiles": len(individual_evals),
        "overall": overall_stats,
        "discipline_breakdown": discipline_stats,
        "top_5_psas": [
            {
                "researcher": r["researcher"],
                "discipline": r["discipline"],
                "PSAS": r["profile_scores"]["PSAS"],
                "summary_cosine": r["profile_scores"]["summary_cosine"]
            }
            for r in sorted_by_psas[:5]
        ],
        "bottom_5_psas": [
            {
                "researcher": r["researcher"],
                "discipline": r["discipline"],
                "PSAS": r["profile_scores"]["PSAS"],
                "summary_cosine": r["profile_scores"]["summary_cosine"]
            }
            for r in sorted_by_psas[-5:]
        ],
        "top_5_summary_cosine": [
            {
                "researcher": r["researcher"],
                "discipline": r["discipline"],
                "PSAS": r["profile_scores"]["PSAS"],
                "summary_cosine": r["profile_scores"]["summary_cosine"]
            }
            for r in sorted_by_summary[:5]
        ],
        "bottom_5_summary_cosine": [
            {
                "researcher": r["researcher"],
                "discipline": r["discipline"],
                "PSAS": r["profile_scores"]["PSAS"],
                "summary_cosine": r["profile_scores"]["summary_cosine"]
            }
            for r in sorted_by_summary[-5:]
        ],
        "all_researchers": [
            {
                "researcher": r["researcher"],
                "discipline": r["discipline"],
                "PSAS": r["profile_scores"]["PSAS"],
                "summary_cosine": r["profile_scores"]["summary_cosine"],
                "field_scores": {
                    k: {
                        "avg_similarity": v["avg_similarity"],
                        "tp": v["tp_total"],
                        "fp": v["fp"],
                        "fn": v["fn"]
                    }
                    for k, v in r["field_metrics"].items()
                }
            }
            for r in sorted_by_psas
        ]
    }

    results_file = output_dir / "comprehensive_evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n==========================================")
    print(f"Overall PSAS Mean: {overall_stats['PSAS_mean']:.4f} +/- {overall_stats['PSAS_std']:.4f} (95% CI: [{overall_stats['PSAS_95CI'][0]:.4f}, {overall_stats['PSAS_95CI'][1]:.4f}])")
    print(f"Overall Summary Cosine: {overall_stats['summary_cosine_mean']:.4f} +/- {overall_stats['summary_cosine_std']:.4f} (95% CI: [{overall_stats['summary_cosine_95CI'][0]:.4f}, {overall_stats['summary_cosine_95CI'][1]:.4f}])")
    print(f"==========================================\n")

    for disc, s in discipline_stats.items():
        print(f"[{disc}] N={s['count']} | PSAS: {s['PSAS_mean']:.4f} (std: {s['PSAS_std']:.4f}) | Summary Cosine: {s['summary_cosine_mean']:.4f} (std: {s['summary_cosine_std']:.4f})")

if __name__ == "__main__":
    main()
