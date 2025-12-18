# evaluate_profiles.py
import json
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

#from profile_io import load_profile_json
#from matching import match_profile_list_field, compute_summary_similarity
#from metrics import compute_profile_aggregate_metrics

import utils.profile_io as profile_io
import utils.matching as matching
import utils.metrics as metrics


# Define the schema: Which fields are lists, which is the summary?
LIST_FIELDS = [
    "Research Domains",
    "Techniques Used",
    "Data & Platforms",
    "Application Areas",
    "Key Research Thinking Patterns"
]
SUMMARY_FIELD = "Summary Description"


def evaluate_single_profile(
    llm_profile_path: str,
    gold_profile_path: str,
    model: SentenceTransformer,
    tau: float = 0.65
) -> Dict:
    """
    Evaluates a single researcher's LLM-extracted profile against Ground Truth.
    """
    pred = load_profile_json(llm_profile_path)
    gold = load_profile_json(gold_profile_path)
    
    field_metrics = {}
    avg_sims = []

    # 1. Evaluate List Fields (Hybrid Matching)
    for field in LIST_FIELDS:
        # Safely get lists, default to empty
        p_list = pred.get(field, [])
        g_list = gold.get(field, [])
        
        # Run matching
        res = match_profile_list_field(p_list, g_list, model, tau)
        
        field_metrics[field] = res
        avg_sims.append(res["avg_similarity"])

    # 2. Evaluate Summary (Cosine)
    p_sum = pred.get(SUMMARY_FIELD, "")
    g_sum = gold.get(SUMMARY_FIELD, "")
    sum_res = compute_summary_similarity(p_sum, g_sum, model)

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
        
    # Aggregate
    print("Aggregating metrics...")
    final_report = compute_profile_aggregate_metrics(all_results)
    
    # Print Headline Metrics
    print("\n=== Experiment 1 Results ===")
    print(f"Mean PSAS (Semantic Alignment): {final_report['profile_level']['PSAS_mean']:.4f}")
    if "bootstrap" in final_report:
        ci = final_report["bootstrap"]["PSAS_95CI"]
        print(f"PSAS 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    print("\n--- Per-Key Micro F1 ---")
    for k, v in final_report["per_key"].items():
        print(f"{k}: {v['micro_f1']:.4f}")

    return final_report
