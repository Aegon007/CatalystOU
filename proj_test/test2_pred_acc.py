# evaluate_collaboration.py
import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Re-use modules from Experiment 1
from profile_io import load_profile_json, save_json
from matching import match_profile_list_field, compute_summary_similarity
from metrics import compute_profile_aggregate_metrics
from llm_reasoner import call_llm_reasoner, COLLAB_CATEGORIES


def evaluate_single_collaboration(
    a_pre_path: str,
    b_pre_path: str,
    groundtruth_path: str,
    llm_config: Dict,
    model: SentenceTransformer,
    tau: float = 0.65
) -> Dict[str, Any]:
    """
    Evaluates ONE historical collaboration case.
    
    1. Loads Pre-Collab Profiles A & B.
    2. Loads Manual Ground Truth Collaboration Profile.
    3. LLM predicts collaboration mechanisms.
    4. Compares Prediction vs. Ground Truth using Semantic Matching.
    """
    
    # 1. Load Inputs
    prof_a = load_profile_json(a_pre_path)
    prof_b = load_profile_json(b_pre_path)
    gt_collab = load_profile_json(groundtruth_path)
    
    # 2. Generate Prediction (The Reasoning Step)
    pred_collab = call_llm_reasoner(prof_a, prof_b, llm_config)
    
    # 3. Evaluate Semantic Alignment
    field_metrics = {}
    avg_sims = []
    
    # Iterate over the 10 reasoning categories
    for field in COLLAB_CATEGORIES:
        p_list = pred_collab.get(field, [])
        g_list = gt_collab.get(field, [])
        
        # Reuse Experiment 1's robust matching logic
        res = match_profile_list_field(p_list, g_list, model, tau)
        
        field_metrics[field] = res
        avg_sims.append(res["avg_similarity"])
        
    # 4. Evaluate Summary Description
    p_sum = pred_collab.get("Summary Collaboration Themes", "")
    g_sum = gt_collab.get("Summary Collaboration Themes", "") # Ensure GT has this key
    sum_res = compute_summary_similarity(p_sum, g_sum, model)
    
    # 5. Compute MAS (Mechanistic Alignment Score)
    # This is effectively the PSAS for collaboration
    mas = float(np.mean(avg_sims)) if avg_sims else 0.0
    
    return {
        "prediction": pred_collab, # Save prediction for qualitative analysis
        "field_metrics": field_metrics,
        "profile_scores": {
            "MAS": mas, # Naming it MAS for Exp 2 clarity
            "PSAS": mas, # Kept for compatibility with metrics.py
            "summary_cosine": sum_res["cosine_sim"]
        }
    }

def run_experiment_2(case_list: List[Dict[str, str]], llm_config: Dict, model_name="all-mpnet-base-v2"):
    """
    Driver for Experiment 2.
    case_list: List of dicts [{"a_path":..., "b_path":..., "gt_path":...}]
    """
    print(f"Loading SBERT model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    all_results = []
    print(f"Backtesting {len(case_list)} historical collaborations...")
    
    for case in case_list:
        print(f"Processing Case: {case.get('id', 'unknown')}")
        res = evaluate_single_collaboration(
            case["a_path"], 
            case["b_path"], 
            case["gt_path"], 
            llm_config, 
            model
        )
        all_results.append(res)
        
        # Optional: Save individual prediction for inspection
        if "out_path" in case:
            save_json(res, case["out_path"])
            
    # Aggregate using the shared metrics module
    # Note: metrics.py calculates per-key F1 and mean PSAS/MAS
    print("Aggregating collaboration metrics...")
    final_report = compute_profile_aggregate_metrics(all_results)
    
    print("\n=== Experiment 2 Results (Collaboration Backtesting) ===")
    print(f"Mean MAS (Mechanistic Alignment): {final_report['profile_level']['PSAS_mean']:.4f}")
    
    # Print per-category breakdown (crucial for "Method-Application" vs "Shared Domain" analysis)
    print("\n--- Per-Reasoning-Category Micro F1 ---")
    for k, v in final_report["per_key"].items():
        if k in COLLAB_CATEGORIES:
            print(f"{k}: {v['micro_f1']:.4f}")
            
    return final_report


# Example Run Block
if __name__ == "__main__":
    # Define a dummy case based on your file structure
    cases = [
        {
            "id": "Lan_Example_Collab",
            "a_path": "data/pre_collab_profiles/ChaoLan_before.json",
            "b_path": "data/pre_collab_profiles/Collaborator_before.json",
            "gt_path": "data/groundtruth_collab/Lan_Collaborator_GT.json",
            "out_path": "results/exp2_lan_collab_eval.json"
        }
    ]
    
    config = {"model": "gpt-4o", "temperature": 0.0}
    
    # Note: This requires the files to actually exist. 
    # run_experiment_2(cases, config)
