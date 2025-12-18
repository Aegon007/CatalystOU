# metrics.py
import numpy as np
from typing import List, Dict, Any


def compute_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def compute_profile_aggregate_metrics(
    all_results: List[Dict[str, Any]], 
    bootstrap_samples: int = 1000
) -> Dict[str, Any]:
    """
    Aggregates results across multiple researchers.
    
    Key Metrics Produced:
    1. Per-Key Micro F1: Aggregates counts per category (e.g., 'Research Domains').
    2. Per-Key Macro F1: Average of individual researcher F1s per category.
    3. PSAS: Profile Semantic Alignment Score (mean of average similarities).
    4. Summary Similarity: Average cosine similarity of summary descriptions.
    """
    if not all_results:
        return {}

    # Identify all keys present in the results
    keys = list(all_results[0]["field_metrics"].keys())
    
    # Storage for aggregation
    micro_counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in keys}
    macro_scores = {k: {"f1": [], "sim": []} for k in keys}
    psas_values = []
    summary_cosines = []

    for res in all_results:
        # 1. PSAS (Profile Level)
        psas_values.append(res["profile_scores"]["PSAS"])
        
        # 2. Summary Description
        summary_cosines.append(res["profile_scores"]["summary_cosine"])

        # 3. Per-key stats
        for k in keys:
            if k in res["field_metrics"]:
                m = res["field_metrics"][k]
                # Micro accumulation
                micro_counts[k]["tp"] += m["tp_total"]
                micro_counts[k]["fp"] += m["fp"]
                micro_counts[k]["fn"] += m["fn"]
                
                # Macro collection
                _, _, f1 = compute_f1(m["tp_total"], m["fp"], m["fn"])
                macro_scores[k]["f1"].append(f1)
                macro_scores[k]["sim"].append(m["avg_similarity"])

    # --- Compile Final Report ---
    report = {
        "per_key": {},
        "profile_level": {},
        "bootstrap": {}
    }

    # 1. Per-Key Metrics
    for k in keys:
        # Micro
        tp, fp, fn = micro_counts[k]["tp"], micro_counts[k]["fp"], micro_counts[k]["fn"]
        mic_p, mic_r, mic_f1 = compute_f1(tp, fp, fn)
        
        # Macro
        mac_f1 = np.mean(macro_scores[k]["f1"])
        avg_sim = np.mean(macro_scores[k]["sim"])
        
        report["per_key"][k] = {
            "micro_f1": mic_f1,
            "macro_f1": mac_f1,
            "avg_semantic_similarity": avg_sim,
            "tp_total": tp,
            "fp_total": fp,
            "fn_total": fn
        }

    # 2. Profile Level Metrics (PSAS)
    report["profile_level"] = {
        "PSAS_mean": float(np.mean(psas_values)),
        "PSAS_std": float(np.std(psas_values)),
        "summary_cosine_mean": float(np.mean(summary_cosines))
    }

    # 3. Bootstrap CI (Optional, for PSAS only)
    if bootstrap_samples > 0:
        sim_means = []
        n = len(psas_values)
        data = np.array(psas_values)
        for _ in range(bootstrap_samples):
            sample = np.random.choice(data, size=n, replace=True)
            sim_means.append(np.mean(sample))
        
        report["bootstrap"]["PSAS_95CI"] = (
            float(np.percentile(sim_means, 2.5)),
            float(np.percentile(sim_means, 97.5))
        )

    return report
