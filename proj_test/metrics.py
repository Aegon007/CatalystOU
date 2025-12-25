# metrics.py
import numpy as np
from typing import List, Dict


def compute_profile_aggregate_metrics(all_results: List[Dict], n_boot: int = 1000, alpha: float = 0.05) -> Dict:
    """
    Aggregate metrics across evaluated profiles.

    Input: list of outputs from evaluate_single_profile()
    """

    if not all_results:
        return {}

    # -------------------------
    # Profile-level PSAS
    # -------------------------
    psas_vals = np.array([
        r["profile_scores"]["PSAS"] for r in all_results
    ])

    profile_level = {
        "PSAS_mean": float(np.mean(psas_vals)),
        "PSAS_std": float(np.std(psas_vals))
    }

    # -------------------------
    # Field-level aggregation
    # -------------------------
    field_stats = {}

    for r in all_results:
        for field, fm in r["field_metrics"].items():
            if field not in field_stats:
                field_stats[field] = {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "avg_sims": []
                }

            field_stats[field]["tp"] += fm["tp_total"]
            field_stats[field]["fp"] += fm["fp"]
            field_stats[field]["fn"] += fm["fn"]

            if fm["avg_similarity"] > 0:
                field_stats[field]["avg_sims"].append(fm["avg_similarity"])

    per_key = {}
    for field, s in field_stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        per_key[field] = {
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "avg_similarity_mean": float(np.mean(s["avg_sims"])) if s["avg_sims"] else 0.0
        }

    # -------------------------
    # Summary cosine similarity
    # -------------------------
    summary_sims = [
        r["profile_scores"]["summary_cosine"]
        for r in all_results
    ]

    summary_level = {
        "summary_cosine_mean": float(np.mean(summary_sims)),
        "summary_cosine_std": float(np.std(summary_sims))
    }

    # -------------------------
    # Bootstrap CI for PSAS
    # -------------------------
    boot = []
    n = len(psas_vals)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        boot.append(np.mean(psas_vals[idx]))

    boot = np.sort(boot)
    lo = boot[int((alpha / 2) * n_boot)]
    hi = boot[int((1 - alpha / 2) * n_boot)]

    bootstrap = {
        "PSAS_95CI": (float(lo), float(hi))
    }

    return {
        "profile_level": profile_level,
        "per_key": per_key,
        "summary_level": summary_level,
        "bootstrap": bootstrap
    }
