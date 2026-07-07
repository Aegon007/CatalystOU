"""Aggregate metrics for profile extraction and collaboration experiments."""

from typing import List, Dict, Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency fallback
    np = None


def compute_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_profile_aggregate_metrics(
    all_results: List[Dict[str, Any]],
    bootstrap_samples: int = 1000,
    seed: int = 42,
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

    keys = sorted({
        key
        for result in all_results
        for key in result.get("field_metrics", {}).keys()
    })

    # Storage for aggregation
    micro_counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in keys}
    macro_scores = {k: {"f1": [], "sim": []} for k in keys}
    psas_values = []
    summary_cosines = []

    for res in all_results:
        # 1. PSAS (Profile Level)
        scores = res.get("profile_scores", {})
        psas_values.append(scores.get("PSAS", scores.get("MAS", 0.0)))

        # 2. Summary Description
        summary_cosines.append(scores.get("summary_cosine", 0.0))

        # 3. Per-key stats
        for k in keys:
            if k in res["field_metrics"]:
                m = res["field_metrics"][k]
                # Micro accumulation
                micro_counts[k]["tp"] += m.get("tp_total", 0)
                micro_counts[k]["fp"] += m.get("fp", 0)
                micro_counts[k]["fn"] += m.get("fn", 0)

                # Macro collection
                _, _, f1 = compute_f1(m.get("tp_total", 0), m.get("fp", 0), m.get("fn", 0))
                macro_scores[k]["f1"].append(f1)
                macro_scores[k]["sim"].append(m.get("avg_similarity", 0.0))

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
        mac_f1 = float(np.mean(macro_scores[k]["f1"])) if macro_scores[k]["f1"] and np is not None else 0.0
        avg_sim = float(np.mean(macro_scores[k]["sim"])) if macro_scores[k]["sim"] and np is not None else 0.0
        sim_std = float(np.std(macro_scores[k]["sim"])) if macro_scores[k]["sim"] and np is not None else 0.0

        report["per_key"][k] = {
            "micro_precision": mic_p,
            "micro_recall": mic_r,
            "micro_f1": mic_f1,
            "macro_f1": mac_f1,
            "avg_semantic_similarity": avg_sim,
            "avg_similarity_mean": avg_sim,
            "avg_similarity_std": sim_std,
            "tp_total": tp,
            "fp_total": fp,
            "fn_total": fn
        }

    # 2. Profile Level Metrics (PSAS)
    report["profile_level"] = {
        "PSAS_mean": float(np.mean(psas_values)) if np is not None else 0.0,
        "PSAS_std": float(np.std(psas_values)) if np is not None else 0.0,
        "summary_cosine_mean": float(np.mean(summary_cosines)) if np is not None else 0.0
    }
    report["summary_level"] = {
        "summary_cosine_mean": float(np.mean(summary_cosines)) if np is not None else 0.0,
        "summary_cosine_std": float(np.std(summary_cosines)) if np is not None else 0.0,
    }

    # 3. Bootstrap CI (Optional, for PSAS only)
    if bootstrap_samples > 0:
        sim_means = []
        n = len(psas_values)
        data = np.array(psas_values)
        rng = np.random.default_rng(seed)
        for _ in range(bootstrap_samples):
            sample = rng.choice(data, size=n, replace=True)
            sim_means.append(np.mean(sample)) if np is not None else sim_means.append(float(sample.mean()))

        if np is not None:
            report["bootstrap"]["PSAS_95CI"] = (
                float(np.percentile(sim_means, 2.5)),
                float(np.percentile(sim_means, 97.5))
            )
        else:
            report["bootstrap"]["PSAS_95CI"] = (0.0, 0.0)

    return report
