# metrics.py

import numpy as np
from typing import List, Dict


def aggregate_scores(all_results: List[dict]) -> dict:
    """
    aggregate MAS and average per-category scores.
    """
    if not all_results:
        return {}

    mas_list = [r["MAS"] for r in all_results]

    # collect per-category
    category_avgs = {}
    for r in all_results:
        for cat, scores in r["category_scores"].items():
            if cat not in category_avgs:
                category_avgs[cat] = []
            category_avgs[cat].append(scores["avg_similarity"])

    aggregated_category_scores = {
        cat: float(np.mean(vals)) for cat, vals in category_avgs.items()
    }

    return {
        "MAS_mean": float(np.mean(mas_list)),
        "MAS_std": float(np.std(mas_list)),
        "category_avg_similarities": aggregated_category_scores
    }


def bootstrap_ci(all_results: List[dict], n_boot=1000, alpha=0.05):
    """
    Bootstrap CI for MAS.
    """
    if not all_results:
        return {}

    mas_list = np.array([r["MAS"] for r in all_results])
    n = len(mas_list)

    boot_samples = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        boot_samples.append(np.mean(mas_list[idx]))

    boot_samples = np.sort(boot_samples)
    lower = boot_samples[int(alpha/2 * n_boot)]
    upper = boot_samples[int((1 - alpha/2) * n_boot)]

    return {
        "MAS_CI_lower": float(lower),
        "MAS_CI_upper": float(upper)
    }
