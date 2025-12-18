# orchestrator.py

import json
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from profile_io import load_profile_json, save_json
from llm_reasoner import call_llm_reasoner
from matching import exact_normalized_match, hungarian_semantic_matching
from metrics import aggregate_scores, bootstrap_ci

CATEGORIES = [
  "Shared Domains",
  "Method-Application Synergies",
  "Complementary Technique Synergies",
  "Data-Method Synergies",
  "Cross-Domain Fusion Topics",
  "Shared Application Areas",
  "Joint Technique Development",
  "Theory-Application Synergy",
  "Thinking Pattern Synergies",
  "Future Research Directions"
]

def evaluate_one_pair(a_pre_path, b_pre_path, gt_path, llm_cfg, model, tau=0.65):

    A = load_profile_json(a_pre_path)
    B = load_profile_json(b_pre_path)
    GT = load_profile_json(gt_path)

    pred = call_llm_reasoner(A, B, llm_cfg)

    category_matches = {}
    category_scores = {}
    MAS_components = []

    for cat in CATEGORIES:
        preds = pred.get(cat, [])
        golds = GT.get(cat, [])

        tp_e, fp_e, fn_e, exact_list = exact_normalized_match(preds, golds)

        remaining_preds = [p for p, m in exact_list if m is None]
        matched_gold_set = {m for _, m in exact_list if m is not None}
        remaining_golds = [g for g in golds if g not in matched_gold_set]

        if remaining_preds and remaining_golds:
            sem_matches = hungarian_semantic_matching(remaining_preds, remaining_golds, model)
        else:
            sem_matches = []

        combined = []
        for p, m in exact_list:
            if m is not None:
                combined.append({"pred": p, "gold": m, "sim": 1.0})
            else:
                combined.append({"pred": p, "gold": None, "sim": 0.0})

        for p, g, sim in sem_matches:
            for entry in combined:
                if entry["pred"] == p and entry["gold"] is None:
                    entry["gold"] = g
                    entry["sim"] = sim
                    break

        sims = [x["sim"] for x in combined if x["gold"] is not None]
        avg_sim = float(np.mean(sims)) if sims else 0.0
        max_sim = float(max(sims)) if sims else 0.0
        coverage = (
            sum(1 for x in combined if x["gold"] is not None and x["sim"] >= tau)
            / max(1, len(golds))
        )

        category_matches[cat] = combined
        category_scores[cat] = {
            "avg_similarity": avg_sim,
            "max_similarity": max_sim,
            "coverage_at_tau": coverage
        }

        MAS_components.append(avg_sim)

    MAS = float(np.mean(MAS_components))

    return {
        "predicted": pred,
        "groundtruth": GT,
        "category_matches": category_matches,
        "category_scores": category_scores,
        "MAS": MAS
    }


def evaluate_many_pairs(pairs_list, llm_cfg, model_name="all-mpnet-base-v2", tau=0.65):
    model = SentenceTransformer(model_name)
    all_results = []

    for p in pairs_list:
        res = evaluate_one_pair(
            p["a_pre"],
            p["b_pre"],
            p["groundtruth"],
            llm_cfg,
            model,
            tau=tau
        )
        res["pair_id"] = p.get("pair_id", "unknown")
        all_results.append(res)

    summary = aggregate_scores(all_results)
    ci = bootstrap_ci(all_results, n_boot=1000)

    return {
        "per_pair_results": all_results,
        "aggregate": summary,
        "bootstrap": ci
    }


if __name__ == "__main__":
    # Example pair (you will replace with real file paths)
    llm_cfg = {"model": "placeholder"}

    pairs = [
        {
            "pair_id": "Lan_Example",
            "a_pre": "/mnt/data/ChaoLan_ComputerScience_Profile.json",   # your provided profile
            "b_pre": "data/example_coauthor_before.json",
            "groundtruth": "data/lan_example_groundtruth.json"
        }
    ]

    results = evaluate_many_pairs(pairs, llm_cfg)
    print(json.dumps(results["aggregate"], indent=2))
