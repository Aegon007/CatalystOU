import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
from utils.data_utils import COLLABORATION_CATEGORIES

def evaluate_recommendation1(cases_file: str, model_name: str = "all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    with open(cases_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    category_aggregates = {cat: {"max_sims": [], "hits_50": [], "hits_45": [], "coverage_50": [], "coverage_45": []} 
                           for cat in COLLABORATION_CATEGORIES}

    print("=" * 80)
    print("RECOMMENDATION 1 EVALUATION: SOFT RECALL, HITS@K & MAX-SIMILARITY")
    print("=" * 80)

    for case in cases:
        pred_path = case["out_path"]
        gt_path = case["gt_path"]
        case_id = case.get("id", "case")

        with open(pred_path, "r", encoding="utf-8") as f:
            pred = json.load(f)
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        case_max_sims = {}
        case_hits_50 = {}
        case_hits_45 = {}
        case_cov_50 = {}
        case_cov_45 = {}

        for cat in COLLABORATION_CATEGORIES:
            p_items = pred.get(cat, [])
            g_items = gt.get(cat, [])
            if not p_items or not g_items:
                continue

            p_emb = model.encode(p_items, normalize_embeddings=True)
            g_emb = model.encode(g_items, normalize_embeddings=True)
            sim = np.matmul(p_emb, g_emb.T)  # (n_preds, n_golds)

            max_s = float(sim.max())
            hit_50 = 1 if max_s >= 0.50 else 0
            hit_45 = 1 if max_s >= 0.45 else 0

            cov_50 = float(np.mean(np.any(sim >= 0.50, axis=0)))
            cov_45 = float(np.mean(np.any(sim >= 0.45, axis=0)))

            case_max_sims[cat] = max_s
            case_hits_50[cat] = hit_50
            case_hits_45[cat] = hit_45
            case_cov_50[cat] = cov_50
            case_cov_45[cat] = cov_45

            category_aggregates[cat]["max_sims"].append(max_s)
            category_aggregates[cat]["hits_50"].append(hit_50)
            category_aggregates[cat]["hits_45"].append(hit_45)
            category_aggregates[cat]["coverage_50"].append(cov_50)
            category_aggregates[cat]["coverage_45"].append(cov_45)

        # Summary Cosine
        p_sum = pred.get("Summary Collaboration Themes", "")
        g_sum = gt.get("Summary Collaboration Themes", "")
        pe = model.encode([p_sum], normalize_embeddings=True)
        ge = model.encode([g_sum], normalize_embeddings=True)
        sum_cos = float(np.dot(pe, ge.T)[0, 0])

        case_res = {
            "id": case_id,
            "collab_type": case.get("collab_type", "Unknown"),
            "summary_cosine": sum_cos,
            "mean_max_sim": float(np.mean(list(case_max_sims.values()))),
            "peak_sim": float(max(case_max_sims.values())),
            "hits_50_pct": float(np.mean(list(case_hits_50.values())) * 100),
            "hits_45_pct": float(np.mean(list(case_hits_45.values())) * 100),
            "soft_recall_50_pct": float(np.mean(list(case_cov_50.values())) * 100),
            "soft_recall_45_pct": float(np.mean(list(case_cov_45.values())) * 100),
            "category_max_sims": case_max_sims,
        }
        results.append(case_res)

        print(f"\n[{case_id}] ({case_res['collab_type']})")
        print(f"  Summary Cosine       : {sum_cos:.4f}")
        print(f"  Mean Max-Similarity  : {case_res['mean_max_sim']:.4f} (Peak: {case_res['peak_sim']:.4f})")
        print(f"  Hits@0.50 (Category) : {case_res['hits_50_pct']:.1f}% | Hits@0.45: {case_res['hits_45_pct']:.1f}%")
        print(f"  Soft Recall@0.50     : {case_res['soft_recall_50_pct']:.1f}% | Soft Recall@0.45: {case_res['soft_recall_45_pct']:.1f}%")

    # Global summary
    mean_sum_cos = float(np.mean([r["summary_cosine"] for r in results]))
    mean_max_sim = float(np.mean([r["mean_max_sim"] for r in results]))
    mean_hits_50 = float(np.mean([r["hits_50_pct"] for r in results]))
    mean_hits_45 = float(np.mean([r["hits_45_pct"] for r in results]))
    mean_cov_50 = float(np.mean([r["soft_recall_50_pct"] for r in results]))
    mean_cov_45 = float(np.mean([r["soft_recall_45_pct"] for r in results]))

    print("\n" + "=" * 80)
    print("GLOBAL AGGREGATE SUMMARY (N = 5 Cases)")
    print("=" * 80)
    print(f"Mean Summary Cosine Similarity : {mean_sum_cos:.4f}")
    print(f"Mean Max-Similarity across Cats : {mean_max_sim:.4f}")
    print(f"Mean Category Hits@0.50        : {mean_hits_50:.1f}%")
    print(f"Mean Category Hits@0.45        : {mean_hits_45:.1f}%")
    print(f"Mean Soft Recall (Cov@0.50)    : {mean_cov_50:.1f}%")
    print(f"Mean Soft Recall (Cov@0.45)    : {mean_cov_45:.1f}%")

    print("\nPER-CATEGORY PERFORMANCE BREAKDOWN:")
    for cat in COLLABORATION_CATEGORIES:
        m_sim = np.mean(category_aggregates[cat]["max_sims"])
        h50 = np.mean(category_aggregates[cat]["hits_50"]) * 100
        h45 = np.mean(category_aggregates[cat]["hits_45"]) * 100
        c50 = np.mean(category_aggregates[cat]["coverage_50"]) * 100
        c45 = np.mean(category_aggregates[cat]["coverage_45"]) * 100
        print(f"  {cat:34s} : Mean MaxSim={m_sim:.4f} | Hits@0.50={h50:4.1f}% | Hits@0.45={h45:4.1f}% | Recall@0.45={c45:4.1f}%")

    out_file = Path("results/exp2_qwen/recommendation1_soft_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "global_aggregates": {
                "mean_summary_cosine": mean_sum_cos,
                "mean_max_sim": mean_max_sim,
                "mean_hits_50_pct": mean_hits_50,
                "mean_hits_45_pct": mean_hits_45,
                "mean_soft_recall_50_pct": mean_cov_50,
                "mean_soft_recall_45_pct": mean_cov_45,
            },
            "cases": results,
            "category_aggregates": {
                cat: {
                    "mean_max_sim": float(np.mean(vals["max_sims"])),
                    "hits_50_pct": float(np.mean(vals["hits_50"]) * 100),
                    "hits_45_pct": float(np.mean(vals["hits_45"]) * 100),
                    "soft_recall_50_pct": float(np.mean(vals["coverage_50"]) * 100),
                    "soft_recall_45_pct": float(np.mean(vals["coverage_45"]) * 100),
                }
                for cat, vals in category_aggregates.items()
            }
        }, f, indent=2)

    print(f"\nSaved detailed Recommendation 1 metrics to: {out_file}")

if __name__ == "__main__":
    cases_p = "cases/cases.json" if Path("cases/cases.json").exists() else "cases.json"
    evaluate_recommendation1(cases_p)
