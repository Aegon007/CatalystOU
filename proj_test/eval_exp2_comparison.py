"""
Comprehensive Evaluation & Comparative Analysis for Experiment 2
Compares Qwen 3.8-27B (Dense) vs Qwen 3.6-35B-A3B (MoE) across:
1. Historical Positive Benchmark (5 pairs)
2. Negative Controls (5 pairs)
3. Contrastive Discrimination Margin
4. Table 2 (Backtesting Overview) & Table 3 (Reasoning Mode Alignment)
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GT_DIR = REPO_ROOT / "collaboration_ground_truth"

POS_CASES = [
    {"id": "case_001_ebert_xiao_cs_bio", "gt": "case_001_ebert_xiao_cs_bio_gt.json", "collab_type": "Cross-Discipline"},
    {"id": "case_002_zhao_cheng_cs_ece", "gt": "case_002_zhao_cheng_cs_ece_gt.json", "collab_type": "Cross-Discipline"},
    {"id": "case_003_cai_song_psych_psych", "gt": "case_003_cai_song_psych_psych_gt.json", "collab_type": "Same-Discipline"},
    {"id": "case_004_yeary_havlicek_ece_ece", "gt": "case_004_yeary_havlicek_ece_ece_gt.json", "collab_type": "Same-Discipline"},
    {"id": "case_005_radhakrishnan_antonio_cs_cs", "gt": "case_005_radhakrishnan_antonio_cs_cs_gt.json", "collab_type": "Same-Discipline"},
]

NEG_CASES = [
    {"id": "neg_001_maiti_cartersowell", "filename": "neg_001_maiti_cartersowell_prediction.json"},
    {"id": "neg_002_yeary_dunn", "filename": "neg_002_yeary_dunn_prediction.json"},
    {"id": "neg_003_forester_quadri", "filename": "neg_003_forester_quadri_prediction.json"},
    {"id": "neg_004_karr_cokely", "filename": "neg_004_karr_cokely_prediction.json"},
    {"id": "neg_005_jiang_martin", "filename": "neg_005_jiang_martin_prediction.json"},
]

COLLAB_CATEGORIES = [
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

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def to_text(val):
    if isinstance(val, list):
        return " ".join(str(x) for x in val)
    return str(val) if val is not None else ""

def to_list(val):
    if isinstance(val, list):
        return [str(x) for x in val if x]
    elif isinstance(val, str) and val.strip():
        return [val.strip()]
    return []

def run_evaluation():
    print("=" * 80)
    print("EXPERIMENT 2: SBERT EVALUATION & MODEL COMPARISON")
    print("Embedding Model: all-mpnet-base-v2")
    print("=" * 80)

    embedder = SentenceTransformer("all-mpnet-base-v2")

    models = {
        "Qwen 3.8 27B (Dense)": {
            "pos_dir": REPO_ROOT / "results" / "exp2_qwen3.8-27b",
            "neg_dir": REPO_ROOT / "results" / "exp2_qwen3.8-27b_negative",
        },
        "Qwen 3.6-35B-A3B (MoE)": {
            "pos_dir": REPO_ROOT / "results" / "exp2_qwen3.6-35b-a3b",
            "neg_dir": REPO_ROOT / "results" / "exp2_qwen3.6-35b-a3b_negative",
        }
    }

    eval_results = {}

    for model_name, paths in models.items():
        print(f"\n>>> Evaluating: {model_name} <<<")
        pos_dir = paths["pos_dir"]
        neg_dir = paths["neg_dir"]

        pos_case_scores = []
        neg_case_scores = []
        category_hits_45 = {cat: 0 for cat in COLLAB_CATEGORIES}
        category_hits_50 = {cat: 0 for cat in COLLAB_CATEGORIES}
        category_sims = {cat: [] for cat in COLLAB_CATEGORIES}

        # 1. Positive Evaluation
        print("  Evaluating 5 Positive Benchmark Cases...")
        for case in POS_CASES:
            cid = case["id"]
            gt_file = GT_DIR / case["gt"]
            pred_file = pos_dir / f"{cid}_prediction.json"

            gt = load_json(gt_file)
            pred = load_json(pred_file)

            # Summary narrative cosine
            g_sum = to_text(gt.get("Summary Collaboration Themes", ""))
            p_sum = to_text(pred.get("Summary Collaboration Themes", ""))
            e_g = embedder.encode([g_sum], normalize_embeddings=True)
            e_p = embedder.encode([p_sum], normalize_embeddings=True)
            sum_cos = float(np.dot(e_p, e_g.T)[0, 0])

            # Category-level max similarities
            cat_maxs = {}
            for cat in COLLAB_CATEGORIES:
                g_items = to_list(gt.get(cat, []))
                p_items = to_list(pred.get(cat, []))
                if g_items and p_items:
                    eg_cat = embedder.encode(g_items, normalize_embeddings=True)
                    ep_cat = embedder.encode(p_items, normalize_embeddings=True)
                    sim_matrix = np.matmul(ep_cat, eg_cat.T)
                    max_sim = float(sim_matrix.max())
                else:
                    max_sim = 0.0
                cat_maxs[cat] = max_sim
                category_sims[cat].append(max_sim)
                if max_sim >= 0.45:
                    category_hits_45[cat] += 1
                if max_sim >= 0.50:
                    category_hits_50[cat] += 1

            pos_case_scores.append({
                "id": cid,
                "collab_type": case["collab_type"],
                "summary_cosine": sum_cos,
                "cat_maxs": cat_maxs
            })
            print(f"    [{cid}] Summary Cosine: {sum_cos:.4f} | Peak Cat Sim: {max(cat_maxs.values()):.4f}")

        # 2. Negative Evaluation (Baseline Affinity against Ground Truth)
        print("  Evaluating 5 Negative Control Pairs...")
        # For negative pairs, compare each negative prediction against each positive GT
        all_neg_sims = []
        per_case_neg_sims = {case["id"]: [] for case in POS_CASES}

        for n_item in NEG_CASES:
            n_file = neg_dir / n_item["filename"]
            n_data = load_json(n_file)
            n_sum = to_text(n_data.get("Summary Collaboration Themes", ""))
            e_n = embedder.encode([n_sum], normalize_embeddings=True)

            for case in POS_CASES:
                gt = load_json(GT_DIR / case["gt"])
                g_sum = to_text(gt.get("Summary Collaboration Themes", ""))
                e_g = embedder.encode([g_sum], normalize_embeddings=True)
                sim = float(np.dot(e_n, e_g.T)[0, 0])
                all_neg_sims.append(sim)
                per_case_neg_sims[case["id"]].append(sim)

        # Aggregate metrics
        pos_sims = [p["summary_cosine"] for p in pos_case_scores]
        mean_pos_sim = float(np.mean(pos_sims))
        r_45 = float(np.mean([1 if s >= 0.45 else 0 for s in pos_sims]) * 100)
        r_50 = float(np.mean([1 if s >= 0.50 else 0 for s in pos_sims]) * 100)

        mean_neg_sim = float(np.mean(all_neg_sims))
        neg_r_45 = float(np.mean([1 if s >= 0.45 else 0 for s in all_neg_sims]) * 100)
        neg_r_50 = float(np.mean([1 if s >= 0.50 else 0 for s in all_neg_sims]) * 100)

        selectivity_margin = mean_pos_sim - mean_neg_sim
        margin_pct = (selectivity_margin / mean_neg_sim) * 100 if mean_neg_sim > 0 else 0.0

        # Discrimination Accuracy (% of cases where true positive > all 5 negative similarities)
        discriminated = 0
        for p in pos_case_scores:
            cid = p["id"]
            pos_s = p["summary_cosine"]
            neg_max = max(per_case_neg_sims[cid])
            if pos_s > neg_max:
                discriminated += 1
        discrim_acc = (discriminated / len(POS_CASES)) * 100

        # Table 3 Reasoning Mode Alignment (% at tau=0.45)
        n_pos = len(POS_CASES)
        mode_alignment = {
            "Method-App": (category_hits_45["Method-Application Synergies"] / n_pos) * 100,
            "Shared Domains": (category_hits_45["Shared Domains"] / n_pos) * 100,
            "Cross-Disciplinary": (category_hits_45["Cross-Domain Fusion Topics"] / n_pos) * 100,
            "Tech Transfer": (category_hits_45["Complementary Technique Synergies"] / n_pos) * 100,
            "Joint Tech Dev": (category_hits_45["Joint Technique Development"] / n_pos) * 100,
            "Data Reuse": (category_hits_45["Data-Method Synergies"] / n_pos) * 100,
            "Shared App Areas": (category_hits_45["Shared Application Areas"] / n_pos) * 100,
            "Theory-App": (category_hits_45["Theory-Application Synergy"] / n_pos) * 100,
            "Thinking Patterns": (category_hits_45["Thinking Pattern Synergies"] / n_pos) * 100,
            "Future Directions": (category_hits_45["Future Research Directions"] / n_pos) * 100,
        }

        eval_results[model_name] = {
            "mean_pos_sim": mean_pos_sim,
            "pos_sims": pos_sims,
            "r_45": r_45,
            "r_50": r_50,
            "mean_neg_sim": mean_neg_sim,
            "neg_r_45": neg_r_45,
            "neg_r_50": neg_r_50,
            "selectivity_margin": selectivity_margin,
            "margin_pct": margin_pct,
            "discrim_acc": discrim_acc,
            "mode_alignment": mode_alignment,
            "category_sims_mean": {cat: float(np.mean(sims)) for cat, sims in category_sims.items()},
            "case_scores": pos_case_scores
        }

    # Print Comparative Summary
    print("\n" + "=" * 80)
    print(" COMPARATIVE SUMMARY: TABLE 2 (HISTORICAL BACKTESTING OVERVIEW)")
    print("=" * 80)
    print(f"{'Model / Baseline':<32} | {'Avg. Sim':<10} | {'R@0.45':<10} | {'R@0.50':<10} | {'Selectivity Margin'}")
    print("-" * 85)
    for mname, res in eval_results.items():
        print(f"{mname:<32} | {res['mean_pos_sim']:<10.4f} | {res['r_45']:<9.1f}% | {res['r_50']:<9.1f}% | +{res['selectivity_margin']:.4f} (+{res['margin_pct']:.1f}%)")
    # Add negative baseline row
    q36_neg = eval_results["Qwen 3.6-35B-A3B (MoE)"]["mean_neg_sim"]
    print(f"{'Negative Controls (35B MoE)':<32} | {q36_neg:<10.4f} | {eval_results['Qwen 3.6-35B-A3B (MoE)']['neg_r_45']:<9.1f}% | {eval_results['Qwen 3.6-35B-A3B (MoE)']['neg_r_50']:<9.1f}% | Reference Baseline")

    print("\n" + "=" * 80)
    print(" COMPARATIVE SUMMARY: TABLE 3 (REASONING MODE ALIGNMENT AT tau = 0.45)")
    print("=" * 80)
    headers = ["Method-App", "Shared Domains", "Cross-Disciplinary", "Tech Transfer", "Data Reuse"]
    print(f"{'Model':<30} | " + " | ".join([f"{h:<18}" for h in headers]))
    print("-" * 125)
    for mname, res in eval_results.items():
        ma = res["mode_alignment"]
        row = [f"{ma['Method-App']:.1f}%", f"{ma['Shared Domains']:.1f}%", f"{ma['Cross-Disciplinary']:.1f}%", f"{ma['Tech Transfer']:.1f}%", f"{ma['Data Reuse']:.1f}%"]
        print(f"{mname:<30} | " + " | ".join([f"{r:<18}" for r in row]))

    # Save comparison json
    out_eval_file = REPO_ROOT / "results" / "exp2_comparative_evaluation.json"
    with open(out_eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[+] Full comparison metrics saved to: {out_eval_file}")

if __name__ == "__main__":
    run_evaluation()
