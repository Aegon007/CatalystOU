"""Canonical Experiment 2 Hungarian bipartite matching & selectivity evaluator."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
GT_DIR = REPO_ROOT / "collaboration_ground_truth"
DEFAULT_OUT_PATH = REPO_ROOT / "results" / "exp2_hungarian_evaluation_gemini_gt.json"

POS_CASES = [
    {"id": "case_001_ebert_xiao_cs_bio", "gt": "case_001_ebert_xiao_cs_bio_gt.json", "tier": "OU Tier-1", "collab_type": "Cross-Discipline (CS-Bio)"},
    {"id": "case_002_zhao_cheng_cs_ece", "gt": "case_002_zhao_cheng_cs_ece_gt.json", "tier": "OU Tier-1", "collab_type": "Cross-Discipline (CS-ECE)"},
    {"id": "case_003_cai_ding_psych_ece", "gt": "case_003_cai_ding_psych_ece_gt.json", "tier": "OU Tier-1", "collab_type": "Cross-Discipline (Psych-ECE)"},
    {"id": "case_004_yeary_havlicek_ece_ece", "gt": "case_004_yeary_havlicek_ece_ece_gt.json", "tier": "OU Tier-1", "collab_type": "Intra-Discipline (ECE-ECE)"},
    {"id": "case_005_radhakrishnan_antonio_cs_cs", "gt": "case_005_radhakrishnan_antonio_cs_cs_gt.json", "tier": "OU Tier-1", "collab_type": "Intra-Discipline (CS-CS)"},
    {"id": "case_006_chitambar_leditzky_ece_math", "gt": "case_006_chitambar_leditzky_ece_math_gt.json", "tier": "UIUC Tier-2", "collab_type": "Cross-Discipline (ECE-Math)"},
    {"id": "case_007_rhodes_tsai_psych_bio", "gt": "case_007_rhodes_tsai_psych_bio_gt.json", "tier": "UIUC Tier-2", "collab_type": "Cross-Discipline (Psych-Bio)"},
    {"id": "case_008_federmeier_lavalle_psych_cs", "gt": "case_008_federmeier_lavalle_psych_cs_gt.json", "tier": "UIUC Tier-2", "collab_type": "Cross-Discipline (Psych-CS)"},
    {"id": "case_009_raginsky_ringer_ece_cs", "gt": "case_009_raginsky_ringer_ece_cs_gt.json", "tier": "UIUC Tier-2", "collab_type": "Cross-Discipline (ECE-CS)"},
    {"id": "case_010_whitaker_rapti_bio_math", "gt": "case_010_whitaker_rapti_bio_math_gt.json", "tier": "UIUC Tier-2", "collab_type": "Cross-Discipline (Bio-Math)"},
]

NEG_CASES = [
    {"id": "neg_001_maiti_cartersowell", "filename": "neg_001_maiti_cartersowell_prediction.json"},
    {"id": "neg_002_yeary_dunn", "filename": "neg_002_yeary_dunn_prediction.json"},
    {"id": "neg_003_forester_quadri", "filename": "neg_003_forester_quadri_prediction.json"},
    {"id": "neg_004_karr_cokely", "filename": "neg_004_karr_cokely_prediction.json"},
    {"id": "neg_005_jiang_martin", "filename": "neg_005_jiang_martin_prediction.json"},
]

CORE_6_CATEGORIES = [
    "Data-Method Synergies",
    "Complementary Technique Synergies",
    "Method-Application Synergies",
    "Shared Domains",
    "Cross-Domain Fusion Topics",
    "Future Research Directions",
]

ALL_10_CATEGORIES = [
    "Shared Domains",
    "Method-Application Synergies",
    "Complementary Technique Synergies",
    "Data-Method Synergies",
    "Cross-Domain Fusion Topics",
    "Shared Application Areas",
    "Joint Technique Development",
    "Theory-Application Synergy",
    "Thinking Pattern Synergies",
    "Future Research Directions",
]

MODEL_SPECS = {
    "dense": {
        "name": "Qwen 3.8-27B (Dense)",
        "pos_dir": REPO_ROOT / "results" / "exp2_qwen3.8-27b_true",
        "neg_dir": REPO_ROOT / "results" / "exp2_qwen3.8-27b_true_negative",
    },
    "pure_moe": {
        "name": "Qwen 3.6-35B-A3B (Pure MoE)",
        "pos_dir": REPO_ROOT / "results" / "exp2_qwen3.6-35b-a3b_true",
        "neg_dir": REPO_ROOT / "results" / "exp2_qwen3.6-35b-a3b_true_negative",
    },
    "agent_moe": {
        "name": "Qwen 3.6-35B-A3B (Agent MoE)",
        "pos_dir": REPO_ROOT / "results" / "exp2_qwen3.6-35b-a3b_agent_true",
        "neg_dir": REPO_ROOT / "results" / "exp2_qwen3.6-35b-a3b_agent_true_negative",
    },
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def hungarian_bipartite_match(preds: List[str], golds: List[str], embedder: SentenceTransformer, tau: float) -> Dict[str, Any]:
    """Perform 1-to-1 optimal Hungarian bipartite matching on normalized SBERT embeddings."""
    if not preds or not golds:
        return {"tp": 0, "fp": len(preds), "fn": len(golds), "matches": [], "mean_sim": 0.0}

    ep = embedder.encode(preds, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    eg = embedder.encode(golds, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    sim_matrix = np.matmul(ep, eg.T)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    matches = []
    matched_sims = []
    tp = 0
    for r, c in zip(row_ind, col_ind):
        score = float(sim_matrix[r, c])
        if score >= tau:
            tp += 1
            matches.append((preds[r], golds[c], score))
            matched_sims.append(score)

    return {
        "tp": tp,
        "fp": len(preds) - tp,
        "fn": len(golds) - tp,
        "matches": matches,
        "mean_sim": float(np.mean(matched_sims)) if matched_sims else 0.0,
    }


def _summarize_tier(case_evals: List[Dict[str, Any]], per_case_neg: Dict[str, List[float]], tier: str) -> Optional[Dict[str, Any]]:
    sub = [c for c in case_evals if c["tier"] == tier]
    if not sub:
        return None
    prf = compute_prf(sum(c["tp"] for c in sub), sum(c["fp"] for c in sub), sum(c["fn"] for c in sub))
    pos_mean = float(np.mean([c["summary_cosine"] for c in sub]))
    neg_vals = [s for c in sub for s in per_case_neg.get(c["id"], [])]
    neg_mean = float(np.mean(neg_vals)) if neg_vals else 0.0
    margin = pos_mean - neg_mean
    margin_pct = (margin / neg_mean) * 100 if neg_mean > 0 else 0.0
    disc = sum(1 for c in sub if per_case_neg.get(c["id"]) and all(c["summary_cosine"] > ns for ns in per_case_neg[c["id"]]))
    return {
        "n": len(sub),
        "mean_pos_summary_cosine": pos_mean,
        "mean_neg_summary_cosine": neg_mean,
        "margin_pct": margin_pct,
        "discrimination_accuracy": (disc / len(sub)) * 100,
        "core6_tau45": prf,
    }


def evaluate_exp2(
    models: Optional[List[str]] = None,
    save_json: bool = True,
    out_path: Path = DEFAULT_OUT_PATH,
    local_only: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run unified Experiment 2 Hungarian bipartite matching & selectivity evaluation."""
    if local_only is None:
        local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    selected_keys = models if models and "all" not in models else list(MODEL_SPECS.keys())

    print("=" * 96)
    print(" CatalystOU: Experiment 2 Hungarian Bipartite Opportunity Evaluation (N=10 Pos + 5 Neg)")
    print(" Metric: 1-to-1 scipy.optimize.linear_sum_assignment on all-mpnet-base-v2")
    print("=" * 96)

    embedder = SentenceTransformer("all-mpnet-base-v2", local_files_only=local_only)

    results: Dict[str, Any] = {}
    if save_json and out_path.exists() and len(selected_keys) < len(MODEL_SPECS):
        try:
            existing = load_json(out_path)
            if isinstance(existing, dict):
                results.update(existing)
        except Exception:
            pass

    for mkey in selected_keys:
        if mkey not in MODEL_SPECS:
            raise ValueError(f"Unknown model key '{mkey}'. Valid keys: {list(MODEL_SPECS.keys())}")
        spec = MODEL_SPECS[mkey]
        mname = spec["name"]
        pos_dir = spec["pos_dir"]
        neg_dir = spec["neg_dir"]

        available_cases = [c for c in POS_CASES if (GT_DIR / c["gt"]).exists() and (pos_dir / f"{c['id']}_prediction.json").exists()]
        if not available_cases:
            print(f"\n[!] Skipping {mname}: no matching prediction JSONs found in {pos_dir}")
            continue

        print(f"\n{'=' * 80}")
        print(f"EVALUATING MODEL: {mname} ({len(available_cases)}/{len(POS_CASES)} positive cases present)")
        print(f"{'=' * 80}")

        core6_45 = {"tp": 0, "fp": 0, "fn": 0}
        core6_50 = {"tp": 0, "fp": 0, "fn": 0}
        all10_45 = {"tp": 0, "fp": 0, "fn": 0}
        all10_50 = {"tp": 0, "fp": 0, "fn": 0}
        category_stats = {cat: {"tp_45": 0, "fp_45": 0, "fn_45": 0, "sims": [], "pair_hits_45": 0, "valid_pairs": 0} for cat in ALL_10_CATEGORIES}
        pos_case_evals = []

        for case in available_cases:
            cid = case["id"]
            gt = load_json(GT_DIR / case["gt"])
            pred = load_json(pos_dir / f"{cid}_prediction.json")

            g_sum = gt.get("Summary Collaboration Themes", "")
            p_sum = pred.get("Summary Collaboration Themes", "")
            if g_sum and p_sum:
                eg = embedder.encode([g_sum], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                ep = embedder.encode([p_sum], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                sum_cos = float(np.dot(ep, eg.T)[0, 0])
            else:
                sum_cos = 0.0

            c6_tp, c6_fp, c6_fn = 0, 0, 0
            cat_results = {}
            for cat in ALL_10_CATEGORIES:
                preds_list = to_list(pred.get(cat, []))
                golds_list = to_list(gt.get(cat, []))
                m45 = hungarian_bipartite_match(preds_list, golds_list, embedder, 0.45)
                m50 = hungarian_bipartite_match(preds_list, golds_list, embedder, 0.50)

                category_stats[cat]["tp_45"] += m45["tp"]
                category_stats[cat]["fp_45"] += m45["fp"]
                category_stats[cat]["fn_45"] += m45["fn"]
                if golds_list:
                    category_stats[cat]["valid_pairs"] += 1
                    if m45["tp"] > 0:
                        category_stats[cat]["pair_hits_45"] += 1
                if m45["matches"]:
                    category_stats[cat]["sims"].extend([x[2] for x in m45["matches"]])

                all10_45["tp"] += m45["tp"]; all10_45["fp"] += m45["fp"]; all10_45["fn"] += m45["fn"]
                all10_50["tp"] += m50["tp"]; all10_50["fp"] += m50["fp"]; all10_50["fn"] += m50["fn"]

                if cat in CORE_6_CATEGORIES:
                    core6_45["tp"] += m45["tp"]; core6_45["fp"] += m45["fp"]; core6_45["fn"] += m45["fn"]
                    core6_50["tp"] += m50["tp"]; core6_50["fp"] += m50["fp"]; core6_50["fn"] += m50["fn"]
                    c6_tp += m45["tp"]; c6_fp += m45["fp"]; c6_fn += m45["fn"]

                cat_results[cat] = {"pred_count": len(preds_list), "gold_count": len(golds_list), "tp_45": m45["tp"], "mean_sim": m45["mean_sim"]}

            case_prf = compute_prf(c6_tp, c6_fp, c6_fn)
            pos_case_evals.append({
                "id": cid,
                "tier": case["tier"],
                "collab_type": case["collab_type"],
                "summary_cosine": sum_cos,
                "core6_p": case_prf["precision"],
                "core6_r": case_prf["recall"],
                "core6_f1": case_prf["f1"],
                "tp": c6_tp,
                "fp": c6_fp,
                "fn": c6_fn,
                "category_breakdown": cat_results,
            })
            print(f"  [{cid}] ({case['tier']}) SumCos={sum_cos:.4f} | Hungarian Core-6 P={case_prf['precision']*100:.1f}%, R={case_prf['recall']*100:.1f}%, F1={case_prf['f1']*100:.1f}%")

        # Negative Controls & Cross-Pair Discrimination
        all_neg_sims = []
        per_case_neg = {c["id"]: [] for c in available_cases}
        for neg in NEG_CASES:
            nfile = neg_dir / neg["filename"]
            if not nfile.exists():
                continue
            nsum = load_json(nfile).get("Summary Collaboration Themes", "")
            if not nsum:
                continue
            en = embedder.encode([nsum], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            for case in available_cases:
                gsum = load_json(GT_DIR / case["gt"]).get("Summary Collaboration Themes", "")
                if not gsum:
                    continue
                eg = embedder.encode([gsum], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                sim = float(np.dot(en, eg.T)[0, 0])
                all_neg_sims.append(sim)
                per_case_neg[case["id"]].append(sim)

        mean_pos = float(np.mean([c["summary_cosine"] for c in pos_case_evals]))
        mean_neg = float(np.mean(all_neg_sims)) if all_neg_sims else 0.0
        margin = mean_pos - mean_neg
        margin_pct = (margin / mean_neg) * 100 if mean_neg > 0 else 0.0
        discriminated = sum(1 for c in pos_case_evals if per_case_neg[c["id"]] and all(c["summary_cosine"] > ns for ns in per_case_neg[c["id"]]))
        disc_pct = (discriminated / len(available_cases)) * 100

        c6_45_prf = compute_prf(core6_45["tp"], core6_45["fp"], core6_45["fn"])
        c6_50_prf = compute_prf(core6_50["tp"], core6_50["fp"], core6_50["fn"])
        a10_45_prf = compute_prf(all10_45["tp"], all10_45["fp"], all10_45["fn"])

        tier_breakdown = {}
        for tier in ("OU Tier-1", "UIUC Tier-2"):
            tb = _summarize_tier(pos_case_evals, per_case_neg, tier)
            if tb:
                tier_breakdown[tier] = tb

        results[mname] = {
            "n_pos": len(available_cases),
            "n_pos_expected": len(POS_CASES),
            "mean_pos_summary_cosine": mean_pos,
            "mean_neg_summary_cosine": mean_neg,
            "selectivity_margin": margin,
            "margin_pct": margin_pct,
            "discrimination_accuracy": disc_pct,
            "core6_tau45": c6_45_prf,
            "core6_tau50": c6_50_prf,
            "all10_tau45": a10_45_prf,
            "tier_breakdown": tier_breakdown,
            "category_breakdown": {
                cat: {
                    **compute_prf(category_stats[cat]["tp_45"], category_stats[cat]["fp_45"], category_stats[cat]["fn_45"]),
                    "pair_alignment_rate_45": (category_stats[cat]["pair_hits_45"] / category_stats[cat]["valid_pairs"]) if category_stats[cat]["valid_pairs"] else 0.0,
                    "mean_sim": float(np.mean(category_stats[cat]["sims"])) if category_stats[cat]["sims"] else 0.0,
                }
                for cat in ALL_10_CATEGORIES
            },
            "cases": pos_case_evals,
        }

    if save_json and results:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved Hungarian matching evaluation results to: {out_path}")

    if results:
        print("\n" + "=" * 106)
        print(f"{'Model':<30} | {'N':<5} | {'Pos Cos':<8} | {'Neg Cos':<8} | {'Margin':<9} | {'Disc':<6} | {'Core6 P':<8} | {'Core6 R':<8} | {'Core6 F1':<8}")
        print("-" * 106)
        for mname, d in results.items():
            n_str = f"{d.get('n_pos', len(d.get('cases', [])))}/10"
            c6 = d["core6_tau45"]
            print(
                f"{mname:<30} | {n_str:<5} | {d['mean_pos_summary_cosine']:.4f}   | {d['mean_neg_summary_cosine']:.4f}   | "
                f"+{d['margin_pct']:.1f}%   | {d['discrimination_accuracy']:.0f}%  | {c6['precision']*100:5.1f}%   | {c6['recall']*100:5.1f}%   | {c6['f1']*100:5.1f}%"
            )
        print("=" * 106)

    return results


def main():
    parser = argparse.ArgumentParser(description="CatalystOU Exp 2 Hungarian Bipartite Evaluation")
    parser.add_argument("--models", nargs="+", default=["all"], choices=["all", "dense", "pure_moe", "agent_moe"])
    parser.add_argument("--no-save", action="store_true", help="Print metrics without writing results JSON")
    parser.add_argument("--local-only", action="store_true", help="Force local cached SBERT model only")
    args = parser.parse_args()
    evaluate_exp2(models=args.models, save_json=not args.no_save, local_only=args.local_only or None)


if __name__ == "__main__":
    main()
