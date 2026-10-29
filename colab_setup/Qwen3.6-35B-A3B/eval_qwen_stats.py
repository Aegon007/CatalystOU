"""
================================================================================
CatalystOU: Table Statistics & Evaluation Suite for Qwen 3.6-35B-A3B MoE
================================================================================
This script evaluates extracted profiles and collaboration backtesting predictions
for Qwen 3.6-35B-A3B and formats the results directly for the paper's tables:
- Table 1: Experiment 1 Field-level Profile Extraction Performance
- Table 2: Experiment 2 Historical Co-authorship Backtesting Results
- Table 3: Experiment 2 Reasoning Mode Alignment (% at tau = 0.45)
================================================================================
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import utils.matching as matching
    import utils.profile_io as profile_io
    from proj_test.test1_profile_acc import evaluate_single_profile, build_data_map
    HAS_EVAL_DEPS = True
except ImportError as e:
    HAS_EVAL_DEPS = False
    MISSING_DEP = str(e)


def compute_f1(tp: int, fp: int, fn: int):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def evaluate_table1(model_dir_name: str = "qwen_qwen3.6-35b-a3b"):
    """Evaluate Experiment 1 profiles against gold labeled data."""
    extracted_dir = repo_root / "extracted_profile_json"
    gold_dir = repo_root / "profile_labeled_data"
    output_dir = repo_root / "results" / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(" EVALUATION: TABLE 1 (PROFILE EXTRACTION PERFORMANCE)")
    print("=" * 80)

    target_profiles = list((extracted_dir / model_dir_name).glob("*/*.json"))
    if not target_profiles:
        print(f"[!] No extracted profiles found at: {extracted_dir / model_dir_name}")
        print("    Please run Code Cell 1 in 'catalyst_ou_pipeline.ipynb' to generate profiles first.")
        return None

    print(f"[*] Found {len(target_profiles)} extracted profiles. Loading SBERT ('all-mpnet-base-v2')...")
    sbert_model = SentenceTransformer("all-mpnet-base-v2")
    data_map = build_data_map(str(extracted_dir), str(gold_dir), model_dir_name)

    individual_evals = []
    for item in data_map:
        res = evaluate_single_profile(
            item["llm_path"],
            item["gold_path"],
            sbert_model,
            tau=0.65,
            thinking_patterns_tau=0.50
        )
        individual_evals.append(res)

    if not individual_evals:
        print("[!] No matching profiles could be paired with gold ground truth.")
        return None

    # Aggregate counts
    keys = sorted({k for res in individual_evals for k in res["field_metrics"].keys()})
    micro_counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in keys}
    psas_values = [r["profile_scores"]["PSAS"] for r in individual_evals]
    summary_cosines = [r["profile_scores"]["summary_cosine"] for r in individual_evals]

    for res in individual_evals:
        for k in keys:
            if k in res["field_metrics"]:
                m = res["field_metrics"][k]
                micro_counts[k]["tp"] += m.get("tp_total", 0)
                micro_counts[k]["fp"] += m.get("fp", 0)
                micro_counts[k]["fn"] += m.get("fn", 0)

    print(f"\n--- Table 1: Field-Level Extraction Performance (N = {len(individual_evals)}) ---")
    print(f"{'Field Name':<30} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 70)

    total_tp = sum(v["tp"] for v in micro_counts.values())
    total_fp = sum(v["fp"] for v in micro_counts.values())
    total_fn = sum(v["fn"] for v in micro_counts.values())

    table1_rows = []
    for k in keys:
        tp, fp, fn = micro_counts[k]["tp"], micro_counts[k]["fp"], micro_counts[k]["fn"]
        p, r, f1 = compute_f1(tp, fp, fn)
        print(f"{k:<30} | {p*100:6.1f}%    | {r*100:6.1f}%    | {f1*100:6.1f}%")
        table1_rows.append((k, p*100, r*100, f1*100))

    mean_summary_cos = float(np.mean(summary_cosines))
    print(f"{'Researcher Summary (Sim)':<30} | {'--':<10} | {'--':<10} | {mean_summary_cos:.4f}")
    
    overall_p, overall_r, overall_f1 = compute_f1(total_tp, total_fp, total_fn)
    print("-" * 70)
    print(f"{'OVERALL':<30} | {overall_p*100:6.1f}%    | {overall_r*100:6.1f}%    | {overall_f1*100:6.1f}%")
    print(f"Mean PSAS: {float(np.mean(psas_values)):.4f} +/- {float(np.std(psas_values)):.4f}")
    print("=" * 80)

    # Save summary json
    summary_path = output_dir / "comprehensive_evaluation_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen/Qwen3.6-35B-A3B",
            "num_profiles": len(individual_evals),
            "PSAS_mean": float(np.mean(psas_values)),
            "PSAS_std": float(np.std(psas_values)),
            "summary_cosine_mean": mean_summary_cos,
            "overall_precision": overall_p,
            "overall_recall": overall_r,
            "overall_f1": overall_f1,
            "per_key": micro_counts
        }, f, indent=2)
    print(f"[+] Saved evaluation metrics to: {summary_path}")
    return table1_rows


def evaluate_table2_and_3():
    """Evaluate Experiment 2 collaboration predictions."""
    exp2_dir = repo_root / "results" / "exp2_qwen3.6-35b-a3b"
    gt_dir = repo_root / "collaboration_ground_truth"

    print("\n" + "=" * 80)
    print(" EVALUATION: TABLES 2 & 3 (HISTORICAL COLLABORATION BACKTESTING)")
    print("=" * 80)

    if not exp2_dir.exists() or not list(exp2_dir.glob("*.json")):
        print(f"[!] No Experiment 2 predictions found in: {exp2_dir}")
        print("    Please run Code Cell 2 in 'catalyst_ou_pipeline.ipynb' to generate predictions first.")
        return

    print(f"[*] Found predictions in {exp2_dir}. Computing alignment metrics against ground truth...")
    # Reference cases
    cases = [
        ("case_001_ebert_xiao_cs_bio", "Cross-Discipline"),
        ("case_002_zhao_cheng_cs_ece", "Cross-Discipline"),
        ("case_003_cai_song_psych_psych", "Same-Discipline"),
        ("case_004_yeary_havlicek_ece_ece", "Same-Discipline"),
        ("case_005_radhakrishnan_antonio_cs_cs", "Same-Discipline"),
    ]

    sbert_model = SentenceTransformer("all-mpnet-base-v2")
    similarities = []
    hits_45 = 0
    hits_50 = 0
    mode_hits = {
        "Method-Application Synergies": 0,
        "Shared Domains": 0,
        "Cross-Domain Fusion Topics": 0,
        "Complementary Technique Synergies": 0,
        "Data-Method Synergies": 0,
    }

    for cid, ctype in cases:
        pred_file = exp2_dir / f"{cid}_prediction.json"
        gt_file = gt_dir / f"{cid}_gt.json"
        if not pred_file.exists() or not gt_file.exists():
            continue

        with open(pred_file, "r", encoding="utf-8") as f: pred = json.load(f)
        with open(gt_file, "r", encoding="utf-8") as f: gt = json.load(f)

        pred_summary = pred.get("Summary Collaboration Themes", "")
        gt_summary = gt.get("Summary Collaboration Themes", "")

        if pred_summary and gt_summary:
            emb_p = sbert_model.encode(pred_summary, convert_to_tensor=True)
            emb_g = sbert_model.encode(gt_summary, convert_to_tensor=True)
            cos = float(torch.nn.functional.cosine_similarity(emb_p.unsqueeze(0), emb_g.unsqueeze(0)).item())
            similarities.append(cos)
            if cos >= 0.45: hits_45 += 1
            if cos >= 0.50: hits_50 += 1

        for mode_key in mode_hits:
            p_items = pred.get(mode_key, [])
            g_items = gt.get(mode_key, [])
            if p_items and g_items:
                emb_p_items = sbert_model.encode(p_items, convert_to_tensor=True)
                emb_g_items = sbert_model.encode(g_items, convert_to_tensor=True)
                # Max pairwise similarity
                sim_mat = torch.mm(
                    torch.nn.functional.normalize(emb_p_items, p=2, dim=1),
                    torch.nn.functional.normalize(emb_g_items, p=2, dim=1).t()
                )
                max_sim = float(torch.max(sim_mat).item())
                if max_sim >= 0.45:
                    mode_hits[mode_key] += 1

    n_cases = len(similarities) if similarities else 1
    avg_sim = float(np.mean(similarities)) if similarities else 0.0
    r_45 = (hits_45 / n_cases) * 100
    r_50 = (hits_50 / n_cases) * 100

    print("\n--- Table 2: Backtesting Results on Historical Coauthorship Reasoning ---")
    print(f"Avg. Sim: {avg_sim:.4f} | R@0.45: {r_45:.1f}% | R@0.50: {r_50:.1f}%")

    print("\n--- Table 3: Reasoning Mode Alignment (% at tau = 0.45) ---")
    for mode, count in mode_hits.items():
        pct = (count / n_cases) * 100
        print(f"  {mode:<35}: {pct:.1f}%")
    print("=" * 80)


def print_latex_templates():
    print("\n" + "=" * 80)
    print(" LATEX TABLE SNIPPETS FOR PAPER (experiments.tex)")
    print("=" * 80)
    print(r"""
% Snippet for Table 1 (Update in experiments.tex):
% Add Qwen 3.6-35B-A3B MoE column alongside Llama 3.1 8B and Qwen 3.8 27B:
\begin{table*}[htbp]
\centering
\caption{Experiment~1 results: field-level profile extraction performance across 50 evaluated researchers ($N = 50$).}
\begin{tabular}{lccccccccc}
    \hline
    \textbf{Field Name} & \multicolumn{3}{c}{\textbf{Meta-Llama 3.1 8B}} & \multicolumn{3}{c}{\textbf{Qwen 3.8 27B (Dense)}} & \multicolumn{3}{c}{\textbf{Qwen 3.6 35B-A3B (MoE)}} \\
    & \textbf{Prec} & \textbf{Rec} & \textbf{F1} & \textbf{Prec} & \textbf{Rec} & \textbf{F1} & \textbf{Prec} & \textbf{Rec} & \textbf{F1} \\ \hline
    Research Domains        & 49.1\% & 47.7\% & 48.4\% & 51.1\% & 60.3\% & 55.3\% & [P]\% & [R]\% & [F1]\% \\
    Techniques Used         & 44.6\% & 34.0\% & 38.6\% & 50.9\% & 47.1\% & 48.9\% & [P]\% & [R]\% & [F1]\% \\
    Datasets \& Platforms   & 22.1\% & 31.0\% & 25.8\% & 28.4\% & 47.4\% & 35.5\% & [P]\% & [R]\% & [F1]\% \\
    Application Areas       & 17.5\% & 14.5\% & 15.9\% & 21.4\% & 19.8\% & 20.5\% & [P]\% & [R]\% & [F1]\% \\
    Thinking Patterns       & 50.7\% & 50.5\% & 50.6\% & 49.8\% & 50.0\% & 49.9\% & [P]\% & [R]\% & [F1]\% \\ 
    Researcher Summary (Sim)& --     & --     & 0.7913 & --     & --     & 0.8270 & --    & --    & [SIM] \\ \hline
    \textbf{Overall}        & 36.8\% & 35.5\% & 35.9\% & 40.3\% & 44.9\% & 42.0\% & \textbf{[P]\%} & \textbf{[R]\%} & \textbf{[F1]\%} \\ \hline
\end{tabular}
\label{tab:exp1_results}
\end{table*}
""")


if __name__ == "__main__":
    if not HAS_EVAL_DEPS:
        print(f"[!] Evaluation dependencies missing: {MISSING_DEP}")
        print("    Install required packages via: pip install sentence-transformers numpy")
    else:
        evaluate_table1()
        evaluate_table2_and_3()
    print_latex_templates()
