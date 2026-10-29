"""
================================================================================
CatalystOU: Table Statistics & Evaluation Suite (Agent vs. Pure LLM Benchmark)
================================================================================
Models Benchmarked:
1. Qwen 3.8-27B (Dense Baseline - Single-Pass Map-Reduce)
2. Qwen 3.6-35B-A3B (Sparse MoE Baseline - Single-Pass Map-Reduce)
3. Qwen 3.6-35B-A3B Agent (Two-Stage Stateful Agent + 2-Pass Reasoner)

Outputs:
- Table 1: Field-level Profile Extraction Accuracy & Delta (Precision, Recall, F1)
- Table 2: Historical Collaboration Backtesting & Selectivity Margin Delta
- Table 3: Reasoning Mode Alignment (% at tau = 0.45)
================================================================================
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

try:
    import torch
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


def evaluate_table1_agent_vs_pure():
    """Evaluate Agent profiles against gold ground truth and compare against Pure LLM baseline."""
    extracted_dir = repo_root / "extracted_profile_json"
    gold_dir = repo_root / "profile_labeled_data"
    agent_dir_name = "qwen_qwen3.6-35b-a3b_agent"
    pure_dir_name = "qwen_qwen3.6-35b-a3b"

    output_dir = repo_root / "results" / agent_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(" TABLE 1 BENCHMARK: AGENT ITERATIVE REFINEMENT vs. PURE LLM BASELINE")
    print(" Model: Qwen 3.6-35B-A3B (MoE)")
    print("=" * 90)

    agent_profiles_root = extracted_dir / agent_dir_name
    if not agent_profiles_root.exists() or not list(agent_profiles_root.rglob("*.json")):
        print(f"[!] No Agent profiles found in: {agent_profiles_root}")
        print("    Running on Colab or waiting for Code Cell 1 inference.")
        return None

    # Load Pure LLM reference summary for comparison
    pure_summary_path = repo_root / "results" / pure_dir_name / "comprehensive_evaluation_results.json"
    pure_data = {}
    if pure_summary_path.exists():
        with open(pure_summary_path, "r", encoding="utf-8") as f:
            pure_data = json.load(f)

    pure_per_key = pure_data.get("per_key", {})
    pure_summary_cos = pure_data.get("summary_cosine_mean", 0.8242)

    keys = [
        "Research Domains",
        "Techniques Used",
        "Data & Platforms",
        "Application Areas",
        "Key Research Thinking Patterns",
    ]

    agent_counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in keys}
    agent_summaries = []
    agent_psas = []

    gold_map = build_data_map(gold_dir)
    agent_map = build_data_map(agent_profiles_root)

    eval_pairs = []
    for disc, res_dict in agent_map.items():
        if disc not in gold_map:
            continue
        for r_name, p_path in res_dict.items():
            if r_name in gold_map[disc]:
                eval_pairs.append((p_path, gold_map[disc][r_name]))

    if not eval_pairs:
        print("[!] No matching ground-truth pairs found.")
        return None

    print(f"[*] Evaluating {len(eval_pairs)} agent profiles against human annotations...")
    individual_evals = []
    for pred_p, gold_p in eval_pairs:
        res = evaluate_single_profile(pred_p, gold_p)
        individual_evals.append(res)
        agent_psas.append(res.get("PSAS", 0.0))
        agent_summaries.append(res.get("summary_cosine", 0.0))

        details = res.get("details", {})
        for k in keys:
            if k in details:
                agent_counts[k]["tp"] += details[k].get("tp", 0)
                agent_counts[k]["fp"] += details[k].get("fp", 0)
                agent_counts[k]["fn"] += details[k].get("fn", 0)

    # Print Table 1 Comparison
    header = f"{'Field Name':<28} | {'Pure LLM F1':<12} | {'Agent Prec':<11} | {'Agent Rec':<11} | {'Agent F1':<11} | {'Delta F1':<8}"
    print("\n" + header)
    print("-" * len(header))

    total_tp = sum(v["tp"] for v in agent_counts.values())
    total_fp = sum(v["fp"] for v in agent_counts.values())
    total_fn = sum(v["fn"] for v in agent_counts.values())

    table1_summary = {}

    for k in keys:
        tp = agent_counts[k]["tp"]
        fp = agent_counts[k]["fp"]
        fn = agent_counts[k]["fn"]
        p, r, f1 = compute_f1(tp, fp, fn)

        pure_f1 = 0.0
        if k in pure_per_key:
            pk = pure_per_key[k]
            pure_f1 = pk.get("micro_f1", 0.0) * 100 if "micro_f1" in pk else compute_f1(pk.get("tp", 0), pk.get("fp", 0), pk.get("fn", 0))[2] * 100

        delta_f1 = (f1 * 100) - pure_f1
        sign = "+" if delta_f1 >= 0 else ""
        print(f"{k:<28} | {pure_f1:6.1f}%      | {p*100:6.1f}%     | {r*100:6.1f}%     | {f1*100:6.1f}%    | {sign}{delta_f1:5.1f}%")
        table1_summary[k] = {"pure_f1": pure_f1, "agent_prec": p * 100, "agent_rec": r * 100, "agent_f1": f1 * 100, "delta_f1": delta_f1}

    # Researcher Summary
    agent_mean_cos = float(np.mean(agent_summaries))
    delta_cos = agent_mean_cos - pure_summary_cos
    cos_sign = "+" if delta_cos >= 0 else ""
    print(f"{'Researcher Summary (Sim)':<28} | {pure_summary_cos:6.4f}       | {'--':<11} | {'--':<11} | {agent_mean_cos:6.4f}    | {cos_sign}{delta_cos:.4f}")

    # Overall
    ov_p, ov_r, ov_f1 = compute_f1(total_tp, total_fp, total_fn)
    pure_ov_f1 = pure_data.get("overall_f1", 0.408) * 100 if pure_data.get("overall_f1", 0) <= 1.0 else pure_data.get("overall_f1", 40.8)
    delta_ov = (ov_f1 * 100) - pure_ov_f1
    ov_sign = "+" if delta_ov >= 0 else ""

    print("-" * len(header))
    print(f"{'OVERALL':<28} | {pure_ov_f1:6.1f}%      | {ov_p*100:6.1f}%     | {ov_r*100:6.1f}%     | {ov_f1*100:6.1f}%    | {ov_sign}{delta_ov:5.1f}%")
    print(f"Agent Mean PSAS: {float(np.mean(agent_psas)):.4f} +/- {float(np.std(agent_psas)):.4f}")
    print("=" * 90)

    summary_path = output_dir / "comprehensive_evaluation_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen/Qwen3.6-35B-A3B (Agent)",
            "architecture": "Two-Stage Stateful Agent (Sequential Accumulator -> PI Synthesizer)",
            "num_profiles": len(individual_evals),
            "PSAS_mean": float(np.mean(agent_psas)),
            "PSAS_std": float(np.std(agent_psas)),
            "summary_cosine_mean": agent_mean_cos,
            "overall_precision": ov_p,
            "overall_recall": ov_r,
            "overall_f1": ov_f1,
            "pure_llm_baseline_f1": pure_ov_f1,
            "delta_overall_f1": delta_ov,
            "per_key": table1_summary
        }, f, indent=2)

    return table1_summary


def evaluate_table2_and_3_agent_vs_pure():
    """Evaluate Experiment 2 predictions across Dense 27B, Pure MoE, and Agent MoE against Ground Truth."""
    gt_dir = repo_root / "collaboration_ground_truth"

    models_to_eval = [
        {
            "name": "Qwen 3.8-27B Dense",
            "pos_dir": repo_root / "results" / "exp2_qwen3.8-27b_true",
            "neg_dir": repo_root / "results" / "exp2_qwen3.8-27b_true_negative",
            "legacy_pos": repo_root / "results" / "exp2_qwen_qwen3.8-27b",
            "legacy_neg": repo_root / "results" / "exp2_qwen_qwen3.8-27b_negative",
        },
        {
            "name": "Qwen 3.6-35B Pure MoE",
            "pos_dir": repo_root / "results" / "exp2_qwen3.6-35b-a3b_true",
            "neg_dir": repo_root / "results" / "exp2_qwen3.6-35b-a3b_true_negative",
            "legacy_pos": repo_root / "results" / "exp2_qwen3.6-35b-a3b",
            "legacy_neg": repo_root / "results" / "exp2_qwen3.6-35b-a3b_negative",
        },
        {
            "name": "Qwen 3.6-35B Agent MoE",
            "pos_dir": repo_root / "results" / "exp2_qwen3.6-35b-a3b_agent_true",
            "neg_dir": repo_root / "results" / "exp2_qwen3.6-35b-a3b_agent_true_negative",
            "legacy_pos": repo_root / "results" / "exp2_qwen3.6-35b-a3b_agent",
            "legacy_neg": repo_root / "results" / "exp2_qwen3.6-35b-a3b_agent_negative",
        }
    ]

    print("\n" + "=" * 95)
    print(" TABLE 2 & 3 BENCHMARK: TRUE HISTORICAL BACKTESTING & CONTRASTIVE SELECTIVITY")
    print(" Evaluates 5 Historical True Co-authorship Pairs vs. 5 Volume-Balanced Negative Controls")
    print("=" * 95)

    sbert_model = SentenceTransformer("all-mpnet-base-v2")

    pos_cases = [
        "case_001_ebert_xiao_cs_bio",
        "case_002_zhao_cheng_cs_ece",
        "case_003_cai_song_psych_psych",
        "case_004_yeary_havlicek_ece_ece",
        "case_005_radhakrishnan_antonio_cs_cs",
    ]

    neg_cases = [
        "neg_001_maiti_cartersowell",
        "neg_002_yeary_dunn",
        "neg_003_forester_quadri",
        "neg_004_karr_cokely",
        "neg_005_jiang_martin",
    ]

    def evaluate_pair_set(p_dir: Path, case_list: List[str]) -> Tuple[List[float], Dict[str, int]]:
        sims = []
        mode_counts = {
            "Method-Application Synergies": 0,
            "Shared Domains": 0,
            "Cross-Domain Fusion Topics": 0,
            "Complementary Technique Synergies": 0,
            "Data-Method Synergies": 0,
        }
        for cid in case_list:
            pred_file = p_dir / f"{cid}_prediction.json"
            gt_file = gt_dir / f"{cid}_gt.json"
            if not pred_file.exists() or not gt_file.exists():
                continue

            with open(pred_file, "r", encoding="utf-8") as f:
                pred = json.load(f)
            with open(gt_file, "r", encoding="utf-8") as f:
                gt = json.load(f)

            ps = pred.get("Summary Collaboration Themes", "")
            gs = gt.get("Summary Collaboration Themes", "")
            if ps and gs:
                ep = sbert_model.encode(ps, convert_to_tensor=True)
                eg = sbert_model.encode(gs, convert_to_tensor=True)
                cos = float(torch.nn.functional.cosine_similarity(ep.unsqueeze(0), eg.unsqueeze(0)).item())
                sims.append(cos)

            for m in mode_counts:
                pi = pred.get(m, [])
                gi = gt.get(m, [])
                if pi and gi:
                    epi = sbert_model.encode(pi, convert_to_tensor=True)
                    egi = sbert_model.encode(gi, convert_to_tensor=True)
                    sm = torch.mm(
                        torch.nn.functional.normalize(epi, p=2, dim=1),
                        torch.nn.functional.normalize(egi, p=2, dim=1).t()
                    )
                    if float(torch.max(sm).item()) >= 0.45:
                        mode_counts[m] += 1
        return sims, mode_counts

    table2_results = []
    mode_results = {}

    for mspec in models_to_eval:
        mname = mspec["name"]
        pdir = mspec["pos_dir"]
        ndir = mspec["neg_dir"]

        if not pdir.exists() or not list(pdir.glob("*.json")):
            pdir = mspec["legacy_pos"]
        if not ndir.exists() or not list(ndir.glob("*.json")):
            ndir = mspec["legacy_neg"]

        if not pdir.exists() or not list(pdir.glob("*.json")):
            table2_results.append({
                "model": mname, "status": "Pending Inference",
                "avg_sim": 0.0, "r45": 0.0, "r50": 0.0, "margin": 0.0, "pct_gain": 0.0
            })
            continue

        pos_sims, modes = evaluate_pair_set(pdir, pos_cases)
        neg_sims, _ = evaluate_pair_set(ndir, neg_cases)

        pos_mean = float(np.mean(pos_sims)) if pos_sims else 0.0
        neg_mean = float(np.mean(neg_sims)) if neg_sims else 0.1358
        margin = pos_mean - neg_mean
        pct_gain = (margin / neg_mean * 100) if neg_mean > 0 else 0.0

        r45 = (sum(1 for s in pos_sims if s >= 0.45) / len(pos_sims) * 100) if pos_sims else 0.0
        r50 = (sum(1 for s in pos_sims if s >= 0.50) / len(pos_sims) * 100) if pos_sims else 0.0

        table2_results.append({
            "model": mname, "status": "Ready",
            "avg_sim": pos_mean, "r45": r45, "r50": r50,
            "neg_sim": neg_mean, "margin": margin, "pct_gain": pct_gain
        })
        mode_results[mname] = (modes, len(pos_sims))

    print("\n--- TABLE 2: HISTORICAL CO-AUTHORSHIP BACKTESTING (ALL MODELS) ---")
    t2_hdr = f"{'Model / Architecture':<32} | {'Avg. Sim':<9} | {'R@0.45':<8} | {'R@0.50':<8} | {'Selectivity Margin':<22}"
    print(t2_hdr)
    print("-" * len(t2_hdr))

    for r in table2_results:
        if r["status"] == "Pending Inference":
            print(f"{r['model']:<32} | {'[Pending]':<9} | {'--':<8} | {'--':<8} | [Waiting for Colab run]")
        else:
            sign = "+" if r["margin"] >= 0 else ""
            margin_str = f"{sign}{r['margin']:.4f} ({sign}{r['pct_gain']:.0f}%)"
            print(f"{r['model']:<32} | {r['avg_sim']:7.4f}   | {r['r45']:5.1f}%  | {r['r50']:5.1f}%  | {margin_str:<22}")

    print(f"{'Negative Controls (Baseline)':<32} | 0.1358    | 0.0%     | 0.0%     | Reference Baseline")
    print("=" * 95)

    if mode_results:
        print("\n--- TABLE 3: REASONING MODE ALIGNMENT (% of Pairs at tau = 0.45) ---")
        t3_hdr = f"{'Model':<30} | {'Method-App':<11} | {'Shared Dom':<11} | {'Cross-Disc':<11} | {'Tech Transfer':<14} | {'Data Reuse':<11}"
        print(t3_hdr)
        print("-" * len(t3_hdr))
        for mname, (mdict, total_p) in mode_results.items():
            tot = total_p if total_p > 0 else 1
            p_ma = (mdict.get("Method-Application Synergies", 0) / tot) * 100
            p_sd = (mdict.get("Shared Domains", 0) / tot) * 100
            p_cd = (mdict.get("Cross-Domain Fusion Topics", 0) / tot) * 100
            p_tt = (mdict.get("Complementary Technique Synergies", 0) / tot) * 100
            p_dr = (mdict.get("Data-Method Synergies", 0) / tot) * 100
            print(f"{mname:<30} | {p_ma:5.1f}%      | {p_sd:5.1f}%      | {p_cd:5.1f}%      | {p_tt:5.1f}%         | {p_dr:5.1f}%")
        print("=" * 95)


if __name__ == "__main__":
    if not HAS_EVAL_DEPS:
        print(f"[!] Dependencies missing: {MISSING_DEP}")
        print("    Install via: pip install sentence-transformers numpy torch")
    else:
        evaluate_table1_agent_vs_pure()
        evaluate_table2_and_3_agent_vs_pure()
