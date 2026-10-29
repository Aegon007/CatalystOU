import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
from proj_test.llm_reasoner import call_llm_reasoner
from utils.data_utils import COLLABORATION_CATEGORIES
from utils.llm_utils import get_llm_config
import utils.profile_io as profile_io

async def run_negative_contrastive_experiment():
    print("=" * 80)
    print("CONTRASTIVE NEGATIVE PAIR EXPERIMENT EXECUTION")
    print("=" * 80)

    # 1. Load cases
    pos_file = Path("cases/cases.json") if Path("cases/cases.json").exists() else Path("cases.json")
    neg_file = Path("cases/negative_cases.json") if Path("cases/negative_cases.json").exists() else Path("negative_cases.json")
    with open(pos_file, "r", encoding="utf-8") as f:
        pos_cases = json.load(f)
    with open(neg_file, "r", encoding="utf-8") as f:
        neg_cases = json.load(f)

    out_dir = Path("results/exp2_negative")
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = get_llm_config("qwen3.8-27b")
    model = SentenceTransformer("all-mpnet-base-v2")

    # 2. Run LLM Reasoner for Negative Pairs (if not already cached)
    print(f"\n[1/3] Generating Synergy Predictions for {len(neg_cases)} Negative Control Pairs...")
    for idx, case in enumerate(neg_cases, 1):
        out_p = Path(case["out_path"])
        if out_p.exists():
            print(f"  [{idx}/{len(neg_cases)}] Loading cached prediction for {case['id']}")
            continue

        print(f"  [{idx}/{len(neg_cases)}] Querying LLM Reasoner for: {case['id']} ({case['pair_desc']})...")
        prof_a = profile_io.load_profile_json(case["a_path"])
        prof_b = profile_io.load_profile_json(case["b_path"])

        pred = await call_llm_reasoner(prof_a, prof_b, llm_cfg)
        profile_io.save_json(pred, str(out_p))
        print(f"  [{idx}/{len(neg_cases)}] Saved prediction to {out_p}")

    # 3. Compute Profile-to-Profile Baseline Semantic Affinity (Pre-Collaboration Divergence)
    print("\n[2/3] Computing Pre-Collaboration Baseline Affinity (A vs B Summary Distance)...")
    def get_summary_sim(p_a_path, p_b_path):
        p_a = profile_io.load_profile_json(p_a_path)
        p_b = profile_io.load_profile_json(p_b_path)
        s_a = p_a.get("summary", "") or p_a.get("Summary Description", "")
        s_b = p_b.get("summary", "") or p_b.get("Summary Description", "")
        e_a = model.encode([s_a], normalize_embeddings=True)
        e_b = model.encode([s_b], normalize_embeddings=True)
        return float(np.dot(e_a, e_b.T)[0, 0])

    pos_baseline_sims = []
    for c in pos_cases:
        sim = get_summary_sim(c["a_path"], c["b_path"])
        pos_baseline_sims.append(sim)
        print(f"  Positive Pair {c['id']}: Pre-Collab Baseline Cosine = {sim:.4f}")

    neg_baseline_sims = []
    for c in neg_cases:
        sim = get_summary_sim(c["a_path"], c["b_path"])
        neg_baseline_sims.append(sim)
        print(f"  Negative Pair {c['id']}: Pre-Collab Baseline Cosine = {sim:.4f}")

    # 4. Contrastive Cross-Validation / Discriminator Test
    # For each Ground Truth (positive co-authored paper), compute:
    # - Similarity with the True Positive Pair's predicted synergy
    # - Similarity with each of the 5 Negative Control Pairs' predicted synergies
    def to_text(val):
        if isinstance(val, list):
            return " ".join(str(x) for x in val)
        return str(val) if val is not None else ""

    print("\n[3/3] Contrastive Ground Truth Discriminator Matrix...")
    discriminator_results = []
    for pos_idx, pos_case in enumerate(pos_cases):
        gt = profile_io.load_profile_json(pos_case["gt_path"])
        gt_summary = to_text(gt.get("Summary Collaboration Themes", ""))
        gt_emb = model.encode([gt_summary], normalize_embeddings=True)

        # True positive pair prediction
        true_pred = profile_io.load_profile_json(pos_case["out_path"])
        true_sum = to_text(true_pred.get("Summary Collaboration Themes", ""))
        true_sim = float(np.dot(model.encode([true_sum], normalize_embeddings=True), gt_emb.T)[0, 0])

        # Negative pairs predictions
        neg_sims = []
        for neg_case in neg_cases:
            neg_pred = profile_io.load_profile_json(neg_case["out_path"])
            neg_sum = to_text(neg_pred.get("Summary Collaboration Themes", ""))
            n_sim = float(np.dot(model.encode([neg_sum], normalize_embeddings=True), gt_emb.T)[0, 0])
            neg_sims.append((neg_case["id"], n_sim))

        mean_neg_sim = float(np.mean([s for _, s in neg_sims]))
        max_neg_sim = float(max([s for _, s in neg_sims]))
        contrastive_margin = true_sim - mean_neg_sim
        is_discriminated = true_sim > max_neg_sim

        print(f"\nGround Truth: {pos_case['id']}")
        print(f"  True Positive Pair Alignment     : {true_sim:.4f}")
        print(f"  Mean Negative Control Alignment  : {mean_neg_sim:.4f}")
        print(f"  Max Negative Control Alignment   : {max_neg_sim:.4f}")
        print(f"  Contrastive Margin (Delta)       : {contrastive_margin:+.4f} (Separation: {'SUCCESS' if is_discriminated else 'OVERLAP'})")

        discriminator_results.append({
            "gt_id": pos_case["id"],
            "true_positive_sim": true_sim,
            "mean_negative_sim": mean_neg_sim,
            "max_negative_sim": max_neg_sim,
            "contrastive_margin": contrastive_margin,
            "is_discriminated": is_discriminated,
            "negative_similarities": dict(neg_sims)
        })

    # Summary Statistics
    mean_pos_baseline = float(np.mean(pos_baseline_sims))
    mean_neg_baseline = float(np.mean(neg_baseline_sims))
    mean_true_sim = float(np.mean([r["true_positive_sim"] for r in discriminator_results]))
    mean_neg_control_sim = float(np.mean([r["mean_negative_sim"] for r in discriminator_results]))
    mean_margin = float(np.mean([r["contrastive_margin"] for r in discriminator_results]))
    success_rate = float(np.mean([1 if r["is_discriminated"] else 0 for r in discriminator_results]) * 100)

    print("\n" + "=" * 80)
    print("FINAL CONTRASTIVE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Pre-Collab Affinity - Positive Pairs : {mean_pos_baseline:.4f}")
    print(f"Pre-Collab Affinity - Negative Pairs : {mean_neg_baseline:.4f}")
    print(f"Mean True Positive Alignment to GT   : {mean_true_sim:.4f}")
    print(f"Mean Negative Pair Alignment to GT   : {mean_neg_control_sim:.4f}")
    print(f"Average Contrastive Margin (Delta)   : {mean_margin:+.4f} (+{((mean_true_sim - mean_neg_control_sim)/mean_neg_control_sim)*100:.1f}%)")
    print(f"Ground Truth Discrimination Accuracy : {success_rate:.1f}%")

    # Save artifact
    out_file = out_dir / "contrastive_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "mean_positive_baseline_affinity": mean_pos_baseline,
                "mean_negative_baseline_affinity": mean_neg_baseline,
                "mean_true_positive_alignment": mean_true_sim,
                "mean_negative_alignment": mean_neg_control_sim,
                "mean_contrastive_margin": mean_margin,
                "discrimination_accuracy_pct": success_rate,
            },
            "cases": discriminator_results
        }, f, indent=2)

    print(f"\nSaved contrastive benchmark results to: {out_file}")

if __name__ == "__main__":
    asyncio.run(run_negative_contrastive_experiment())
