"""
Generate Publication-Quality Figures for CatalystOU Research Paper and Presentations.
Outputs high-resolution 300-DPI charts into figures/ directory.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# Ensure figures output directory exists
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set high-level publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

# Curated modern color palettes
PALETTE_MODELS = ["#718096", "#3182CE", "#38A169"]  # Gray (Baseline), Blue (Llama 8B), Green (Qwen 27B)
PALETTE_DEPTS = ["#319795", "#3182CE", "#805AD5", "#D69E2E", "#DD6B20"]  # Bio, CS, ECE, Math, Psych
PALETTE_CONTRAST = ["#2B6CB0", "#E53E3E"]  # Positive (Blue), Negative (Red)


def generate_figure1_model_comparison():
    """Figure 1: 3-Phase Evolution: Baseline GPT-5 Nano vs Llama 3.1 8B vs Qwen 3.8 27B."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    metrics = [
        "Global PSAS",
        "Summary Cosine",
        "Research Domains",
        "Techniques Used",
        "Thinking Patterns F1"
    ]

    # Scores (Apples-to-apples & benchmark means)
    gpt5_scores = [0.4276, 0.7160, 0.6410, 0.5039, 0.2955]
    llama_scores = [0.7092, 0.7913, 0.9241, 0.8721, 0.5063]
    qwen_scores = [0.6817, 0.8270, 0.8540, 0.8037, 0.4894]

    x = np.arange(len(metrics))
    width = 0.26

    rects1 = ax.bar(x - width, gpt5_scores, width, label="Baseline GPT-5 Nano (10 Bio)", color=PALETTE_MODELS[0], alpha=0.9, edgecolor="black", linewidth=0.6)
    rects2 = ax.bar(x, llama_scores, width, label="Meta-Llama 3.1 8B (Full 50)", color=PALETTE_MODELS[1], alpha=0.9, edgecolor="black", linewidth=0.6)
    rects3 = ax.bar(x + width, qwen_scores, width, label="Qwen 3.8 27B (Full 50)", color=PALETTE_MODELS[2], alpha=0.9, edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Score / Similarity / F1", fontweight="bold")
    ax.set_title("CatalystOU Pipeline Evolution: Profile Extraction Fidelity", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="semibold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower right")

    # Annotate values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig1_pipeline_evolution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure2_discipline_breakdown():
    """Figure 2: Cross-Disciplinary Consistency across 5 Departments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    depts = ["Biology", "Computer\nScience", "ECE", "Mathematics", "Psychology"]
    
    # Llama 3.1 8B discipline data
    psas_means = [0.7302, 0.7063, 0.7324, 0.6558, 0.7212]
    psas_stds = [0.1245, 0.1215, 0.1208, 0.1086, 0.1189]
    cos_means = [0.8124, 0.7850, 0.7963, 0.7712, 0.7918]
    cos_stds = [0.0482, 0.0512, 0.0495, 0.0612, 0.0541]

    # Bar chart 1: PSAS
    bars1 = ax1.bar(depts, psas_means, yerr=psas_stds, capsize=4, color=PALETTE_DEPTS, alpha=0.88, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Profile Semantic Alignment Score (PSAS)", fontweight="bold")
    ax1.set_title("Profile Alignment across Disciplines", fontweight="bold")
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval - 0.12, f"{yval:.3f}", ha="center", va="bottom", color="white", fontweight="bold", fontsize=9.5)

    # Bar chart 2: Summary Cosine
    bars2 = ax2.bar(depts, cos_means, yerr=cos_stds, capsize=4, color=PALETTE_DEPTS, alpha=0.88, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("Summary Narrative Cosine Similarity", fontweight="bold")
    ax2.set_title("Summary Alignment across Disciplines", fontweight="bold")
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval - 0.12, f"{yval:.3f}", ha="center", va="bottom", color="white", fontweight="bold", fontsize=9.5)

    fig.suptitle("Experiment 1: Zero-Shot Generalization Across Academic Domains", fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig2_discipline_breakdown.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure3_tau_sensitivity():
    """Figure 3: Threshold Sensitivity Sweep (The Generative Prediction Discovery)."""
    fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=300)

    taus = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    mean_mas = [0.3082, 0.2316, 0.1480, 0.0984, 0.0518, 0.0136]
    shared_domains_f1 = [0.4667, 0.3222, 0.3222, 0.2333, 0.1000, 0.0000]
    shared_apps_f1 = [0.2722, 0.2722, 0.2722, 0.1333, 0.0889, 0.0444]
    method_app_f1 = [0.2444, 0.1500, 0.0500, 0.0000, 0.0000, 0.0000]

    ax1.plot(taus, mean_mas, marker="o", linewidth=2.5, color="#1A365D", label="Mean MAS (Overall Alignment)")
    ax1.plot(taus, shared_domains_f1, marker="s", linewidth=2.0, linestyle="--", color="#2B6CB0", label="Shared Domains F1")
    ax1.plot(taus, shared_apps_f1, marker="^", linewidth=2.0, linestyle="-.", color="#38A169", label="Shared Application Areas F1")
    ax1.plot(taus, method_app_f1, marker="d", linewidth=2.0, linestyle=":", color="#D69E2E", label="Method-Application F1")

    # Annotate the cliff effect
    ax1.axvspan(0.58, 0.66, color="#FED7D7", alpha=0.45, label="Literal Entity Regime (Cliff Effect)")
    ax1.axvspan(0.44, 0.51, color="#C6F6D5", alpha=0.40, label="Optimal Generative Prediction Zone")

    ax1.set_xlabel("Semantic Cosine Threshold (τ)", fontweight="bold")
    ax1.set_ylabel("Score / F1 Measure", fontweight="bold")
    ax1.set_title("Experiment 2: Threshold Sensitivity Analysis (The Generative Prediction Cliff)", fontweight="bold", pad=15)
    ax1.set_xticks(taus)
    ax1.set_ylim(-0.02, 0.52)
    ax1.grid(True)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig3_exp2_tau_sensitivity.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure4_contrastive_discrimination():
    """Figure 4: Negative Control Contrastive Benchmark & Ground Truth Discriminator."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    cases = [
        "Case 1 (CS + Bio)\nEbert & Xiao",
        "Case 2 (CS + ECE)\nZhao & Cheng",
        "Case 3 (Psych + Psych)\nCai & Song",
        "Case 4 (ECE + ECE)\nYeary & Havlicek",
        "Case 5 (CS + CS)\nRadhakrishnan & Antonio"
    ]

    true_sims = [0.3173, 0.5830, 0.4350, 0.4337, 0.3316]
    mean_neg_sims = [0.2180, 0.1568, 0.0790, 0.1275, 0.0833]
    max_neg_sims = [0.3369, 0.2354, 0.2754, 0.2716, 0.1477]

    x = np.arange(len(cases))
    width = 0.28

    rects1 = ax.bar(x - width/2, true_sims, width, label="True Positive Collaborators", color="#2B6CB0", alpha=0.92, edgecolor="black", linewidth=0.6)
    rects2 = ax.bar(x + width/2, mean_neg_sims, width, label="Mean Negative Control Pairs", color="#E53E3E", alpha=0.85, edgecolor="black", linewidth=0.6)
    
    # Plot max negative as scatter points
    ax.scatter(x + width/2, max_neg_sims, color="#742A2A", s=65, zorder=5, label="Max Negative Control (Threshold)")

    # Connect true vs neg with margin delta labels
    for i in range(len(cases)):
        delta = true_sims[i] - mean_neg_sims[i]
        color = "#22543D" if true_sims[i] > max_neg_sims[i] else "#742A2A"
        ax.annotate(f"Δ = +{delta:.2f}",
                    xy=(x[i] - width/2, true_sims[i]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color)

    ax.set_ylabel("Semantic Alignment to Historical Ground Truth", fontweight="bold")
    ax.set_title("Experiment 2 Contrastive Discrimination: True Collaborators vs. Negative Controls\n(+216% Separation Margin | 80% Discrimination Accuracy)", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontweight="semibold")
    ax.set_ylim(0, 0.70)
    ax.grid(axis="y")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig4_contrastive_discrimination.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def generate_figure5_category_hits_recall():
    """Figure 5: Recommendation 1 Category Hits@k and Soft Recall."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    categories = [
        "Shared Domains",
        "Shared Application Areas",
        "Method-Application Synergies",
        "Cross-Domain Fusion Topics",
        "Complementary Technique Synergies",
        "Thinking Pattern Synergies",
        "Data-Method Synergies",
        "Future Research Directions",
        "Theory-Application Synergy",
        "Joint Technique Development"
    ]
    # Reverse for clean top-to-bottom bar order
    categories = categories[::-1]

    hits_50 = [100.0, 80.0, 40.0, 20.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0][::-1]
    hits_45 = [100.0, 80.0, 40.0, 40.0, 60.0, 20.0, 20.0, 40.0, 40.0, 0.0][::-1]
    soft_recall = [58.3, 36.7, 20.0, 15.0, 16.7, 5.0, 5.0, 9.0, 15.0, 0.0][::-1]

    y = np.arange(len(categories))
    height = 0.28

    rects1 = ax.barh(y + height, hits_45, height, label="Category Hits@0.45 (% Cases)", color="#3182CE", alpha=0.9, edgecolor="black", linewidth=0.5)
    rects2 = ax.barh(y, hits_50, height, label="Category Hits@0.50 (% Cases)", color="#38A169", alpha=0.9, edgecolor="black", linewidth=0.5)
    rects3 = ax.barh(y - height, soft_recall, height, label="Soft Recall@0.45 (Ground Truth Coverage %)", color="#DD6B20", alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Percentage (%)", fontweight="bold")
    ax.set_title("Experiment 2: Semantic Category Hits and Ground Truth Coverage", fontweight="bold", pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontweight="semibold")
    ax.set_xlim(0, 115)
    ax.grid(axis="x")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    # Annotate top bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            width = rect.get_width()
            if width > 0:
                ax.annotate(f"{width:.0f}%",
                            xy=(width, rect.get_y() + rect.get_height() / 2),
                            xytext=(3, 0),
                            textcoords="offset points",
                            ha="left", va="center", fontsize=8, fontweight="semibold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig5_exp2_category_hits_recall.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def main():
    print("=" * 80)
    print("GENERATING PUBLICATION-QUALITY FIGURES (300 DPI)...")
    print("=" * 80)
    generate_figure1_model_comparison()
    generate_figure2_discipline_breakdown()
    generate_figure3_tau_sensitivity()
    generate_figure4_contrastive_discrimination()
    generate_figure5_category_hits_recall()
    print("All 5 publication figures successfully generated in figures/ directory!")


if __name__ == "__main__":
    main()
