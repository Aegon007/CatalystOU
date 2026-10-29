"""
================================================================================
Google BERT Extractive Baseline for CatalystOU Experiment 1
================================================================================
Model: bert-base-uncased (Google AI, Devlin et al., 2018)
Task: Classical non-generative extractive keyphrase & entity baseline across 
      all 50 faculty publications in CatalystOU-pdf/.

Evaluates against profile_labeled_data/ using the identical Hungarian matching
metric and SBERT evaluator (all-mpnet-base-v2) to establish the empirical lower bound.
================================================================================
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Set
import numpy as np

# Ensure project root is in path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from utils.logger_utils import setup_logger
import utils.profile_io as profile_io
import utils.matching as matching
from proj_test.test1_profile_acc import evaluate_single_profile, build_data_map
from proj_test.comprehensive_analysis import aggregate_group

logger = setup_logger("bert_baseline", log_file="bert_baseline.log")

# Standard English stopwords
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "paper", "study", "results", "using", "method", "approach", "data", "based", "et", "al", "figure", "table"
}

# Anchor queries defining the 4 entity fields in semantic space
ANCHOR_QUERIES = {
    "Research Domains": "academic discipline specialized research domain scientific field theoretical branch subfield",
    "Techniques Used": "methodology algorithm experimental assay computational technique mathematical framework analytical pipeline",
    "Data & Platforms": "curated dataset software library tool benchmark database physical instrument repository package",
    "Application Areas": "practical problem translational goal real world application industrial deployment clinical technological target"
}


# ==============================================================================
# 1. TEXT EXTRACTION UTILITIES
# ==============================================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF file using PyMuPDF or PyPDF2/pypdf."""
    text = ""
    # Try fitz / pymupdf
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        text = "\n".join([page.get_text() or "" for page in doc])
        doc.close()
    except Exception:
        pass

    if not text.strip():
        # Try PyPDF2 / pypdf
        try:
            import PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() or "" for page in reader.pages[:15]])
        except Exception:
            pass

    # Basic cleaning
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_candidate_phrases(text: str, max_candidates: int = 400) -> List[str]:
    """Extract candidate n-grams (1 to 3 words) from raw text."""
    # Split text into sentences/clauses
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
    if not tokens:
        return []

    candidates = set()
    n_tokens = len(tokens)

    # Extract 1-gram, 2-gram, 3-gram
    for n in (1, 2, 3):
        for i in range(min(n_tokens - n + 1, 4000)):
            gram = tokens[i : i + n]
            gram_lower = [w.lower() for w in gram]

            # Skip if starting or ending with stopword
            if gram_lower[0] in STOPWORDS or gram_lower[-1] in STOPWORDS:
                continue

            # Skip if all are numbers or single characters
            if all(len(w) <= 2 for w in gram):
                continue

            phrase = " ".join(gram)
            if len(phrase) >= 4 and len(phrase) <= 45:
                candidates.add(phrase)

            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    return sorted(list(candidates))


# ==============================================================================
# 2. BERT EMBEDDING EXTRACTOR
# ==============================================================================

class BertKeyphraseExtractor:
    """Uses pre-trained Google BERT (bert-base-uncased) to embed and rank technical phrases."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cpu"):
        self.device = device
        print(f"[*] Initializing Google BERT ({model_name}) on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Precompute anchor embeddings
        print("[*] Encoding category anchor vectors with BERT...")
        self.anchor_keys = list(ANCHOR_QUERIES.keys())
        anchor_texts = [ANCHOR_QUERIES[k] for k in self.anchor_keys]
        self.anchor_embeddings = self.encode_texts(anchor_texts)

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Compute mean-pooled, normalized BERT embeddings for a list of texts."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                # Mean pooling using attention mask
                mask = encoded["attention_mask"].unsqueeze(-1)
                token_embs = outputs.last_hidden_state
                sum_embs = torch.sum(token_embs * mask, dim=1)
                lens = torch.clamp(mask.sum(dim=1), min=1e-9)
                pooled = sum_embs / lens
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu())

        return torch.cat(all_embeddings, dim=0)

    def extract_profile_from_texts(self, researcher_name: str, paper_texts: List[str]) -> Dict[str, Any]:
        """Extract profile entities using BERT cosine similarity to category anchors."""
        full_text = " ".join(paper_texts)
        candidate_phrases = extract_candidate_phrases(full_text, max_candidates=500)

        if not candidate_phrases:
            return {
                "Researcher Profile:": researcher_name,
                "Affiliation:": "University of Oklahoma",
                "Research Domains": [],
                "Techniques Used": [],
                "Data & Platforms": [],
                "Application Areas": [],
                "Key Research Thinking Patterns": [],
                "Summary Description": full_text[:250]
            }

        # Compute embeddings for candidates
        cand_embs = self.encode_texts(candidate_phrases)

        # Compute cosine similarity matrix: [num_candidates, num_categories]
        sim_matrix = torch.mm(cand_embs, self.anchor_embeddings.t())  # shape: [N, 4]

        profile_fields: Dict[str, List[str]] = {k: [] for k in self.anchor_keys}

        # Assign candidate phrases to top-matching category
        # Threshold: similarity >= 0.55
        for idx, phrase in enumerate(candidate_phrases):
            sims = sim_matrix[idx]
            best_cat_idx = int(torch.argmax(sims).item())
            best_score = float(sims[best_cat_idx].item())
            cat_name = self.anchor_keys[best_cat_idx]

            if best_score >= 0.50:
                profile_fields[cat_name].append((phrase, best_score))

        # Sort each field by similarity score and take top unique items
        top_k = {
            "Research Domains": 12,
            "Techniques Used": 20,
            "Data & Platforms": 15,
            "Application Areas": 8,
        }

        final_fields = {}
        for cat, pairs in profile_fields.items():
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
            # Deduplicate case-insensitively
            seen = set()
            cleaned = []
            for p, _ in pairs_sorted:
                key = p.lower()
                if key not in seen:
                    seen.add(key)
                    cleaned.append(p)
                if len(cleaned) >= top_k.get(cat, 15):
                    break
            final_fields[cat] = cleaned

        # Extractive Lead-3 Summary (first 3 sentences)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if len(s.strip()) > 20]
        extractive_summary = " ".join(sentences[:3]) if sentences else full_text[:300]

        # Thinking Patterns: Extractive models cannot synthesize structured patterns
        # Output empty or generic extracted sentences (illustrates non-generative limitation)
        extractive_thinking = []

        return {
            "Researcher Profile:": researcher_name,
            "Affiliation:": "University of Oklahoma",
            "Research Domains": final_fields.get("Research Domains", []),
            "Techniques Used": final_fields.get("Techniques Used", []),
            "Data & Platforms": final_fields.get("Data & Platforms", []),
            "Application Areas": final_fields.get("Application Areas", []),
            "Key Research Thinking Patterns": extractive_thinking,
            "Summary Description": extractive_summary
        }


# ==============================================================================
# 3. MAIN RUNNER & EVALUATION PIPELINE
# ==============================================================================

def run_bert_baseline_pipeline():
    print("=" * 80)
    print(" CatalystOU: Google BERT Extractive Baseline Evaluation (Table 1)")
    print(" Model: bert-base-uncased (110M Encoder)")
    print("=" * 80)

    pdf_root = repo_root / "CatalystOU-pdf"
    output_dir = repo_root / "extracted_profile_json" / "google_bert_base_uncased"
    results_dir = repo_root / "results" / "google_bert_base_uncased"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all 50 researchers
    researcher_entries = []
    for disc_dir in sorted(pdf_root.iterdir()):
        if not disc_dir.is_dir() or disc_dir.name.startswith(".") or disc_dir.name.startswith("~"):
            continue
        for r_dir in sorted(disc_dir.iterdir()):
            if not r_dir.is_dir() or r_dir.name.startswith(".") or r_dir.name.startswith("~"):
                continue
            pdfs = list(r_dir.glob("*.pdf"))
            if pdfs:
                researcher_entries.append({
                    "discipline": disc_dir.name,
                    "researcher_name": r_dir.name,
                    "researcher_dir": r_dir,
                    "pdfs": pdfs
                })

    print(f"[+] Discovered {len(researcher_entries)} researchers across 5 disciplines.")

    # 2. Initialize BERT extractor
    extractor = BertKeyphraseExtractor("bert-base-uncased", device="cpu")

    # 3. Extract profiles
    print("\n[*] Extracting profiles with Google BERT...")
    start_time = time.time()
    for item in researcher_entries:
        r_name = item["researcher_name"]
        disc = item["discipline"]
        sanitized_name = re.sub(r"[^\w\s-]", "", r_name).strip().replace(" ", "_")
        dest_file = output_dir / disc / f"{sanitized_name}_profile.json"
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        if dest_file.exists():
            continue

        paper_texts = []
        for pdf_path in item["pdfs"]:
            t = extract_text_from_pdf(pdf_path)
            if t:
                paper_texts.append(t)

        profile = extractor.extract_profile_from_texts(r_name, paper_texts)
        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"[+] Extraction complete in {time.time() - start_time:.2f}s! All profiles saved to:")
    print(f"    {output_dir}")

    # 4. Evaluate using standard CatalystOU evaluation suite
    print("\n[*] Evaluating BERT profiles against ground truth using SBERT (all-mpnet-base-v2)...")
    data_map = build_data_map(
        str(repo_root / "extracted_profile_json"),
        str(repo_root / "profile_labeled_data"),
        "google_bert_base_uncased"
    )
    print(f"[+] Mapped profile pairs: {len(data_map)}")

    sbert_model = SentenceTransformer("all-mpnet-base-v2")
    individual_evals = []
    by_discipline = {"Biology": [], "CS": [], "ECE": [], "Mathematics": [], "Psychology": []}

    for item in data_map:
        llm_p = Path(item["llm_path"])
        gold_p = Path(item["gold_path"])
        disc = "Unknown"
        for d in by_discipline:
            if d in str(llm_p):
                disc = d
                break

        res = evaluate_single_profile(
            str(llm_p),
            str(gold_p),
            sbert_model,
            tau=0.65,
            thinking_patterns_tau=0.50
        )
        res["researcher"] = llm_p.stem.replace("_profile", "")
        res["discipline"] = disc
        individual_evals.append(res)
        if disc in by_discipline:
            by_discipline[disc].append(res)

    overall_stats = aggregate_group(individual_evals)
    discipline_stats = {d: aggregate_group(records) for d, records in by_discipline.items()}

    final_payload = {
        "model": "Google BERT (Extractive Baseline)",
        "num_profiles": len(individual_evals),
        "overall": overall_stats,
        "discipline_breakdown": discipline_stats
    }

    results_file = results_dir / "comprehensive_evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"[+] Saved evaluation results to: {results_file}")

    # 5. Print Table 1 Comparison
    per_key = overall_stats.get("per_key", {})
    print("\n" + "=" * 80)
    print(" TABLE 1: GOOGLE BERT (EXTRACTIVE BASELINE) VS. GENERATIVE LLMs")
    print("=" * 80)
    print(f"{'Field Name':<28} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9}")
    print("-" * 65)

    total_tp = sum(v.get("tp_total", 0) for v in per_key.values())
    total_fp = sum(v.get("fp_total", 0) for v in per_key.values())
    total_fn = sum(v.get("fn_total", 0) for v in per_key.values())

    for k, v in per_key.items():
        p = v.get("micro_precision", 0.0) * 100
        r = v.get("micro_recall", 0.0) * 100
        f1 = v.get("micro_f1", 0.0) * 100
        print(f"{k:<28} | {p:6.1f}%   | {r:6.1f}%   | {f1:6.1f}%")

    sum_cos = overall_stats.get("summary_cosine_mean", 0.0)
    print(f"{'Researcher Summary (Sim)':<28} | {'--':<9} | {'--':<9} | {sum_cos:.4f}")

    ov_p = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
    ov_r = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0.0
    ov_f1 = (2 * ov_p * ov_r / (ov_p + ov_r)) if (ov_p + ov_r) > 0 else 0.0
    print("-" * 65)
    print(f"{'OVERALL':<28} | {ov_p:6.1f}%   | {ov_r:6.1f}%   | {ov_f1:6.1f}%")
    print(f"Mean PSAS: {overall_stats.get('PSAS_mean', 0.0):.4f} +/- {overall_stats.get('PSAS_std', 0.0):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_bert_baseline_pipeline()
