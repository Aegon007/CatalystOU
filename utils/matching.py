# matching.py
import re
import pdb
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.optimize import linear_sum_assignment
# Note: sentence_transformers should be imported in the orchestrator 
# and the model object passed in to avoid reloading it frequently.


# ------------------------------------------------------------
# Schema-aware normalization for researcher profile evaluation
# ------------------------------------------------------------
def get_matching_fields():
    LIST_FIELDS = [
        "Research Domains",
        "Techniques Used",
        "Data & Platforms",
        "Application Areas",
        "Key Research Thinking Patterns"
    ]
    SUMMARY_FIELD = "Summary Description"

    return LIST_FIELDS, SUMMARY_FIELD


LIST_FIELDS, SUMMARY_FIELD = get_matching_fields()


def normalize_profile_field(field_name: str, field_value: Any) -> List[str]:
    """
    Normalize a profile field into List[str] suitable for matching.

    Rules:
    - List[str] → unchanged
    - List[dict] → extract all string values
    - str → wrap into singleton list
    - Anything else → ignored
    """

    if field_name not in LIST_FIELDS:
        return []

    if field_value is None:
        return []

    # Case 1: already List[str]
    if isinstance(field_value, list) and all(isinstance(x, str) for x in field_value):
        return field_value

    normalized = []

    # Case 2: List[dict]
    if isinstance(field_value, list):
        for item in field_value:
            if isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str) and v.strip():
                        normalized.append(v.strip())
            elif isinstance(item, str):
                normalized.append(item.strip())

        return normalized

    # Case 3: single string
    if isinstance(field_value, str):
        return [field_value.strip()]

    # Everything else: drop
    return []


def normalize_phrase(s: str) -> str:
    """
    Lightweight normalization for exact matching.
    Lowercases, removes special punctuation, collapses whitespace.
    """
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # Keep alphanumeric, hyphens, spaces. Remove others.
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _exact_match_helper(preds: List[str], golds: List[str]) -> Tuple[int, int, int, List[Tuple[str, Optional[str]]]]:
    """
    Internal helper for normalized exact matching.
    Returns: tp, fp, fn, matches_list
    """
    gold_norm_map = {normalize_phrase(g): g for g in golds}
    matched_gold_norms = set()
    matches = []
    tp = 0

    for p in preds:
        p_norm = normalize_phrase(p)
        if p_norm in gold_norm_map and p_norm not in matched_gold_norms:
            matches.append((p, gold_norm_map[p_norm]))
            matched_gold_norms.add(p_norm)
            tp += 1
        else:
            matches.append((p, None))

    # Calculate FP (unmatched preds) and FN (unmatched golds)
    fp = len([m for m in matches if m[1] is None])
    fn = len(golds) - tp
    
    return tp, fp, fn, matches


def _semantic_match_helper(preds: List[str], golds: List[str], model, tau: float) -> List[Tuple[str, str, float]]:
    """
    Internal helper for Hungarian semantic matching on unmatched items.
    Returns list of (pred, gold, similarity).
    """
    if not preds or not golds:
        return []

    # Encode - assumes model is a SentenceTransformer object
    pred_emb = model.encode(preds, convert_to_numpy=True, normalize_embeddings=True)
    gold_emb = model.encode(golds, convert_to_numpy=True, normalize_embeddings=True)

    # Compute Similarity Matrix (Cosine)
    sim_matrix = np.matmul(pred_emb, gold_emb.T)

    # Hungarian Algorithm: minimize cost (-similarity)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        score = float(sim_matrix[r, c])
        if score >= tau:
            matches.append((preds[r], golds[c], score))
            
    return matches


def match_profile_list_field(pred_list: List[str], gold_list: List[str], embedding_model=None, tau: float = 0.65) -> Dict[str, Any]:
    """
    Main matching logic for list-based profile fields (e.g., "Research Domains").
    
    1. Validates inputs are List[str].
    2. Performs normalized exact matching.
    3. Performs Hungarian semantic matching on residuals.
    4. Aggregates results separating exact vs semantic counts.
    """
    # 0. Schema Validation

    if not isinstance(pred_list, list) or not isinstance(gold_list, list):
        raise ValueError("Inputs to match_profile_list_field must be lists.")
    if pred_list and not isinstance(pred_list[0], str):
        raise ValueError("Input lists must contain strings.")
    
    # 1. Exact Matching
    tp_exact, _, _, exact_matches_raw = _exact_match_helper(pred_list, gold_list)
    
    # Identify residuals
    matched_preds_indices = {i for i, m in enumerate(exact_matches_raw) if m[1] is not None}
    matched_gold_vals = {m[1] for m in exact_matches_raw if m[1] is not None}
    
    unmatched_preds = [p for i, p in enumerate(pred_list) if i not in matched_preds_indices]
    unmatched_golds = [g for g in gold_list if g not in matched_gold_vals]
    
    # 2. Semantic Matching (on residuals)
    sem_matches_raw = []
    if embedding_model and unmatched_preds and unmatched_golds:
        sem_matches_raw = _semantic_match_helper(unmatched_preds, unmatched_golds, embedding_model, tau)
    
    tp_sem = len(sem_matches_raw)
    
    # 3. Combine Results
    total_tp = tp_exact + tp_sem
    total_fp = len(pred_list) - total_tp
    total_fn = len(gold_list) - total_tp
    
    # Construct unified match list [ {"pred":..., "gold":..., "sim":...} ]
    # Start with exact matches (sim=1.0)
    final_matches = []
    for p, g in exact_matches_raw:
        if g is not None:
            final_matches.append({"pred": p, "gold": g, "sim": 1.0, "type": "exact"})
        else:
            # Check if this unmatched pred was matched semantically
            sem_match = next((x for x in sem_matches_raw if x[0] == p), None)
            if sem_match:
                final_matches.append({"pred": p, "gold": sem_match[1], "sim": sem_match[2], "type": "semantic"})
            else:
                final_matches.append({"pred": p, "gold": None, "sim": 0.0, "type": "none"})

    # Calculate field-level stats
    sims = [m["sim"] for m in final_matches if m["gold"] is not None]
    avg_sim = float(np.mean(sims)) if sims else 0.0
    
    return {
        "tp_total": total_tp,
        "tp_exact": tp_exact,
        "tp_semantic": tp_sem,
        "fp": total_fp,
        "fn": total_fn,
        "avg_similarity": avg_sim,
        "matches": final_matches
    }


def compute_summary_similarity(pred_text: str, gold_text: str, model) -> Dict[str, float]:
    """
    Evaluates the free-text Summary Description using Cosine Similarity.
    BERTScore can be added optionally.
    """
    if not pred_text or not gold_text:
        return {"cosine_sim": 0.0}
    
    emb_pred = model.encode([pred_text], convert_to_numpy=True, normalize_embeddings=True)
    emb_gold = model.encode([gold_text], convert_to_numpy=True, normalize_embeddings=True)
    
    cosine_sim = float(np.dot(emb_pred, emb_gold.T)[0,0])
    
    return {"cosine_sim": cosine_sim}
