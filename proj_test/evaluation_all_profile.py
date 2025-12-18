import os
import sys
import re
import json
import argparse
import numpy as np

import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class ResearcherProfileEvaluator:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.list_keys = [
            "Research Domains", 
            "Techniques Used", 
            "Data & Platforms", 
            "Application Areas",
            "Key Research Thinking Patterns"
        ]
        self.text_key = "Summary Description"
        self.model_name = model_name
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load SBERT model for semantic matching"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.has_embeddings = True
        except ImportError:
            print("sentence-transformers not available, using exact matching only")
            self.has_embeddings = False
            self.model = None

    def normalize_text(self, text: str) -> str:
        """Normalize text for exact matching"""
        if text is None:
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\-]", " ", text)  # Keep hyphens
        text = re.sub(r"\s+", " ", text)
        return text

    def exact_matching(self, pred_list: List[str], gold_list: List[str]) -> Tuple[int, int, int, List]:
        """Perform exact normalized matching"""
        pred_norm = [self.normalize_text(p) for p in pred_list if p and p.strip()]
        gold_norm = [self.normalize_text(g) for g in gold_list if g and g.strip()]
        
        gold_set = set(gold_norm)
        pred_set = set(pred_norm)
        
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        
        # Create match details
        matches = []
        for p in pred_list:
            p_norm = self.normalize_text(p)
            matched_gold = None
            for g in gold_list:
                if self.normalize_text(g) == p_norm:
                    matched_gold = g
                    break
            matches.append((p, matched_gold, 1.0 if matched_gold else 0.0))
        
        return tp, fp, fn, matches

    def semantic_matching(self, pred_list: List[str], gold_list: List[str], tau: float = 0.65) -> Tuple[int, int, int, List]:
        """Perform semantic matching using SBERT embeddings"""
        if not self.has_embeddings or not pred_list or not gold_list:
            return 0, len(pred_list), len(gold_list), [(p, None, 0.0) for p in pred_list]
        
        try:
            from scipy.optimize import linear_sum_assignment
            
            # Encode all texts
            pred_emb = self.model.encode(pred_list, convert_to_tensor=False, show_progress_bar=False)
            gold_emb = self.model.encode(gold_list, convert_to_tensor=False, show_progress_bar=False)
            
            # Normalize embeddings
            pred_emb = pred_emb / np.linalg.norm(pred_emb, axis=1, keepdims=True)
            gold_emb = gold_emb / np.linalg.norm(gold_emb, axis=1, keepdims=True)
            
            # Compute similarity matrix
            similarity_matrix = np.dot(pred_emb, gold_emb.T)
            
            # Hungarian matching
            cost_matrix = -similarity_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Apply threshold and collect matches
            matches = []
            matched_pred_indices = set()
            matched_gold_indices = set()
            
            for i, j in zip(row_ind, col_ind):
                sim_score = similarity_matrix[i, j]
                if sim_score >= tau:
                    matches.append((pred_list[i], gold_list[j], float(sim_score)))
                    matched_pred_indices.add(i)
                    matched_gold_indices.add(j)
            
            # Add unmatched predictions as FP
            for i, pred in enumerate(pred_list):
                if i not in matched_pred_indices:
                    matches.append((pred, None, 0.0))
            
            # Calculate counts
            tp = len([m for m in matches if m[1] is not None])
            fp = len(pred_list) - tp
            fn = len(gold_list) - tp
            
            return tp, fp, fn, matches
            
        except Exception as e:
            print(f"Semantic matching failed: {e}, falling back to exact matching")
            return self.exact_matching(pred_list, gold_list)

    def hybrid_matching(self, pred_list: List[str], gold_list: List[str], tau: float = 0.65) -> Tuple[int, int, int, List]:
        """Combine exact and semantic matching"""
        # First do exact matching
        tp_exact, fp_exact, fn_exact, exact_matches = self.exact_matching(pred_list, gold_list)
        
        # Get unmatched items for semantic matching
        matched_preds = {m[0] for m in exact_matches if m[1] is not None}
        matched_golds = {m[1] for m in exact_matches if m[1] is not None}
        
        unmatched_preds = [p for p in pred_list if p not in matched_preds]
        unmatched_golds = [g for g in gold_list if g not in matched_golds]
        
        # Semantic matching on unmatched items
        tp_sem, fp_sem, fn_sem, sem_matches = self.semantic_matching(unmatched_preds, unmatched_golds, tau)
        
        # Combine results
        total_tp = tp_exact + tp_sem
        total_fp = fp_exact + fp_sem
        total_fn = fn_exact + fn_sem
        
        # Combine match details
        all_matches = exact_matches + sem_matches
        
        return total_tp, total_fp, total_fn, all_matches

    def evaluate_summary(self, pred_summary: str, gold_summary: str) -> Dict[str, float]:
        """Evaluate summary description using embedding similarity"""
        if not pred_summary or not gold_summary:
            return {"cosine_similarity": 0.0, "bertscore_f1": 0.0}
        
        if self.has_embeddings:
            # Cosine similarity
            emb_pred = self.model.encode([pred_summary], convert_to_tensor=False)
            emb_gold = self.model.encode([gold_summary], convert_to_tensor=False)
            emb_pred = emb_pred / np.linalg.norm(emb_pred)
            emb_gold = emb_gold / np.linalg.norm(emb_gold)
            cosine_sim = float(np.dot(emb_pred, emb_gold.T)[0, 0])
        else:
            cosine_sim = 0.0
        
        # BERTScore (optional)
        bertscore_f1 = 0.0
        try:
            from bert_score import BERTScorer
            scorer = BERTScorer(lang="en")
            P, R, F1 = scorer.score([pred_summary], [gold_summary])
            bertscore_f1 = float(F1[0])
        except ImportError:
            pass
        
        return {
            "cosine_similarity": cosine_sim,
            "bertscore_f1": bertscore_f1
        }

    def precision_recall_f1(self, tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        """Calculate precision, recall, F1"""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def evaluate_researcher(self, pred_profile: Dict, gold_profile: Dict, tau: float = 0.65) -> Dict:
        """Evaluate a single researcher's profile"""
        results = {
            "per_key": {},
            "summary": {},
            "pairwise_matches": {}
        }
        
        # Evaluate list keys
        for key in self.list_keys:
            pred_list = pred_profile.get(key, [])
            gold_list = gold_profile.get(key, [])
            
            if not isinstance(pred_list, list):
                pred_list = []
            if not isinstance(gold_list, list):
                gold_list = []
            
            tp, fp, fn, matches = self.hybrid_matching(pred_list, gold_list, tau)
            precision, recall, f1 = self.precision_recall_f1(tp, fp, fn)
            
            results["per_key"][key] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
                "n_pred": len(pred_list), "n_gold": len(gold_list)
            }
            results["pairwise_matches"][key] = matches
        
        # Evaluate summary
        pred_summary = pred_profile.get(self.text_key, "")
        gold_summary = gold_profile.get(self.text_key, "")
        results["summary"] = self.evaluate_summary(pred_summary, gold_summary)
        
        return results

    def evaluate_all(self, pred_profiles: Dict[str, Dict], gold_profiles: Dict[str, Dict], 
                    tau: float = 0.65, bootstrap_samples: int = 1000) -> Dict:
        """Evaluate all researchers and compute aggregate metrics"""
        
        researcher_ids = list(gold_profiles.keys())
        all_results = {}
        
        # Individual researcher evaluations
        for researcher_id in researcher_ids:
            if researcher_id in pred_profiles:
                all_results[researcher_id] = self.evaluate_researcher(
                    pred_profiles[researcher_id], 
                    gold_profiles[researcher_id], 
                    tau
                )
        
        # Aggregate metrics
        return self._aggregate_metrics(all_results, bootstrap_samples)

    def _aggregate_metrics(self, all_results: Dict, bootstrap_samples: int = 1000) -> Dict:
        """Compute aggregate metrics across all researchers"""
        
        # Initialize aggregators
        micro_counts = {key: {"tp": 0, "fp": 0, "fn": 0} for key in self.list_keys}
        macro_metrics = {key: {"precisions": [], "recalls": [], "f1s": []} for key in self.list_keys}
        summary_similarities = {"cosine": [], "bertscore": []}
        
        # Collect metrics
        for researcher_id, results in all_results.items():
            for key in self.list_keys:
                if key in results["per_key"]:
                    key_results = results["per_key"][key]
                    micro_counts[key]["tp"] += key_results["tp"]
                    micro_counts[key]["fp"] += key_results["fp"]
                    micro_counts[key]["fn"] += key_results["fn"]
                    
                    macro_metrics[key]["precisions"].append(key_results["precision"])
                    macro_metrics[key]["recalls"].append(key_results["recall"])
                    macro_metrics[key]["f1s"].append(key_results["f1"])
            
            # Summary metrics
            if "cosine_similarity" in results["summary"]:
                summary_similarities["cosine"].append(results["summary"]["cosine_similarity"])
            if "bertscore_f1" in results["summary"]:
                summary_similarities["bertscore"].append(results["summary"]["bertscore_f1"])
        
        # Compute final metrics
        final_results = {
            "per_key": {},
            "global": {},
            "summary": {},
            "bootstrap_ci": {}
        }
        
        # Per-key metrics
        for key in self.list_keys:
            # Micro-averaged
            tp, fp, fn = micro_counts[key]["tp"], micro_counts[key]["fp"], micro_counts[key]["fn"]
            micro_prec, micro_rec, micro_f1 = self.precision_recall_f1(tp, fp, fn)
            
            # Macro-averaged
            macro_prec = np.mean(macro_metrics[key]["precisions"]) if macro_metrics[key]["precisions"] else 0.0
            macro_rec = np.mean(macro_metrics[key]["recalls"]) if macro_metrics[key]["recalls"] else 0.0
            macro_f1 = np.mean(macro_metrics[key]["f1s"]) if macro_metrics[key]["f1s"] else 0.0
            
            final_results["per_key"][key] = {
                "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1, "tp": tp, "fp": fp, "fn": fn},
                "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1}
            }
        
        # Global metrics (across all keys)
        global_tp = sum(micro_counts[key]["tp"] for key in self.list_keys)
        global_fp = sum(micro_counts[key]["fp"] for key in self.list_keys)
        global_fn = sum(micro_counts[key]["fn"] for key in self.list_keys)
        global_prec, global_rec, global_f1 = self.precision_recall_f1(global_tp, global_fp, global_fn)
        
        macro_f1s = [final_results["per_key"][key]["macro"]["f1"] for key in self.list_keys]
        global_macro_f1 = np.mean(macro_f1s) if macro_f1s else 0.0
        
        final_results["global"] = {
            "micro": {"precision": global_prec, "recall": global_rec, "f1": global_f1},
            "macro_f1": global_macro_f1
        }
        
        # Summary metrics
        final_results["summary"] = {
            "cosine_similarity_mean": np.mean(summary_similarities["cosine"]) if summary_similarities["cosine"] else 0.0,
            "bertscore_f1_mean": np.mean(summary_similarities["bertscore"]) if summary_similarities["bertscore"] else 0.0
        }
        
        # Bootstrap confidence intervals
        if bootstrap_samples > 0:
            final_results["bootstrap_ci"] = self._compute_bootstrap_ci(all_results, bootstrap_samples)
        
        return final_results

    def _compute_bootstrap_ci(self, all_results: Dict, n_bootstrap: int = 1000) -> Dict:
        """Compute bootstrap confidence intervals for micro-F1 per key"""
        researcher_ids = list(all_results.keys())
        n_researchers = len(researcher_ids)
        
        bootstrap_ci = {}
        
        for key in self.list_keys:
            f1_scores = []
            rng = np.random.RandomState(42)
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                sample_ids = rng.choice(researcher_ids, size=n_researchers, replace=True)
                
                # Compute micro-F1 for this bootstrap sample
                tp_total, fp_total, fn_total = 0, 0, 0
                for rid in sample_ids:
                    if key in all_results[rid]["per_key"]:
                        r_results = all_results[rid]["per_key"][key]
                        tp_total += r_results["tp"]
                        fp_total += r_results["fp"]
                        fn_total += r_results["fn"]
                
                _, _, f1 = self.precision_recall_f1(tp_total, fp_total, fn_total)
                f1_scores.append(f1)
            
            # Compute 95% CI
            lower = np.percentile(f1_scores, 2.5)
            upper = np.percentile(f1_scores, 97.5)
            bootstrap_ci[key] = {"micro_f1_95ci": (float(lower), float(upper))}
        
        return bootstrap_ci



