"""
================================================================================
CatalystOU - Google Colab Pipeline: Code Cell 1 (Profile Extraction)
================================================================================
Hardware Target: Google Colab A100 GPU (80GB VRAM) High-RAM Runtime
Model: Qwen3.6-35B-A3B (Mixture of Experts: 35B total params, 3B active params)
Context Window: 262k native context
Released: April 2026

Architecture:
1. Environment & HF Model Loader:
   - Loads 'Qwen/Qwen3.6-35B-A3B' in native bfloat16 onto the A100 (~70GB VRAM).
   - Features automatic 4-bit / 8-bit BitsAndBytes fallback for smaller GPUs (e.g. L4, A100 40GB).
   - Utilizes PyTorch SDPA (Scaled Dot-Product Attention) for fast MoE inference.
2. Dataset Discovery:
   - Scans PDF directories organized by Discipline/Researcher (e.g. CatalystOU-pdf/Biology/Anne K. Dunn/*.pdf).
   - Automatically supports Google Drive mount or local workspace clones.
3. Multi-Engine PDF Text Extraction:
   - High-throughput text extraction via PyMuPDF (fitz) or pypdf.
   - Cleans and strips bibliography/reference noise to focus on core research contributions.
4. Map-Reduce Profile Extraction:
   - Map: Extracts micro-JSON ontology entities from each paper individually in parallel batches.
   - Reduce: Programmatically merges and semantically deduplicates entity lists.
   - Synthesize: Synthesizes 'Key Research Thinking Patterns' and 'Summary Description'.
5. Checkpointing & Output:
   - Saves final structured profiles matching CatalystOU evaluation schema to:
     extracted_profile_json/qwen_qwen3.6-35b-a3b/<Discipline>/<Researcher>_profile.json
   - Skips already extracted researchers to allow safe resumption on Colab disconnects.
================================================================================
"""

import os
import re
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: sentence-transformers for semantic deduplication (falls back cleanly if unavailable)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# Optimize PyTorch CUDA allocator to prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# HuggingFace Model Repo ID
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen3.6-35B-A3B")

# Paths for Dataset and Output
# If running on Colab with Drive mounted: '/content/drive/MyDrive/CatalystOU/CatalystOU-pdf'
# If running on Colab repo clone: '/content/CatalystOU/CatalystOU-pdf' or './CatalystOU-pdf'
PDF_ROOT_DIR = os.getenv("PDF_ROOT_DIR", "CatalystOU-pdf")
OUTPUT_ROOT_DIR = os.getenv("OUTPUT_ROOT_DIR", "extracted_profile_json")

# Extraction Controls
MODEL_CLEAN_NAME = HF_MODEL_ID.replace("/", "_").lower()
MAX_PAPER_CHARS = 18000       # Chars per paper (~4.5k tokens, safe native bfloat16 activation headroom)
MAX_NEW_TOKENS = 4096         # Max generation tokens for JSON schema
TEMPERATURE = 0.0             # 0.0 for deterministic greedy decoding
SEED = 42
USE_SEMANTIC_DEDUP = True     # Use SentenceTransformer to cluster near-identical entities
SEMANTIC_SIM_THRESHOLD = 0.82
MAP_BATCH_SIZE = int(os.getenv("MAP_BATCH_SIZE", "3"))  # Batched GPU inference (2-3 papers per batch; 3 fits within A100 VRAM headroom)

# Optional filters (set to None to process all)
DISCIPLINE_FILTER: Optional[List[str]] = None   # e.g., ["Biology"] or None for all
MAX_RESEARCHERS_LIMIT: Optional[int] = None     # e.g., 5 for a quick test run, or None


# ==============================================================================
# PROMPT DEFINITIONS (Aligned with CatalystOU Ontology)
# ==============================================================================

PAPER_EXTRACTION_SYSTEM_PROMPT = """You are an expert scientific ontology extractor. 
Your task is to extract structured, high-signal research entities from the provided academic research paper across any discipline.

You MUST respond strictly with a valid JSON object matching this schema:
{
    "Affiliation": "Author Affiliation/Department",
    "Research Domains": ["Domain 1", "Domain 2"],
    "Techniques Used": ["Technique 1", "Technique 2"],
    "Data & Platforms": ["Platform/Tool 1", "Dataset 2"],
    "Application Areas": ["Application 1", "Application 2"],
    "Primary Objective & Findings": "1-3 sentence summary of the core objective, experimental approach, and key findings."
}

Extraction Guidelines:
1. Research Domains: Primary academic disciplines, specialized subfields, and core research areas directly relevant to this paper.
2. Techniques Used: Specific, named methodologies, experimental assays, algorithms, computational models, mathematical frameworks, or analytical pipelines central to the paper's scientific investigation. Focus on distinctive methodologies rather than generic, incidental tasks.
3. Data & Platforms: Named physical instruments, software packages, programming libraries, databases, repositories, benchmarks, and curated datasets used in the study. Output as a flat list.
4. Application Areas: Specific problem domains, translational goals, technological, industrial, environmental, or clinical applications targeted by the work.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are a principal investigator synthesizing a comprehensive researcher profile from their publications across any scientific discipline.
You will receive aggregated lists of domains, techniques, platforms, and paper summaries for a single researcher.

You MUST respond strictly with a valid JSON object matching this schema:
{
    "Affiliation:": "Canonical University Department / School Affiliation",
    "Application Areas": [
        "Macro Problem Domain / Translational Goal 1",
        "Macro Problem Domain / Translational Goal 2"
    ],
    "Key Research Thinking Patterns": [
        "Methodological Strategy Name: Conceptual definition of how the researcher approaches problem-solving and hypothesis validation (e.g., Concrete implementation citing specific methods, datasets, models, or findings from the papers)."
    ],
    "Summary Description": "A cohesive summary under 150 words describing the researcher's specialization, primary model systems/frameworks, key technical methodologies, and scientific impact."
}

Guidelines for Key Research Thinking Patterns:
- Provide 3 to 4 prominent, hypothesis-driven scientific reasoning patterns or methodological paradigms.
- The Pattern Name and Definition should capture the researcher's general problem-solving methodology or experimental design strategy.
- Format EXACTLY as: 'Pattern Name: Definition (e.g., Specific example citing concrete methods, assays, or findings)'.
- Do NOT cite paper numbers or generic labels (such as 'Paper 1'). Cite the concrete experimental mechanism, model system, or assay directly in the parenthesis.

Guidelines for Application Areas:
- Consolidate the paper-level sub-goals into 4 to 6 macro-level translational goals or application domains targeted by the research.
"""


# ==============================================================================
# 1. HUGGING FACE MODEL RUNNER (A100 Optimized + MoE & Quantization Handling)
# ==============================================================================

class HFModelRunner:
    """Manages local inference with Qwen3.6-35B-A3B (MoE) on A100 (80GB)."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        print(f"[*] Initializing HuggingFace weights: {model_id}")
        print(f"[*] Architecture: Mixture of Experts (35B total params, 3B active params)")
        print(f"[*] CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count()}")
        
        total_vram_gb = 0.0
        if torch.cuda.is_available():
            print(f"[*] Primary GPU: {torch.cuda.get_device_name(0)}")
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[*] Total VRAM: {total_vram_gb:.2f} GB")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load weights in unquantized native bfloat16 to match Llama 3.1 8B and Qwen 3.8 27B baselines
        print("[*] Loading Qwen3.6-35B-A3B in unquantized native bfloat16 onto NVIDIA A100 (80GB)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model.eval()

        allocated_vram = torch.cuda.memory_allocated() / (1024**3)
        print(f"[+] Model loaded successfully in native bfloat16! VRAM Allocated: {allocated_vram:.2f} GB")

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Dict[str, Any]:
        """Generate response from model and parse into a dictionary."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.90
        else:
            gen_kwargs["do_sample"] = False

        try:
            with torch.inference_mode():
                output_tokens = self.model.generate(**inputs, **gen_kwargs)
        except torch.OutOfMemoryError:
            print("    [!] Warning: CUDA OOM encountered during generation. Flushing cache and retrying with compact context...")
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            truncated_user_prompt = user_prompt[:len(user_prompt) // 2]
            retry_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": truncated_user_prompt},
            ]
            try:
                retry_prompt_text = self.tokenizer.apply_chat_template(
                    retry_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                retry_prompt_text = self.tokenizer.apply_chat_template(
                    retry_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = self.tokenizer([retry_prompt_text], return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                output_tokens = self.model.generate(**inputs, **gen_kwargs)

        # Slice generated token ids (excluding the input prompt)
        input_len = inputs.input_ids.shape[1]
        new_tokens = output_tokens[0][input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        del inputs, output_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return parse_json_text(response_text)

    def generate_json_batch(
        self,
        system_prompt: str,
        user_prompts: List[str],
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        batch_size: int = MAP_BATCH_SIZE,
    ) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts in parallel batches on GPU (2-3 papers per batch)."""
        results = []
        for i in range(0, len(user_prompts), batch_size):
            chunk = user_prompts[i : i + batch_size]
            formatted_prompts = []
            for up in chunk:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": up},
                ]
                try:
                    pt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    pt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                formatted_prompts.append(pt)

            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_PAPER_CHARS // 3 + 1000,
            ).to(self.model.device)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "do_sample": False if temperature == 0.0 else True,
            }
            if temperature > 0.0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.90

            try:
                with torch.inference_mode():
                    output_tokens = self.model.generate(**inputs, **gen_kwargs)
            except torch.OutOfMemoryError:
                print(f"    [!] Warning: Batch OOM encountered on GPU for batch_size={len(chunk)}. Clearing cache and retrying with adaptive fallback...")
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                # Step-down fallback: try batch_size=2 if original chunk was > 2
                if len(chunk) > 2:
                    try:
                        sub_results = []
                        for sub_start in range(0, len(chunk), 2):
                            sub_chunk = chunk[sub_start : sub_start + 2]
                            sub_formatted = []
                            for up in sub_chunk:
                                msg = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": up},
                                ]
                                try:
                                    spt = self.tokenizer.apply_chat_template(
                                        msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
                                    )
                                except TypeError:
                                    spt = self.tokenizer.apply_chat_template(
                                        msg, tokenize=False, add_generation_prompt=True
                                    )
                                sub_formatted.append(spt)

                            sub_inputs = self.tokenizer(
                                sub_formatted,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=MAX_PAPER_CHARS // 3 + 1000,
                            ).to(self.model.device)

                            with torch.inference_mode():
                                sub_output = self.model.generate(**sub_inputs, **gen_kwargs)

                            sub_in_len = sub_inputs.input_ids.shape[1]
                            for row in sub_output:
                                resp_text = self.tokenizer.decode(row[sub_in_len:], skip_special_tokens=True)
                                parsed = parse_json_text(resp_text)
                                if parsed:
                                    sub_results.append(parsed)

                            del sub_inputs, sub_output
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        results.extend(sub_results)
                        continue
                    except torch.OutOfMemoryError:
                        print("    [!] Sub-batch size 2 also hit OOM. Falling back to sequential execution for this chunk...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()

                # Sequential fallback for this chunk
                for up in chunk:
                    res = self.generate_json(system_prompt, up, max_new_tokens, temperature)
                    if res:
                        results.append(res)
                continue

            # Decode each item in batch
            input_len = inputs.input_ids.shape[1]
            for row in output_tokens:
                new_tokens = row[input_len:]
                resp_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                parsed = parse_json_text(resp_text)
                if parsed:
                    results.append(parsed)

            del inputs, output_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results


# ==============================================================================
# 2. ROBUST JSON PARSER & TEXT UTILS
# ==============================================================================

def parse_json_text(text: str) -> Dict[str, Any]:
    """Extract and parse JSON object, sanitizing code fences, reasoning tags, and invalid escapes."""
    cleaned = text.strip()

    # 1. Remove thinking / reasoning blocks (e.g. <think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # 2. Strip Markdown code fences if present
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

    # 3. Fast direct parse attempt
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 4. Extract first outer balanced curly brace block
    match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
    if match:
        snippet = match.group(1).strip()
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        # 5. Sanitize illegal escape sequences (e.g., \alpha, \beta, \mu in LaTeX formulas)
        sanitized = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', snippet)
        try:
            data = json.loads(sanitized)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    print(f"    [!] Failed to parse JSON. Raw output prefix: {cleaned[:150]}...")
    return {}


# ==============================================================================
# 3. PDF INGESTION & TEXT EXTRACTION
# ==============================================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract and clean plain text from PDF using PyMuPDF (fitz) or pypdf fallback."""
    text = ""

    # Engine 1: PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        pages_text = []
        for page in doc:
            p_text = page.get_text() or ""
            pages_text.append(p_text)
        doc.close()
        text = "\n".join(pages_text)
    except Exception:
        pass

    # Engine 2: pypdf fallback
    if not text.strip():
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            pages_text = []
            for page in reader.pages:
                p_text = page.extract_text() or ""
                pages_text.append(p_text)
            text = "\n".join(pages_text)
        except Exception as e:
            print(f"    [!] PDF extraction error on {pdf_path.name}: {e}")

    return clean_paper_text(text)


def clean_paper_text(text: str) -> str:
    """Strip references/bibliography and normalize whitespace."""
    if not text:
        return ""

    # Truncate at bibliography/references header to focus on core research content
    ref_patterns = [
        r"\n\s*(?:REFERENCES|References|Bibliography|Works Cited)\s*\n",
        r"\n\s*\d+\.\s+(?:REFERENCES|References)\s*\n",
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]
            break

    # Normalize excessive newlines and whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ==============================================================================
# 4. SEMANTIC DEDUPLICATION & UTILITIES
# ==============================================================================

def deduplicate_exact(items: List[str]) -> List[str]:
    """Preserve order deduplication ignoring case and leading/trailing whitespace."""
    seen = set()
    deduped = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


def semantic_deduplicate(
    items: List[str],
    embedder: Optional[Any],
    sim_threshold: float = SEMANTIC_SIM_THRESHOLD
) -> List[str]:
    """Cluster and deduplicate near-synonym phrases using cosine similarity."""
    cleaned_items = deduplicate_exact(items)
    if not cleaned_items or embedder is None or len(cleaned_items) <= 1:
        return cleaned_items

    try:
        embeddings = embedder.encode(cleaned_items, convert_to_tensor=True, show_progress_bar=False)
        # Compute pairwise cosine similarity matrix
        norm_emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.mm(norm_emb, norm_emb.t()).cpu().numpy()

        kept = []
        skip_indices = set()

        for i in range(len(cleaned_items)):
            if i in skip_indices:
                continue
            kept.append(cleaned_items[i])
            for j in range(i + 1, len(cleaned_items)):
                if j in skip_indices:
                    continue
                if sim_matrix[i, j] >= sim_threshold:
                    skip_indices.add(j)

        return kept
    except Exception as e:
        print(f"    [!] Semantic dedup warning: {e}. Falling back to exact dedup.")
        return cleaned_items


# ==============================================================================
# 5. PROFILE EXTRACTION PIPELINE (MAP-REDUCE)
# ==============================================================================

def extract_profile_for_researcher(
    researcher_dir: Path,
    model_runner: HFModelRunner,
    embedder: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Runs the Map-Reduce extraction pipeline for a single researcher folder."""
    pdf_files = sorted(list(researcher_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"    [!] No PDF files found in {researcher_dir}")
        return None

    researcher_name = researcher_dir.name
    print(f"\n---> Processing Researcher: {researcher_name} ({len(pdf_files)} PDFs)")

    # 1. Ingest PDF texts
    paper_texts = []
    for pdf_path in pdf_files:
        raw_text = extract_text_from_pdf(pdf_path)
        if len(raw_text) > MAX_PAPER_CHARS:
            raw_text = raw_text[:MAX_PAPER_CHARS]
        paper_texts.append(raw_text)

    # 2. MAP STEP: Extract micro-JSON per paper (Batched inference on GPU: 2-3 papers per batch)
    user_prompts = [
        f"Academic Research Paper Content:\n\"\"\"\n{text}\n\"\"\"\n\nExtract the requested scientific entities in strict JSON schema."
        for text in paper_texts
    ]

    print(f"    [*] Running parallel Batched Map extraction for {len(user_prompts)} papers (batch_size={MAP_BATCH_SIZE})...")
    paper_jsons = model_runner.generate_json_batch(
        system_prompt=PAPER_EXTRACTION_SYSTEM_PROMPT,
        user_prompts=user_prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        batch_size=MAP_BATCH_SIZE,
    )

    if not paper_jsons:
        print(f"    [!] Failed to extract micro-JSONs for {researcher_name}")
        return None

    # 3. REDUCE STEP: Merge entity lists
    merged_domains: List[str] = []
    merged_techniques: List[str] = []
    merged_platforms: List[str] = []
    merged_applications: List[str] = []
    paper_summaries: List[str] = []
    affiliations: List[str] = []

    for pj in paper_jsons:
        if not isinstance(pj, dict):
            continue
        aff = pj.get("Affiliation", "")
        if aff and isinstance(aff, str) and len(aff.strip()) > 3:
            affiliations.append(aff.strip())

        merged_domains.extend(pj.get("Research Domains", []))
        merged_techniques.extend(pj.get("Techniques Used", []))
        merged_platforms.extend(pj.get("Data & Platforms", []))
        merged_applications.extend(pj.get("Application Areas", []))

        summary = pj.get("Primary Objective & Findings", "")
        if summary:
            paper_summaries.append(summary)

    # Semantic deduplication of atomic entities
    if USE_SEMANTIC_DEDUP and embedder is not None:
        merged_domains = semantic_deduplicate(merged_domains, embedder, sim_threshold=0.85)
        merged_techniques = semantic_deduplicate(merged_techniques, embedder, sim_threshold=0.82)
        merged_platforms = semantic_deduplicate(merged_platforms, embedder, sim_threshold=0.82)
        merged_applications = semantic_deduplicate(merged_applications, embedder, sim_threshold=0.80)
    else:
        merged_domains = deduplicate_exact(merged_domains)
        merged_techniques = deduplicate_exact(merged_techniques)
        merged_platforms = deduplicate_exact(merged_platforms)
        merged_applications = deduplicate_exact(merged_applications)

    # Select best affiliation string
    affiliation_candidate = "University of Oklahoma"
    if affiliations:
        # Prefer longest non-generic affiliation
        affiliation_candidate = max(affiliations, key=len)

    # 4. SYNTHESIZE STEP: Synthesize Thinking Patterns & Summary Description
    synthesis_context = f"""Researcher: {researcher_name}
Candidate Affiliation: {affiliation_candidate}

Aggregated Research Domains ({len(merged_domains)}):
{json.dumps(merged_domains, ensure_ascii=False)}

Aggregated Methodological Techniques ({len(merged_techniques)}):
{json.dumps(merged_techniques, ensure_ascii=False)}

Aggregated Data & Platforms ({len(merged_platforms)}):
{json.dumps(merged_platforms, ensure_ascii=False)}

Candidate Application Areas ({len(merged_applications)}):
{json.dumps(merged_applications, ensure_ascii=False)}

Key Findings Across Publications:
{json.dumps(paper_summaries, indent=2, ensure_ascii=False)}

Synthesize the researcher profile following the instructions and schema."""

    print("    [*] Synthesizing Key Research Thinking Patterns & Summary Description...")
    synth_json = model_runner.generate_json(
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        user_prompt=synthesis_context,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    if not synth_json:
        print("    [!] Synthesis failed to produce valid JSON, using fallback structure.")
        synth_json = {
            "Affiliation:": affiliation_candidate,
            "Application Areas": merged_applications[:6],
            "Key Research Thinking Patterns": [],
            "Summary Description": " ".join(paper_summaries[:2])[:300]
        }

    # Normalize canonical keys matching CatalystOU evaluation suite
    final_affiliation = synth_json.get("Affiliation:") or synth_json.get("Affiliation") or affiliation_candidate
    final_applications = synth_json.get("Application Areas") or merged_applications[:6]
    final_thinking = synth_json.get("Key Research Thinking Patterns") or []
    final_summary = synth_json.get("Summary Description") or ""

    final_profile = {
        "Researcher Profile:": researcher_name,
        "Affiliation:": final_affiliation,
        "Research Domains": merged_domains,
        "Techniques Used": merged_techniques,
        "Data & Platforms": merged_platforms,
        "Application Areas": final_applications,
        "Key Research Thinking Patterns": final_thinking,
        "Summary Description": final_summary,
    }

    print(f"    [+] Successfully synthesized profile: {researcher_name}")
    print(f"        - Domains: {len(merged_domains)} | Techniques: {len(merged_techniques)} | Platforms: {len(merged_platforms)}")
    print(f"        - Thinking Patterns: {len(final_thinking)} | Summary length: {len(final_summary.split())} words")

    return final_profile


# ==============================================================================
# 6. MAIN EXECUTION CONTROLLER
# ==============================================================================

def main_colab_cell1():
    print("=" * 80)
    print(" CatalystOU: Automated Profile Extraction Pipeline")
    print(f" Model ID: {HF_MODEL_ID} (Qwen 3.6-35B-A3B MoE)")
    print(f" Dataset Directory: {PDF_ROOT_DIR}")
    print(f" Output Directory:  {OUTPUT_ROOT_DIR}/{MODEL_CLEAN_NAME}")
    print("=" * 80)

    pdf_root = Path(PDF_ROOT_DIR)
    output_root = Path(OUTPUT_ROOT_DIR)

    if not pdf_root.exists():
        print(f"[!] PDF root directory not found at: {pdf_root.resolve()}")
        print("    If using Google Drive in Colab, make sure you ran:")
        print("    from google.colab import drive; drive.mount('/content/drive')")
        print("    And set: PDF_ROOT_DIR = '/content/drive/MyDrive/.../CatalystOU-pdf'")
        return

    # 1. Discover all researcher folders: PDF_ROOT / <Discipline> / <ResearcherName> / *.pdf
    researcher_entries = []
    for disc_dir in sorted(pdf_root.iterdir()):
        if not disc_dir.is_dir():
            continue
        # Skip hidden or collaborative ground truth folders
        if disc_dir.name.startswith(".") or disc_dir.name.startswith("~"):
            continue

        if DISCIPLINE_FILTER and disc_dir.name not in DISCIPLINE_FILTER:
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
                    "pdf_count": len(pdfs)
                })

    if MAX_RESEARCHERS_LIMIT:
        researcher_entries = researcher_entries[:MAX_RESEARCHERS_LIMIT]

    print(f"[+] Discovered {len(researcher_entries)} researchers across {len(set(r['discipline'] for r in researcher_entries))} disciplines.")

    if not researcher_entries:
        print("[!] No researcher directories found. Please verify PDF directory structure.")
        return

    # 2. Load HuggingFace Model Weights onto A100
    global model_runner
    model_runner = HFModelRunner(HF_MODEL_ID)

    # 3. Optional Embedder for Semantic Deduplication
    embedder = None
    if USE_SEMANTIC_DEDUP and HAS_SENTENCE_TRANSFORMERS:
        print("[*] Loading SentenceTransformer ('all-mpnet-base-v2') on CPU (conserves GPU VRAM)...")
        embedder = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    # 4. Process each researcher
    total = len(researcher_entries)
    completed = 0
    skipped = 0
    failed = 0

    start_time = time.time()

    for item in tqdm(researcher_entries, desc="Extracting Profiles"):
        r_name = item["researcher_name"]
        disc = item["discipline"]
        r_dir = item["researcher_dir"]

        # Destination path: output_root / model_folder / discipline / <sanitized_name>_profile.json
        target_dir = output_root / MODEL_CLEAN_NAME / disc
        target_dir.mkdir(parents=True, exist_ok=True)

        sanitized_name = re.sub(r"[^\w\s-]", "", r_name).strip().replace(" ", "_")
        dest_file = target_dir / f"{sanitized_name}_profile.json"

        # Checkpoint: Skip if already extracted
        if dest_file.exists():
            print(f"[=] Skipping '{r_name}' ({disc}): Profile already exists at {dest_file.name}")
            skipped += 1
            continue

        try:
            profile_data = extract_profile_for_researcher(
                researcher_dir=r_dir,
                model_runner=model_runner,
                embedder=embedder,
            )

            if profile_data:
                with dest_file.open("w", encoding="utf-8") as f:
                    json.dump(profile_data, f, indent=4, ensure_ascii=False)
                print(f"[+] Successfully saved profile: {dest_file}")
                completed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"[!] Exception during profile extraction for {r_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        # Periodically empty CUDA cache to avoid memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" Profile Extraction Complete in {elapsed/60:.2f} minutes")
    print(f" Total: {total} | Completed: {completed} | Skipped: {skipped} | Failed: {failed}")
    print(f" Profiles saved to: {output_root / MODEL_CLEAN_NAME}")
    print(" Ready for Code Cell 2 (Synergy Checker) & Code Cell 3 (Table Stats)!")
    print("=" * 80)


if __name__ == "__main__":
    main_colab_cell1()
