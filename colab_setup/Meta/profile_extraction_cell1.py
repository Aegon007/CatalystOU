"""
================================================================================
CatalystOU - Google Colab Pipeline: Code Cell 1 (Profile Extraction)
================================================================================
Hardware Target: Google Colab A100 GPU (80GB VRAM) / Any CUDA GPU >= 24GB
Model: Meta-Llama-3.1-8B-Instruct (loaded via HuggingFace transformers in bfloat16)

Architecture:
1. Environment & HF Model Loader:
   - Loads 'meta-llama/Meta-Llama-3.1-8B-Instruct' in native bfloat16 (~16GB VRAM).
   - Supports HuggingFace authentication token (gated repo access).
   - Utilizes PyTorch SDPA (Scaled Dot-Product Attention) for high-speed inference.
2. Dataset Discovery:
   - Scans PDF directories organized by Discipline/Researcher (e.g. CatalystOU-pdf/Biology/Anne K. Dunn/*.pdf).
   - Automatically supports Google Drive mount or local workspace clones.
3. Multi-Engine PDF Text Extraction:
   - High-throughput text extraction via PyMuPDF (fitz) or pypdf.
   - Cleans and strips bibliography/reference noise to focus on core research contributions.
4. Map-Reduce Profile Extraction:
   - Map: Extracts micro-JSON ontology entities from each paper individually.
   - Reduce: Programmatically merges and semantically deduplicates entity lists.
   - Synthesize: Synthesizes 'Key Research Thinking Patterns' and 'Summary Description'.
5. Checkpointing & Output:
   - Saves final structured profiles matching CatalystOU evaluation schema.
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


# Optimize PyTorch CUDA allocator to prevent memory fragmentation on 22GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# HuggingFace Model Repo ID (Meta Llama 3.1 8B Instruct)
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")

# Paths for Dataset and Output
# If running on Colab with Drive mounted: '/content/drive/MyDrive/CatalystOU/CatalystOU-pdf'
# If running on Colab repo clone: '/content/CatalystOU/CatalystOU-pdf' or './CatalystOU-pdf'
PDF_ROOT_DIR = os.getenv("PDF_ROOT_DIR", "CatalystOU-pdf")
OUTPUT_ROOT_DIR = os.getenv("OUTPUT_ROOT_DIR", "extracted_profile_json")

# Extraction Controls
MODEL_CLEAN_NAME = HF_MODEL_ID.replace("/", "_").lower()
MAX_PAPER_CHARS = 28000       # Chars per paper sent to LLM context (~7k tokens, prevents OOM on 22GB GPUs)
MAX_NEW_TOKENS = 4096         # Max generation tokens for JSON schema
TEMPERATURE = 0.0             # 0.0 for deterministic greedy decoding
SEED = 42
USE_SEMANTIC_DEDUP = True     # Use SentenceTransformer to cluster near-identical entities
SEMANTIC_SIM_THRESHOLD = 0.82

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
# 1. HUGGING FACE AUTHENTICATION & MODEL RUNNER (Meta Llama 3.1)
# ==============================================================================

def resolve_hf_token() -> Optional[str]:
    """Retrieve Hugging Face token from environment, Colab secrets, or prompt."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token

    # Check Google Colab secrets if available
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass

    return None


class MetaLlamaModelRunner:
    """Manages local inference with Meta-Llama-3.1-8B-Instruct on GPU."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        print(f"[*] Initializing Hugging Face weights: {model_id}")
        print(f"[*] CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"[*] Primary GPU: {torch.cuda.get_device_name(0)}")
            print(f"[*] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

        hf_token = resolve_hf_token()
        if not hf_token:
            print("[!] Note: Meta Llama 3.1 is a gated repository.")
            print("    If download fails, set your HF_TOKEN in Colab Secrets or via os.environ['HF_TOKEN'].")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setup terminator tokens for Llama 3.1 (<|eot_id|> and <|end_of_text|>)
        terminators = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id not in terminators:
            terminators.append(eot_id)
        self.terminators = terminators

        # Load weights in bfloat16 (occupies ~16GB VRAM)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()

        allocated_vram = torch.cuda.memory_allocated() / (1024**3)
        print(f"[+] Model loaded successfully! VRAM Allocated: {allocated_vram:.2f} GB")

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> Dict[str, Any]:
        """Generate response from Llama 3.1 and parse into a dictionary."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.terminators,
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

            # Truncate prompt to half and retry with compact context
            truncated_user_prompt = user_prompt[:len(user_prompt) // 2]
            retry_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": truncated_user_prompt},
            ]
            retry_prompt_text = self.tokenizer.apply_chat_template(
                retry_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer([retry_prompt_text], return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                output_tokens = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        new_tokens = output_tokens[0][input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Cleanup memory after each generation
        del inputs, output_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return parse_json_text(response_text)


# ==============================================================================
# 2. ROBUST JSON PARSER & TEXT UTILS
# ==============================================================================

def parse_json_text(text: str) -> Dict[str, Any]:
    """Extract and parse JSON object, sanitizing code fences and invalid escapes."""
    cleaned = text.strip()

    # 1. Remove thinking / reasoning blocks if present
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # 2. Strip Markdown code fences if present
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    # 3. Find outer JSON boundaries
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1].strip()

    # 4. Direct JSON parsing attempt
    try:
        data = json.loads(cleaned, strict=False)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 5. Repair unescaped backslashes (common in LaTeX or scientific nomenclature e.g. \beta, \mu, \d)
    repaired = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", cleaned)

    try:
        data = json.loads(repaired, strict=False)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as err:
        print(f"    [!] JSON Parse Warning: {err}. Returning partial empty structure.")

    return {}


# ==============================================================================
# 3. PDF EXTRACTION & TEXT CLEANING
# ==============================================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from PDF using PyMuPDF (fitz) or pypdf."""
    # 1. Try PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages_text = [page.get_text() for page in doc]
        doc.close()
        full_text = "\n".join(pages_text).strip()
        if full_text:
            return clean_extracted_text(full_text)
    except Exception:
        pass

    # 2. Try pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages_text).strip()
        if full_text:
            return clean_extracted_text(full_text)
    except Exception:
        pass

    return ""


def clean_extracted_text(text: str) -> str:
    """Normalize whitespace and strip the bibliography/references section."""
    if not text:
        return ""

    text = re.sub(r"\n\s*\n+", "\n\n", text)

    cutoff_patterns = [
        r"\n(?:References|REFERENCES|Literature Cited|LITERATURE CITED|Bibliography)\s*\n",
    ]
    halfway_point = len(text) // 2
    for pattern in cutoff_patterns:
        match = re.search(pattern, text[halfway_point:])
        if match:
            text = text[: halfway_point + match.start()]
            break

    return text.strip()


# ==============================================================================
# 4. DEDUPLICATION (EXACT & SEMANTIC)
# ==============================================================================

def deduplicate_list(items: List[Any]) -> List[str]:
    """Deduplicate strings case-insensitively while preserving original casing."""
    seen = set()
    result = []
    for item in items:
        if not item or not isinstance(item, str):
            continue
        cleaned = item.strip().strip("-*• \t\r\n")
        if not cleaned or cleaned.lower() in ("not available", "none", "n/a", "null", "unknown"):
            continue
        norm = re.sub(r"\s+", " ", cleaned).lower()
        if norm not in seen:
            seen.add(norm)
            result.append(cleaned)
    return result


def semantic_deduplicate_list(
    items: List[Any],
    similarity_threshold: float = SEMANTIC_SIM_THRESHOLD,
    embedder: Optional[Any] = None,
) -> List[str]:
    """Deduplicate exact matches first, then cluster near-synonyms using embeddings."""
    base_items = deduplicate_list(items)
    if len(base_items) <= 1 or not USE_SEMANTIC_DEDUP:
        return base_items

    if not HAS_SENTENCE_TRANSFORMERS and embedder is None:
        return base_items

    try:
        if embedder is None:
            embedder = SentenceTransformer("all-mpnet-base-v2", device="cuda" if torch.cuda.is_available() else "cpu")

        embeddings = embedder.encode(base_items, normalize_embeddings=True, show_progress_bar=False)
        sim_matrix = embeddings @ embeddings.T

        merged = []
        dropped = set()
        for i in range(len(base_items)):
            if i in dropped:
                continue
            merged.append(base_items[i])
            for j in range(i + 1, len(base_items)):
                if j not in dropped and sim_matrix[i, j] >= similarity_threshold:
                    dropped.add(j)
        return merged
    except Exception as e:
        print(f"    [!] Semantic dedup fallback: {e}")
        return base_items


# ==============================================================================
# 5. RESEARCHER PIPELINE LOGIC (MAP-REDUCE)
# ==============================================================================

def extract_profile_for_researcher(
    researcher_dir: Path,
    model_runner: MetaLlamaModelRunner,
    embedder: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Execute Map-Reduce extraction across all PDFs for one researcher."""
    researcher_name = researcher_dir.name
    pdf_files = sorted(list(researcher_dir.glob("*.pdf")))

    if not pdf_files:
        print(f"[-] No PDFs found in {researcher_dir.name}. Skipping.")
        return None

    print(f"\n---> Processing Researcher: '{researcher_name}' ({len(pdf_files)} publications)")

    # 1. MAP: Extract structured micro-JSON from each publication
    valid_results = []
    for pdf_path in pdf_files:
        print(f"    [Paper Map] Reading '{pdf_path.name}'...")
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"    [!] Warning: Could not extract text from {pdf_path.name}")
            continue

        truncated_text = text[:MAX_PAPER_CHARS]
        user_prompt = f"Extract all structured entities from this research paper ({pdf_path.name}):\n\n{truncated_text}"

        paper_json = model_runner.generate_json(
            system_prompt=PAPER_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        if paper_json:
            valid_results.append(paper_json)
        else:
            print(f"    [!] Warning: Model extraction returned empty for {pdf_path.name}")

    if not valid_results:
        print(f"[-] Error: No valid extractions obtained for {researcher_name}.")
        return None

    # 2. REDUCE: Aggregate and deduplicate across publications
    all_domains: List[str] = []
    all_techniques: List[str] = []
    all_platforms: List[str] = []
    all_applications: List[str] = []
    paper_findings: List[str] = []
    affiliations: List[str] = []

    for res in valid_results:
        all_domains.extend(res.get("Research Domains", []) if isinstance(res.get("Research Domains"), list) else [])
        all_techniques.extend(res.get("Techniques Used", []) if isinstance(res.get("Techniques Used"), list) else [])
        all_platforms.extend(res.get("Data & Platforms", []) if isinstance(res.get("Data & Platforms"), list) else [])
        all_applications.extend(res.get("Application Areas", []) if isinstance(res.get("Application Areas"), list) else [])

        finding = res.get("Primary Objective & Findings")
        if finding and isinstance(finding, str):
            paper_findings.append(finding.strip())

        aff = res.get("Affiliation")
        if aff and isinstance(aff, str) and aff.strip().lower() not in ("not available", "none"):
            affiliations.append(aff.strip())

    merged_domains = semantic_deduplicate_list(all_domains, similarity_threshold=0.85, embedder=embedder)
    merged_techniques = semantic_deduplicate_list(all_techniques, similarity_threshold=0.80, embedder=embedder)
    merged_platforms = semantic_deduplicate_list(all_platforms, similarity_threshold=0.85, embedder=embedder)
    merged_applications = semantic_deduplicate_list(all_applications, similarity_threshold=0.85, embedder=embedder)
    candidate_affiliations = deduplicate_list(affiliations)

    # 3. SYNTHESIZE: Thinking Patterns & Summary Description
    print(f"    [Synthesizing Profile] Generating Key Research Thinking Patterns & Summary...")
    synthesis_user_prompt = f"""Researcher: {researcher_name}

Candidate Affiliations: {', '.join(candidate_affiliations) if candidate_affiliations else 'University of Oklahoma'}
Aggregated Research Domains: {', '.join(merged_domains)}
Aggregated Key Techniques: {', '.join(merged_techniques[:45])}
Platforms & Tools: {', '.join(merged_platforms[:30])}
Application Areas: {', '.join(merged_applications)}

Key Methodological Findings & Evidence Across Publications:
"""
    for finding in paper_findings:
        synthesis_user_prompt += f"- Finding & Evidence: {finding}\n"

    synthesis = model_runner.generate_json(
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        user_prompt=synthesis_user_prompt,
    )

    # 4. ASSEMBLE FINAL PROFILE (Exact Schema for CatalystOU Evaluation)
    affiliation_val = (
        synthesis.get("Affiliation:")
        or synthesis.get("Affiliation")
        or (candidate_affiliations[0] if candidate_affiliations else "University of Oklahoma")
    )

    # Consolidate Application Areas from synthesis if valid, otherwise fallback to top deduplicated
    synth_apps = synthesis.get("Application Areas")
    if isinstance(synth_apps, list) and len(synth_apps) >= 2:
        final_applications = deduplicate_list(synth_apps)
    else:
        final_applications = merged_applications[:6]

    final_profile = {
        "Researcher Profile:": researcher_name,
        "Affiliation:": affiliation_val,
        "Research Domains": merged_domains,
        "Techniques Used": merged_techniques,
        "Data & Platforms": merged_platforms,
        "Application Areas": final_applications,
        "Key Research Thinking Patterns": synthesis.get("Key Research Thinking Patterns", []),
        "Summary Description": synthesis.get("Summary Description", ""),
    }

    return final_profile


# ==============================================================================
# 6. DATASET DISCOVERY & EXECUTION ORCHESTRATION
# ==============================================================================

def discover_researcher_folders(pdf_root: Path) -> List[Dict[str, Any]]:
    """Scan and discover researcher folders formatted as: pdf_root/Discipline/Researcher_Name/."""
    candidates = []
    if not pdf_root.exists():
        print(f"[!] Warning: PDF root path does not exist: {pdf_root}")
        return []

    for path in sorted(pdf_root.rglob("*.pdf")):
        # Cleanly ignore any hidden (.), template, or collaborative (~) directories
        if any(part.startswith((".", "~")) for part in path.parts):
            continue

        r_dir = path.parent
        parent_dir = r_dir.parent
        discipline = parent_dir.name if parent_dir != pdf_root else "General"

        if DISCIPLINE_FILTER and discipline not in DISCIPLINE_FILTER:
            continue

        entry = {
            "researcher_name": r_dir.name,
            "discipline": discipline,
            "researcher_dir": r_dir,
        }
        if entry not in candidates:
            candidates.append(entry)

    unique_candidates = []
    seen_dirs = set()
    for item in candidates:
        if item["researcher_dir"] not in seen_dirs:
            seen_dirs.add(item["researcher_dir"])
            unique_candidates.append(item)

    return unique_candidates


def main_colab_cell1():
    """Main execution function for Code Cell 1 in Google Colab."""
    print("=" * 80)
    print(" CatalystOU: Step 1 - Researcher Profile Extraction Pipeline (Meta Llama 3.1 8B)")
    print("=" * 80)

    # 1. Resolve Dataset Paths (Auto-detecting Colab environment)
    pdf_path_candidates = [
        Path(PDF_ROOT_DIR),
        Path("/content/CatalystOU/CatalystOU-pdf"),
        Path("/content/drive/MyDrive/CatalystOU/CatalystOU-pdf"),
        Path("/content/drive/MyDrive/CatalystOU-pdf"),
        Path("../CatalystOU-pdf"),
    ]

    resolved_pdf_root = None
    for cand in pdf_path_candidates:
        if cand.exists() and cand.is_dir():
            resolved_pdf_root = cand.resolve()
            break

    if resolved_pdf_root is None:
        raise FileNotFoundError(
            f"Could not locate PDF dataset directory. Checked: {[str(c) for c in pdf_path_candidates]}.\n"
            f"Please update PDF_ROOT_DIR or mount Google Drive."
        )

    output_root = Path(OUTPUT_ROOT_DIR).resolve()
    print(f"[*] PDF Dataset Root: {resolved_pdf_root}")
    print(f"[*] Output Target Root: {output_root}")
    print(f"[*] HF Model: {HF_MODEL_ID}")

    # 2. Discover researchers
    researcher_entries = discover_researcher_folders(resolved_pdf_root)
    print(f"[*] Discovered {len(researcher_entries)} researcher directory(ies).")

    if MAX_RESEARCHERS_LIMIT is not None and MAX_RESEARCHERS_LIMIT > 0:
        researcher_entries = researcher_entries[:MAX_RESEARCHERS_LIMIT]
        print(f"[*] Limited to first {MAX_RESEARCHERS_LIMIT} researcher(s) for this run.")

    if not researcher_entries:
        print("[!] No researcher directories found. Please verify PDF directory structure.")
        return

    # 3. Load HuggingFace Model Weights onto GPU
    model_runner = MetaLlamaModelRunner(HF_MODEL_ID)

    # 4. Optional Embedder for Semantic Deduplication
    embedder = None
    if USE_SEMANTIC_DEDUP and HAS_SENTENCE_TRANSFORMERS:
        print("[*] Loading SentenceTransformer ('all-mpnet-base-v2') on CPU (conserves GPU VRAM)...")
        embedder = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    # 5. Process each researcher
    total = len(researcher_entries)
    completed = 0
    skipped = 0
    failed = 0

    start_time = time.time()

    for item in tqdm(researcher_entries, desc="Extracting Profiles"):
        r_name = item["researcher_name"]
        disc = item["discipline"]
        r_dir = item["researcher_dir"]

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
    print(" Ready for Code Cell 2 (Synergy Checker)!")
    print("=" * 80)


if __name__ == "__main__":
    main_colab_cell1()
