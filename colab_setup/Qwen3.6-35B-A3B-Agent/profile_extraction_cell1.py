"""
================================================================================
CatalystOU - Google Colab Pipeline: Code Cell 1 (Agent Profile Extraction)
================================================================================
Hardware Target: Google Colab A100 GPU (80GB VRAM) High-RAM Runtime
Model: Qwen3.6-35B-A3B (Mixture of Experts: 35B total params, 3B active params)
Context Window: 262k native context
Agent Paradigm: Two-Stage Stateful Agent (Sequential Accumulator -> Reflective PI Synthesizer)

Architecture & Operational Flow:
1. Environment & HF Model Loader:
   - Loads 'Qwen/Qwen3.6-35B-A3B' in native bfloat16 onto the A100 (~70GB VRAM).
   - Features automatic 4-bit BitsAndBytes fallback (NF4) for smaller GPUs (e.g. L4, A100 40GB).
   - Utilizes PyTorch SDPA (Scaled Dot-Product Attention) for fast MoE inference.
2. Dataset Discovery:
   - Scans PDF directories organized by Discipline/Researcher (e.g. CatalystOU-pdf/Biology/Anne K. Dunn/*.pdf).
   - Features PILOT_MODE = True by default (3 diverse faculty across Bio, CS, ECE for rapid ~2-3 min validation).
   - Set PILOT_MODE = False to run the full 50-researcher benchmark across all disciplines.
3. Multi-Engine PDF Text Extraction:
   - PyMuPDF (fitz) with automatic pypdf fallback.
   - Cleans and strips bibliography/reference noise to focus on core research contributions.
4. Two-Stage Stateful Agent Architecture:
   - STAGE 1 (The Investigator Mode - Sequential Accumulation):
     * Turn 1: Paper 1 establishes the foundational Living Dossier.
     * Turns 2..N: Papers 2..N are evaluated sequentially with 100% focused attention.
     * Extracts fine-grained techniques, experimental assays, software packages, instruments, datasets.
     * Monotonically appends newly discovered assets to guarantee 100% recall preservation.
     * Continuously evolves the career summary narrative turn-by-turn.
   - STAGE 2 (The Executive PI Mode - Reflective Macro-Synthesis):
     * Evaluates the complete accumulated working memory dossier.
     * Synthesizes and consolidates laboratory sub-tasks into cohesive, macro-level translational Application Areas.
     * Calibrates 3-4 overarching Key Research Thinking Patterns with strict 'Pattern Name: Definition (e.g., Concrete method)' syntax.
     * Polishes the cumulative Summary Description (<150 words).
5. Final Deduplication & Output:
   - SentenceTransformer (all-mpnet-base-v2) semantic clustering on Techniques, Platforms, and Domains.
   - Preserves 100% of accumulated tools while merging near-synonyms.
   - Saves final structured profiles matching CatalystOU evaluation schema to:
     extracted_profile_json/qwen_qwen3.6-35b-a3b_agent/<Discipline>/<Researcher>_profile.json
================================================================================
"""

import os
import re
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
PDF_ROOT_DIR = os.getenv("PDF_ROOT_DIR", "CatalystOU-pdf")
OUTPUT_ROOT_DIR = os.getenv("OUTPUT_ROOT_DIR", "extracted_profile_json")

# Extraction Controls
MODEL_CLEAN_NAME = "qwen_qwen3.6-35b-a3b_agent"
MAX_PAPER_CHARS = 18000       # Chars per paper (~4.5k tokens, safe native bfloat16 activation headroom)
MAX_NEW_TOKENS = 4096         # Max generation tokens for JSON schema
TEMPERATURE = 0.0             # 0.0 for deterministic greedy decoding
SEED = 42
USE_SEMANTIC_DEDUP = True     # Use SentenceTransformer to cluster near-identical entities
SEMANTIC_SIM_THRESHOLD = 0.82

# Set OVERWRITE_EXISTING = False to automatically resume and continue from existing profiles
OVERWRITE_EXISTING = os.getenv("OVERWRITE_EXISTING", "False").lower() in ("true", "1", "yes")

# ==============================================================================
# PILOT MODE CONTROL:
# - Set PILOT_MODE = True to run a quick ~2-3 min test on 3 diverse researchers.
# - Set PILOT_MODE = False to run the complete 50-researcher benchmark suite.
# ==============================================================================
PILOT_MODE = os.getenv("PILOT_MODE", "False").lower() in ("true", "1", "yes")
PILOT_LIMIT = 3
PILOT_PREFERRED_RESEARCHERS = ["Anne_K_Dunn", "David_Ebert", "Mark_Yeary"]

# Optional discipline filter (e.g. ["CS", "Biology"] or None for all)
DISCIPLINE_FILTER: Optional[List[str]] = None


# ==============================================================================
# PROMPT DEFINITIONS: TWO-STAGE STATEFUL AGENT
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 1 PROMPTS: Sequential Working Memory Accumulator
# ------------------------------------------------------------------------------

# Turn 1: Initializing the Living Researcher Dossier from Paper 1
DOSSIER_INIT_SYSTEM_PROMPT = """You are an expert scientific ontology extractor and profiling agent.
Your task is to establish a foundational researcher profile from an author's initial publication.

You MUST respond strictly with a valid JSON object matching this schema:
{
    "Affiliation": "Canonical Department / School / University Affiliation",
    "Research Domains": ["Primary Academic Discipline", "Core Subfield 1", "Core Subfield 2"],
    "Techniques Used": ["Specific Algorithm / Experimental Assay / Computational Pipeline 1", "Technique 2"],
    "Data & Platforms": ["Specific Software / Dataset / Instrument / Library 1", "Tool 2"],
    "Application Areas": ["Targeted Problem Domain / Translational Goal 1", "Application 2"],
    "Key Research Thinking Patterns": [
        "Methodological Strategy Name: Conceptual definition of how the researcher approaches problem-solving (e.g., Concrete implementation citing specific methods, datasets, or findings from the paper)."
    ],
    "Summary Description": "Cohesive summary under 150 words describing the researcher's specialization, primary model systems/frameworks, key technical methodologies, and scientific impact."
}

Extraction Guidelines:
1. Research Domains: Primary academic disciplines, specialized subfields, and domain areas directly represented in the paper.
2. Techniques Used: Specific, named methodologies, experimental assays, algorithms, computational models, mathematical frameworks, or analytical pipelines central to the paper. Extract fine-grained methodologies rather than generic labels.
3. Data & Platforms: Named physical instruments, software packages, programming libraries, databases, repositories, benchmarks, and curated datasets used in the study.
4. Application Areas: Specific problem domains, translational goals, or technological/clinical/scientific applications targeted by the work.
5. Key Research Thinking Patterns: Provide 2 to 3 prominent, hypothesis-driven scientific reasoning patterns or methodological paradigms. Each pattern MUST follow the format: 'Pattern Name: Definition (e.g., Concrete method/system cited in the paper)'.
6. Summary Description: Cohesive, concise summary (<150 words) synthesizing the researcher's core focus and methodology in this work.
"""

# Turns 2..N: Sequential Accumulator on Subsequent Papers
DOSSIER_ACCUMULATOR_SYSTEM_PROMPT = """You are an expert scientific ontology accumulator agent.
You maintain an evolving Living Researcher Dossier across an author's publication history.

You will receive:
1. The Current Living Dossier (accumulated from previous publications).
2. A New Academic Publication by this researcher.

Your mission is to read the new publication with focused attention, extract newly introduced entities that expand the researcher's portfolio, and evolve the profile synthesis.

You MUST respond strictly with a valid JSON object matching this schema:
{
    "Affiliation": "Updated canonical affiliation if this paper specifies a more detailed department, else existing affiliation",
    "New Research Domains": ["New academic subfields or domains introduced in this paper not already in the dossier"],
    "New Techniques Used": ["New specific methodologies, experimental assays, algorithms, models, or pipelines introduced in this paper not already in the dossier"],
    "New Data & Platforms": ["New named instruments, software packages, libraries, databases, or datasets used in this paper not already in the dossier"],
    "New Application Areas": ["New problem domains, experimental objectives, or translational goals targeted in this paper not already in the dossier"],
    "Updated Key Research Thinking Patterns": [
        "Methodological Strategy Name: Conceptual definition of problem-solving approach (e.g., Concrete implementation citing specific methods, datasets, or findings across the accumulated papers)."
    ],
    "Updated Summary Description": "Cohesive, cumulative summary under 150 words synthesizing the researcher's expanding specialization, methods, and scientific impact across all publications seen so far."
}

Critical Instructions:
1. ADDITIVE FOCUS FOR ENTITIES:
   - Do NOT repeat techniques, platforms, or domains already present in the Current Living Dossier.
   - Dedicate your attention to discovering fine-grained tools, specialized datasets, software packages, instrumentation, experimental assays, and mathematical algorithms in this new paper that were NOT in previous papers.
   - If this paper uses the same tools as before and introduces no new ones in a category, return an empty list [] for that field.

2. THINKING PATTERN EVOLUTION:
   - Review the researcher's cumulative trajectory. Maintain, refine, or introduce hypothesis-driven reasoning patterns so that the final list contains 3 to 4 prominent, distinct methodological paradigms across the researcher's cumulative body of work.
   - Every pattern MUST strictly follow the format: 'Pattern Name: Methodological Definition (e.g., Concrete method, assay, model system, or algorithm cited in the papers)'.
   - The parenthetical example must cite concrete technical methods or datasets, never generic placeholders like 'Paper 2'.

3. SUMMARY DESCRIPTION REFINEMENT:
   - Write a polished, cohesive summary under 150 words that captures the full spectrum of the researcher's work across all publications seen up to this point.
"""

DOSSIER_ACCUMULATOR_USER_TEMPLATE = """### RESEARCHER: {researcher_name}

### CURRENT LIVING DOSSIER (Accumulated across previous {prev_paper_count} publication(s)):
{current_dossier_json}

### NEW PUBLICATION (Publication {paper_idx} of {total_papers}):
\"\"\"
{paper_text}
\"\"\"

Analyze this new publication against the living dossier:
1. Extract newly introduced Research Domains, Techniques Used, Data & Platforms, and Application Areas that expand beyond the current dossier.
2. Evolve the 3 to 4 Key Research Thinking Patterns to incorporate new methodologies.
3. Update the cumulative Summary Description (<150 words).

Output strictly the valid JSON object."""


# ------------------------------------------------------------------------------
# STAGE 2 PROMPTS: Reflective Executive PI Macro-Synthesis
# ------------------------------------------------------------------------------

PI_SYNTHESIS_SYSTEM_PROMPT = """You are a Principal Investigator and senior academic evaluator synthesizing a comprehensive researcher profile from their accumulated publication dossier.
You will receive the complete aggregated portfolio of research techniques, platforms, subfields, experimental objectives, and publication summaries discovered across all of the author's papers.

You MUST respond strictly with a valid JSON object matching this schema:
{
    "Affiliation:": "Canonical University Department / School Affiliation",
    "Application Areas": [
        "Macro Translational Goal / Problem Domain 1",
        "Macro Translational Goal / Problem Domain 2"
    ],
    "Key Research Thinking Patterns": [
        "Methodological Strategy Name: Conceptual definition of how the researcher approaches problem-solving and hypothesis validation (e.g., Concrete implementation citing specific methods, datasets, models, or findings from the papers)."
    ],
    "Summary Description": "A cohesive summary under 150 words describing the researcher's specialization, primary model systems/frameworks, key technical methodologies, and scientific impact across their career."
}

Guidelines for Application Areas:
- Consolidate fine-grained laboratory experiments, assays, or paper-level sub-tasks into cohesive, macro-level translational goals or application domains targeted by the research.
- Do NOT list individual biochemical pathways, clinical tasks, or diagnostic assays (those belong under Techniques). Represent the broader scientific fields, industries, societal problems, or clinical domains the research addresses.
- Synthesize the portfolio into its natural overarching translational themes based on the accumulated evidence.

Guidelines for Key Research Thinking Patterns:
- Provide 3 to 4 prominent, hypothesis-driven scientific reasoning patterns or methodological paradigms that define how the researcher designs experiments, solves technical bottlenecks, or tests hypotheses.
- Format EXACTLY as: 'Pattern Name: Definition (e.g., Concrete example citing specific methods, assays, or findings from the papers)'.
- The parenthetical example must cite concrete technical methods or models from the dossier. Do NOT use paper numbers or generic labels.

Guidelines for Summary Description:
- Write a polished, cohesive summary under 150 words that weaves together the researcher's core specialization, primary frameworks/systems, methodologies, and overall scientific trajectory.
"""

PI_SYNTHESIS_USER_TEMPLATE = """### RESEARCHER: {researcher_name}
### CANDIDATE AFFILIATION: {candidate_affiliation}

### ACCUMULATED RESEARCH METHODOLOGIES ({num_techniques}):
{accumulated_techniques_json}

### ACCUMULATED DATA & PLATFORMS ({num_platforms}):
{accumulated_platforms_json}

### ACCUMULATED RESEARCH DOMAINS ({num_domains}):
{accumulated_domains_json}

### ACCUMULATED EXPERIMENTAL OBJECTIVES & FINDINGS ACROSS PAPERS:
{accumulated_findings_json}

### WORKING CAREER SUMMARY DRAFT:
"{current_summary_draft}"

Perform the Stage 2 Executive PI Synthesis:
1. Consolidate the accumulated experimental objectives into cohesive, macro-level translational Application Areas.
2. Synthesize 3 to 4 prominent Key Research Thinking Patterns with strict format: 'Pattern Name: Definition (e.g., Concrete method/system)'.
3. Polish the final Summary Description (<150 words).

Output strictly the valid JSON object matching the requested schema."""


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

        # Auto-detect if quantization is required (e.g., on L4 or A100 40GB)
        force_4bit = os.getenv("LOAD_IN_4BIT", "0") == "1"
        if (total_vram_gb > 0 and total_vram_gb < 75.0) or force_4bit:
            print(f"[*] Detected VRAM = {total_vram_gb:.2f} GB (< 75 GB threshold or force_4bit set).")
            print("[*] Loading with 4-bit BitsAndBytes quantization (NF4) to fit safely in VRAM...")
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
        else:
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
        print(f"[+] Model loaded successfully! VRAM Allocated: {allocated_vram:.2f} GB")

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

        input_len = inputs.input_ids.shape[1]
        new_tokens = output_tokens[0][input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        del inputs, output_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return parse_json_text(response_text)


# ==============================================================================
# 2. ROBUST JSON PARSER & AUTO-REPAIR
# ==============================================================================

def parse_json_text(text: str) -> Optional[Dict[str, Any]]:
    """Robustly parse JSON from LLM output with fence stripping, regex extraction, and auto-repair."""
    if not text:
        return None

    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    elif "{" in cleaned:
        cleaned = cleaned[cleaned.find("{"):].strip()

    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        repaired = re.sub(r'\\(?![/"\\bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", cleaned)
        try:
            return json.loads(repaired, strict=False)
        except Exception:
            if repaired.count('"') % 2 != 0:
                repaired += '"'
            open_brackets = repaired.count("[") - repaired.count("]")
            open_braces = repaired.count("{") - repaired.count("}")
            if open_brackets > 0:
                repaired += "]" * open_brackets
            if open_braces > 0:
                repaired += "}" * open_braces
            try:
                return json.loads(repaired, strict=False)
            except Exception:
                return None


# ==============================================================================
# 3. MULTI-ENGINE PDF TEXT EXTRACTION & SANITIZATION
# ==============================================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract clean text from PDF using PyMuPDF (fitz), pypdf, or pdfplumber."""
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
    except ImportError:
        pass
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
        except ImportError:
            # Auto-install pypdf and pymupdf if missing
            try:
                import subprocess, sys
                print("    [*] Auto-installing missing PDF parser (pypdf)...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypdf", "pymupdf"], check=True)
                import pypdf
                reader = pypdf.PdfReader(str(pdf_path))
                pages_text = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(pages_text)
            except Exception as inst_err:
                print(f"    [!] PDF auto-install failed: {inst_err}")
        except Exception as e:
            print(f"    [!] PDF extraction error on {pdf_path.name}: {e}")

    # Engine 3: pdfplumber fallback
    if not text.strip():
        try:
            import pdfplumber
            with pdfplumber.open(str(pdf_path)) as pdf:
                pages_text = [page.extract_text() or "" for page in pdf.pages]
            text = "\n".join(pages_text)
        except Exception:
            pass

    return clean_paper_text(text)


def clean_paper_text(text: str) -> str:
    """Strip references/bibliography and normalize whitespace."""
    if not text:
        return ""

    ref_patterns = [
        r"\n\s*(?:REFERENCES|References|Bibliography|Works Cited)\s*\n",
        r"\n\s*\d+\.\s+(?:REFERENCES|References)\s*\n",
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]
            break

    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ==============================================================================
# 4. DEDUPLICATION & UTILITIES
# ==============================================================================

def deduplicate_exact(items: List[str]) -> List[str]:
    """Preserve order deduplication ignoring case and whitespace."""
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
# 5. TWO-STAGE STATEFUL AGENT: ACCUMULATOR -> REFLECTIVE PI SYNTHESIZER
# ==============================================================================

def extract_profile_for_researcher_agent(
    researcher_dir: Path,
    model_runner: HFModelRunner,
    embedder: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """
    Executes the Two-Stage Stateful Agent:
    - Stage 1 (Investigator Mode): Sequential Working Memory Accumulation across papers.
    - Stage 2 (PI Reflection Mode): Macro translational consolidation and thinking pattern synthesis.
    - Final: SBERT semantic deduplication preserving 100% of accumulated tools.
    """
    pdf_files = sorted(list(researcher_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"    [!] No PDF files found in {researcher_dir}")
        return None

    researcher_name = researcher_dir.name
    total_papers = len(pdf_files)
    print(f"\n---> Processing Researcher [Two-Stage Agent]: {researcher_name} ({total_papers} PDFs)")

    # 1. Ingest all PDF texts
    paper_texts = []
    for pdf_path in pdf_files:
        raw_text = extract_text_from_pdf(pdf_path)
        if len(raw_text) > MAX_PAPER_CHARS:
            raw_text = raw_text[:MAX_PAPER_CHARS]
        paper_texts.append((pdf_path.name, raw_text))

    valid_papers = [(name, text) for name, text in paper_texts if len(text.strip()) > 100]
    if not valid_papers:
        print(f"    [!] CRITICAL ERROR: Could not extract text from any PDF for {researcher_name}!")
        print(f"        Please verify that 'pymupdf' or 'pypdf' is installed and that PDFs contain readable text.")
        return None
    total_papers = len(valid_papers)

    # --------------------------------------------------------------------------
    # STAGE 1, TURN 1: Initialize Living Dossier with Paper 1
    # --------------------------------------------------------------------------
    p1_name, p1_text = valid_papers[0]
    print(f"    [*] [Stage 1: Turn 1/{total_papers}] Ingesting Paper 1 ({p1_name}) to initialize living dossier...")

    init_user_prompt = f"""### RESEARCHER: {researcher_name}
### PUBLICATION (Paper 1 of {total_papers}):
\"\"\"
{p1_text}
\"\"\"

Extract the foundational researcher profile matching the requested schema in strict JSON."""

    init_json = model_runner.generate_json(
        system_prompt=DOSSIER_INIT_SYSTEM_PROMPT,
        user_prompt=init_user_prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    if not init_json:
        print(f"    [!] Warning: Failed to parse Paper 1 JSON, using baseline defaults.")
        init_json = {}

    affiliation_candidate = (
        init_json.get("Affiliation:")
        or init_json.get("Affiliation")
        or "University of Oklahoma"
    )

    living_dossier: Dict[str, Any] = {
        "Researcher Profile:": researcher_name,
        "Affiliation:": affiliation_candidate,
        "Research Domains": deduplicate_exact(init_json.get("Research Domains", [])),
        "Techniques Used": deduplicate_exact(init_json.get("Techniques Used", [])),
        "Data & Platforms": deduplicate_exact(init_json.get("Data & Platforms", [])),
        "Application Areas": deduplicate_exact(init_json.get("Application Areas", [])),
        "Key Research Thinking Patterns": init_json.get("Key Research Thinking Patterns", []),
        "Summary Description": init_json.get("Summary Description", ""),
    }

    print(f"    [+] [Turn 1 Initialized] Domains: {len(living_dossier['Research Domains'])}, "
          f"Techniques: {len(living_dossier['Techniques Used'])}, Platforms: {len(living_dossier['Data & Platforms'])}, "
          f"Findings/Apps: {len(living_dossier['Application Areas'])}, Patterns: {len(living_dossier['Key Research Thinking Patterns'])}")

    # --------------------------------------------------------------------------
    # STAGE 1, TURNS 2..N: Sequential Accumulation on Subsequent Papers
    # --------------------------------------------------------------------------
    for idx in range(1, total_papers):
        p_name, p_text = valid_papers[idx]
        turn_num = idx + 1
        print(f"    [*] [Stage 1: Turn {turn_num}/{total_papers}] Ingesting Paper {turn_num} ({p_name}) + Living Dossier...")

        compact_dossier = {
            "Affiliation": living_dossier["Affiliation:"],
            "Current Research Domains": living_dossier["Research Domains"],
            "Current Techniques Used": living_dossier["Techniques Used"],
            "Current Data & Platforms": living_dossier["Data & Platforms"],
            "Current Application Areas": living_dossier["Application Areas"],
            "Current Key Research Thinking Patterns": living_dossier["Key Research Thinking Patterns"],
            "Current Summary Description": living_dossier["Summary Description"],
        }

        acc_user_prompt = DOSSIER_ACCUMULATOR_USER_TEMPLATE.format(
            researcher_name=researcher_name,
            prev_paper_count=idx,
            current_dossier_json=json.dumps(compact_dossier, indent=2, ensure_ascii=False),
            paper_idx=turn_num,
            total_papers=total_papers,
            paper_text=p_text,
        )

        acc_json = model_runner.generate_json(
            system_prompt=DOSSIER_ACCUMULATOR_SYSTEM_PROMPT,
            user_prompt=acc_user_prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )

        if not acc_json:
            print(f"    [!] Warning: Turn {turn_num} returned empty JSON. Preserving current dossier.")
            continue

        def get_items(res: Dict[str, Any], *keys: str) -> List[str]:
            for k in keys:
                v = res.get(k)
                if isinstance(v, list):
                    return [str(x).strip() for x in v if x and str(x).strip()]
            return []

        new_tech = get_items(acc_json, "New Techniques Used", "Techniques Used", "Additional Techniques")
        new_plat = get_items(acc_json, "New Data & Platforms", "Data & Platforms", "Additional Platforms")
        new_dom = get_items(acc_json, "New Research Domains", "Research Domains", "Additional Domains")
        new_app = get_items(acc_json, "New Application Areas", "Application Areas", "Additional Applications")

        # Set-difference against current dossier (case-insensitive)
        curr_tech_lower = {t.lower().strip() for t in living_dossier["Techniques Used"]}
        curr_plat_lower = {p.lower().strip() for p in living_dossier["Data & Platforms"]}
        curr_dom_lower = {d.lower().strip() for d in living_dossier["Research Domains"]}
        curr_app_lower = {a.lower().strip() for a in living_dossier["Application Areas"]}

        added_tech = [t for t in new_tech if t.lower().strip() not in curr_tech_lower]
        added_plat = [p for p in new_plat if p.lower().strip() not in curr_plat_lower]
        added_dom = [d for d in new_dom if d.lower().strip() not in curr_dom_lower]
        added_app = [a for a in new_app if a.lower().strip() not in curr_app_lower]

        # Monotonic accumulation: 100% preservation + append newly discovered items
        living_dossier["Techniques Used"].extend(added_tech)
        living_dossier["Data & Platforms"].extend(added_plat)
        living_dossier["Research Domains"].extend(added_dom)
        living_dossier["Application Areas"].extend(added_app)

        living_dossier["Techniques Used"] = deduplicate_exact(living_dossier["Techniques Used"])
        living_dossier["Data & Platforms"] = deduplicate_exact(living_dossier["Data & Platforms"])
        living_dossier["Research Domains"] = deduplicate_exact(living_dossier["Research Domains"])
        living_dossier["Application Areas"] = deduplicate_exact(living_dossier["Application Areas"])

        # Evolve Thinking Patterns
        updated_thinking = (
            acc_json.get("Updated Key Research Thinking Patterns")
            or acc_json.get("Key Research Thinking Patterns")
        )
        if isinstance(updated_thinking, list) and len(updated_thinking) >= 2:
            living_dossier["Key Research Thinking Patterns"] = updated_thinking

        # Evolve Summary Description
        updated_summary = (
            acc_json.get("Updated Summary Description")
            or acc_json.get("Summary Description")
        )
        if isinstance(updated_summary, str) and len(updated_summary.strip()) > 50:
            living_dossier["Summary Description"] = updated_summary.strip()

        # Update Affiliation if more specific
        new_aff = acc_json.get("Affiliation") or acc_json.get("Affiliation:")
        if isinstance(new_aff, str) and len(new_aff.strip()) > len(living_dossier["Affiliation:"]):
            living_dossier["Affiliation:"] = new_aff.strip()

        print(f"    [+] [Stage 1: Turn {turn_num} Delta] Added: +{len(added_tech)} techniques (total {len(living_dossier['Techniques Used'])}), "
              f"+{len(added_plat)} platforms (total {len(living_dossier['Data & Platforms'])}), "
              f"+{len(added_dom)} domains (total {len(living_dossier['Research Domains'])}), "
              f"+{len(added_app)} findings (total {len(living_dossier['Application Areas'])})")

    # --------------------------------------------------------------------------
    # STAGE 2: REFLECTIVE EXECUTIVE PI SYNTHESIS PASS
    # --------------------------------------------------------------------------
    print(f"\n    [*] [Stage 2: PI Reflection] Synthesizing macro Application Areas & calibrating Thinking Patterns...")
    synthesis_context = PI_SYNTHESIS_USER_TEMPLATE.format(
        researcher_name=researcher_name,
        candidate_affiliation=living_dossier["Affiliation:"],
        num_techniques=len(living_dossier["Techniques Used"]),
        accumulated_techniques_json=json.dumps(living_dossier["Techniques Used"], ensure_ascii=False),
        num_platforms=len(living_dossier["Data & Platforms"]),
        accumulated_platforms_json=json.dumps(living_dossier["Data & Platforms"], ensure_ascii=False),
        num_domains=len(living_dossier["Research Domains"]),
        accumulated_domains_json=json.dumps(living_dossier["Research Domains"], ensure_ascii=False),
        accumulated_findings_json=json.dumps(living_dossier["Application Areas"], ensure_ascii=False),
        current_summary_draft=living_dossier["Summary Description"]
    )

    synth_json = model_runner.generate_json(
        system_prompt=PI_SYNTHESIS_SYSTEM_PROMPT,
        user_prompt=synthesis_context,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    if synth_json:
        # 1. Update Application Areas with consolidated macro translational domains
        macro_apps = synth_json.get("Application Areas")
        if isinstance(macro_apps, list) and len(macro_apps) >= 2:
            living_dossier["Application Areas"] = deduplicate_exact(macro_apps)
            print(f"    [+] [Stage 2 Synthesized] Consolidated into {len(living_dossier['Application Areas'])} macro translational domains.")

        # 2. Update Thinking Patterns with calibrated overarching paradigms
        synth_thinking = synth_json.get("Key Research Thinking Patterns")
        if isinstance(synth_thinking, list) and len(synth_thinking) >= 2:
            living_dossier["Key Research Thinking Patterns"] = synth_thinking
            print(f"    [+] [Stage 2 Synthesized] Calibrated {len(living_dossier['Key Research Thinking Patterns'])} overarching thinking patterns.")

        # 3. Polish Summary Description
        synth_summary = synth_json.get("Summary Description")
        if isinstance(synth_summary, str) and len(synth_summary.strip()) > 50:
            living_dossier["Summary Description"] = synth_summary.strip()

        # 4. Canonical affiliation
        synth_aff = synth_json.get("Affiliation:") or synth_json.get("Affiliation")
        if synth_aff and len(synth_aff) > len(living_dossier["Affiliation:"]):
            living_dossier["Affiliation:"] = synth_aff.strip()
    else:
        print("    [!] Warning: Stage 2 PI synthesis returned empty JSON, preserving Stage 1 accumulation.")

    # --------------------------------------------------------------------------
    # FINAL POST-PROCESSING: Semantic Deduplication on Accumulated Entity Lists
    # --------------------------------------------------------------------------
    print(f"    [*] Running final Semantic Deduplication on accumulated methodologies and platforms...")
    if USE_SEMANTIC_DEDUP and embedder is not None:
        final_techniques = semantic_deduplicate(living_dossier["Techniques Used"], embedder, sim_threshold=0.82)
        final_platforms = semantic_deduplicate(living_dossier["Data & Platforms"], embedder, sim_threshold=0.82)
        final_domains = semantic_deduplicate(living_dossier["Research Domains"], embedder, sim_threshold=0.85)
    else:
        final_techniques = deduplicate_exact(living_dossier["Techniques Used"])
        final_platforms = deduplicate_exact(living_dossier["Data & Platforms"])
        final_domains = deduplicate_exact(living_dossier["Research Domains"])

    final_applications = deduplicate_exact(living_dossier["Application Areas"])
    final_thinking = living_dossier["Key Research Thinking Patterns"]
    if not isinstance(final_thinking, list) or len(final_thinking) == 0:
        final_thinking = [
            f"Hypothesis-Driven Methodological Modeling: Applying rigorous computational and experimental approaches (e.g., {final_techniques[0] if final_techniques else 'Empirical analysis'})."
        ]

    final_profile = {
        "Researcher Profile:": researcher_name,
        "Affiliation:": living_dossier["Affiliation:"],
        "Research Domains": final_domains,
        "Techniques Used": final_techniques,
        "Data & Platforms": final_platforms,
        "Application Areas": final_applications,
        "Key Research Thinking Patterns": final_thinking,
        "Summary Description": living_dossier["Summary Description"],
        "_agent_metadata": {
            "model": HF_MODEL_ID,
            "architecture": "Two-Stage Stateful Agent: Sequential Accumulator -> Reflective PI Synthesizer",
            "papers_processed": total_papers,
            "stages": {
                "stage1_accumulation_turns": total_papers,
                "stage2_pi_reflection_executed": bool(synth_json),
            },
            "counts": {
                "domains": len(final_domains),
                "techniques": len(final_techniques),
                "platforms": len(final_platforms),
                "applications": len(final_applications),
                "thinking_patterns": len(final_thinking),
            }
        }
    }

    print(f"    [✓] [Final Profile Generated] Domains={len(final_domains)}, Techniques={len(final_techniques)}, "
          f"Platforms={len(final_platforms)}, Applications={len(final_applications)}, Thinking={len(final_thinking)}")
    return final_profile


def is_valid_profile_json(filepath: Path) -> bool:
    """Validate that a profile JSON exists, has non-empty valid content, and wasn't generated from failed PDF reads."""
    if not filepath.exists() or filepath.stat().st_size <= 200:
        return False
    try:
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        domains = data.get("Research Domains", [])
        techs = data.get("Techniques Used", [])
        summary = data.get("Summary Description", "")
        # Corrupted extractions when PDF parsing failed had only <= 1 dummy item
        if len(domains) <= 1 and len(techs) <= 1:
            return False
        if not summary or len(summary.strip()) < 40:
            return False
        return True
    except Exception:
        return False


# ==============================================================================
# 6. MAIN EXECUTION CONTROLLER
# ==============================================================================

def main_colab_cell1():
    print("=" * 80)
    print(" CatalystOU: Agentic Profile Extraction Pipeline (Qwen 3.6-35B-A3B MoE)")
    print(" Paradigm: Two-Stage Stateful Agent (Sequential Accumulator -> Reflective PI Synthesizer)")
    print(f" Model ID: {HF_MODEL_ID}")
    print(f" Dataset Directory: {PDF_ROOT_DIR}")
    print(f" Output Directory:  {OUTPUT_ROOT_DIR}/{MODEL_CLEAN_NAME}")
    print(f" Pilot Mode Active: {PILOT_MODE} (Pilot Limit: {PILOT_LIMIT} researchers)")
    print(f" Overwrite Existing: {OVERWRITE_EXISTING}")
    print("=" * 80)

    pdf_root = Path(PDF_ROOT_DIR)
    output_root = Path(OUTPUT_ROOT_DIR)

    if not pdf_root.exists():
        print(f"[!] PDF root directory not found at: {pdf_root.resolve()}")
        print("    If using Google Drive in Colab, please run:")
        print("    from google.colab import drive; drive.mount('/content/drive')")
        print("    And set: PDF_ROOT_DIR = '/content/drive/MyDrive/.../CatalystOU-pdf'")
        return

    # 1. Discover all researcher folders: PDF_ROOT / <Discipline> / <ResearcherName> / *.pdf
    researcher_entries = []
    for disc_dir in sorted(pdf_root.iterdir()):
        if not disc_dir.is_dir() or disc_dir.name.startswith((".", "~")):
            continue

        if DISCIPLINE_FILTER and disc_dir.name not in DISCIPLINE_FILTER:
            continue

        for r_dir in sorted(disc_dir.iterdir()):
            if not r_dir.is_dir() or r_dir.name.startswith((".", "~")):
                continue
            pdfs = list(r_dir.glob("*.pdf"))
            if pdfs:
                researcher_entries.append({
                    "discipline": disc_dir.name,
                    "researcher_name": r_dir.name,
                    "researcher_dir": r_dir,
                    "pdf_count": len(pdfs)
                })

    print(f"[+] Discovered {len(researcher_entries)} total researchers across {len(set(r['discipline'] for r in researcher_entries))} disciplines.")

    if not researcher_entries:
        print("[!] No researcher directories found. Please verify PDF directory structure.")
        return

    # Filter for PILOT MODE if enabled
    if PILOT_MODE:
        print(f"[*] Filtering to PILOT dataset ({PILOT_LIMIT} diverse researchers across disciplines)...")
        pilot_entries = []
        for pref in PILOT_PREFERRED_RESEARCHERS:
            matched = [r for r in researcher_entries if pref.lower() in r["researcher_name"].lower().replace(" ", "_")]
            if matched:
                pilot_entries.append(matched[0])
        
        # If preferred not found, take first researcher of diverse disciplines
        if len(pilot_entries) < PILOT_LIMIT:
            seen_discs = set(r["discipline"] for r in pilot_entries)
            for r in researcher_entries:
                if r["discipline"] not in seen_discs and r not in pilot_entries:
                    pilot_entries.append(r)
                    seen_discs.add(r["discipline"])
                if len(pilot_entries) >= PILOT_LIMIT:
                    break

        researcher_entries = pilot_entries[:PILOT_LIMIT]
        print(f"[+] Active Pilot Researchers ({len(researcher_entries)}):")
        for idx, r in enumerate(researcher_entries, 1):
            print(f"    {idx}. {r['discipline']} / {r['researcher_name']} ({r['pdf_count']} PDFs)")

    # 2. Load HuggingFace Model Weights onto A100 (reuse if already in memory)
    global model_runner
    if "model_runner" in globals() and hasattr(globals()["model_runner"], "generate_json"):
        print("[*] Reusing active model_runner from notebook memory.")
    else:
        model_runner = HFModelRunner(HF_MODEL_ID)
        globals()["model_runner"] = model_runner

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

    for item in tqdm(researcher_entries, desc="Agent Profile Extraction"):
        r_name = item["researcher_name"]
        disc = item["discipline"]
        r_dir = item["researcher_dir"]

        target_dir = output_root / MODEL_CLEAN_NAME / disc
        target_dir.mkdir(parents=True, exist_ok=True)

        sanitized_name = re.sub(r"[^\w\s-]", "", r_name).strip().replace(" ", "_")
        dest_file = target_dir / f"{sanitized_name}_profile.json"

        # Checkpoint / Resume check:
        if dest_file.exists() and not OVERWRITE_EXISTING:
            if is_valid_profile_json(dest_file):
                print(f"[=] RESUME SKIP: '{r_name}' ({disc}) profile exists & is valid -> Skipping ({dest_file.name})")
                skipped += 1
                continue
            else:
                print(f"[*] RE-EXTRACTING: Existing profile for '{r_name}' ({disc}) is corrupt/empty ({dest_file.stat().st_size} bytes). Re-extracting...")
                dest_file.unlink(missing_ok=True)

        try:
            profile_data = extract_profile_for_researcher_agent(
                researcher_dir=r_dir,
                model_runner=model_runner,
                embedder=embedder,
            )

            if profile_data:
                with dest_file.open("w", encoding="utf-8") as f:
                    json.dump(profile_data, f, indent=4, ensure_ascii=False)
                print(f"[+] Successfully saved agent profile: {dest_file}")
                completed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"[!] Exception during agent profile extraction for {r_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" Agent Profile Extraction Complete in {elapsed/60:.2f} minutes")
    print(f" Total: {total} | Completed: {completed} | Skipped: {skipped} | Failed: {failed}")
    print(f" Profiles saved to: {output_root / MODEL_CLEAN_NAME}")
    print(" Ready for Code Cell 2 (Agent Collaboration Backtesting) & Code Cell 3 (Table Stats)!")
    print("=" * 80)


if __name__ == "__main__":
    main_colab_cell1()
