# ==============================================================================
# STEP 2: TRUE HISTORICAL BACKTESTING PIPELINE (PURE MOE MAP-REDUCE)
# ==============================================================================
# Model: Qwen/Qwen3.6-35B-A3B (MoE)
# Extraction Paradigm: Single-Pass Map-Reduce (Paper Extraction -> PI Synthesis)
#   [EXACT SAME EXTRACTION METHOD & PROMPTS AS EXP 1 CELL 1]
# Reasoning Paradigm: Single-Pass Synergy Reasoner (10 Ontological Categories)
#
# Datasets:
# - Positive Pairs: CatalystOU-pdf/~Collaborative Datasets/<Pair>/Publications Before Collaboration/<Author>/
# - Negative Controls: CatalystOU-pdf/<Discipline>/<Author>/ (volume-balanced to top 3 PDFs)
#
# Outputs:
# - Pre-Collab Profiles: extracted_profile_json/exp2_true_pre_collab_pure_moe/<Author>_profile.json
# - Positive Predictions: results/exp2_qwen3.6-35b-a3b_true/
# - Negative Predictions: results/exp2_qwen3.6-35b-a3b_true_negative/
# - Packaged Download: qwen3.6_exp2_true_results.zip
# ==============================================================================

import os
import re
import gc
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

try:
    from google.colab import files
    HAS_COLAB_FILES = True
except ImportError:
    HAS_COLAB_FILES = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

# ------------------------------------------------------------------------------
# 1. MODEL RUNNER & HARDWARE INTEGRATION
# ------------------------------------------------------------------------------
HF_MODEL_ID = globals().get('HF_MODEL_ID', 'Qwen/Qwen3.6-35B-A3B')

class ColabCollabRunner:
    def __init__(self, m, tok):
        self.model = m
        self.tokenizer = tok

    def generate_json(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 4096, temperature: float = 0.0):
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self.tokenizer([prompt_text], return_tensors='pt').to(self.model.device)
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'do_sample': False if temperature == 0.0 else True,
        }
        if temperature > 0.0:
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = 0.90

        with torch.inference_mode():
            output_tokens = self.model.generate(**inputs, **gen_kwargs)

        gen_slice = output_tokens[0][inputs['input_ids'].shape[1]:]
        raw_resp = self.tokenizer.decode(gen_slice, skip_special_tokens=True).strip()

        cleaned = raw_resp
        if '```json' in cleaned:
            cleaned = cleaned.split('```json')[1].split('```')[0].strip()
        elif '```' in cleaned:
            cleaned = cleaned.split('```')[1].split('```')[0].strip()

        match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        elif '{' in cleaned:
            cleaned = cleaned[cleaned.find('{'):].strip()

        try:
            return json.loads(cleaned, strict=False)
        except Exception:
            repaired = re.sub(r'\\(?![/"\\bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', cleaned)
            try:
                return json.loads(repaired, strict=False)
            except Exception:
                if repaired.count('"') % 2 != 0:
                    repaired += '"'
                open_brackets = repaired.count('[') - repaired.count(']')
                open_braces = repaired.count('{') - repaired.count('}')
                if open_brackets > 0:
                    repaired += ']' * open_brackets
                if open_braces > 0:
                    repaired += '}' * open_braces
                try:
                    return json.loads(repaired, strict=False)
                except Exception as parse_err:
                    print(f'    [!] JSON Parse error: {parse_err}. Preview: {raw_resp[:120]}')
                    return None

runner = None
if 'model_runner' in globals() and hasattr(globals()['model_runner'], 'generate_json'):
    print('[*] Reusing active model_runner from notebook memory.')
    runner = globals()['model_runner']
elif 'model' in globals() and 'tokenizer' in globals() and globals()['model'] is not None:
    print('[*] Wrapping active model & tokenizer from notebook memory.')
    runner = ColabCollabRunner(globals()['model'], globals()['tokenizer'])
else:
    print('[*] Loading Qwen 3.6-35B-A3B MoE into GPU memory (native bfloat16)...')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True, padding_side='left')
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='sdpa',
        trust_remote_code=True,
    )
    mdl.eval()
    globals()['model'] = mdl
    globals()['tokenizer'] = tok
    runner = ColabCollabRunner(mdl, tok)
    globals()['model_runner'] = runner

# ------------------------------------------------------------------------------
# 2. PDF TEXT EXTRACTION & SBERT DEDUPLICATION
# ------------------------------------------------------------------------------
sbert_embedder = None
if HAS_SBERT:
    try:
        sbert_embedder = SentenceTransformer('all-mpnet-base-v2')
    except Exception:
        sbert_embedder = None

def extract_pdf_clean_text(pdf_path: Path, max_chars: int = 18000) -> str:
    """Extract clean publication text, stripping bibliography."""
    text = ""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        text = "\n".join([page.get_text() or "" for page in doc])
        doc.close()
    except Exception:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            text = "\n".join([p.extract_text() or "" for p in reader.pages[:15]])
        except Exception:
            text = ""

    text = re.sub(r'\s+', ' ', text).strip()
    ref_match = re.search(r'\b(references|bibliography|works cited)\b', text, re.IGNORECASE)
    if ref_match and ref_match.start() > 2000:
        text = text[:ref_match.start()].strip()
    return text[:max_chars]

def deduplicate_exact(items: List[str]) -> List[str]:
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

def semantic_deduplicate(items: List[str], embedder: Optional[Any], sim_threshold: float = 0.82) -> List[str]:
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
    except Exception:
        return cleaned_items

# ------------------------------------------------------------------------------
# 3. EXACT EXP 1 CELL 1 PURE MOE EXTRACTION PROMPTS (MAP-REDUCE)
# ------------------------------------------------------------------------------
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

def extract_profile_for_researcher_pure_moe(
    researcher_name: str,
    discipline: str,
    pdf_files: List[Path],
    runner_inst: Any,
    embedder: Optional[Any]
) -> Dict[str, Any]:
    """Runs the exact Map-Reduce pipeline from pure MoE Exp 1 on ~3 pre-collaboration papers."""
    paper_texts = []
    for pdf_path in pdf_files:
        raw_text = extract_pdf_clean_text(pdf_path, max_chars=18000)
        paper_texts.append(raw_text)

    valid_texts = [t for t in paper_texts if len(t.strip()) > 100]
    if not valid_texts:
        return {
            "Researcher Profile:": researcher_name,
            "Affiliation:": "University of Oklahoma",
            "Research Domains": [discipline],
            "Techniques Used": [],
            "Data & Platforms": [],
            "Application Areas": [],
            "Key Research Thinking Patterns": [],
            "Summary Description": f"Faculty researcher in {discipline}."
        }

    # Map Step: Paper-level micro-JSONs
    paper_jsons = []
    for text in valid_texts:
        user_prompt = f"Academic Research Paper Content:\n\"\"\"\n{text}\n\"\"\"\n\nExtract the requested scientific entities in strict JSON schema."
        pj = runner_inst.generate_json(PAPER_EXTRACTION_SYSTEM_PROMPT, user_prompt, max_new_tokens=4096, temperature=0.0)
        if pj:
            paper_jsons.append(pj)

    # Reduce Step: Merge entity lists
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

    # Semantic deduplication
    if embedder is not None:
        merged_domains = semantic_deduplicate(merged_domains, embedder, 0.85)
        merged_techniques = semantic_deduplicate(merged_techniques, embedder, 0.82)
        merged_platforms = semantic_deduplicate(merged_platforms, embedder, 0.82)
        merged_applications = semantic_deduplicate(merged_applications, embedder, 0.80)
    else:
        merged_domains = deduplicate_exact(merged_domains)
        merged_techniques = deduplicate_exact(merged_techniques)
        merged_platforms = deduplicate_exact(merged_platforms)
        merged_applications = deduplicate_exact(merged_applications)

    affiliation_candidate = "University of Oklahoma"
    if affiliations:
        longest_aff = max(affiliations, key=len)
        if len(longest_aff) > 10:
            affiliation_candidate = longest_aff

    # Synthesis Pass
    synthesis_user_prompt = f"""Researcher: {researcher_name}
Affiliation: {affiliation_candidate}

Aggregated Research Domains ({len(merged_domains)} total):
{json.dumps(merged_domains, indent=2, ensure_ascii=False)}

Aggregated Techniques Used ({len(merged_techniques)} total):
{json.dumps(merged_techniques, indent=2, ensure_ascii=False)}

Aggregated Data & Platforms ({len(merged_platforms)} total):
{json.dumps(merged_platforms, indent=2, ensure_ascii=False)}

Aggregated Paper-Level Application Areas & Findings ({len(merged_applications)} items):
{json.dumps(merged_applications, indent=2, ensure_ascii=False)}

Paper Summaries across publications:
{json.dumps(paper_summaries, indent=2, ensure_ascii=False)}

Synthesize the final Researcher Profile matching the requested JSON schema."""

    synth_json = runner_inst.generate_json(SYNTHESIS_SYSTEM_PROMPT, synthesis_user_prompt, max_new_tokens=4096, temperature=0.0) or {}

    final_profile = {
        "Researcher Profile:": researcher_name,
        "Affiliation:": affiliation_candidate,
        "Research Domains": merged_domains,
        "Techniques Used": merged_techniques,
        "Data & Platforms": merged_platforms,
        "Application Areas": synth_json.get("Application Areas", merged_applications[:6]),
        "Key Research Thinking Patterns": synth_json.get("Key Research Thinking Patterns", []),
        "Summary Description": synth_json.get("Summary Description", "")
    }
    return final_profile

# ------------------------------------------------------------------------------
# 4. EXP 2 REASONING PROMPTS: SINGLE-PASS SYNERGY REASONER
# ------------------------------------------------------------------------------
COLLAB_SYSTEM_PROMPT = "You are an expert Research Collaboration Analyst. Output strictly a valid JSON object."

COLLAB_USER_PROMPT_TEMPLATE = """You have been provided with the pre-collaboration research profiles of two researchers.
Your task is to infer plausible, mechanistic collaboration opportunities based ONLY on their past work before they collaborated.

### PROFILE A:
{profile_a_str}

### PROFILE B:
{profile_b_str}

Generate specific collaboration mechanisms across these exact 10 categories in strict JSON format:
{{
  "Shared Domains": [],
  "Method-Application Synergies": [],
  "Complementary Technique Synergies": [],
  "Data-Method Synergies": [],
  "Cross-Domain Fusion Topics": [],
  "Shared Application Areas": [],
  "Joint Technique Development": [],
  "Theory-Application Synergy": [],
  "Thinking Pattern Synergies": [],
  "Future Research Directions": [],
  "Summary Collaboration Themes": ""
}}
"""

# ------------------------------------------------------------------------------
# 5. ON-THE-FLY PRE-COLLABORATION PROFILE EXTRACTION & CACHING
# ------------------------------------------------------------------------------
pre_collab_profiles_dir = Path("extracted_profile_json/exp2_true_pre_collab_pure_moe")
pre_collab_profiles_dir.mkdir(parents=True, exist_ok=True)

def get_or_extract_pure_moe_profile(researcher_name: str, discipline: str, pdf_files: List[Path]) -> Dict[str, Any]:
    clean_stem = re.sub(r'[^a-zA-Z0-9_]', '_', researcher_name.replace(' ', '_'))
    profile_path = pre_collab_profiles_dir / f"{clean_stem}_profile.json"

    if profile_path.exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data and "Research Domains" in data and len(data["Research Domains"]) > 0:
                return data
        except Exception:
            pass

    print(f"  [*] Extracting Pre-Collab Profile [Pure MoE Map-Reduce] for: {researcher_name} ({discipline}) from {len(pdf_files)} PDFs...")
    profile = extract_profile_for_researcher_pure_moe(
        researcher_name=researcher_name,
        discipline=discipline,
        pdf_files=pdf_files,
        runner_inst=runner,
        embedder=sbert_embedder
    )

    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"  [+] Saved Pre-Collab Profile: {profile_path.name}")
    return profile

# ------------------------------------------------------------------------------
# 6. BENCHMARK CASES DEFINITION (5 POSITIVE + 5 VOLUME-BALANCED NEGATIVE)
# ------------------------------------------------------------------------------
pos_out_dir = Path("results/exp2_qwen3.6-35b-a3b_true")
pos_out_dir.mkdir(parents=True, exist_ok=True)

neg_out_dir = Path("results/exp2_qwen3.6-35b-a3b_true_negative")
neg_out_dir.mkdir(parents=True, exist_ok=True)

root_pdf_dir = Path("CatalystOU-pdf")
collab_datasets_dir = root_pdf_dir / "~Collaborative Datasets"

def get_pre_collab_pdfs(pair_folder_name: str, author_folder_name: str) -> List[Path]:
    p = collab_datasets_dir / pair_folder_name / "Publications Before Collaboration" / author_folder_name
    if p.exists():
        pdfs = sorted(list(p.glob("*.pdf")))
        if pdfs:
            return pdfs
    return []

def get_negative_control_pdfs(disc: str, author_folder_name: str, max_papers: int = 3) -> List[Path]:
    p = root_pdf_dir / disc / author_folder_name
    if p.exists():
        pdfs = sorted(list(p.glob("*.pdf")))
        return pdfs[:max_papers]
    disc_dir = root_pdf_dir / disc
    if disc_dir.exists():
        for d in disc_dir.glob("*"):
            clean_search = author_folder_name.lower().replace("_", "").replace(".", "").replace(" ", "")
            clean_target = d.name.lower().replace("_", "").replace(".", "").replace(" ", "")
            if d.is_dir() and clean_search in clean_target:
                pdfs = sorted(list(d.glob("*.pdf")))
                return pdfs[:max_papers]
    return []

all_cases = [
    # 5 Positive Historical Cases
    {
        "type": "positive",
        "id": "case_001_ebert_xiao_cs_bio",
        "desc": "CS (David Ebert) + Bio (Xiangming Xiao)",
        "a_name": "David S. Ebert", "a_disc": "CS",
        "a_pdfs": get_pre_collab_pdfs("David S. Ebert & Xiangming Xiao (CS-Bio)", "David S. Ebert"),
        "b_name": "Xiangming Xiao", "b_disc": "Biology",
        "b_pdfs": get_pre_collab_pdfs("David S. Ebert & Xiangming Xiao (CS-Bio)", "Xiangming Xiao"),
        "out_file": pos_out_dir / "case_001_ebert_xiao_cs_bio_prediction.json"
    },
    {
        "type": "positive",
        "id": "case_002_zhao_cheng_cs_ece",
        "desc": "CS (Shangqing Zhao) + ECE (Samuel Cheng)",
        "a_name": "Shangqing Zhao", "a_disc": "CS",
        "a_pdfs": get_pre_collab_pdfs("Shangqing Zhao & Samuel Cheng (CS-ECE)", "Shangqing Zhao"),
        "b_name": "Samuel Cheng", "b_disc": "ECE",
        "b_pdfs": get_pre_collab_pdfs("Shangqing Zhao & Samuel Cheng (CS-ECE)", "Samuel Cheng"),
        "out_file": pos_out_dir / "case_002_zhao_cheng_cs_ece_prediction.json"
    },
    {
        "type": "positive",
        "id": "case_003_cai_song_psych_psych",
        "desc": "Psych (Huajian Cai) + Psych (Hairong Song)",
        "a_name": "Huajian Cai", "a_disc": "Psychology",
        "a_pdfs": get_pre_collab_pdfs("Huajian Cai & Hairong Song (Psych-Psych)", "Huajian Cai"),
        "b_name": "Hairong Song", "b_disc": "Psychology",
        "b_pdfs": get_pre_collab_pdfs("Huajian Cai & Hairong Song (Psych-Psych)", "Hairong Song"),
        "out_file": pos_out_dir / "case_003_cai_song_psych_psych_prediction.json"
    },
    {
        "type": "positive",
        "id": "case_004_yeary_havlicek_ece_ece",
        "desc": "ECE (Mark Yeary) + ECE (Joseph Havlicek)",
        "a_name": "Mark Yeary", "a_disc": "ECE",
        "a_pdfs": get_pre_collab_pdfs("Mark Yeary & Joseph Havlicek (ECE-ECE)", "Mark Yeary"),
        "b_name": "Joseph Havlicek", "b_disc": "ECE",
        "b_pdfs": get_pre_collab_pdfs("Mark Yeary & Joseph Havlicek (ECE-ECE)", "Joseph Havlicek"),
        "out_file": pos_out_dir / "case_004_yeary_havlicek_ece_ece_prediction.json"
    },
    {
        "type": "positive",
        "id": "case_005_radhakrishnan_antonio_cs_cs",
        "desc": "CS (Sridhar Radhakrishnan) + CS (John Antonio)",
        "a_name": "Sridhar Radhakrishnan", "a_disc": "CS",
        "a_pdfs": get_pre_collab_pdfs("Sridhar Radhakrishnan & John Antonio (CS-CS)", "Sridhar Radhakrishnan"),
        "b_name": "John Antonio", "b_disc": "CS",
        "b_pdfs": get_pre_collab_pdfs("Sridhar Radhakrishnan & John Antonio (CS-CS)", "John Antonio"),
        "out_file": pos_out_dir / "case_005_radhakrishnan_antonio_cs_cs_prediction.json"
    },
    # 5 Volume-Balanced Negative Controls (from benchmark cases/negative_cases.json)
    {
        "type": "negative",
        "id": "neg_001_maiti_cartersowell",
        "desc": "CS (Anindya Maiti) + Psych (Adrienne R. Carter-Sowell)",
        "a_name": "Anindya Maiti", "a_disc": "CS",
        "a_pdfs": get_negative_control_pdfs("CS", "Anindya Maiti"),
        "b_name": "Adrienne R. Carter-Sowell", "b_disc": "Psychology",
        "b_pdfs": get_negative_control_pdfs("Psychology", "Adrienne R. Carter-Sowell"),
        "out_file": neg_out_dir / "neg_001_maiti_cartersowell_prediction.json"
    },
    {
        "type": "negative",
        "id": "neg_002_yeary_dunn",
        "desc": "ECE (Mark Yeary) + Bio (Anne K. Dunn)",
        "a_name": "Mark Yeary", "a_disc": "ECE",
        "a_pdfs": get_negative_control_pdfs("ECE", "Mark Yeary"),
        "b_name": "Anne K. Dunn", "b_disc": "Biology",
        "b_pdfs": get_negative_control_pdfs("Biology", "Anne K. Dunn"),
        "out_file": neg_out_dir / "neg_002_yeary_dunn_prediction.json"
    },
    {
        "type": "negative",
        "id": "neg_003_forester_quadri",
        "desc": "Math (Max Forester) + CS (Ghulam Quadri)",
        "a_name": "Max Forester", "a_disc": "Mathematics",
        "a_pdfs": get_negative_control_pdfs("Mathematics", "Max Forester"),
        "b_name": "Ghulam Quadri", "b_disc": "CS",
        "b_pdfs": get_negative_control_pdfs("CS", "Ghulam Quadri"),
        "out_file": neg_out_dir / "neg_003_forester_quadri_prediction.json"
    },
    {
        "type": "negative",
        "id": "neg_004_karr_cokely",
        "desc": "Bio (Elizabeth Karr) + Psych (Edward T. Cokely)",
        "a_name": "Elizabeth Karr", "a_disc": "Biology",
        "a_pdfs": get_negative_control_pdfs("Biology", "Elizabeth Karr"),
        "b_name": "Edward T. Cokely", "b_disc": "Psychology",
        "b_pdfs": get_negative_control_pdfs("Psychology", "Edward T. Cokely"),
        "out_file": neg_out_dir / "neg_004_karr_cokely_prediction.json"
    },
    {
        "type": "negative",
        "id": "neg_005_jiang_martin",
        "desc": "ECE (John Jiang) + Math (Kimball Martin)",
        "a_name": "John Jiang", "a_disc": "ECE",
        "a_pdfs": get_negative_control_pdfs("ECE", "John Jiang"),
        "b_name": "Kimball Martin", "b_disc": "Mathematics",
        "b_pdfs": get_negative_control_pdfs("Mathematics", "Kimball Martin"),
        "out_file": neg_out_dir / "neg_005_jiang_martin_prediction.json"
    }
]

# ------------------------------------------------------------------------------
# 7. EXECUTION LOOP: PURE MOE MAP-REDUCE EXTRACTION -> SINGLE-PASS REASONER
# ------------------------------------------------------------------------------
print("=" * 80)
print(" RUNNING PURE MOE EXPERIMENT 2: TRUE PRE-COLLABORATION HISTORICAL BACKTESTING")
print(" Model: Qwen 3.6-35B-A3B (MoE)")
print(" Extraction Paradigm: Map-Reduce (Exact Exp 1 Cell 1 Parity)")
print(" Reasoning Paradigm: Single-Pass Synergy Reasoner (10 Ontological Categories)")
print("=" * 80)

total_completed = 0
for idx, case in enumerate(all_cases):
    case_id = case["id"]
    ptype = case["type"].upper()
    out_file = case["out_file"]
    print(f"\n[{idx+1}/10] [{ptype}] {case_id}: {case['desc']}")

    if out_file.exists():
        print(f"         [✓] Prediction already exists: {out_file.name} (Skipping)")
        total_completed += 1
        continue

    # 1. On-the-fly pre-collab extraction
    prof_a = get_or_extract_pure_moe_profile(case["a_name"], case["a_disc"], case["a_pdfs"])
    prof_b = get_or_extract_pure_moe_profile(case["b_name"], case["b_disc"], case["b_pdfs"])

    prof_a_str = json.dumps(prof_a, indent=2, ensure_ascii=False)
    prof_b_str = json.dumps(prof_b, indent=2, ensure_ascii=False)

    # 2. Single-pass collaboration reasoning
    print(f"         [*] Inferring collaboration opportunities across 10 ontology categories...")
    collab_prompt = COLLAB_USER_PROMPT_TEMPLATE.format(profile_a_str=prof_a_str, profile_b_str=prof_b_str)
    pred_collab = runner.generate_json(COLLAB_SYSTEM_PROMPT, collab_prompt, max_new_tokens=4096, temperature=0.0)

    if not pred_collab:
        pred_collab = {"Summary Collaboration Themes": "Interdisciplinary collaboration across shared domains."}

    pred_collab["_metadata"] = {
        "model": HF_MODEL_ID,
        "extraction_method": "Single-Pass Map-Reduce (Paper Extraction -> PI Synthesis)",
        "reasoning_method": "Single-Pass Synergy Reasoner (10 Ontological Categories)",
        "pair_type": ptype.lower(),
        "temporal_cutoff": "pre-collaboration historical publication split"
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(pred_collab, f, indent=2, ensure_ascii=False)
    print(f"         [+] Successfully saved prediction to: {out_file.name}")
    total_completed += 1

print("\n" + "=" * 80)
print(f" PURE MOE EXPERIMENT 2 COMPLETED: {total_completed}/10 PAIRS READY")
print("=" * 80)

# ------------------------------------------------------------------------------
# 8. PACKAGE & DOWNLOAD RESULTS ZIP
# ------------------------------------------------------------------------------
zip_path = Path("qwen3.6_exp2_true_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in sorted(pos_out_dir.glob("*.json")):
        zipf.write(f, arcname=f"exp2_qwen3.6-35b-a3b_true/{f.name}")
    for f in sorted(neg_out_dir.glob("*.json")):
        zipf.write(f, arcname=f"exp2_qwen3.6-35b-a3b_true_negative/{f.name}")

print(f"[+] Packaged true backtest pure MoE results into: {zip_path.name} ({zip_path.stat().st_size / 1024:.1f} KB)")
if HAS_COLAB_FILES:
    print("[*] Initiating automatic download...")
    files.download(str(zip_path))
else:
    print(f"[*] Results ready at: {zip_path.resolve()}")

