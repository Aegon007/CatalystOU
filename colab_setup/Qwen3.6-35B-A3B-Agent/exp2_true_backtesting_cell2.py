# ==============================================================================
# STEP 2: TRUE HISTORICAL BACKTESTING PIPELINE (TWO-STAGE STATEFUL AGENT)
# ==============================================================================
# Model: Qwen/Qwen3.6-35B-A3B (MoE)
# Extraction Paradigm: Two-Stage Stateful Agent (Sequential Accumulator -> PI Synthesis)
#   [EXACT SAME EXTRACTION METHOD & PROMPTS AS EXP 1 CELL 1]
# Reasoning Paradigm: 2-Pass Synthesize-and-Review Reasoner (Draft -> Skeptic Audit)
#
# Datasets:
# - Positive Pairs: CatalystOU-pdf/~Collaborative Datasets/<Pair>/Publications Before Collaboration/<Author>/
# - Negative Controls: CatalystOU-pdf/<Discipline>/<Author>/ (volume-balanced to top 3 PDFs)
#
# Outputs:
# - Pre-Collab Profiles: extracted_profile_json/exp2_true_pre_collab_agent/<Author>_profile.json
# - Positive Predictions: results/exp2_qwen3.6-35b-a3b_agent_true/
# - Negative Predictions: results/exp2_qwen3.6-35b-a3b_agent_true_negative/
# - Packaged Download: qwen3.6_agent_exp2_true_results.zip
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

class ColabAgentRunner:
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
    runner = ColabAgentRunner(globals()['model'], globals()['tokenizer'])
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
    runner = ColabAgentRunner(mdl, tok)
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
# 3. EXACT EXP 1 CELL 1 AGENT EXTRACTION PROMPTS (TWO-STAGE STATEFUL AGENT)
# ------------------------------------------------------------------------------
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

def extract_profile_for_researcher_agent(
    researcher_name: str,
    discipline: str,
    pdf_files: List[Path],
    runner_inst: Any,
    embedder: Optional[Any]
) -> Dict[str, Any]:
    """Runs the exact Two-Stage Stateful Agent pipeline on ~3 pre-collaboration papers."""
    total_papers = len(pdf_files)
    paper_texts = []
    for pdf_path in pdf_files:
        raw_text = extract_pdf_clean_text(pdf_path, max_chars=18000)
        paper_texts.append((pdf_path.name, raw_text))

    valid_papers = [(name, text) for name, text in paper_texts if len(text.strip()) > 100]
    if not valid_papers:
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

    total_papers = len(valid_papers)

    # Stage 1, Turn 1: Init with Paper 1
    p1_name, p1_text = valid_papers[0]
    init_user_prompt = f"""### RESEARCHER: {researcher_name}\n### PUBLICATION (Paper 1 of {total_papers}):\n\"\"\"\n{p1_text}\n\"\"\"\n\nExtract the foundational researcher profile matching the requested schema in strict JSON."""
    init_json = runner_inst.generate_json(DOSSIER_INIT_SYSTEM_PROMPT, init_user_prompt, max_new_tokens=4096, temperature=0.0) or {}

    affiliation_candidate = init_json.get("Affiliation:") or init_json.get("Affiliation") or "University of Oklahoma"
    living_dossier = {
        "Researcher Profile:": researcher_name,
        "Affiliation:": affiliation_candidate,
        "Research Domains": deduplicate_exact(init_json.get("Research Domains", [])),
        "Techniques Used": deduplicate_exact(init_json.get("Techniques Used", [])),
        "Data & Platforms": deduplicate_exact(init_json.get("Data & Platforms", [])),
        "Application Areas": deduplicate_exact(init_json.get("Application Areas", [])),
        "Key Research Thinking Patterns": init_json.get("Key Research Thinking Patterns", []),
        "Summary Description": init_json.get("Summary Description", ""),
    }

    # Stage 1, Turns 2..N: Sequential Accumulation
    for idx in range(1, total_papers):
        p_name, p_text = valid_papers[idx]
        turn_num = idx + 1
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
        acc_json = runner_inst.generate_json(DOSSIER_ACCUMULATOR_SYSTEM_PROMPT, acc_user_prompt, max_new_tokens=4096, temperature=0.0) or {}

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

        curr_tech_lower = {t.lower().strip() for t in living_dossier["Techniques Used"]}
        curr_plat_lower = {p.lower().strip() for p in living_dossier["Data & Platforms"]}
        curr_dom_lower = {d.lower().strip() for d in living_dossier["Research Domains"]}
        curr_app_lower = {a.lower().strip() for a in living_dossier["Application Areas"]}

        added_tech = [t for t in new_tech if t.lower().strip() not in curr_tech_lower]
        added_plat = [p for p in new_plat if p.lower().strip() not in curr_plat_lower]
        added_dom = [d for d in new_dom if d.lower().strip() not in curr_dom_lower]
        added_app = [a for a in new_app if a.lower().strip() not in curr_app_lower]

        living_dossier["Techniques Used"].extend(added_tech)
        living_dossier["Data & Platforms"].extend(added_plat)
        living_dossier["Research Domains"].extend(added_dom)
        living_dossier["Application Areas"].extend(added_app)

        updated_thinking = acc_json.get("Updated Key Research Thinking Patterns") or acc_json.get("Key Research Thinking Patterns")
        if isinstance(updated_thinking, list) and len(updated_thinking) >= 2:
            living_dossier["Key Research Thinking Patterns"] = updated_thinking

        updated_summary = acc_json.get("Updated Summary Description") or acc_json.get("Summary Description")
        if isinstance(updated_summary, str) and len(updated_summary.strip()) > 50:
            living_dossier["Summary Description"] = updated_summary.strip()

    # Stage 2: PI Reflection Pass
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
    synth_json = runner_inst.generate_json(PI_SYNTHESIS_SYSTEM_PROMPT, synthesis_context, max_new_tokens=4096, temperature=0.0)
    if synth_json:
        macro_apps = synth_json.get("Application Areas")
        if isinstance(macro_apps, list) and len(macro_apps) >= 2:
            living_dossier["Application Areas"] = deduplicate_exact(macro_apps)
        synth_thinking = synth_json.get("Key Research Thinking Patterns")
        if isinstance(synth_thinking, list) and len(synth_thinking) >= 2:
            living_dossier["Key Research Thinking Patterns"] = synth_thinking
        synth_summary = synth_json.get("Summary Description")
        if isinstance(synth_summary, str) and len(synth_summary.strip()) > 50:
            living_dossier["Summary Description"] = synth_summary.strip()

    # Final Semantic Deduplication
    living_dossier["Research Domains"] = semantic_deduplicate(living_dossier["Research Domains"], embedder, 0.85)
    living_dossier["Techniques Used"] = semantic_deduplicate(living_dossier["Techniques Used"], embedder, 0.82)
    living_dossier["Data & Platforms"] = semantic_deduplicate(living_dossier["Data & Platforms"], embedder, 0.82)

    return living_dossier

# ------------------------------------------------------------------------------
# 4. EXP 2 REASONING PROMPTS: 2-PASS SYNTHESIZE-AND-REVIEW REASONER
# ------------------------------------------------------------------------------
COLLAB_DRAFT_SYSTEM_PROMPT = "You are an expert Research Collaboration Analyst. Output strictly a valid JSON object."

COLLAB_DRAFT_USER_TEMPLATE = """You have been provided with the pre-collaboration research profiles of two researchers.
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

COLLAB_REVIEW_SYSTEM_PROMPT = """You are a senior scientific collaboration reviewer. 
Your mission is to critically review prospective collaboration hypotheses between two researchers to eliminate generic platitudes.

Review Guidelines:
1. Contrastive Specificity: Ensure each synergy connects a specific tool/method from Researcher A to a concrete domain problem/data in Researcher B.
2. Eliminate generic fluff that could apply to arbitrary unrelated researchers.
3. Keep high-conviction, mechanically grounded joint pathways.
4. Output strictly a valid JSON object with the exact same 10 categories."""

COLLAB_REVIEW_USER_TEMPLATE = """### RESEARCHER A PROFILE (PRE-COLLABORATION):
{profile_a_str}

### RESEARCHER B PROFILE (PRE-COLLABORATION):
{profile_b_str}

### DRAFT COLLABORATION PROPOSAL (PASS 1):
{draft_collab_str}

Perform the review and refinement pass:
1. Prune generic claims across all 10 categories.
2. Ensure explicit causal bridges between Researcher A's tools and Researcher B's domain targets.
3. Refine 'Summary Collaboration Themes' into a focused, high-selectivity narrative.

Output strictly the final refined JSON object."""

# ------------------------------------------------------------------------------
# 5. ON-THE-FLY PRE-COLLABORATION PROFILE EXTRACTION & CACHING
# ------------------------------------------------------------------------------
pre_collab_profiles_dir = Path("extracted_profile_json/exp2_true_pre_collab_agent")
pre_collab_profiles_dir.mkdir(parents=True, exist_ok=True)

def get_or_extract_agent_profile(researcher_name: str, discipline: str, pdf_files: List[Path]) -> Dict[str, Any]:
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

    print(f"  [*] Extracting Pre-Collab Profile [Two-Stage Agent] for: {researcher_name} ({discipline}) from {len(pdf_files)} PDFs...")
    profile = extract_profile_for_researcher_agent(
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
pos_out_dir = Path("results/exp2_qwen3.6-35b-a3b_agent_true")
pos_out_dir.mkdir(parents=True, exist_ok=True)

neg_out_dir = Path("results/exp2_qwen3.6-35b-a3b_agent_true_negative")
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
# 7. EXECUTION LOOP: TWO-STAGE AGENT EXTRACTION -> 2-PASS REASONER
# ------------------------------------------------------------------------------
print("=" * 80)
print(" RUNNING AGENT EXPERIMENT 2: TRUE PRE-COLLABORATION HISTORICAL BACKTESTING")
print(" Model: Qwen 3.6-35B-A3B (MoE)")
print(" Extraction Paradigm: Two-Stage Stateful Agent (Exact Exp 1 Cell 1 Parity)")
print(" Reasoning Paradigm: 2-Pass Synthesize-and-Review Reasoner (Draft -> Review)")
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
    prof_a = get_or_extract_agent_profile(case["a_name"], case["a_disc"], case["a_pdfs"])
    prof_b = get_or_extract_agent_profile(case["b_name"], case["b_disc"], case["b_pdfs"])

    prof_a_str = json.dumps(prof_a, indent=2, ensure_ascii=False)
    prof_b_str = json.dumps(prof_b, indent=2, ensure_ascii=False)

    # 2. Pass 1: Hypothesis Synthesis
    print(f"         [*] [Pass 1] Synthesizing initial collaboration hypothesis...")
    draft_prompt = COLLAB_DRAFT_USER_TEMPLATE.format(profile_a_str=prof_a_str, profile_b_str=prof_b_str)
    draft_collab = runner.generate_json(COLLAB_DRAFT_SYSTEM_PROMPT, draft_prompt, max_new_tokens=4096, temperature=0.0)
    if not draft_collab:
        draft_collab = {"Summary Collaboration Themes": "Interdisciplinary collaboration across shared domains."}

    # 3. Pass 2: Skeptic Review & Pruning
    print(f"         [*] [Pass 2] Skeptic review & selectivity audit...")
    review_prompt = COLLAB_REVIEW_USER_TEMPLATE.format(
        profile_a_str=prof_a_str,
        profile_b_str=prof_b_str,
        draft_collab_str=json.dumps(draft_collab, indent=2, ensure_ascii=False)
    )
    final_collab = runner.generate_json(COLLAB_REVIEW_SYSTEM_PROMPT, review_prompt, max_new_tokens=4096, temperature=0.0) or draft_collab

    final_collab["_metadata"] = {
        "model": HF_MODEL_ID,
        "extraction_method": "Two-Stage Stateful Agent (Sequential Accumulator -> PI Synthesizer)",
        "reasoning_method": "2-Pass Synthesize-and-Review Reasoner (Draft -> Review)",
        "pair_type": ptype.lower(),
        "temporal_cutoff": "pre-collaboration historical publication split"
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_collab, f, indent=2, ensure_ascii=False)
    print(f"         [+] Successfully saved verified prediction to: {out_file.name}")
    total_completed += 1

print("\n" + "=" * 80)
print(f" AGENT EXPERIMENT 2 COMPLETED: {total_completed}/10 PAIRS READY")
print("=" * 80)

# ------------------------------------------------------------------------------
# 8. PACKAGE & DOWNLOAD RESULTS ZIP
# ------------------------------------------------------------------------------
zip_path = Path("qwen3.6_agent_exp2_true_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in sorted(pos_out_dir.glob("*.json")):
        zipf.write(f, arcname=f"exp2_qwen3.6-35b-a3b_agent_true/{f.name}")
    for f in sorted(neg_out_dir.glob("*.json")):
        zipf.write(f, arcname=f"exp2_qwen3.6-35b-a3b_agent_true_negative/{f.name}")

print(f"[+] Packaged true backtest agent results into: {zip_path.name} ({zip_path.stat().st_size / 1024:.1f} KB)")
if HAS_COLAB_FILES:
    print("[*] Initiating automatic download...")
    files.download(str(zip_path))
else:
    print(f"[*] Results ready at: {zip_path.resolve()}")
