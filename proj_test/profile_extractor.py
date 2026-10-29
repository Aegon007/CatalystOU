"""
Profile Extractor for CatalystOU.

Map-Reduce Architecture:
1. Map: Extracts structured micro-JSONs (Domains, Techniques, Platforms, Applications, Findings) 
   from each paper individually to maximize exhaustive recall.
2. Reduce: Programmatically merges and deduplicates entity lists across papers.
3. Synthesize: Invokes the LLM to generate mechanistic 'Key Research Thinking Patterns' 
   and the final cohesive 'Summary Description' while preserving high-recall lists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger_utils import setup_logger
from utils.llm_utils import call_llm_json

logger = setup_logger(__name__, log_file="profile_extractor.log")

CONCURRENT_LIMIT = 10
DEFAULT_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")


# ----------------------------------------------------------------------
# Robust Multi-Engine PDF Extraction
# ----------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text using whichever PDF reader is installed in the active environment."""
    text = ""

    # 1. PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        if text.strip():
            return clean_extracted_text(text)
    except Exception:
        pass

    # 2. pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return clean_extracted_text(text)
    except Exception:
        pass

    # 3. PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(str(pdf_path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return clean_extracted_text(text)
    except Exception:
        pass

    # 4. pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return clean_extracted_text(text)
    except Exception:
        pass

    # 5. pdfminer.six
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(str(pdf_path))
        if text.strip():
            return clean_extracted_text(text)
    except Exception:
        pass

    logger.warning(f"Could not extract readable text from '{pdf_path.name}'.")
    return ""


def clean_extracted_text(text: str) -> str:
    """Normalize whitespace and remove reference sections to optimize context window."""
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


def deduplicate_list(items: List[Any]) -> List[str]:
    """Deduplicate strings case-insensitively while preserving original casing."""
    seen = set()
    result = []
    for item in items:
        if not item or not isinstance(item, str):
            continue
        cleaned = item.strip().strip("-*• \t\r\n")
        if not cleaned or cleaned.lower() in ("not available", "none", "n/a", "null"):
            continue
        norm = re.sub(r"\s+", " ", cleaned).lower()
        if norm not in seen:
            seen.add(norm)
            result.append(cleaned)
    return result


def semantic_deduplicate_list(
    items: List[Any], 
    similarity_threshold: float = 0.80,
    embedder: Optional[Any] = None
) -> List[str]:
    """Deduplicate exact matches first, then cluster near-synonyms using embeddings."""
    base_items = deduplicate_list(items)
    if len(base_items) <= 1:
        return base_items

    try:
        from sentence_transformers import SentenceTransformer
        if embedder is None:
            embedder = SentenceTransformer("all-mpnet-base-v2")

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
        logger.warning(f"Semantic deduplication fallback: {e}")
        return base_items


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Extraction Functions
# ----------------------------------------------------------------------

async def extract_single_paper_json(
    paper_text: str,
    paper_name: str,
    model_name: str,
    max_chars: int = 45000,
) -> Optional[Dict[str, Any]]:
    """Extract structured micro-JSON from an individual paper."""
    if not paper_text.strip():
        return None

    truncated_text = paper_text[:max_chars]
    user_prompt = f"Extract all structured entities from this research paper ({paper_name}):\n\n{truncated_text}"

    try:
        data = await call_llm_json(
            model_name=model_name,
            system_prompt=PAPER_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return data
    except Exception as e:
        logger.error(f"Failed to extract JSON from {paper_name}: {e}")
        return None


async def synthesize_global_profile(
    researcher_name: str,
    merged_domains: List[str],
    merged_techniques: List[str],
    merged_platforms: List[str],
    merged_applications: List[str],
    paper_findings: List[str],
    candidate_affiliations: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """Synthesize Thinking Patterns and Summary while preserving high-recall entity lists."""
    user_prompt = f"""Researcher: {researcher_name}

Candidate Affiliations: {', '.join(candidate_affiliations) if candidate_affiliations else 'University of Oklahoma'}
Aggregated Research Domains: {', '.join(merged_domains)}
Aggregated Key Techniques: {', '.join(merged_techniques[:45])}
Platforms & Tools: {', '.join(merged_platforms[:30])}
Application Areas: {', '.join(merged_applications)}

Key Methodological Findings & Evidence Across Publications:
"""
    for finding in paper_findings:
        user_prompt += f"- Finding & Evidence: {finding}\n"

    try:
        synthesis = await call_llm_json(
            model_name=model_name,
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as e:
        logger.error(f"Failed to generate profile synthesis for {researcher_name}: {e}")
        synthesis = {}

    # Consolidate Application Areas from synthesis if valid, otherwise fallback to top deduplicated
    synth_apps = synthesis.get("Application Areas")
    if isinstance(synth_apps, list) and len(synth_apps) >= 2:
        final_applications = deduplicate_list(synth_apps)
    else:
        final_applications = merged_applications[:6]

    final_profile = {
        "Researcher Profile:": researcher_name,
        "Affiliation:": synthesis.get("Affiliation:") or (candidate_affiliations[0] if candidate_affiliations else "University of Oklahoma"),
        "Research Domains": merged_domains,
        "Techniques Used": merged_techniques,
        "Data & Platforms": merged_platforms,
        "Application Areas": final_applications,
        "Key Research Thinking Patterns": synthesis.get("Key Research Thinking Patterns", []),
        "Summary Description": synthesis.get("Summary Description", ""),
    }

    return final_profile


# ----------------------------------------------------------------------
# Pipeline Orchestration
# ----------------------------------------------------------------------

async def process_researcher(
    researcher_dir: Path,
    output_dir: Path,
    model_name: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Process all PDFs for a single researcher folder and write the final profile JSON."""
    researcher_name = researcher_dir.name
    dept_name = researcher_dir.parent.name
    
    model_clean = model_name.replace("/", "_")
    target_dir = output_dir / model_clean / dept_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    sanitized_name = re.sub(r"[^\w\s-]", "", researcher_name).strip().replace(" ", "_")
    output_file = target_dir / f"{sanitized_name}_profile.json"

    if output_file.exists():
        logger.info(f"Profile already exists for {researcher_name} at {output_file}. Skipping.")
        return

    pdf_files = sorted(list(researcher_dir.glob("*.pdf")))
    if not pdf_files:
        logger.warning(f"No PDFs found in {researcher_dir}. Skipping.")
        return

    logger.info(f"Processing {len(pdf_files)} PDF(s) for {researcher_name}...")

    async def process_pdf(pdf_path: Path):
        async with semaphore:
            text = extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(f"Skipping {pdf_path.name} (no text extracted).")
                return None
            logger.info(f"Extracting entities from {pdf_path.name}...")
            return await extract_single_paper_json(text, pdf_path.name, model_name)

    tasks = [process_pdf(pdf) for pdf in pdf_files]
    paper_results = await asyncio.gather(*tasks)

    valid_results = [res for res in paper_results if res is not None]
    if not valid_results:
        logger.error(f"Failed to extract structured data from any PDFs for {researcher_name}.")
        return

    all_domains = []
    all_techniques = []
    all_platforms = []
    all_applications = []
    paper_findings = []
    affiliations = []

    for res in valid_results:
        all_domains.extend(res.get("Research Domains", []))
        all_techniques.extend(res.get("Techniques Used", []))
        all_platforms.extend(res.get("Data & Platforms", []))
        all_applications.extend(res.get("Application Areas", []))
        
        finding = res.get("Primary Objective & Findings")
        if finding:
            paper_findings.append(finding)
            
        aff = res.get("Affiliation")
        if aff and aff.strip() and aff.lower() not in ("not available", "none"):
            affiliations.append(aff.strip())

    merged_domains = semantic_deduplicate_list(all_domains, similarity_threshold=0.85)
    merged_techniques = semantic_deduplicate_list(all_techniques, similarity_threshold=0.80)
    merged_platforms = semantic_deduplicate_list(all_platforms, similarity_threshold=0.85)
    merged_applications = semantic_deduplicate_list(all_applications, similarity_threshold=0.85)
    candidate_affiliations = deduplicate_list(affiliations)

    logger.info(f"Synthesizing profile for {researcher_name} ({len(merged_techniques)} unique techniques found)...")
    async with semaphore:
        final_profile = await synthesize_global_profile(
            researcher_name=researcher_name,
            merged_domains=merged_domains,
            merged_techniques=merged_techniques,
            merged_platforms=merged_platforms,
            merged_applications=merged_applications,
            paper_findings=paper_findings,
            candidate_affiliations=candidate_affiliations,
            model_name=model_name,
        )

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(final_profile, f, indent=4, ensure_ascii=False)

    logger.info(f"Successfully saved profile for {researcher_name} -> {output_file}")


async def main_async(args: argparse.Namespace) -> None:
    pdf_root = Path(args.pdf_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    if not pdf_root.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_root}")

    # Discover all researcher directories containing PDFs (excluding hidden and collaborative benchmark folders)
    pdf_files = [p for p in pdf_root.rglob("*.pdf") if not any(part.startswith((".", "~")) for part in p.parts)]
    if not pdf_files:
        logger.warning(f"No PDF files found under {pdf_root}.")
        return

    researcher_dirs = sorted(list({p.parent for p in pdf_files}))

    logger.info(f"Found {len(researcher_dirs)} researcher directory(ies) to process.")
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        process_researcher(
            researcher_dir=r_dir,
            output_dir=output_root,
            model_name=args.model_name,
            semaphore=semaphore,
        )
        for r_dir in researcher_dirs
    ]

    await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="Extract researcher profiles from PDFs using Map-Reduce LLM pipeline.")
    parser.add_argument(
        "-p", "--pdf-dir",
        type=str,
        required=True,
        help="Root folder containing department/author subdirectories with PDFs (e.g. CatalystOU-pdf/Biology).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="extracted_profile_json",
        help="Root directory where output JSON profiles will be saved.",
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model identifier to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=CONCURRENT_LIMIT,
        help=f"Max concurrent LLM requests (default: {CONCURRENT_LIMIT}).",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()