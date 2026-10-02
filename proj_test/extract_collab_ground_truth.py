"""
Extract Ground Truth Collaboration Profiles from Co-authored Publications.
Processes the 5 historical collaboration pairs in CatalystOU-pdf/~Collaborative Datasets/,
reads the joint co-authored PDFs, synthesizes the 10-category collaboration schema via LLM,
and creates the ground truth JSON files and cases.json for Experiment 2.
"""

import os
import sys
import json
import re
from pathlib import Path
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
COLLAB_PDF_ROOT = BASE_DIR / "CatalystOU-pdf" / "~Collaborative Datasets"
OUTPUT_GT_DIR = BASE_DIR / "collaboration_ground_truth"
OUTPUT_GT_DIR.mkdir(parents=True, exist_ok=True)

# Provider & API configuration from .env
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
api_url = os.getenv("LLM_API_URL") or "https://openrouter.ai/api/v1"
model_name = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")

client = OpenAI(
    base_url=api_url,
    api_key=api_key,
    default_headers={"HTTP-Referer": "https://catalystou.local", "X-Title": "CatalystOU"}
)

PAIRS_CONFIG = [
    {
        "id": "case_001_ebert_xiao_cs_bio",
        "dir_name": "David S. Ebert & Xiangming Xiao (CS-Bio)",
        "researcher_a": "David S. Ebert",
        "dept_a": "CS",
        "stem_a": "David_Ebert_profile.json",
        "gold_stem_a": "David Ebert/DavidEbert_ComputerScience_Profile.json",
        "researcher_b": "Xiangming Xiao",
        "dept_b": "Biology",
        "stem_b": "Xiangming_Xiao_profile.json",
        "gold_stem_b": "Xiangming Xiao/XiangmingXiao_Biology_Profile.json",
        "collab_type": "Cross-Discipline"
    },
    {
        "id": "case_002_zhao_cheng_cs_ece",
        "dir_name": "Shangqing Zhao & Samuel Cheng (CS-ECE)",
        "researcher_a": "Shangqing Zhao",
        "dept_a": "CS",
        "stem_a": "Shangqing_Zhao_profile.json",
        "gold_stem_a": "Shangqing Zhao/ShangqingZhao_ComputerScience_Profile.json",
        "researcher_b": "Samuel Cheng",
        "dept_b": "ECE",
        "stem_b": "Samuel_Cheng_profile.json",
        "gold_stem_b": "Samuel Cheng/SamuelCheng_ECE_Profile.json",
        "collab_type": "Cross-Discipline"
    },
    {
        "id": "case_003_cai_song_psych_psych",
        "dir_name": "Huajian Cai & Hairong Song (Psych-Psych)",
        "researcher_a": "Huajian Cai",
        "dept_a": "Psychology",
        "stem_a": "Huajian_Cai_profile.json",
        "gold_stem_a": "Huajian Cai/HuajianCai_Psychology_Profile.json",
        "researcher_b": "Hairong Song",
        "dept_b": "Psychology",
        "stem_b": "Hairong_Song_profile.json",
        "gold_stem_b": "Hairong Song/HairongSong_Psychology_Profile.json",
        "collab_type": "Same-Discipline"
    },
    {
        "id": "case_004_yeary_havlicek_ece_ece",
        "dir_name": "Mark Yeary & Joseph Havlicek (ECE-ECE)",
        "researcher_a": "Mark Yeary",
        "dept_a": "ECE",
        "stem_a": "Mark_Yeary_profile.json",
        "gold_stem_a": "Mark Yeary/MarkYeary_ECE_Profile.json",
        "researcher_b": "Joseph Havlicek",
        "dept_b": "ECE",
        "stem_b": "Joseph_Havlicek_profile.json",
        "gold_stem_b": "Joseph Havlicek/JosephHavlicek_ECE_Profile.json",
        "collab_type": "Same-Discipline"
    },
    {
        "id": "case_005_radhakrishnan_antonio_cs_cs",
        "dir_name": "Sridhar Radhakrishnan & John Antonio (CS-CS)",
        "researcher_a": "Sridhar Radhakrishnan",
        "dept_a": "CS",
        "stem_a": "Sridhar_Radhakrishnan_profile.json",
        "gold_stem_a": "Sridhar Radhakrishnan/SridharRadhakrishnan_ComputerScience_Profile.json",
        "researcher_b": "John Antonio",
        "dept_b": "CS",
        "stem_b": "John_Antonio_profile.json",
        "gold_stem_b": "John Antonio/JohnAntonio_ComputerScience_Profile.json",
        "collab_type": "Same-Discipline"
    }
]


def extract_text_from_pdf(pdf_path: Path, max_chars_per_pdf: int = 25000) -> str:
    """Extract text from a PDF file using PyPDF2."""
    text_chunks = []
    total_chars = 0
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_chunks.append(page_text)
                total_chars += len(page_text)
                if total_chars >= max_chars_per_pdf:
                    break
        full_text = "\n".join(text_chunks)
        return full_text[:max_chars_per_pdf]
    except Exception as e:
        print(f"Warning: Failed to extract text from {pdf_path.name}: {e}")
        return ""


def clean_json_response(raw_text: str) -> dict:
    """Sanitize and parse JSON response from LLM."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        # Repair unescaped LaTeX backslashes
        repaired = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", cleaned)
        return json.loads(repaired, strict=False)


def extract_collab_profile(pair: dict) -> dict:
    """Process all co-authored papers for a pair and extract the ground-truth profile."""
    pair_dir = COLLAB_PDF_ROOT / pair["dir_name"]
    co_pub_dir = pair_dir / "Co-authored Publications"

    pdf_files = sorted(list(co_pub_dir.glob("*.pdf")))
    print(f"\n==========================================")
    print(f"Processing: {pair['dir_name']}")
    print(f"Found {len(pdf_files)} co-authored publication(s)")

    combined_paper_text = ""
    for idx, pdf in enumerate(pdf_files, 1):
        txt = extract_text_from_pdf(pdf)
        combined_paper_text += f"\n\n--- CO-AUTHORED PAPER {idx}: {pdf.name} ---\n{txt}\n"

    prompt = f"""You are an expert scientific collaboration analyst establishing the ground-truth collaboration profile for two researchers based strictly on their actual co-authored publications.

Researchers:
- Researcher A: {pair['researcher_a']} ({pair['dept_a']})
- Researcher B: {pair['researcher_b']} ({pair['dept_b']})
Collaboration Type: {pair['collab_type']}

Co-authored Papers Text:
{combined_paper_text[:60000]}

TASK:
Analyze the co-authored publications above to extract the exact collaboration mechanisms, synergies, and joint achievements established in this published work.
You MUST output ONLY a valid JSON object matching this exact schema:

{{
  "Shared Domains": [
    "Primary shared academic domains or overlapping subfields where this collaboration operated"
  ],
  "Method-Application Synergies": [
    "Specific instances where methods/tools from one researcher were applied to problems/domains of the other"
  ],
  "Complementary Technique Synergies": [
    "Specific technical pairings where distinct experimental or analytical tools were merged together"
  ],
  "Data-Method Synergies": [
    "Specific datasets, sensors, instruments, or benchmarks analyzed via computational/modeling methods"
  ],
  "Cross-Domain Fusion Topics": [
    "Novel hybrid topics or research intersections emerging from this collaboration"
  ],
  "Shared Application Areas": [
    "Target domains, clinical, environmental, industrial, or societal applications addressed"
  ],
  "Joint Technique Development": [
    "Newly adapted or jointly developed algorithms, models, pipelines, or experimental protocols"
  ],
  "Theory-Application Synergy": [
    "How foundational or mathematical theories directly informed empirical applications"
  ],
  "Thinking Pattern Synergies": [
    "How their complementary scientific reasoning or problem-solving approaches reinforced each other"
  ],
  "Future Research Directions": [
    "Extensions, open problems, or future avenues identified in these publications"
  ],
  "Summary Collaboration Themes": "A cohesive narrative summary under 150 words describing the core scientific contribution, combined methodologies, and impact of this collaboration."
}}

Guidelines:
- Each list must contain 2 to 5 high-signal, specific phrases (5-25 words each).
- Base all entries strictly on the provided co-authored publication text. Do NOT hallucinate methods not present in the papers.
- Do not output markdown code blocks or explanatory notes outside the JSON.
"""

    print("Querying LLM via OpenRouter for Ground Truth synthesis...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert research analyst establishing scientific collaboration ground truth. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        seed=42,
        max_tokens=4096
    )

    raw_out = response.choices[0].message.content or ""
    gt_data = clean_json_response(raw_out)
    return gt_data


def main():
    print("=== CatalystOU Experiment 2: Ground Truth Collaboration Extractor ===")
    print(f"LLM Engine: {model_name} via {api_url}")

    cases_qwen = []
    cases_llama = []
    cases_gold = []

    for pair in PAIRS_CONFIG:
        out_gt_path = OUTPUT_GT_DIR / f"{pair['id']}_gt.json"
        
        # Extract if not exists
        if not out_gt_path.exists():
            gt_data = extract_collab_profile(pair)
            with open(out_gt_path, "w", encoding="utf-8") as f:
                json.dump(gt_data, f, indent=2, ensure_ascii=False)
            print(f"Saved Ground Truth: {out_gt_path.name}")
        else:
            print(f"Ground Truth already exists: {out_gt_path.name} (skipping extraction)")

        # Case for Qwen extracted profiles
        case_qwen = {
            "id": pair["id"],
            "collab_type": pair["collab_type"],
            "researcher_a": pair["researcher_a"],
            "researcher_b": pair["researcher_b"],
            "a_path": str(BASE_DIR / "extracted_profile_json" / "qwen_qwen3.8-27b" / pair["dept_a"] / pair["stem_a"]),
            "b_path": str(BASE_DIR / "extracted_profile_json" / "qwen_qwen3.8-27b" / pair["dept_b"] / pair["stem_b"]),
            "gt_path": str(out_gt_path),
            "out_path": str(BASE_DIR / "results" / "exp2_qwen" / f"{pair['id']}_prediction.json")
        }
        cases_qwen.append(case_qwen)

        # Case for Llama extracted profiles
        case_llama = {
            "id": pair["id"],
            "collab_type": pair["collab_type"],
            "researcher_a": pair["researcher_a"],
            "researcher_b": pair["researcher_b"],
            "a_path": str(BASE_DIR / "extracted_profile_json" / "meta-llama_meta-llama-3.1-8b-instruct" / pair["dept_a"] / pair["stem_a"]),
            "b_path": str(BASE_DIR / "extracted_profile_json" / "meta-llama_meta-llama-3.1-8b-instruct" / pair["dept_b"] / pair["stem_b"]),
            "gt_path": str(out_gt_path),
            "out_path": str(BASE_DIR / "results" / "exp2_llama" / f"{pair['id']}_prediction.json")
        }
        cases_llama.append(case_llama)

        # Case for Gold Ground Truth profiles
        case_gold = {
            "id": pair["id"],
            "collab_type": pair["collab_type"],
            "researcher_a": pair["researcher_a"],
            "researcher_b": pair["researcher_b"],
            "a_path": str(BASE_DIR / "profile_labeled_data" / pair["dept_a"] / pair["gold_stem_a"]),
            "b_path": str(BASE_DIR / "profile_labeled_data" / pair["dept_b"] / pair["gold_stem_b"]),
            "gt_path": str(out_gt_path),
            "out_path": str(BASE_DIR / "results" / "exp2_gold" / f"{pair['id']}_prediction.json")
        }
        cases_gold.append(case_gold)

    # Save cases files
    cases_qwen_path = BASE_DIR / "cases_qwen.json"
    cases_llama_path = BASE_DIR / "cases_llama.json"
    cases_gold_path = BASE_DIR / "cases_gold.json"

    with open(cases_qwen_path, "w", encoding="utf-8") as f:
        json.dump(cases_qwen, f, indent=2)
    with open(cases_llama_path, "w", encoding="utf-8") as f:
        json.dump(cases_llama, f, indent=2)
    with open(cases_gold_path, "w", encoding="utf-8") as f:
        json.dump(cases_gold, f, indent=2)

    # Standard cases.json points to Qwen
    with open(BASE_DIR / "cases.json", "w", encoding="utf-8") as f:
        json.dump(cases_qwen, f, indent=2)

    print("\n==========================================")
    print("SUCCESS: All 5 Ground Truth Collaboration profiles generated!")
    print(f"Generated cases files:")
    print(f"  • {cases_qwen_path.name} (Qwen 3.8 27B extracted profiles -> cases.json)")
    print(f"  • {cases_llama_path.name} (Meta-Llama 3.1 8B extracted profiles)")
    print(f"  • {cases_gold_path.name} (Human Gold profiles)")
    print("==========================================\n")


if __name__ == "__main__":
    main()
