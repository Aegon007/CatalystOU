# llm_reasoner.py

import json
from typing import Dict

def build_structured_prompt(profile_a: dict, profile_b: dict) -> str:
    """
    Convert profiles into a structured reasoning prompt.
    """
    return f"""
You are an expert research collaboration analyst.

Two researchers' historical profiles are provided below in JSON.

Your task: infer all plausible collaboration mechanisms,
and output ONLY a JSON object with the following structure:

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

Here are the profiles:

PROFILE_A:
{json.dumps(profile_a, indent=2)}

PROFILE_B:
{json.dumps(profile_b, indent=2)}

Now output ONLY the JSON object.
"""


def call_llm_reasoner(profile_a: dict, profile_b: dict, llm_cfg: dict) -> dict:
    """
    Wrapper for the LLM call.
    For now: placeholder that returns empty fields.
    You will integrate your GPT / Claude / local LLM here.
    """
    prompt = build_structured_prompt(profile_a, profile_b)

    # --- PLACEHOLDER for now ---
    # replace this with your actual LLM API call
    # Example if using OpenAI:
    #
    # from openai import OpenAI
    # client = OpenAI()
    # resp = client.chat.completions.create(
    #     model=llm_cfg["model"],
    #     messages=[{"role": "system", "content": "You output only JSON."},
    #               {"role": "user", "content": prompt}],
    #     temperature=0
    # )
    # raw_output = resp.choices[0].message["content"]
    # ----------------------------------------

    raw_output = """
{
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
}
"""

    try:
        parsed = json.loads(raw_output)
    except:
        parsed = {  # fallback empty
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
        }

    return parsed
