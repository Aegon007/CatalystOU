# llm_reasoner.py
import json
import time
from typing import Dict, Any


# Taxonomy of reasoning types we expect the LLM to generate
COLLAB_CATEGORIES = [
    "Shared Domains",
    "Method-Application Synergies",
    "Complementary Technique Synergies",
    "Data-Method Synergies",
    "Cross-Domain Fusion Topics",
    "Shared Application Areas",
    "Joint Technique Development",
    "Theory-Application Synergy",
    "Thinking Pattern Synergies",
    "Future Research Directions"
]

def build_collaboration_prompt(profile_a: Dict, profile_b: Dict) -> str:
    """
    Constructs a structured prompt forcing the LLM to identify mechanistic synergies.
    """
    # Helper to convert profile dict to a compact string for the prompt
    def compact_dump(d):
        return json.dumps(d, indent=2, ensure_ascii=False)

    return f"""
You are an expert Research Collaboration Analyst.
You are provided with the historical research profiles of two researchers (Author A and Author B).
Your task is to infer plausible, mechanistic collaboration opportunities based ONLY on their past work.

---
### PROFILE A (Historical)
{compact_dump(profile_a)}

### PROFILE B (Historical)
{compact_dump(profile_b)}

---
### INSTRUCTIONS
Analyze the profiles to identify how their skills, data, and domains could combine.
Generate specific collaboration mechanisms in the following categories:

1. Shared Domains: Overlapping research questions.
2. Method-Application Synergies: Author A's methods applied to Author B's problems (or vice versa).
3. Data-Method Synergies: One author's unique data enabling the other's modeling techniques.
4. Cross-Domain Fusion: Novel topics emerging from combining their distinct fields.
5. Thinking Pattern Synergies: How their cognitive research styles (e.g., theoretical vs empirical) complement each other.
6. (Include other standard categories: Complementary Techniques, Shared Applications, Joint Development, Theory-Application, Future Directions).

### OUTPUT FORMAT
You must output ONLY a valid JSON object. Do not output markdown code blocks.
The JSON must have exactly these keys, with values being lists of short, specific phrases (5-15 words each).

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
  "Summary Collaboration Themes": "A short prose summary of the collaboration logic."
}}
"""

def call_llm_reasoner(
    profile_a: Dict, 
    profile_b: Dict, 
    llm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Orchestrates the LLM call.
    Current version: PLACEHOLDER. 
    Replace the mocked section with your actual API client (OpenAI/Anthropic/Local).
    """
    prompt = build_collaboration_prompt(profile_a, profile_b)
    
    # --- MOCK LLM CALL START ---
    # In a real run, you would do:
    # response = client.chat.completions.create(model=llm_config['model'], messages=[...])
    # raw_json = response.choices[0].message.content
    
    print(f"[LLM Reasoner] Generating prediction for {llm_config.get('model', 'mock-model')}...")
    time.sleep(0.5) # Simulate latency
    
    # Return a dummy prediction structure for testing code flow
    raw_json = json.dumps({
        "Shared Domains": ["Adversarial Machine Learning"],
        "Method-Application Synergies": ["Applying Author A's random projections to Author B's social network data"],
        "Data-Method Synergies": ["Author B's Twitter dataset processed by Author A's fairness-aware algorithms"],
        "Cross-Domain Fusion Topics": ["Robustness in Social Network Analysis"],
        "Thinking Pattern Synergies": ["Combining theoretical bounds with empirical validation"],
        "Summary Collaboration Themes": "The authors will likely collaborate on robust and fair social network analysis."
    })
    # --- MOCK LLM CALL END ---

    try:
        # cleanup markdown if present
        clean_json = raw_json.replace("```json", "").replace("```", "").strip()
        parsed_output = json.loads(clean_json)
        
        # Validate schema keys exist (fill missing with empty lists)
        for cat in COLLAB_CATEGORIES:
            if cat not in parsed_output:
                parsed_output[cat] = []
        
        return parsed_output
        
    except json.JSONDecodeError:
        print("Error: LLM output was not valid JSON.")
        return {}
