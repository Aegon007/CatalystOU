"""
LLM-based reasoning module for inferring collaboration mechanisms.
Identifies synergies between researcher profiles using structured prompting.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_utils import create_provider, ensure_collaboration_schema
from utils.logger_utils import setup_logger

logger = setup_logger(__name__, log_file="llm_reasoner.log")


def build_collaboration_prompt(profile_a: Dict[str, Any], profile_b: Dict[str, Any]) -> str:
    """
    Construct a structured prompt for collaboration analysis.

    Args:
        profile_a: First researcher's profile
        profile_b: Second researcher's profile

    Returns:
        Formatted prompt string
    """
    return f"""
You are an expert Research Collaboration Analyst.
You have been provided with the historical research profiles of two researchers.
Your task is to infer plausible, mechanistic collaboration opportunities based ONLY on their past work.

---
### PROFILE A (Researcher)
{json.dumps(profile_a, indent=2, ensure_ascii=False)}

### PROFILE B (Researcher)
{json.dumps(profile_b, indent=2, ensure_ascii=False)}

---
### TASK
Analyze these profiles to identify how their skills, data, methods, and domains could combine productively.
Generate specific collaboration mechanisms across these 10 categories.

### OUTPUT FORMAT
You must output ONLY a valid JSON object. Do not output markdown code blocks or explanatory text.
The JSON must contain these exact keys with values as lists of specific phrases (5-20 words each):

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

Generate the JSON now:
"""


async def call_llm_reasoner(
    profile_a: Dict[str, Any],
    profile_b: Dict[str, Any],
    llm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use LLM to infer collaboration mechanisms between two researchers.

    Args:
        profile_a: First researcher's profile
        profile_b: Second researcher's profile
        llm_config: LLM configuration with provider and model details

    Returns:
        Dictionary with 10 collaboration categories and summary

    Raises:
        ValueError: If LLM provider not configured
        Exception: On API errors
    """
    provider_type = llm_config.get("provider", "openai")

    try:
        logger.info(f"Creating {provider_type} LLM provider")
        provider = create_provider(provider_type, llm_config)
    except Exception as e:
        logger.error(f"Failed to initialize LLM provider: {e}")
        raise

    prompt = build_collaboration_prompt(profile_a, profile_b)

    try:
        logger.info("Sending collaboration analysis request to LLM")
        output = await provider.generate_json(
            prompt=prompt,
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 4096)
        )

        output = ensure_collaboration_schema(output)
        logger.info("Successfully generated collaboration analysis")
        return output

    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}")
        raise ValueError(f"LLM response was not valid JSON: {e}")
    except Exception as e:
        logger.error(f"Error during LLM reasoning: {e}")
        raise


def get_default_llm_config() -> Dict[str, Any]:
    """Get default LLM configuration."""
    from utils.llm_utils import get_default_llm_config as shared_default

    return shared_default(model_name="gpt-4", provider="openai")
