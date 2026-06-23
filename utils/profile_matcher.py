"""Compatibility wrapper for collaboration profile matching.

The publication experiments use ``proj_test.llm_reasoner`` because it returns
the 10-category collaboration mechanism schema used by Experiment 2. This file
keeps the older ``analyze_collaboration_synergy`` entry point available without
duplicating provider/client code.
"""

import asyncio
import json
from typing import Any, Dict, Optional

from proj_test.llm_reasoner import call_llm_reasoner, get_default_llm_config
from utils.profile_io import load_profile_json

# --- Part 2: Function to Read JSON Profiles ---
# This function can remain synchronous as it's a simple file operation.
def read_json_profile(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Reads a JSON file from the given path and returns it as a Python dictionary.
    """
    try:
        return load_profile_json(file_path)
    except Exception as exc:
        print(f"Error reading {file_path}: {exc}")
        return None

# --- Part 3: Synergy Analysis Function (Now Asynchronous) ---
async def analyze_collaboration_synergy(
    profile1: Dict[str, Any],
    profile2: Dict[str, Any],
    llm_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Infer collaboration mechanisms between two researcher profiles.
    """
    try:
        prediction = await call_llm_reasoner(
            profile1,
            profile2,
            llm_config or get_default_llm_config(),
        )
        return {
            "profile_a": profile1,
            "profile_b": profile2,
            "collaboration_mechanisms": prediction,
        }
    except Exception as exc:
        print(f"An error occurred during synergy analysis: {exc}")
        return None

# --- Part 4: Main Execution for Standalone Testing ---
async def main():
    """An async main function for testing this script directly."""
    profile1_filepath = "GM3.json"  # Replace with your first test file
    profile2_filepath = "SC3.json"  # Replace with your second test file

    researcher_a_profile = read_json_profile(profile1_filepath)
    researcher_b_profile = read_json_profile(profile2_filepath)

    if researcher_a_profile and researcher_b_profile:
        synergy_report_data = await analyze_collaboration_synergy(researcher_a_profile, researcher_b_profile)
        if synergy_report_data:
            print("\n--- Collaboration Synergy Report Data ---")
            print(json.dumps(synergy_report_data, indent=2))
        else:
            print("Analysis failed.")
    else:
        print("\nCould not read one or both profile files.")

if __name__ == "__main__":
    # Use asyncio.run() to execute the async main function
    asyncio.run(main())
