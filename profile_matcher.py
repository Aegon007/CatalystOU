import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# --- Part 1: Client Setup ---
# Securely get the API key from Colab Secrets and set up the client.
try:
    api_key = userdata.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("API key not found in Colab Secrets. Please add it.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://llm.nrp-nautilus.io/"
    )
except Exception as e:
    print(f"Error setting up the client: {e}")
    client = None

# --- Part 2: Function to Read JSON Profiles ---

def read_json_profile(file_path: str) -> dict:
    """
    Reads a JSON file from the given path and returns it as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None

# --- Part 3: Synergy Analysis Function ---

def analyze_collaboration_synergy(profile1: dict, profile2: dict) -> str:
    """
    Sends two researcher profiles to the LLM for a detailed synergy analysis.

    Args:
        profile1: A dictionary representing the first researcher's profile.
        profile2: A dictionary representing the second researcher's profile.

    Returns:
        A string containing the detailed analysis report from the LLM.
    """
    if not client:
        return "Error: OpenAI client is not initialized."

    # Convert the profile dictionaries into nicely formatted JSON strings for the prompt.
    profile1_str = json.dumps(profile1, indent=2)
    profile2_str = json.dumps(profile2, indent=2)

    # The specific, detailed prompt provided by the user.
    prompt = f"""
Act as an expert research analyst and scientific collaboration strategist.

Your task is to conduct a comprehensive synergy analysis of the two research profiles provided below. Your goal is to identify specific, actionable collaboration opportunities by evaluating their research interests, methodologies, and available resources.
Provide a detailed report structured exactly as follows using markdown headings. Base your analysis strictly on the information given in the profiles.

**Researcher Profile A:**
```json
{profile1_str}
```

**Researcher Profile B:**
```json
{profile2_str}
```

---

## 1. Collaboration Synergy Score
Assign a numerical score from 1 (Low Potential) to 10 (High Potential) that represents the overall potential for a fruitful and impactful scientific collaboration between Researcher A and Researcher B. Provide a single sentence justifying the score.

## 2. Executive Summary
In a concise paragraph, summarize the core areas of potential synergy. Briefly state why a collaboration between these two researchers would be mutually beneficial and scientifically valuable.

## 3. Detailed Analysis of Synergies
### A. Overlapping and Complementary Research Interests
- Identify the primary research questions or long-term goals shared by both researchers.
- Point out any complementary areas where one researcher's focus fills a gap in the other's.

### B. Methodological & Technical Synergy
- Can Researcher A's methods, techniques, or technologies be applied to advance Researcher B's projects? If so, how? (e.g., "A's machine learning model could analyze B's large-scale genomic data to identify new patterns.")
- Can Researcher B's methods, techniques, or technologies be applied to advance Researcher A's projects? If so, how? (e.g., "B's novel imaging technique could be used to validate the cellular-level changes predicted by A's computational simulations.")

### C. Data, Resources, and Model Sharing
- Does Researcher A possess datasets, patient cohorts, software, or unique equipment that could be valuable for Researcher B's research? Explain the potential use case.
- Does Researcher B possess datasets, patient cohorts, software, or unique equipment that could be valuable for Researcher A's research? Explain the potential use case.
"""

    try:
        print("Sending profiles to LLM for synergy analysis...")
        completion = client.chat.completions.create(
            model="gemma3",
            messages=[
                {"role": "system", "content": "You are an expert research analyst and scientific collaboration strategist that outputs reports in Markdown format."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6, # A higher temperature for more creative, analytical text
            timeout=300.0, # 5-minute timeout
        )
        report = completion.choices[0].message.content
        print("Successfully generated synergy report.")
        return report
    except openai.APITimeoutError:
        print(f"Error: The request timed out after 5 minutes.")
        return "Analysis failed due to a timeout."
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return "Analysis failed due to an unexpected error."

# --- Part 4: Main Execution ---
if __name__ == "__main__":
    # IMPORTANT: Update these file paths to your uploaded JSON files.
    profile1_filepath = "GM3.json"
    profile2_filepath = "SC3.json"

    if client:
        # Read the two JSON profiles from their files.
        researcher_a_profile = read_json_profile(profile1_filepath)
        researcher_b_profile = read_json_profile(profile2_filepath)

        if researcher_a_profile and researcher_b_profile:
            # If both profiles were read successfully, run the analysis.
            synergy_report = analyze_collaboration_synergy(researcher_a_profile, researcher_b_profile)

            # Print the final report.
            print("\n--- Collaboration Synergy Report ---")
            print(synergy_report)
        else:
            print("\nCould not proceed with analysis because one or both profile files could not be read.")
    else:
        print("Script did not run because the OpenAI client could not be initialized.")
