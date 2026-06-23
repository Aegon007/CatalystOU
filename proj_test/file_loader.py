"""Legacy profile-loading helpers used by older evaluation scripts.

The main paper experiments now use ``test1_profile_acc.py`` and
``run_experiments.py``. This module remains as a small compatibility wrapper
around the shared loader and evaluator.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from proj_test.evaluation_all_profile import ResearcherProfileEvaluator
from utils.data import ProfileLoader
from utils.profile_io import save_json


def load_profiles_from_directory(directory_path: str) -> Dict[str, Dict]:
    """Load all JSON profiles below a directory, keyed by normalized filename."""
    profiles: Dict[str, Dict] = {}
    discovered = ProfileLoader.discover_profiles(Path(directory_path), recursive=True)

    for researcher_id, file_path in discovered.items():
        profile_data = ProfileLoader.load_json_profile(file_path)
        if profile_data is not None:
            profiles[researcher_id] = profile_data

    return profiles


def run_evaluation(
    gold_dir: str,
    pred_dir: str,
    output_dir: str,
    tau_values: Optional[List[float]] = None,
) -> Dict[str, Dict]:
    """Run the legacy evaluator across several semantic thresholds."""
    if tau_values is None:
        tau_values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading gold profiles...")
    gold_profiles = load_profiles_from_directory(gold_dir)
    print(f"Loaded {len(gold_profiles)} gold profiles")

    print("Loading predicted profiles...")
    pred_profiles = load_profiles_from_directory(pred_dir)
    print(f"Loaded {len(pred_profiles)} predicted profiles")

    evaluator = ResearcherProfileEvaluator()
    all_results: Dict[str, Dict] = {}

    for tau in tau_values:
        print(f"Evaluating with tau = {tau}...")
        results = evaluator.evaluate_all(pred_profiles, gold_profiles, tau=tau)
        all_results[f"tau_{tau}"] = results

        output_file = output_path / f"evaluation_results_tau_{tau}.json"
        save_json(results, output_file)
        print(f"Results for tau={tau} saved to {output_file}")

    comprehensive_output = output_path / "comprehensive_evaluation_results.json"
    save_json(all_results, comprehensive_output)
    print(f"Comprehensive results saved to {comprehensive_output}")
    return all_results


if __name__ == "__main__":
    run_evaluation(
        "path/to/gold_profiles",
        "path/to/llm_profiles",
        "path/to/evaluation_results",
    )
