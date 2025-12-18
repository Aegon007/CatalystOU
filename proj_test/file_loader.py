import os
import glob


def load_profiles_from_directory(directory_path: str) -> Dict[str, Dict]:
    """Load all JSON profiles from a directory"""
    profiles = {}
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
                researcher_id = os.path.splitext(os.path.basename(file_path))[0]
                profiles[researcher_id] = profile_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return profiles

def run_evaluation(gold_dir: str, pred_dir: str, output_dir: str, tau_values: List[float] = None):
    """Main evaluation runner"""
    if tau_values is None:
        tau_values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    
    # Load profiles
    print("Loading gold profiles...")
    gold_profiles = load_profiles_from_directory(gold_dir)
    print(f"Loaded {len(gold_profiles)} gold profiles")
    
    print("Loading predicted profiles...")
    pred_profiles = load_profiles_from_directory(pred_dir)
    print(f"Loaded {len(pred_profiles)} predicted profiles")
    
    # Initialize evaluator
    evaluator = ResearcherProfileEvaluator()
    
    # Run evaluation for different tau values
    all_results = {}
    
    for tau in tau_values:
        print(f"Evaluating with tau = {tau}...")
        results = evaluator.evaluate_all(pred_profiles, gold_profiles, tau=tau)
        all_results[f"tau_{tau}"] = results
        
        # Save individual tau results
        output_file = os.path.join(output_dir, f"evaluation_results_tau_{tau}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results for tau={tau} saved to {output_file}")
    
    # Save comprehensive results
    comprehensive_output = os.path.join(output_dir, "comprehensive_evaluation_results.json")
    with open(comprehensive_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Comprehensive results saved to {comprehensive_output}")
    return all_results

# Usage example
if __name__ == "__main__":
    gold_directory = "path/to/gold_profiles"
    pred_directory = "path/to/llm_profiles" 
    output_directory = "path/to/evaluation_results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Run evaluation
    results = run_evaluation(gold_directory, pred_directory, output_directory)

