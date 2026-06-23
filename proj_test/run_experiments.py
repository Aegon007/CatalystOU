"""
Main entry point for running CatalystOU experiments.

Provides a unified CLI for:
- Experiment 1: Profile extraction accuracy evaluation
- Experiment 2: Collaboration prediction backtesting
- Utility: Building experiment 2 case files from pair specifications

Experiments are designed by calling functions from test1_profile_acc.py and test2_pred_acc.py.

"""

import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger

# Import experiment drivers directly for synchronous calling
from proj_test.test1_profile_acc import (
    build_data_map,
    print_formatted_report,
    run_experiment_1,
)
from proj_test.test2_pred_acc import (
    run_experiment_2,
    build_cases_cli,
)
from proj_test.extract_profiles import extract_profiles


logger = setup_logger("exp_runner", log_file="experiment_runner.log")


def get_llm_config(model: str, provider: str = "openai") -> Dict[str, Any]:
    """
    Get LLM configuration for a given model.

    Args:
        model: Model identifier (e.g., 'gpt-4', 'gpt-4-turbo')
        provider: LLM provider ('openai', 'anthropic', 'local')

    Returns:
        Configuration dictionary
    """
    temperature_support = model not in {"gpt-5-nano"}
    api_mode = "responses" if model.startswith("gpt-5") else "chat"
    return {
        "provider": provider,
        "model_name": model,
        "temperature": 0.3,
        "max_tokens": 4096,
        "temperature_support": temperature_support,
        "max_retries": 3,
        "timeout": 600.0,
        "api_mode": api_mode,
    }


def build_collaboration_cases(
    pairs_file: str,
    output_file: str,
    predictions_dir: str | None = None,
) -> List[Dict[str, str]]:
    """
    Build an Experiment 2 cases file from a compact pair specification.

    Input format:
    [
      {
        "id": "case_001",
        "a_path": "path/to/researcher_a_profile.json",
        "b_path": "path/to/researcher_b_profile.json",
        "gt_path": "path/to/collaboration_groundtruth.json"
      }
    ]
    """
    with open(pairs_file, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    if not isinstance(raw_cases, list):
        raise ValueError("Pairs file must contain a JSON list of case objects")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_root = Path(predictions_dir) if predictions_dir else output_path.parent / "predictions"

    cases: List[Dict[str, str]] = []
    required = {"a_path", "b_path", "gt_path"}
    for idx, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"Case at index {idx} must be a JSON object")

        missing = required - set(item)
        if missing:
            raise ValueError(f"Case at index {idx} is missing required keys: {sorted(missing)}")

        case_id = str(item.get("id") or f"case_{idx:03d}")
        case = {
            "id": case_id,
            "a_path": str(item["a_path"]),
            "b_path": str(item["b_path"]),
            "gt_path": str(item["gt_path"]),
            "out_path": str(item.get("out_path") or pred_root / f"{case_id}_prediction.json"),
        }
        cases.append(case)

    from utils.profile_io import save_json



def create_parser():
    """Create and configure argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        description="CatalystOU Collaboration Discovery Experiments"
    )

    subparsers = parser.add_subparsers(dest="experiment", help="Experiment to run")

    # Experiment 1 parser
    exp1_parser = subparsers.add_parser("exp1", help="Profile Extraction Accuracy")
    exp1_parser.add_argument(
        "-e", "--extracted_profiles_dir",
        required=True,
        help="Directory containing LLM-extracted profiles"
    )
    exp1_parser.add_argument(
        "-g", "--gold_profiles_dir",
        required=True,
        help="Directory containing ground truth profiles"
    )
    exp1_parser.add_argument(
        "-m", "--sbert_model",
        default="all-mpnet-base-v2",
        help="Sentence BERT model name"
    )
    exp1_parser.add_argument(
        "-l", "--llm_model_name",
        default="gpt-5-nano",
        help="LLM model identifier"
    )
    exp1_parser.add_argument(
        "-o", "--output_dir",
        default="./results/exp1",
        help="Output directory for results"
    )

    # Profile extraction parser
    extract_parser = subparsers.add_parser(
        "extract-profiles",
        help="Extract researcher profiles from PDF documents"
    )
    extract_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory containing researcher folders with PDFs"
    )
    extract_parser.add_argument(
        "-o", "--output",
        default="./extracted_profiles",
        help="Output directory for extracted profiles"
    )
    extract_parser.add_argument(
        "-m", "--model",
        default="gpt-4-turbo",
        help="LLM model name (default: gpt-4-turbo)"
    )
    extract_parser.add_argument(
        "-p", "--provider",
        default="openai",
        choices=["openai", "anthropic", "local"],
        help="LLM provider (default: openai)"
    )

    # Case builder parser
    build_cases_parser = subparsers.add_parser(
        "build-cases",
        help="Build an Experiment 2 cases file from pair specifications"
    )
    build_cases_parser.add_argument(
        "-p", "--pairs_file",
        required=True,
        help="JSON list with id, a_path, b_path, and gt_path entries"
    )
    build_cases_parser.add_argument(
        "-o", "--output_file",
        required=True,
        help="Output cases JSON file"
    )
    build_cases_parser.add_argument(
        "--predictions_dir",
        default=None,
        help="Optional directory for per-case prediction output paths"
    )

    # Experiment 2 parser
    exp2_parser = subparsers.add_parser("exp2", help="Collaboration Prediction Accuracy")
    exp2_parser.add_argument(
        "-c", "--cases_file",
        required=True,
        help="JSON file with collaboration cases"
    )
    exp2_parser.add_argument(
        "-m", "--sbert_model",
        default="all-mpnet-base-v2",
        help="Sentence BERT model name"
    )
    exp2_parser.add_argument(
        "--llm_model",
        default="gpt-4",
        help="LLM model name"
    )
    exp2_parser.add_argument(
        "--llm_provider",
        default="openai",
        help="LLM provider (openai, anthropic, local)"
    )
    exp2_parser.add_argument(
        "--tau",
        type=float,
        default=0.65,
        help="Similarity threshold for matching"
    )
    exp2_parser.add_argument(
        "-o", "--output_dir",
        default="./results/exp2",
        help="Output directory for results"
    )

    return parser


def main(parser):
    """Main entry point."""

    args = parser.parse_args()

    if not args.experiment:
        parser.print_help()
        return 1

    if args.experiment == "extract-profiles":
        # Extract researcher profiles from PDFs
        try:
            logger.info(f"Starting profile extraction")
            logger.info(f"  Input: {args.input}")
            logger.info(f"  Output: {args.output}")
            logger.info(f"  Model: {args.model}")
            logger.info(f"  Provider: {args.provider}")

            results = extract_profiles(
                input_dir=args.input,
                output_dir=args.output,
                llm_model=args.model,
                llm_provider=args.provider,
            )

            logger.info(f"Profile extraction complete:")
            logger.info(f"  Total: {results.get('total', 0)}")
            logger.info(f"  Successful: {results.get('successful', 0)}")
            logger.info(f"  Failed: {results.get('failed', 0)}")
            success = True
        except Exception as e:
            logger.error(f"Profile extraction failed: {e}", exc_info=True)
            success = False
    elif args.experiment == "exp1":
        # Run Experiment 1: Profile extraction accuracy
        try:
            from utils.profile_io import save_json

            extracted_dir = Path(args.extracted_profiles_dir)
            gold_dir = Path(args.gold_profiles_dir)
            output_dir = Path(args.output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Experiment 1: Profile extraction accuracy")
            logger.info(f"  Extracted profiles: {extracted_dir}")
            logger.info(f"  Ground truth: {gold_dir}")
            logger.info(f"  Output: {output_dir}")

            # Build profile mapping
            data_map = build_data_map(str(extracted_dir), str(gold_dir), args.llm_model_name)

            if not data_map:
                logger.error("No profile pairs found to evaluate")
                success = False
            else:
                logger.info(f"Found {len(data_map)} profile pairs")

                # Run evaluation
                final_report = run_experiment_1(data_map, args.sbert_model)

                # Save results
                output_file = output_dir / "experiment_1_results.json"
                save_json(final_report, output_file)
                print_formatted_report(final_report, str(output_dir))

                logger.info(f"Results saved to {output_file}")
                success = True
        except Exception as e:
            logger.error(f"Experiment 1 failed: {e}", exc_info=True)
            success = False
    elif args.experiment == "build-cases":
        success = build_cases_cli(args)
    elif args.experiment == "exp2":
        # Run Experiment 2: Collaboration prediction backtesting
        try:
            from utils.profile_io import save_json

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Experiment 2: Collaboration prediction backtesting")
            logger.info(f"  Cases file: {args.cases_file}")
            logger.info(f"  Output: {output_dir}")

            # Load collaboration cases
            with open(args.cases_file, "r", encoding="utf-8") as f:
                cases = json.load(f)

            logger.info(f"Loaded {len(cases)} collaboration cases")

            # Ensure output paths are set for predictions
            for case in cases:
                if "out_path" not in case:
                    case["out_path"] = str(
                        output_dir / f"{case.get('id', 'unknown')}_prediction.json"
                    )

            # Prepare LLM configuration
            llm_cfg = get_llm_config(args.llm_model, args.llm_provider)

            # Run evaluation
            logger.info("Running collaboration prediction evaluation")
            results = run_experiment_2(
                cases,
                llm_cfg,
                model_name=args.sbert_model,
                tau=args.tau,
            )

            if not results:
                logger.error("Experiment 2 produced no results")
                success = False
            else:
                # Save results
                output_file = output_dir / "experiment_2_results.json"
                save_json(results, output_file)

                logger.info(f"Results saved to {output_file}")
                success = True

        except Exception as e:
            logger.error(f"Experiment 2 failed: {e}", exc_info=True)
            success = False
    else:
        logger.error(f"Unknown experiment: {args.experiment}")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    parser = create_parser()
    sys.exit(main(parser))
