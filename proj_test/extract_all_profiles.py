"""
Standalone batch profile extraction module, which will extract researcher profiles from a directory of PDFs using the same LLM model.

it will call profile_extractor to perform the extraction, and will save the extracted profiles in a structured output directory.

Provides a CLI and programmatic API for extracting researcher profiles from PDFs
using LLM-powered analysis. This module should be run independently before running
Experiment 1 to ensure extracted profiles are available for evaluation.

Usage:
    python extract_profiles.py -i ./data/papers -o ./extracted_profiles -m gpt-4-turbo
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from proj_test.profile_extractor import BatchProfileExtractor

logger = setup_logger("profile_extraction", log_file="profile_extraction.log")


async def extract_one_dicipline_profiles(input_dir: str, output_dir: str, llm_model: str = "gpt-4-turbo", llm_provider: str = "openai") -> None:
    """
    Extract researcher profiles from a single discipline directory of PDFs.

    Args:
        input_dir: Directory containing PDFs for a single discipline.
        output_dir: Directory where extracted profiles will be saved.
        llm_model: LLM model name to use for extraction (default: gpt-4-turbo)
        llm_provider: LLM provider (default: openai)
    """
    base_dir = Path(one_dicipline_input_dir)
    out_dir = Path(one_dicipline_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists() or not base_dir.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {base_dir}")
        return {"total": 0, "successful": 0, "failed": 0, "profiles": {}}
    
    Departments = [d for d in base_dir.iterdir() if d.is_dir()]

    author_dirs = []
    for department in Departments:
        tmp_author_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        author_dirs.extend(tmp_author_dirs)

    logger.info(f"Found {len(author_dirs)} researcher directories in {base_dir}")

    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)  # Limit concurrent extractions to 5

    for author in author_dirs:
        await profile_extractor.generate_profile_for_one_author(author, out_dir, model_name, semaphore)
    
    logger.info(f"Profile extraction completed for discipline: {base_dir.name}")
        

def extract_all_profiles(input_dir: str, output_dir: str, llm_model: str = "gpt-4-turbo", llm_provider: str = "openai") -> None:
    """
    Extract researcher profiles from a directory of disciplines.
    The data folder structure is expected to be:
    input_dir/
        discipline_1/
            researcher_1/
                paper_1.pdf
                paper_2.pdf
            researcher_2/
                paper_1.pdf
        discipline_2/
            researcher_3/
                paper_1.pdf
    Args:
        input_dir: Directory containing discipline folders with researcher PDFs.
        output_dir: Directory where extracted profiles will be saved.
        llm_model: LLM model name to use for extraction (default: gpt-4-turbo)
        llm_provider: LLM provider (default: openai)
    Returns:
        None. The extracted profiles will be saved in the output directory.
    """
    all_discipline_path = Path(input_dir)
    root_output_path = Path(output_dir)

    if not all_discipline_path.exists():
        raise ValueError(f"Input directory does not exist: {all_discipline_path}")

    logger.info(f"Starting profile extraction")
    logger.info(f"  Input root: {input_path}")
    logger.info(f"  Output root: {output_path}")
    logger.info(f"  LLM Model: {llm_model}")

    llm_config = {
        "provider": llm_provider,
        "model_name": llm_model,
    }

    diciplines = os.listdir(all_discipline_path)
    for dicipline in diciplines:
        extract_one_discipline()

    logger.info("Profile extraction completed for all disciplines.")


def parse_args(argv) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract researcher profiles from PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_profiles.py -i ./data/papers -o ./extracted_profiles -m gpt-4-turbo
  python extract_profiles.py -i ./research_pdfs -o ./profiles -m gpt-4 -p openai
        """
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input directory containing researcher folders with PDFs"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./extracted_profiles",
        help="Output directory for extracted profiles (default: ./extracted_profiles)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gpt-4-turbo",
        help="LLM model name (default: gpt-4-turbo)"
    )
    parser.add_argument(
        "-p", "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "local"],
        help="LLM provider (default: openai)"
    )

    return parser.parse_args(argv)


def main() -> int:
    """CLI entrypoint."""
    try:
        args = parse_args(sys.argv[1:])
        results = extract_profiles(
            input_dir=args.input,
            output_dir=args.output,
            llm_model=args.model,
            llm_provider=args.provider,
        )
        return 0
    except Exception as e:
        logger.error(f"Profile extraction failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
