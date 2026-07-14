"""
Batch profile extraction entry point for the research workflow.

This module is intentionally small: it wraps the lower-level extraction logic in
profile_extractor.py and exposes a simple CLI for running profile extraction from
a directory of PDF folders.

The expected input structure is:
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

The intended saving structure is:
    output_dir/
        model_name/
            discipline_1/
                researcher_1_profile.json
                researcher_2_profile.json
            discipline_2/
                researcher_3_profile.json
Where each discipline folder contains researcher profile JSON files extracted from
the corresponding PDF documents in the input directory.
"""

import os
import sys
import asyncio
import argparse

from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger_utils import setup_logger
import proj_test.profile_extractor as pf_extractor

logger = setup_logger("profile_extraction", log_file="profile_extraction.log")
CONCURRENT_LIMIT = 5  # Limit concurrent extractions to 5


async def extract_one_discipline_profiles(one_discipline_input_dir: str, one_discipline_output_dir: str, model_name: str = "gpt-5") -> None:
    """
    Extract researcher profiles from a single discipline directory of PDFs.

    Args:
        one_dicipline_input_dir: Directory containing PDFs for a single discipline.
        one_dicipline_output_dir: Directory where extracted profiles will be saved for the input dicipline.
        model_name: LLM model name to use for extraction (default: gpt-4-turbo)
    """
    input_dir = Path(one_discipline_input_dir)
    out_dir = Path(one_discipline_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {input_dir}")
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")
    
    author_dirs = [aaa for aaa in input_dir.iterdir() if aaa.is_dir()]

    logger.info(f"Found {len(author_dirs)} researcher directories in {input_dir}")

    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)  # Limit concurrent extractions to 5

    for author in author_dirs:
        await pf_extractor.generate_profile_for_one_author(author, out_dir, model_name, semaphore)
    
    logger.info(f"Profile extraction completed for discipline: {input_dir.name}")
        

def extract_profiles(input_dir: str, output_dir: str, model_name: str = "gpt-4-turbo") -> Dict[str, int]:
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
    
    output_dir/
        model_name/
            discipline_1/
                researcher_1_profile.json
                researcher_2_profile.json
            discipline_2/
                researcher_3_profile.json
    """
    all_discipline_path = Path(input_dir)
    root_output_path = Path(output_dir)
    model_output_path = os.path.join(root_output_path, model_name)

    if not all_discipline_path.exists():
        raise ValueError(f"Input directory does not exist: {all_discipline_path}")

    logger.info(f"Starting profile extraction")
    logger.info(f"  Input root: {all_discipline_path}")
    logger.info(f"  Output root: {model_output_path}")
    logger.info(f"  LLM Model: {model_name}")

    diciplines = os.listdir(all_discipline_path)
    successful = 0
    failed = 0

    for dicipline in diciplines:
        current_dicipline_dir = os.path.join(all_discipline_path, dicipline)
        if not os.path.isdir(current_dicipline_dir):
            logger.warning(f"Skipping non-directory entry in input directory: {current_dicipline_dir}")
            continue

        dicipline_output_dir = os.path.join(model_output_path, dicipline)
        os.makedirs(dicipline_output_dir, exist_ok=True)
        try:
            asyncio.run(
                extract_one_discipline_profiles(
                    one_discipline_input_dir=current_dicipline_dir,
                    one_discipline_output_dir=dicipline_output_dir,
                    model_name=model_name,
                )
            )
            successful += 1
        except Exception as exc:
            failed += 1
            logger.error(f"Profile extraction failed for discipline {dicipline}: {exc}", exc_info=True)

    logger.info("Profile extraction completed for all disciplines.")
    return {"total": successful + failed, "successful": successful, "failed": failed}


def parse_args(argv) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract researcher profiles from PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python extract_profiles.py -i ./data/papers -o ./extracted_profiles -m gpt-4-turbo
            python extract_profiles.py -i ./research_pdfs -o ./profiles -m gpt-4
        """
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory containing researcher folders with PDFs")
    parser.add_argument("-o", "--output", type=str, default="./extracted_profiles", help="Output directory for extracted profiles (default: ./extracted_profiles)")
    parser.add_argument("-m", "--model",type=str,default="gpt-4-turbo",help="LLM model name (default: gpt-4-turbo)")

    return parser.parse_args(argv)


def main(opts) -> int:
    """CLI entrypoint."""
    try:
        extract_profiles(
            input_dir=opts.input,
            output_dir=opts.output,
            model_name=opts.model,
        )
        return 0
    except Exception as e:
        logger.error(f"Profile extraction failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    opts = parse_args(sys.argv[1:])
    main(opts)
