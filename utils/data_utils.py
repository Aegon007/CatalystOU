"""
Data loader module for discovering and loading researcher profiles and labels.
Supports multiple directory structures and file formats.

    Expected structure of hand-labeled profiles:
        root_dir/
        ├── Biology/
        │   ├── Author_1/
        │   │   └── profile.json
        │   └── Author_2/
        │       └── profile.json
        ├── CS/
        │   └── ...

    Expected structure of extracted-labeled profiles:
        root_dir/
        |-- Model_1/
        |   ├── Biology/
        |   │   ├── Author_1/
        |   │   │   └── profile.json
        |   │   └── Author_2/
        |   |       └── profile.json
        |   └── CS/
        |       └── ...
        ├── Model_2/
        |   ├── Biology/
        |   │   ├── Author_1/     
        │   │   │   └── profile.json
        │   │   └── Author_2/
        │   |       └── profile.json
        |   └── CS/
        |       └── ...
"""

import os
import json
import re

from pathlib import Path
from typing import Any, Dict, List, Optional
from .logger import setup_logger
from dataclasses import asdict, dataclass, field

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> bool:
        return False


# Standard field names used by the paper experiments. Keeping these in one place
# prevents the extraction, matching, metrics, and future GraphRAG code from
# drifting into subtly different schemas.
PROFILE_LIST_FIELDS = [
    "Research Domains",
    "Techniques Used",
    "Data & Platforms",
    "Application Areas",
    "Key Research Thinking Patterns",
]
PROFILE_SUMMARY_FIELD = "Summary Description"
PROFILE_FIELDS = [*PROFILE_LIST_FIELDS, PROFILE_SUMMARY_FIELD]

COLLABORATION_CATEGORIES = [
    "Shared Domains",
    "Method-Application Synergies",
    "Complementary Technique Synergies",
    "Data-Method Synergies",
    "Cross-Domain Fusion Topics",
    "Shared Application Areas",
    "Joint Technique Development",
    "Theory-Application Synergy",
    "Thinking Pattern Synergies",
    "Future Research Directions",
]
COLLABORATION_SUMMARY_FIELD = "Summary Collaboration Themes"
COLLABORATION_FIELDS = [*COLLABORATION_CATEGORIES, COLLABORATION_SUMMARY_FIELD]

# --- 配置区域 ---
load_dotenv()
cwd = os.getcwd()
prev_dir = os.path.dirname(cwd)
LOG_FILE = os.path.join(prev_dir, "logs","data_utils.log")
logger = setup_logger("exp_runner", log_file=LOG_FILE)


class ProfileLoader:
    """Load and manage researcher profiles from various directory structures."""

    @staticmethod
    def normalize_researcher_key(name: str) -> str:
        """Normalize author/profile names so extracted and gold paths can align."""
        key = name.replace("_profile", "").replace("-profile", "")
        key = key.replace("_", " ").lower()
        key = re.sub(r"[^a-z0-9]+", " ", key)
        return re.sub(r"\s+", " ", key).strip()

    @staticmethod
    def load_json_profile(file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a single JSON profile from file.

        Args:
            file_path: Path to JSON file

        Returns:
            Dictionary containing profile data, or None on error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded profile from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"Profile file not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in profile file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading profile {file_path}: {e}")
            return None

    @staticmethod
    def load_researcher_profiles_by_department(root_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Load profiles organized by department/discipline.
        """
        data_by_dept = {}
        root_path = Path(root_dir)

        if not root_path.exists():
            logger.error(f"Directory does not exist: {root_dir}")
            return {}

        # Iterate through departments
        for dept_dir in root_path.iterdir():
            if not dept_dir.is_dir():
                continue

            dept_name = dept_dir.name
            data_by_dept[dept_name] = {}

            # Iterate through authors in department
            for author_dir in dept_dir.iterdir():
                if not author_dir.is_dir():
                    continue

                author_name = author_dir.name

                # Find JSON profiles in author directory
                json_files = list(author_dir.glob("*.json"))
                if json_files:
                    profile_data = ProfileLoader.load_json_profile(json_files[0])
                    if profile_data:
                        data_by_dept[dept_name][author_name] = profile_data
                        logger.debug(f"Loaded {dept_name}/{author_name}")

        logger.info(f"Loaded profiles from {len(data_by_dept)} departments")
        return data_by_dept

    @staticmethod
    def save_json_profile(data: Dict[str, Any], file_path: Path) -> bool:
        """
        Save a profile to JSON file.

        Args:
            data: Profile data dictionary
            file_path: Target file path

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved profile to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving profile to {file_path}: {e}")
            return False


"""Shared schema definitions for experiment inputs and outputs."""
@dataclass
class ResearcherProfile:
    """Structured representation of a researcher's profile."""

    researcher_name: str
    department: Optional[str] = None
    research_domains: List[str] = field(default_factory=list)
    techniques_used: List[str] = field(default_factory=list)
    data_platforms: List[str] = field(default_factory=list)
    application_areas: List[str] = field(default_factory=list)
    key_thinking_patterns: List[str] = field(default_factory=list)
    summary_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearcherProfile":
        """Create instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CollaborationProfile:
    """Structured representation of collaboration synergies between two researchers."""

    researcher_a: str
    researcher_b: str
    shared_domains: List[str] = field(default_factory=list)
    method_application_synergies: List[str] = field(default_factory=list)
    complementary_technique_synergies: List[str] = field(default_factory=list)
    data_method_synergies: List[str] = field(default_factory=list)
    cross_domain_fusion_topics: List[str] = field(default_factory=list)
    shared_application_areas: List[str] = field(default_factory=list)
    joint_technique_development: List[str] = field(default_factory=list)
    theory_application_synergy: List[str] = field(default_factory=list)
    thinking_pattern_synergies: List[str] = field(default_factory=list)
    future_research_directions: List[str] = field(default_factory=list)
    summary_collaboration_themes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollaborationProfile":
        """Create instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
