"""
Centralized logging configuration for CatalystOU experiments.
Provides consistent logging across all modules.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional


def _resolve_log_path(log_file: Optional[str], log_dir: str = "logs") -> Path:
    """Resolve log paths so files always land under the project logs directory."""
    if not log_file:
        return Path(log_dir)

    log_path = Path(log_file)
    if log_path.is_absolute():
        return log_path

    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / log_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    if log_path.parent == Path('.'):
        return logs_dir / log_path.name

    return project_root / log_path


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Optional file path for logging. If not provided, only console output.
        level: Logging level (default: INFO)
        log_dir: Directory for log files
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = _resolve_log_path(log_file, log_dir=log_dir)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # File gets more detail
        file_handler.addFilter(logging.Filter(name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
