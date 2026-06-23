"""
Centralized logging configuration for CatalystOU experiments.
Provides consistent logging across all modules.
"""

import sys
import logging
from pathlib import Path
from typing import Optional


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
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets more detail
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
