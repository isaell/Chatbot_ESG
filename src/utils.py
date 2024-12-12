import os
import logging
from typing import Any


def setup_logging(log_level: str = 'INFO') -> None:
    """Configure basic logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def get_size_format(b, factor=1024, suffix="B") -> int:
    """
    Convert bytes to a human-readable format (e.g., KB, MB, GB).
    Base factor defaults to 1024 for conversion, which is standard for binary prefixes (e.g., 1 KB = 1024 bytes).
    No prefix (bytes), kilobytes (KB), megabytes (MB), gigabytes (GB), terabytes (TB), petabytes (PB), 
    exabytes (EB), and zettabytes (ZB)
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        # size 'b' less than the 'factor'
        if b < factor:
            # format size as a strig
            return f"{b:.2f}{unit}{suffix}"
        # else, 'b' is divided by the factor and moves to the next unit in the list
        b /= factor
    return f"{b:.2f}Y{suffix}"


def safe_env_get(key: str, default: Any = None, required: bool = False) -> str:
    """
    Safely retrieve environment variables with optional defaults
    """
    value = os.environ.get(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {key} not set")
    return value