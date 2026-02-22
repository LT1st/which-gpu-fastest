"""Utility functions"""

import gc
from typing import List, Any
import statistics


def clear_memory():
    """Clear memory cache"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def calculate_statistics(values: List[float]) -> dict:
    """Calculate statistics for a list of values"""
    if not values:
        return {
            "mean": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "median": 0,
            "p50": 0,
            "p95": 0,
            "p99": 0,
        }

    sorted_values = sorted(values)
    n = len(sorted_values)

    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if n > 1 else 0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "p50": sorted_values[int(n * 0.5)],
        "p95": sorted_values[int(n * 0.95)],
        "p99": sorted_values[int(n * 0.99)],
    }


def format_memory(mb: float) -> str:
    """Format memory size for display"""
    if mb < 1024:
        return f"{mb:.2f} MB"
    elif mb < 1024 * 1024:
        return f"{mb / 1024:.2f} GB"
    else:
        return f"{mb / (1024 * 1024):.2f} TB"


def format_time(ms: float) -> str:
    """Format time for display"""
    if ms < 1:
        return f"{ms * 1000:.2f} µs"
    elif ms < 1000:
        return f"{ms:.2f} ms"
    elif ms < 60000:
        return f"{ms / 1000:.2f} s"
    else:
        return f"{ms / 60000:.2f} min"


def format_number(n: int) -> str:
    """Format large numbers with suffixes"""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    else:
        return str(n)


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable string"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"
