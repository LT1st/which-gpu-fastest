"""Report generation modules"""

from typing import Dict, Any

from .json_reporter import JSONReporter
from .csv_reporter import CSVReporter
from .text_reporter import TextReporter
from .html_reporter import HTMLReporter


def get_reporter(format_type: str):
    """Get reporter by format type"""
    reporters = {
        "json": JSONReporter,
        "csv": CSVReporter,
        "txt": TextReporter,
        "text": TextReporter,
        "html": HTMLReporter,
    }
    if format_type not in reporters:
        raise ValueError(f"Unknown format: {format_type}. Available: {list(reporters.keys())}")
    return reporters[format_type]()
