"""JSON report generator"""

import json
from typing import Any, Dict
from pathlib import Path


class JSONReporter:
    """Generate JSON format report"""

    def generate(self, data: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate JSON report.

        Args:
            data: Benchmark result dictionary
            output_path: Path to save the report

        Returns:
            JSON string
        """
        json_str = json.dumps(data, indent=2, ensure_ascii=False)

        if output_path:
            Path(output_path).write_text(json_str, encoding="utf-8")

        return json_str
