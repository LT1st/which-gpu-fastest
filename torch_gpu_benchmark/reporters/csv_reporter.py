"""CSV report generator"""

import csv
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List


class CSVReporter:
    """Generate CSV format report"""

    def generate(self, data: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate CSV report.

        Args:
            data: Benchmark result dictionary
            output_path: Path to save the report

        Returns:
            CSV string
        """
        output = StringIO()
        writer = csv.writer(output)

        # Header
        header = [
            "Model",
            "Type",
            "Batch_Size",
            "Mode",
            "Forward_ms",
            "Backward_ms",
            "Optimize_ms",
            "Total_ms",
            "Throughput",
            "Memory_MB",
        ]
        writer.writerow(header)

        # Data rows
        for result in data.get("results", []):
            model_name = result["model_name"]
            model_type = result["model_type"]

            # Inference metrics
            for key, metrics in result.get("inference_metrics", {}).items():
                batch_size = key.replace("batch_size_", "")
                writer.writerow([
                    model_name,
                    model_type,
                    batch_size,
                    "inference",
                    f"{metrics.get('forward_time_mean_ms', 0):.2f}",
                    "N/A",
                    "N/A",
                    "N/A",
                    f"{metrics.get('throughput_samples_per_sec', 0):.2f}",
                    f"{metrics.get('peak_memory_mb', 0):.2f}",
                ])

            # Training metrics
            for key, metrics in result.get("training_metrics", {}).items():
                batch_size = key.replace("batch_size_", "")
                writer.writerow([
                    model_name,
                    model_type,
                    batch_size,
                    "training",
                    f"{metrics.get('forward_time_ms', 0):.2f}",
                    f"{metrics.get('backward_time_ms', 0) or 'N/A'}",
                    f"{metrics.get('optimize_step_time_ms', 0) or 'N/A'}",
                    f"{metrics.get('total_iteration_time_ms', 0) or 'N/A':.2f}",
                    f"{metrics.get('throughput_samples_per_sec', 0):.2f}",
                    f"{metrics.get('peak_memory_mb', 0):.2f}",
                ])

        csv_str = output.getvalue()

        if output_path:
            Path(output_path).write_text(csv_str, encoding="utf-8")

        return csv_str
