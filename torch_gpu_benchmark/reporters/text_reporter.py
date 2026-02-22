"""Text report generator"""

from pathlib import Path
from typing import Any, Dict


class TextReporter:
    """Generate text format report"""

    def generate(self, data: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate text report.

        Args:
            data: Benchmark result dictionary
            output_path: Path to save the report

        Returns:
            Text string
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("                    PyTorch GPU Benchmark Report")
        lines.append("=" * 80)
        lines.append("")

        # Metadata
        metadata = data.get("metadata", {})
        lines.append(f"Test Date: {metadata.get('test_date', 'N/A')}")
        lines.append(f"Tool Version: {metadata.get('tool_version', 'N/A')}")
        lines.append(f"PyTorch Version: {metadata.get('pytorch_version', 'N/A')}")
        lines.append(f"Duration: {data.get('test_duration_seconds', 0):.1f} seconds")
        lines.append("")

        # Device info
        device_info = data.get("device_info", {})
        lines.append("Device Information:")
        if cpu_info := device_info.get("cpu_info"):
            lines.append(f"  CPU: {cpu_info.get('cpu_name', 'N/A')}")
            lines.append(f"  Cores: {cpu_info.get('cpu_cores', 'N/A')}")
            lines.append(f"  Memory: {cpu_info.get('memory_total_gb', 'N/A')} GB")

        gpu_info_list = device_info.get("gpu_info", [])
        if gpu_info_list:
            for gpu_info in gpu_info_list:
                lines.append(f"  GPU: {gpu_info.get('gpu_name', 'N/A')}")
                lines.append(f"  GPU Memory: {gpu_info.get('gpu_memory_total_mb', 'N/A')} MB")
                lines.append(f"  CUDA Capability: {gpu_info.get('cuda_compute_capability', 'N/A')}")
        else:
            lines.append("  GPU: N/A (using CPU)")
        lines.append("")

        # Configuration
        config = data.get("test_configuration", {})
        lines.append("Test Configuration:")
        lines.append(f"  Device: {config.get('device', 'N/A')}")
        lines.append(f"  Precision: {config.get('precision', 'N/A')}")
        lines.append(f"  Warmup Iterations: {config.get('warmup_iterations', 'N/A')}")
        lines.append(f"  Test Iterations: {config.get('test_iterations', 'N/A')}")
        lines.append(f"  Batch Sizes: {config.get('batch_sizes', 'N/A')}")
        lines.append("")

        # Model results
        lines.append("-" * 80)
        lines.append("                         Model Performance Summary")
        lines.append("-" * 80)
        lines.append("")

        for result in data.get("results", []):
            model_name = result.get("model_name", "Unknown")
            model_type = result.get("model_type", "Unknown")
            params = result.get("parameters_count", 0)
            load_time = result.get("load_time_ms", 0)

            lines.append(f"{model_name} ({model_type}, {params / 1e6:.1f}M params)")
            lines.append(f"  Load Time: {load_time:.2f} ms")
            lines.append("")

            # Training metrics
            for key, metrics in result.get("training_metrics", {}).items():
                batch_size = key.replace("batch_size_", "")
                lines.append(f"  Training (batch_size={batch_size}, {config.get('precision', 'fp32')}):")
                lines.append("    +---------------------------+-----------------+")
                lines.append("    | Metric                    | Value           |")
                lines.append("    +---------------------------+-----------------+")

                self._add_metric_row(lines, "Forward Time", metrics.get("forward_time_ms"), "ms")
                self._add_metric_row(lines, "Backward Time", metrics.get("backward_time_ms"), "ms")
                self._add_metric_row(lines, "Optimize Step", metrics.get("optimize_step_time_ms"), "ms")
                self._add_metric_row(lines, "Total Iteration", metrics.get("total_iteration_time_ms"), "ms")
                self._add_metric_row(lines, "Throughput", metrics.get("throughput_samples_per_sec"), "smp/s")
                self._add_metric_row(lines, "Peak Memory", metrics.get("peak_memory_mb"), "MB")

                lines.append("    +---------------------------+-----------------+")
                lines.append("")

            # Inference metrics
            for key, metrics in result.get("inference_metrics", {}).items():
                batch_size = key.replace("batch_size_", "")
                lines.append(f"  Inference (batch_size={batch_size}):")
                lines.append("    +---------------------------+-----------------+")
                lines.append("    | Metric                    | Value           |")
                lines.append("    +---------------------------+-----------------+")

                self._add_metric_row(lines, "Forward Time", metrics.get("forward_time_mean_ms"), "ms")
                self._add_metric_row(lines, "Throughput", metrics.get("throughput_samples_per_sec"), "smp/s")
                self._add_metric_row(lines, "Peak Memory", metrics.get("peak_memory_mb"), "MB")

                lines.append("    +---------------------------+-----------------+")
                lines.append("")

        # Summary
        summary = data.get("summary", {})
        lines.append("-" * 80)
        lines.append("                              Overall Summary")
        lines.append("-" * 80)
        lines.append("")
        lines.append(f"Total Models Tested: {summary.get('total_models_tested', 0)}")

        best = summary.get("best_model_throughput", {})
        if best.get("model"):
            lines.append(f"Best Throughput: {best.get('model')} ({best.get('throughput', 0):.2f} samples/sec)")

        lines.append("")
        lines.append("=" * 80)

        text = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(text, encoding="utf-8")

        return text

    def _add_metric_row(self, lines: list, name: str, value: Any, unit: str):
        """Add a formatted metric row"""
        if value is None:
            formatted = "N/A"
        elif isinstance(value, float):
            formatted = f"{value:.2f} {unit}"
        else:
            formatted = f"{value} {unit}"
        lines.append(f"    | {name:<25} | {formatted:<15} |")
