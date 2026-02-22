"""HTML report generator"""

from pathlib import Path
from typing import Any, Dict


class HTMLReporter:
    """Generate HTML format report"""

    def generate(self, data: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate HTML report.

        Args:
            data: Benchmark result dictionary
            output_path: Path to save the report

        Returns:
            HTML string
        """
        html = self._get_html_template()

        # Replace placeholders
        html = html.replace("{{TITLE}}", "PyTorch GPU Benchmark Report")
        html = html.replace("{{TEST_DATE}}", data.get("metadata", {}).get("test_date", "N/A"))
        html = html.replace("{{DURATION}}", f"{data.get('test_duration_seconds', 0):.1f}s")

        # Device info
        device_info = data.get("device_info", {})
        device_html = self._generate_device_info(device_info)
        html = html.replace("{{DEVICE_INFO}}", device_html)

        # Configuration
        config = data.get("test_configuration", {})
        config_html = self._generate_config_info(config)
        html = html.replace("{{CONFIG_INFO}}", config_html)

        # Results
        results_html = self._generate_results(data.get("results", []))
        html = html.replace("{{RESULTS}}", results_html)

        # Summary
        summary_html = self._generate_summary(data.get("summary", {}))
        html = html.replace("{{SUMMARY}}", summary_html)

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")

        return html

    def _generate_device_info(self, device_info: dict) -> str:
        """Generate device info HTML"""
        html = "<div class='info-card'>"

        cpu_info = device_info.get("cpu_info")
        if cpu_info:
            html += f"""
            <h3>CPU Information</h3>
            <p><strong>Name:</strong> {cpu_info.get('cpu_name', 'N/A')}</p>
            <p><strong>Cores:</strong> {cpu_info.get('cpu_cores', 'N/A')}</p>
            <p><strong>Memory:</strong> {cpu_info.get('memory_total_gb', 'N/A')} GB</p>
            """

        gpu_info_list = device_info.get("gpu_info", [])
        if gpu_info_list:
            html += "<h3>GPU Information</h3>"
            for gpu in gpu_info_list:
                html += f"""
                <p><strong>Device:</strong> {gpu.get('gpu_name', 'N/A')}</p>
                <p><strong>Memory:</strong> {gpu.get('gpu_memory_total_mb', 'N/A')} MB</p>
                <p><strong>CUDA Capability:</strong> {gpu.get('cuda_compute_capability', 'N/A')}</p>
                """
        else:
            html += "<h3>GPU Information</h3><p>No GPU detected (using CPU)</p>"

        html += "</div>"
        return html

    def _generate_config_info(self, config: dict) -> str:
        """Generate config info HTML"""
        return f"""
        <div class='info-card'>
            <h3>Test Configuration</h3>
            <p><strong>Device:</strong> {config.get('device', 'N/A')}</p>
            <p><strong>Precision:</strong> {config.get('precision', 'N/A')}</p>
            <p><strong>Warmup Iterations:</strong> {config.get('warmup_iterations', 'N/A')}</p>
            <p><strong>Test Iterations:</strong> {config.get('test_iterations', 'N/A')}</p>
            <p><strong>Batch Sizes:</strong> {config.get('batch_sizes', 'N/A')}</p>
        </div>
        """

    def _generate_results(self, results: list) -> str:
        """Generate results HTML"""
        if not results:
            return "<p>No results</p>"

        html = ""
        for result in results:
            model_name = result.get("model_name", "Unknown")
            model_type = result.get("model_type", "Unknown")
            params = result.get("parameters_count", 0)

            html += f"""
            <div class='model-result'>
                <h3>{model_name} <span class='model-type'>({model_type}, {params / 1e6:.1f}M params)</span></h3>
            """

            # Training metrics table
            training_metrics = result.get("training_metrics", {})
            if training_metrics:
                html += "<h4>Training Performance</h4>"
                html += self._generate_metrics_table(training_metrics, is_training=True)

            # Inference metrics table
            inference_metrics = result.get("inference_metrics", {})
            if inference_metrics:
                html += "<h4>Inference Performance</h4>"
                html += self._generate_metrics_table(inference_metrics, is_training=False)

            html += "</div>"

        return html

    def _generate_metrics_table(self, metrics: dict, is_training: bool) -> str:
        """Generate metrics table HTML"""
        html = "<table class='metrics-table'>"
        html += "<tr><th>Batch Size</th><th>Forward (ms)</th>"

        if is_training:
            html += "<th>Backward (ms)</th><th>Optimize (ms)</th><th>Total (ms)</th>"

        html += "<th>Throughput (smp/s)</th><th>Memory (MB)</th></tr>"

        for key, m in metrics.items():
            batch_size = key.replace("batch_size_", "")
            html += f"<tr>"
            html += f"<td>{batch_size}</td>"
            html += f"<td>{m.get('forward_time_mean_ms' if not is_training else 'forward_time_ms', 0):.2f}</td>"

            if is_training:
                html += f"<td>{m.get('backward_time_ms', 'N/A') or 'N/A'}</td>"
                html += f"<td>{m.get('optimize_step_time_ms', 'N/A') or 'N/A'}</td>"
                html += f"<td>{m.get('total_iteration_time_ms', 0):.2f}</td>"

            html += f"<td>{m.get('throughput_samples_per_sec', 0):.2f}</td>"
            html += f"<td>{m.get('peak_memory_mb', 0):.2f}</td>"
            html += f"</tr>"

        html += "</table>"
        return html

    def _generate_summary(self, summary: dict) -> str:
        """Generate summary HTML"""
        best = summary.get("best_model_throughput", {})
        return f"""
        <div class='summary-card'>
            <h3>Summary</h3>
            <p><strong>Total Models Tested:</strong> {summary.get('total_models_tested', 0)}</p>
            <p><strong>Best Throughput:</strong> {best.get('model', 'N/A')} ({best.get('throughput', 0):.2f} samples/sec)</p>
        </div>
        """

    def _get_html_template(self) -> str:
        """Get HTML template"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        header h1 { font-size: 2em; margin-bottom: 10px; }
        header .meta { opacity: 0.9; font-size: 0.9em; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .info-card, .summary-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .info-card h3, .summary-card h3 {
            color: #667eea;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .info-card p { margin: 8px 0; }
        .model-result {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .model-result h3 {
            color: #333;
            margin-bottom: 15px;
        }
        .model-type { color: #888; font-weight: normal; font-size: 0.8em; }
        .model-result h4 {
            color: #667eea;
            margin: 15px 0 10px;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        .metrics-table th, .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .metrics-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }
        .metrics-table tr:hover { background: #f8f9fa; }
        footer {
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>PyTorch GPU Benchmark Report</h1>
            <div class="meta">
                Test Date: {{TEST_DATE}} | Duration: {{DURATION}}
            </div>
        </header>

        <div class="grid">
            {{DEVICE_INFO}}
            {{CONFIG_INFO}}
        </div>

        {{RESULTS}}

        {{SUMMARY}}

        <footer>
            Generated by torch-gpu-benchmark
        </footer>
    </div>
</body>
</html>"""
