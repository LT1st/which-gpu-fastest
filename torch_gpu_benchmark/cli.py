"""Command-line interface for torch-gpu-benchmark"""

import logging
import time
from pathlib import Path
from typing import Optional

import click
import torch
import yaml

from .core.benchmark import (
    GPUBenchmark,
    BenchmarkConfig,
    Precision,
    create_benchmark_result,
)
from .core.device_info import DeviceInfo
from .models import get_model_factory, list_models, ALL_MODELS
from .reporters import get_reporter


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(version="1.0.0", prog_name="torch-gpu-benchmark")
def main():
    """PyTorch GPU/CPU Performance Benchmark Tool"""
    pass


@main.command()
@click.option(
    "-m", "--models",
    default="all",
    help="Models to test (comma-separated) or 'all'. Default: all",
)
@click.option(
    "-b", "--batch-sizes",
    default="1,8,32",
    help="Batch sizes to test (comma-separated). Default: 1,8,32",
)
@click.option(
    "-p", "--precision",
    type=click.Choice(["fp32", "fp16", "bf16", "tf32"]),
    default="fp32",
    help="Precision mode. Default: fp32",
)
@click.option(
    "-i", "--iterations",
    default=100,
    help="Number of test iterations. Default: 100",
)
@click.option(
    "-w", "--warmup",
    default=10,
    help="Number of warmup iterations. Default: 10",
)
@click.option(
    "-o", "--output",
    default="benchmark_report.json",
    help="Output file path. Default: benchmark_report.json",
)
@click.option(
    "-f", "--format",
    type=click.Choice(["json", "csv", "txt", "html"]),
    default="json",
    help="Output format. Default: json",
)
@click.option(
    "--device",
    default="auto",
    help="Device: auto, cpu, cuda, cuda:0, mps (Apple Silicon). Default: auto",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML config file",
)
def run(
    models: str,
    batch_sizes: str,
    precision: str,
    iterations: int,
    warmup: int,
    output: str,
    format: str,
    device: str,
    verbose: bool,
    config: Optional[str],
):
    """Run benchmark tests"""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Load config from file if provided
    if config:
        with open(config) as f:
            config_data = yaml.safe_load(f)
        benchmark_config = config_data.get("benchmark", {})
        models = benchmark_config.get("models", models)
        if isinstance(models, list):
            models = ",".join(models)
        batch_sizes = str(benchmark_config.get("batch_sizes", batch_sizes))
        if isinstance(eval(batch_sizes), list):
            batch_sizes = ",".join(map(str, eval(batch_sizes)))
        precision = benchmark_config.get("precision", precision)
        iterations = benchmark_config.get("iterations", iterations)
        warmup = benchmark_config.get("warmup", warmup)
        device = benchmark_config.get("device", device)

    # Parse models
    if models == "all":
        model_list = list(ALL_MODELS.keys())
    else:
        model_list = [m.strip() for m in models.split(",")]

    # Parse batch sizes
    batch_size_list = [int(b.strip()) for b in batch_sizes.split(",")]

    # Create config
    config_obj = BenchmarkConfig(
        models=model_list,
        batch_sizes=batch_size_list,
        precision=Precision(precision),
        iterations=iterations,
        warmup=warmup,
        device=device,
    )

    # Resolve device
    device_info = DeviceInfo.collect()
    if device == "auto":
        actual_device = device_info.get_device_string()
    else:
        actual_device = device

    config_obj.device = actual_device

    logger.info(f"Starting benchmark on device: {actual_device}")
    logger.info(f"Models: {model_list}")
    logger.info(f"Batch sizes: {batch_size_list}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Iterations: {iterations} (warmup: {warmup})")

    # Create benchmark
    benchmark = GPUBenchmark(config_obj)

    # Run benchmarks
    start_time = time.time()
    results = []

    for model_name in model_list:
        try:
            logger.info(f"Testing model: {model_name}")
            factory, model_type = get_model_factory(model_name)

            # Create a wrapper that calls the factory
            def model_factory(fn=factory, name=model_name):
                return fn(name)

            result = benchmark.run(model_factory, model_name, model_type)
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    end_time = time.time()
    duration = end_time - start_time

    # Create result
    benchmark_result = create_benchmark_result(
        results=results,
        config=config_obj,
        device_info=device_info,
        duration_seconds=duration,
    )

    # Generate report
    reporter = get_reporter(format)
    report = reporter.generate(benchmark_result.to_dict(), output)

    logger.info(f"Benchmark completed in {duration:.2f} seconds")
    logger.info(f"Report saved to: {output}")

    # Print summary to console
    click.echo("\n" + "=" * 60)
    click.echo("Benchmark Summary")
    click.echo("=" * 60)
    click.echo(f"Device: {actual_device}")
    click.echo(f"Models tested: {len(results)}")
    click.echo(f"Duration: {duration:.2f} seconds")

    if results:
        click.echo("\nTop Results (Training Throughput):")
        for result in results:
            for bs_key, metrics in result.training_metrics.items():
                if metrics.throughput_samples_per_sec > 0:
                    click.echo(
                        f"  {result.model_name} {bs_key}: "
                        f"{metrics.throughput_samples_per_sec:.2f} samples/sec"
                    )

    click.echo(f"\nReport saved to: {output}")


@main.command()
@click.option(
    "--light",
    is_flag=True,
    help="Light test (1 minute)",
)
@click.option(
    "--standard",
    is_flag=True,
    default=True,
    help="Standard test (3 minutes)",
)
@click.option(
    "--extensive",
    is_flag=True,
    help="Extensive test (10 minutes)",
)
@click.option(
    "-o", "--output",
    default="quick_benchmark.json",
    help="Output file path",
)
@click.option(
    "-f", "--format",
    type=click.Choice(["json", "csv", "txt", "html"]),
    default="json",
    help="Output format",
)
@click.option(
    "--device",
    default="auto",
    help="Device to use",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def quick(
    light: bool,
    standard: bool,
    extensive: bool,
    output: str,
    format: str,
    device: str,
    verbose: bool,
):
    """Run quick benchmark with preset configuration"""
    setup_logging(verbose)

    if extensive:
        models = "resnet18,resnet50,bert-tiny,bert-mini,unet"
        batch_sizes = "1,8,16,32"
        iterations = 200
        warmup = 20
    elif light:
        models = "simple_cnn,bert-tiny"
        batch_sizes = "1,8"
        iterations = 50
        warmup = 5
    else:  # standard
        models = "resnet50,bert-tiny,unet"
        batch_sizes = "1,8,32"
        iterations = 100
        warmup = 10

    # Call run command with preset values
    ctx = click.Context(run)
    ctx.invoke(
        run,
        models=models,
        batch_sizes=batch_sizes,
        precision="fp32",
        iterations=iterations,
        warmup=warmup,
        output=output,
        format=format,
        device=device,
        verbose=verbose,
    )


@main.command("list-models")
@click.option(
    "--type",
    "model_type",
    type=click.Choice(["cnn", "transformer", "generative"]),
    default=None,
    help="Filter by model type",
)
def list_models_cmd(model_type: Optional[str]):
    """List available models"""
    models = list_models(model_type)

    click.echo("\nAvailable Models:")
    click.echo("=" * 60)

    for name, info in models.items():
        click.echo(f"\n  {name}")
        click.echo(f"    Name: {info.get('name', 'N/A')}")
        click.echo(f"    Parameters: {info.get('params', 'N/A')}")
        click.echo(f"    Description: {info.get('description', 'N/A')}")

    click.echo(f"\nTotal: {len(models)} models")


@main.command("list-devices")
def list_devices_cmd():
    """List available compute devices"""
    device_info = DeviceInfo.collect()
    devices = device_info.get_available_devices()

    click.echo("\nAvailable Devices:")
    click.echo("=" * 60)

    for i, device in enumerate(devices):
        if device == "cpu":
            click.echo(f"  [{i}] cpu - CPU")
        elif device.startswith("cuda"):
            idx = device.split(":")[1] if ":" in device else "0"
            gpu_name = "Unknown"
            for gpu in device_info.gpu_info:
                if gpu.gpu_id == int(idx):
                    gpu_name = gpu.gpu_name
                    break
            click.echo(f"  [{i}] {device} - {gpu_name}")
        elif device == "mps":
            click.echo(f"  [{i}] mps - Apple Silicon GPU")

    click.echo(f"\nPrimary: {device_info.get_device_string()}")
    click.echo(f"Device Type: {device_info.get_device_type()}")


@main.command()
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output in JSON format",
)
def info(json_output: bool):
    """Show device information"""
    device_info = DeviceInfo.collect()

    if json_output:
        import json
        click.echo(json.dumps(device_info.to_dict(), indent=2))
    else:
        click.echo("\n" + "=" * 60)
        click.echo("Device Information")
        click.echo("=" * 60)

        # System info
        click.echo(f"\nSystem:")
        click.echo(f"  Hostname: {device_info.hostname}")
        click.echo(f"  Platform: {device_info.platform}")
        click.echo(f"  Python: {device_info.python_version}")
        click.echo(f"  PyTorch: {device_info.pytorch_version}")

        # Framework versions
        click.echo(f"\nFramework Versions:")
        if device_info.cuda_version:
            click.echo(f"  CUDA: {device_info.cuda_version}")
        if device_info.rocm_version:
            click.echo(f"  ROCm: {device_info.rocm_version}")
        if device_info.mps_available:
            click.echo(f"  MPS: Available (Apple Silicon)")

        # CPU info
        if device_info.cpu_info:
            click.echo(f"\nCPU:")
            click.echo(f"  Name: {device_info.cpu_info.cpu_name}")
            click.echo(f"  Cores: {device_info.cpu_info.cpu_cores}")
            click.echo(f"  Threads: {device_info.cpu_info.cpu_threads}")
            click.echo(f"  Frequency: {device_info.cpu_info.cpu_freq_mhz:.0f} MHz")
            click.echo(f"  Memory: {device_info.cpu_info.memory_total_gb:.2f} GB")

        # GPU info
        if device_info.gpu_info:
            click.echo(f"\nGPU:")
            for gpu in device_info.gpu_info:
                gpu_type_str = f" ({gpu.gpu_type.upper()})" if gpu.gpu_type != "cuda" else ""
                click.echo(f"  [{gpu.gpu_id}] {gpu.gpu_name}{gpu_type_str}")
                if gpu.gpu_memory_total_mb > 0:
                    click.echo(f"      Memory: {gpu.gpu_memory_total_mb} MB")
                if gpu.cuda_compute_capability != "N/A":
                    click.echo(f"      Compute Capability: {gpu.cuda_compute_capability}")
        elif device_info.mps_available:
            click.echo(f"\nGPU:")
            click.echo(f"  [0] Apple Silicon GPU (MPS)")
            click.echo(f"      Memory: Unified Memory")
        else:
            click.echo(f"\nGPU: Not available (using CPU)")

        # Available devices
        available_devices = device_info.get_available_devices()
        primary_device = device_info.get_device_string()
        click.echo(f"\nAvailable Devices: {', '.join(available_devices)}")
        click.echo(f"Primary Device: {primary_device}")
        click.echo("")


@main.command()
@click.argument("report1", type=click.Path(exists=True))
@click.argument("report2", type=click.Path(exists=True))
def compare(report1: str, report2: str):
    """Compare two benchmark reports"""
    import json

    with open(report1) as f:
        data1 = json.load(f)
    with open(report2) as f:
        data2 = json.load(f)

    click.echo("\nBenchmark Comparison:")
    click.echo("=" * 60)
    click.echo(f"Report 1: {report1}")
    click.echo(f"Report 2: {report2}")
    click.echo("")

    # Compare device info
    dev1 = data1.get("device_info", {})
    dev2 = data2.get("device_info", {})

    click.echo("Device Comparison:")
    click.echo(f"  Report 1: {dev1.get('cpu_info', {}).get('cpu_name', 'N/A')}")
    if dev1.get("gpu_info"):
        for g in dev1["gpu_info"]:
            click.echo(f"            {g.get('gpu_name', 'N/A')}")

    click.echo(f"  Report 2: {dev2.get('cpu_info', {}).get('cpu_name', 'N/A')}")
    if dev2.get("gpu_info"):
        for g in dev2["gpu_info"]:
            click.echo(f"            {g.get('gpu_name', 'N/A')}")

    click.echo("")

    # Compare results
    results1 = {r["model_name"]: r for r in data1.get("results", [])}
    results2 = {r["model_name"]: r for r in data2.get("results", [])}

    common_models = set(results1.keys()) & set(results2.keys())

    if common_models:
        click.echo("Performance Comparison (Training Throughput):")
        click.echo("-" * 60)

        for model in sorted(common_models):
            r1 = results1[model]
            r2 = results2[model]

            # Get training metrics for batch_size_32 or first available
            tm1 = r1.get("training_metrics", {})
            tm2 = r2.get("training_metrics", {})

            for bs_key in tm1:
                if bs_key in tm2:
                    m1 = tm1[bs_key]
                    m2 = tm2[bs_key]
                    t1 = m1.get("throughput_samples_per_sec", 0)
                    t2 = m2.get("throughput_samples_per_sec", 0)

                    if t1 > 0 and t2 > 0:
                        change = ((t2 - t1) / t1) * 100
                        symbol = "+" if change > 0 else ""
                        click.echo(
                            f"  {model} {bs_key}: {t1:.2f} -> {t2:.2f} "
                            f"({symbol}{change:.1f}%)"
                        )


if __name__ == "__main__":
    main()
