"""Core benchmark implementation"""

import gc
import logging
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from .timer import Timer
from .device_info import DeviceInfo

logger = logging.getLogger(__name__)


class Precision(Enum):
    """Precision modes for training/inference"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    TF32 = "tf32"


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    models: List[str]
    batch_sizes: List[int]
    precision: Precision
    iterations: int
    warmup: int
    device: str  # "cpu", "cuda", "cuda:0", etc.
    monitor_device: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkConfig":
        """Create config from dictionary"""
        precision_map = {
            "fp32": Precision.FP32,
            "fp16": Precision.FP16,
            "bf16": Precision.BF16,
            "tf32": Precision.TF32,
        }
        return cls(
            models=data.get("models", ["resnet50"]),
            batch_sizes=data.get("batch_sizes", [1, 8, 32]),
            precision=precision_map.get(data.get("precision", "fp32"), Precision.FP32),
            iterations=data.get("iterations", 100),
            warmup=data.get("warmup", 10),
            device=data.get("device", "cpu"),
            monitor_device=data.get("monitor_device", True),
        )


@dataclass
class Metrics:
    """Performance metrics for a single test"""
    forward_time_ms: float = 0.0
    backward_time_ms: Optional[float] = None
    optimize_step_time_ms: Optional[float] = None
    total_iteration_time_ms: Optional[float] = None
    throughput_samples_per_sec: float = 0.0
    peak_memory_mb: float = 0.0
    memory_allocated_mb: float = 0.0

    # Statistics
    forward_time_mean_ms: float = 0.0
    forward_time_std_ms: float = 0.0
    forward_time_min_ms: float = 0.0
    forward_time_max_ms: float = 0.0
    forward_time_p50_ms: float = 0.0
    forward_time_p95_ms: float = 0.0
    forward_time_p99_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "forward_time_ms": self.forward_time_ms,
            "backward_time_ms": self.backward_time_ms,
            "optimize_step_time_ms": self.optimize_step_time_ms,
            "total_iteration_time_ms": self.total_iteration_time_ms,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "peak_memory_mb": self.peak_memory_mb,
            "memory_allocated_mb": self.memory_allocated_mb,
            "forward_time_mean_ms": self.forward_time_mean_ms,
            "forward_time_std_ms": self.forward_time_std_ms,
            "forward_time_min_ms": self.forward_time_min_ms,
            "forward_time_max_ms": self.forward_time_max_ms,
            "forward_time_p50_ms": self.forward_time_p50_ms,
            "forward_time_p95_ms": self.forward_time_p95_ms,
            "forward_time_p99_ms": self.forward_time_p99_ms,
        }


@dataclass
class ModelResult:
    """Benchmark result for a single model"""
    model_name: str
    model_type: str
    parameters_count: int
    load_time_ms: float
    inference_metrics: Dict[str, Metrics] = field(default_factory=dict)
    training_metrics: Dict[str, Metrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "parameters_count": self.parameters_count,
            "load_time_ms": self.load_time_ms,
            "inference_metrics": {
                k: v.to_dict() for k, v in self.inference_metrics.items()
            },
            "training_metrics": {
                k: v.to_dict() for k, v in self.training_metrics.items()
            },
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result"""
    metadata: Dict[str, Any]
    device_info: Dict[str, Any]
    test_configuration: Dict[str, Any]
    results: List[ModelResult]
    summary: Dict[str, Any]
    test_duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "device_info": self.device_info,
            "test_configuration": self.test_configuration,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "test_duration_seconds": self.test_duration_seconds,
        }


class GPUBenchmark:
    """Main benchmark class"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device_info = DeviceInfo.collect()

        # Resolve device
        if config.device == "auto":
            self.device = self.device_info.get_device_string()
        else:
            self.device = config.device

        logger.info(f"Using device: {self.device}")

    def run(self, model_factory: Callable, model_name: str, model_type: str) -> ModelResult:
        """
        Run benchmark for a single model.

        Args:
            model_factory: Function that returns (model, input_shape, get_input_fn)
            model_name: Name of the model
            model_type: Type of model (transformer, cnn, etc.)

        Returns:
            ModelResult with benchmark data
        """
        logger.info(f"Benchmarking model: {model_name}")

        # Load model
        model, input_shape, get_input_fn = model_factory()
        model_load_time = self._measure_load_time(model_factory)

        # Count parameters
        params_count = sum(p.numel() for p in model.parameters())

        result = ModelResult(
            model_name=model_name,
            model_type=model_type,
            parameters_count=params_count,
            load_time_ms=model_load_time,
        )

        # Move model to device
        model = model.to(self.device)

        # Test for each batch size
        for batch_size in self.config.batch_sizes:
            logger.info(f"  Testing batch_size={batch_size}")

            # Inference benchmark
            inference_metrics = self._benchmark_inference(
                model, batch_size, get_input_fn
            )
            result.inference_metrics[f"batch_size_{batch_size}"] = inference_metrics

            # Training benchmark
            training_metrics = self._benchmark_training(
                model, batch_size, get_input_fn
            )
            result.training_metrics[f"batch_size_{batch_size}"] = training_metrics

            # Clear memory
            self._clear_memory()

        return result

    def _measure_load_time(self, model_factory: Callable) -> float:
        """Measure model loading time"""
        timer = Timer(self.device)
        timer.start()
        model, _, _ = model_factory()
        timer.stop()
        return timer.elapsed_ms

    def _benchmark_inference(
        self, model: nn.Module, batch_size: int, get_input_fn: Callable
    ) -> Metrics:
        """Benchmark inference performance"""
        model.eval()
        metrics = Metrics()

        # Get autocast context
        autocast_ctx = self._get_autocast_context()

        # Generate input
        inputs = get_input_fn(batch_size, self.device)

        # Warmup
        with torch.no_grad(), autocast_ctx:
            for _ in range(self.config.warmup):
                if isinstance(inputs, dict):
                    _ = model(**inputs)
                elif isinstance(inputs, (list, tuple)):
                    _ = model(*inputs)
                else:
                    _ = model(inputs)

        # Synchronize if CUDA
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        # Measure
        times = []
        with torch.no_grad(), autocast_ctx:
            for _ in range(self.config.iterations):
                timer = Timer(self.device)
                timer.start()

                if isinstance(inputs, dict):
                    _ = model(**inputs)
                elif isinstance(inputs, (list, tuple)):
                    _ = model(*inputs)
                else:
                    _ = model(inputs)

                timer.stop()
                times.append(timer.elapsed_ms)

        # Calculate metrics
        metrics.forward_time_mean_ms = statistics.mean(times)
        metrics.forward_time_std_ms = statistics.stdev(times) if len(times) > 1 else 0
        metrics.forward_time_min_ms = min(times)
        metrics.forward_time_max_ms = max(times)
        metrics.forward_time_p50_ms = statistics.median(times)
        sorted_times = sorted(times)
        n = len(sorted_times)
        metrics.forward_time_p95_ms = sorted_times[int(n * 0.95)]
        metrics.forward_time_p99_ms = sorted_times[int(n * 0.99)]
        metrics.forward_time_ms = metrics.forward_time_mean_ms
        metrics.throughput_samples_per_sec = (
            batch_size / (metrics.forward_time_mean_ms / 1000)
            if metrics.forward_time_mean_ms > 0
            else 0
        )

        # Memory
        metrics.peak_memory_mb = self._get_peak_memory()
        metrics.memory_allocated_mb = self._get_allocated_memory()

        return metrics

    def _benchmark_training(
        self, model: nn.Module, batch_size: int, get_input_fn: Callable
    ) -> Metrics:
        """Benchmark training performance"""
        model.train()
        metrics = Metrics()

        # Enable gradient computation for all parameters
        for param in model.parameters():
            param.requires_grad = True

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler() if self.config.precision == Precision.FP16 else None

        # Get autocast context
        autocast_ctx = self._get_autocast_context()

        # Generate input and target
        inputs = get_input_fn(batch_size, self.device)

        def compute_loss(outputs, inputs):
            """Compute a simple loss that requires gradients"""
            if isinstance(outputs, dict):
                # Get the main output tensor
                tensor = outputs.get("logits", outputs.get("last_hidden_state", None))
                if tensor is None:
                    # Get first tensor value
                    for v in outputs.values():
                        if isinstance(v, torch.Tensor):
                            tensor = v
                            break
                if tensor is None:
                    raise ValueError("Could not find output tensor")
            else:
                tensor = outputs

            # For models that output embeddings/hidden states, create a simple loss
            # by computing mean squared error against a target
            target = torch.zeros_like(tensor)
            loss = torch.nn.functional.mse_loss(tensor, target)
            return loss

        # Warmup
        for _ in range(self.config.warmup):
            optimizer.zero_grad()
            with autocast_ctx:
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                elif isinstance(inputs, (list, tuple)):
                    outputs = model(*inputs)
                else:
                    outputs = model(inputs)

                loss = compute_loss(outputs, inputs)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # Synchronize if CUDA
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        # Measure
        forward_times = []
        backward_times = []
        optimize_times = []
        total_times = []

        for _ in range(self.config.iterations):
            timer_total = Timer(self.device)
            timer_total.start()

            # Forward
            timer_fwd = Timer(self.device)
            timer_fwd.start()
            optimizer.zero_grad()
            with autocast_ctx:
                if isinstance(inputs, dict):
                    outputs = model(**inputs)
                elif isinstance(inputs, (list, tuple)):
                    outputs = model(*inputs)
                else:
                    outputs = model(inputs)

                loss = compute_loss(outputs, inputs)
            timer_fwd.stop()
            forward_times.append(timer_fwd.elapsed_ms)

            # Backward
            timer_bwd = Timer(self.device)
            timer_bwd.start()
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            timer_bwd.stop()
            backward_times.append(timer_bwd.elapsed_ms)

            # Optimize
            timer_opt = Timer(self.device)
            timer_opt.start()
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            timer_opt.stop()
            optimize_times.append(timer_opt.elapsed_ms)

            timer_total.stop()
            total_times.append(timer_total.elapsed_ms)

        # Calculate metrics
        metrics.forward_time_mean_ms = statistics.mean(forward_times)
        metrics.forward_time_ms = metrics.forward_time_mean_ms
        metrics.backward_time_ms = statistics.mean(backward_times)
        metrics.optimize_step_time_ms = statistics.mean(optimize_times)
        metrics.total_iteration_time_ms = statistics.mean(total_times)
        metrics.throughput_samples_per_sec = (
            batch_size / (metrics.total_iteration_time_ms / 1000)
            if metrics.total_iteration_time_ms > 0
            else 0
        )

        # Memory
        metrics.peak_memory_mb = self._get_peak_memory()
        metrics.memory_allocated_mb = self._get_allocated_memory()

        return metrics

    def _get_autocast_context(self):
        """Get autocast context based on precision and device"""
        from contextlib import nullcontext

        device_type = self.device.split(":")[0]  # cuda, mps, cpu

        if self.config.precision == Precision.FP16:
            if device_type == "mps":
                # MPS supports autocast but with some limitations
                return autocast(device_type="mps", dtype=torch.float16)
            return autocast(device_type=device_type, dtype=torch.float16)
        elif self.config.precision == Precision.BF16:
            if device_type == "mps":
                # MPS doesn't fully support BF16, fallback to FP32
                return nullcontext()
            return autocast(device_type=device_type, dtype=torch.bfloat16)
        elif self.config.precision == Precision.TF32:
            # TF32 is enabled by default on Ampere GPUs, no special context needed
            return nullcontext()
        else:
            # FP32 - no special context
            return nullcontext()

    def _get_peak_memory(self) -> float:
        """Get peak memory usage in MB"""
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        elif self.device == "mps" and hasattr(torch.mps, 'current_allocated_memory'):
            try:
                return torch.mps.current_allocated_memory() / (1024 * 1024)
            except Exception:
                pass
        return 0.0

    def _get_allocated_memory(self) -> float:
        """Get currently allocated memory in MB"""
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        elif self.device == "mps" and hasattr(torch.mps, 'current_allocated_memory'):
            try:
                return torch.mps.current_allocated_memory() / (1024 * 1024)
            except Exception:
                pass
        return 0.0

    def _clear_memory(self):
        """Clear device memory cache"""
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif self.device == "mps" and hasattr(torch.mps, 'empty_cache'):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    @staticmethod
    def format_time(ms: float) -> str:
        """Format time in appropriate units"""
        if ms < 1:
            return f"{ms * 1000:.2f} µs"
        elif ms < 1000:
            return f"{ms:.2f} ms"
        else:
            return f"{ms / 1000:.2f} s"


def create_benchmark_result(
    results: List[ModelResult],
    config: BenchmarkConfig,
    device_info: DeviceInfo,
    duration_seconds: float,
) -> BenchmarkResult:
    """Create a complete benchmark result"""

    metadata = {
        "tool_version": "1.0.0",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "test_date": datetime.now().isoformat(),
        "hostname": device_info.hostname,
    }

    test_config = {
        "precision": config.precision.value,
        "warmup_iterations": config.warmup,
        "test_iterations": config.iterations,
        "batch_sizes": config.batch_sizes,
        "device": config.device,
    }

    # Calculate summary
    all_throughputs = []
    for r in results:
        for metrics in r.training_metrics.values():
            if metrics.throughput_samples_per_sec > 0:
                all_throughputs.append(
                    (r.model_name, metrics.throughput_samples_per_sec)
                )

    best_throughput = max(all_throughputs, key=lambda x: x[1]) if all_throughputs else ("N/A", 0)

    summary = {
        "total_models_tested": len(results),
        "best_model_throughput": {
            "model": best_throughput[0],
            "throughput": best_throughput[1],
        },
        "test_duration_seconds": duration_seconds,
    }

    return BenchmarkResult(
        metadata=metadata,
        device_info=device_info.to_dict(),
        test_configuration=test_config,
        results=results,
        summary=summary,
        test_duration_seconds=duration_seconds,
    )
