"""Precise timer for benchmarking"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TimerResult:
    """Timer measurement result"""
    elapsed_ms: float
    device: str


class Timer:
    """Precise timer supporting CPU, CUDA, and MPS devices"""

    def __init__(self, device: str = "cpu"):
        """
        Initialize timer.

        Args:
            device: "cpu", "cuda", "cuda:0", "mps", etc.
        """
        self.device = device
        self._start_event = None
        self._end_event = None
        self._start_time = None
        self._end_time = None
        self._elapsed_ms: Optional[float] = None

    @contextmanager
    def measure(self):
        """Context manager for timing code blocks"""
        self.start()
        yield self
        self.stop()

    def start(self):
        """Start timing"""
        if self.device.startswith("cuda") and torch.cuda.is_available():
            # Use CUDA events for precise GPU timing
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            # Use CPU timer for CPU, MPS, and other devices
            self._start_time = time.perf_counter()

    def stop(self):
        """Stop timing"""
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self._end_event.record()
            torch.cuda.synchronize()
            self._elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            # MPS uses CPU timer with synchronization
            if self.device == "mps" and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            self._end_time = time.perf_counter()
            self._elapsed_ms = (self._end_time - self._start_time) * 1000

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self._elapsed_ms is None:
            raise RuntimeError("Timer has not been stopped yet")
        return self._elapsed_ms

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds"""
        return self.elapsed_ms / 1000.0


def measure_time(func, device: str = "cpu", *args, **kwargs) -> tuple:
    """
    Measure execution time of a function.

    Returns:
        Tuple of (result, elapsed_ms)
    """
    timer = Timer(device)
    timer.start()
    result = func(*args, **kwargs)
    timer.stop()
    return result, timer.elapsed_ms
