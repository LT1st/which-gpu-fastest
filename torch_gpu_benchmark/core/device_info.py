"""Device information collection"""

import platform
from dataclasses import dataclass, field
from typing import List, Optional

import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class GPUInfo:
    """GPU device information"""
    gpu_id: int
    gpu_name: str
    gpu_memory_total_mb: int
    driver_version: str = "N/A"
    cuda_compute_capability: str = "N/A"
    gpu_architecture: str = "N/A"

    def to_dict(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "gpu_name": self.gpu_name,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "driver_version": self.driver_version,
            "cuda_compute_capability": self.cuda_compute_capability,
            "gpu_architecture": self.gpu_architecture,
        }


@dataclass
class CPUInfo:
    """CPU device information"""
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    cpu_freq_mhz: float
    memory_total_gb: float

    def to_dict(self) -> dict:
        return {
            "cpu_name": self.cpu_name,
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "memory_total_gb": self.memory_total_gb,
        }


@dataclass
class DeviceInfo:
    """Combined device information"""
    hostname: str
    platform: str
    python_version: str
    pytorch_version: str
    cuda_version: Optional[str] = None
    cpu_info: Optional[CPUInfo] = None
    gpu_info: List[GPUInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
            "cpu_info": self.cpu_info.to_dict() if self.cpu_info else None,
            "gpu_info": [g.to_dict() for g in self.gpu_info],
        }

    @classmethod
    def collect(cls) -> "DeviceInfo":
        """Collect device information"""
        import socket

        # Basic info
        hostname = socket.gethostname()
        platform_info = platform.platform()
        python_version = platform.python_version()
        pytorch_version = torch.__version__

        # CUDA version
        cuda_version = None
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda

        # CPU info
        cpu_info = cls._collect_cpu_info()

        # GPU info
        gpu_info = cls._collect_gpu_info()

        return cls(
            hostname=hostname,
            platform=platform_info,
            python_version=python_version,
            pytorch_version=pytorch_version,
            cuda_version=cuda_version,
            cpu_info=cpu_info,
            gpu_info=gpu_info,
        )

    @staticmethod
    def _collect_cpu_info() -> CPUInfo:
        """Collect CPU information"""
        cpu_name = platform.processor() or "Unknown CPU"

        if HAS_PSUTIL:
            cpu_cores = psutil.cpu_count(logical=False) or 1
            cpu_threads = psutil.cpu_count(logical=True) or 1
            freq = psutil.cpu_freq()
            cpu_freq_mhz = freq.current if freq else 0
            memory = psutil.virtual_memory()
            memory_total_gb = round(memory.total / (1024**3), 2)
        else:
            import os
            cpu_cores = os.cpu_count() or 1
            cpu_threads = cpu_cores
            cpu_freq_mhz = 0
            memory_total_gb = 0

        return CPUInfo(
            cpu_name=cpu_name,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            cpu_freq_mhz=cpu_freq_mhz,
            memory_total_gb=memory_total_gb,
        )

    @staticmethod
    def _collect_gpu_info() -> List[GPUInfo]:
        """Collect GPU information"""
        gpu_list = []

        if not torch.cuda.is_available():
            return gpu_list

        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_info = GPUInfo(
                gpu_id=i,
                gpu_name=props.name,
                gpu_memory_total_mb=props.total_memory // (1024 * 1024),
                cuda_compute_capability=f"{props.major}.{props.minor}",
            )
            gpu_list.append(gpu_info)

        return gpu_list

    def get_device_string(self, device_id: int = 0) -> str:
        """Get device string for PyTorch"""
        if self.gpu_info and torch.cuda.is_available():
            return f"cuda:{device_id}"
        return "cpu"

    def has_gpu(self) -> bool:
        """Check if GPU is available"""
        return len(self.gpu_info) > 0 and torch.cuda.is_available()
