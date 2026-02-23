"""Device information collection - Cross-platform support"""

import platform
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_platform_info() -> dict:
    """Get detailed platform information"""
    system = platform.system().lower()
    is_windows = system == "windows"
    is_linux = system == "linux"
    is_macos = system == "darwin"

    return {
        "system": system,
        "is_windows": is_windows,
        "is_linux": is_linux,
        "is_macos": is_macos,
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


@dataclass
class GPUInfo:
    """GPU device information"""
    gpu_id: int
    gpu_name: str
    gpu_memory_total_mb: int
    driver_version: str = "N/A"
    cuda_compute_capability: str = "N/A"
    gpu_architecture: str = "N/A"
    gpu_type: str = "cuda"  # cuda, rocm, mps, directml

    def to_dict(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "gpu_name": self.gpu_name,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "driver_version": self.driver_version,
            "cuda_compute_capability": self.cuda_compute_capability,
            "gpu_architecture": self.gpu_architecture,
            "gpu_type": self.gpu_type,
        }


@dataclass
class CPUInfo:
    """CPU device information"""
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    cpu_freq_mhz: float
    memory_total_gb: float
    platform_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cpu_name": self.cpu_name,
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "memory_total_gb": self.memory_total_gb,
            "platform_info": self.platform_info,
        }


@dataclass
class DeviceInfo:
    """Combined device information"""
    hostname: str
    platform: str
    python_version: str
    pytorch_version: str
    platform_info: dict = field(default_factory=dict)
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None
    cpu_info: Optional[CPUInfo] = None
    gpu_info: List[GPUInfo] = field(default_factory=list)
    mps_available: bool = False

    def to_dict(self) -> dict:
        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "platform_info": self.platform_info,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
            "rocm_version": self.rocm_version,
            "mps_available": self.mps_available,
            "cpu_info": self.cpu_info.to_dict() if self.cpu_info else None,
            "gpu_info": [g.to_dict() for g in self.gpu_info],
        }

    @classmethod
    def collect(cls) -> "DeviceInfo":
        """Collect device information"""
        import socket

        # Platform info
        platform_info = get_platform_info()

        # Basic info
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "unknown"

        platform_str = platform.platform()
        python_version = platform.python_version()
        pytorch_version = torch.__version__

        # CUDA version
        cuda_version = None
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda

        # ROCm version
        rocm_version = None
        if hasattr(torch.version, 'hip') and torch.version.hip:
            rocm_version = torch.version.hip

        # MPS (Apple Silicon) availability
        mps_available = False
        if hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()

        # CPU info
        cpu_info = cls._collect_cpu_info(platform_info)

        # GPU info
        gpu_info = cls._collect_gpu_info(platform_info)

        return cls(
            hostname=hostname,
            platform=platform_str,
            platform_info=platform_info,
            python_version=python_version,
            pytorch_version=pytorch_version,
            cuda_version=cuda_version,
            rocm_version=rocm_version,
            cpu_info=cpu_info,
            gpu_info=gpu_info,
            mps_available=mps_available,
        )

    @staticmethod
    def _collect_cpu_info(platform_info: dict) -> CPUInfo:
        """Collect CPU information - cross-platform"""
        cpu_name = "Unknown CPU"
        cpu_cores = 1
        cpu_threads = 1
        cpu_freq_mhz = 0.0
        memory_total_gb = 0.0

        # Get CPU name based on platform
        if platform_info["is_windows"]:
            cpu_name = platform.processor() or "Unknown CPU"
            try:
                import wmi
                c = wmi.WMI()
                for cpu in c.Win32_Processor():
                    cpu_name = cpu.Name
                    break
            except Exception:
                pass
        elif platform_info["is_macos"]:
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    cpu_name = result.stdout.strip()
            except Exception:
                cpu_name = platform.processor() or "Apple Silicon"
        elif platform_info["is_linux"]:
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            break
            except Exception:
                cpu_name = platform.processor() or "Unknown CPU"

        # Get core counts and frequency using psutil
        if HAS_PSUTIL:
            cpu_cores = psutil.cpu_count(logical=False) or 1
            cpu_threads = psutil.cpu_count(logical=True) or 1
            freq = psutil.cpu_freq()
            if freq:
                cpu_freq_mhz = freq.current
            memory = psutil.virtual_memory()
            memory_total_gb = round(memory.total / (1024**3), 2)
        else:
            import os
            cpu_cores = os.cpu_count() or 1
            cpu_threads = cpu_cores

        return CPUInfo(
            cpu_name=cpu_name,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            cpu_freq_mhz=cpu_freq_mhz,
            memory_total_gb=memory_total_gb,
            platform_info=platform_info,
        )

    @staticmethod
    def _collect_gpu_info(platform_info: dict) -> List[GPUInfo]:
        """Collect GPU information - supports CUDA, ROCm, MPS"""
        gpu_list = []

        # Check CUDA/ROCm (PyTorch uses same API)
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)

                # Detect if ROCm or CUDA
                gpu_type = "cuda"
                if hasattr(torch.version, 'hip') and torch.version.hip:
                    gpu_type = "rocm"

                gpu_info = GPUInfo(
                    gpu_id=i,
                    gpu_name=props.name,
                    gpu_memory_total_mb=props.total_memory // (1024 * 1024),
                    cuda_compute_capability=f"{props.major}.{props.minor}",
                    gpu_type=gpu_type,
                )
                gpu_list.append(gpu_info)

        # Check MPS (Apple Silicon)
        if platform_info["is_macos"] and hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                # MPS doesn't provide detailed GPU info like CUDA
                gpu_info = GPUInfo(
                    gpu_id=0,
                    gpu_name="Apple Silicon GPU",
                    gpu_memory_total_mb=0,  # Unified memory
                    gpu_type="mps",
                )
                gpu_list.append(gpu_info)

        return gpu_list

    def get_device_string(self, device_id: int = 0) -> str:
        """Get device string for PyTorch"""
        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available() and self.gpu_info:
            return f"cuda:{device_id}"

        if self.mps_available:
            return "mps"

        return "cpu"

    def get_available_devices(self) -> List[str]:
        """Get list of available devices"""
        devices = ["cpu"]

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")

        if self.mps_available:
            devices.append("mps")

        return devices

    def has_gpu(self) -> bool:
        """Check if any GPU is available"""
        return len(self.gpu_info) > 0 or self.mps_available

    def get_device_type(self) -> str:
        """Get the primary device type"""
        if torch.cuda.is_available():
            if hasattr(torch.version, 'hip') and torch.version.hip:
                return "rocm"
            return "cuda"
        if self.mps_available:
            return "mps"
        return "cpu"
