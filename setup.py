from setuptools import setup, find_packages

setup(
    name="torch-gpu-benchmark",
    version="1.0.0",
    description="PyTorch GPU/CPU Performance Benchmark Tool",
    author="Dev Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "jinja2>=3.1.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
    ],
    entry_points={
        "console_scripts": [
            "torch-gpu-benchmark=torch_gpu_benchmark.cli:main",
        ],
    },
    python_requires=">=3.8",
)
