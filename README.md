# which-gpu-fastest

[![PyPI version](https://badge.fury.io/py/which-gpu-fastest.svg)](https://badge.fury.io/py/which-gpu-fastest)
[![Python](https://img.shields.io/pypi/pyversions/which-gpu-fastest.svg)](https://pypi.org/project/which-gpu-fastest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command-line tool for benchmarking PyTorch performance on different hardware (GPU/CPU).

## Features

- Test training and inference performance for various deep learning models
- Support for CNN (ResNet, VGG, EfficientNet, etc.)
- Support for Transformers (BERT, GPT-2, DistilBERT)
- Support for Generative models (VAE, U-Net, Autoencoder)
- Multiple output formats (JSON, CSV, TXT, HTML)
- Automatic device detection (GPU/CPU)
- Mixed precision support (FP32, FP16, BF16)
- Quick benchmark presets

## Installation

### From PyPI (Recommended)
```bash
pip install which-gpu-fastest
```

### From GitHub
```bash
pip install git+https://github.com/LT1st/which-gpu-fastest.git
```

### From Source
```bash
git clone https://github.com/LT1st/which-gpu-fastest.git
cd which-gpu-fastest
pip install -e .
```

## Quick Start

```bash
# Show device information
torch-gpu-benchmark info

# List available models
torch-gpu-benchmark list-models

# Run quick benchmark
torch-gpu-benchmark quick

# Run custom benchmark
torch-gpu-benchmark run -m resnet50,bert-tiny -b 1,8,32 -p fp32
```

## Commands

### `torch-gpu-benchmark run`

Run benchmark with custom configuration.

```bash
torch-gpu-benchmark run [OPTIONS]

Options:
  -m, --models TEXT          Models to test (comma-separated) or 'all'
  -b, --batch-sizes TEXT     Batch sizes to test (comma-separated)
  -p, --precision TEXT       Precision mode [fp32|fp16|bf16|tf32]
  -i, --iterations INTEGER   Number of test iterations
  -w, --warmup INTEGER       Number of warmup iterations
  -o, --output PATH          Output file path
  -f, --format TEXT          Output format [json|csv|txt|html]
  --device TEXT              Device to use [auto|cpu|cuda]
  -v, --verbose              Enable verbose output
  --config PATH              Path to YAML config file
```

### `torch-gpu-benchmark quick`

Run quick benchmark with preset configuration.

```bash
torch-gpu-benchmark quick [OPTIONS]

Options:
  --light                    Light test (~1 minute)
  --standard                 Standard test (~3 minutes) [default]
  --extensive                Extensive test (~10 minutes)
```

### `torch-gpu-benchmark info`

Show device information.

```bash
torch-gpu-benchmark info [--json]
```

### `torch-gpu-benchmark list-models`

List available models.

```bash
torch-gpu-benchmark list-models [--type cnn|transformer|generative]
```

### `torch-gpu-benchmark compare`

Compare two benchmark reports.

```bash
torch-gpu-benchmark compare report1.json report2.json
```

## Examples

```bash
# Quick test
torch-gpu-benchmark quick

# Test specific models
torch-gpu-benchmark run -m resnet50,bert-base,unet

# Test with different batch sizes
torch-gpu-benchmark run -b 1,16,32,64

# Run on CPU only
torch-gpu-benchmark run --device cpu

# Generate HTML report
torch-gpu-benchmark run -o report.html -f html

# Use config file
torch-gpu-benchmark run --config configs/default_config.yaml
```

## Available Models

### CNN Models
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- `vgg16`, `vgg19`
- `alexnet`
- `mobilenet_v2`, `mobilenet_v3_small`
- `efficientnet_b0`
- `densenet121`
- `simple_cnn` (for quick testing)

### Transformer Models
- `bert-tiny`, `bert-mini`, `bert-small`, `bert-base`
- `gpt2-small`
- `distilbert`

### Generative Models
- `vae`
- `unet`
- `autoencoder`

## Output Formats

### JSON
```json
{
  "metadata": { ... },
  "device_info": { ... },
  "results": [
    {
      "model_name": "resnet50",
      "training_metrics": {
        "batch_size_32": {
          "forward_time_ms": 42.1,
          "throughput_samples_per_sec": 268.5
        }
      }
    }
  ]
}
```

### CSV
```csv
Model,Type,Batch_Size,Mode,Forward_ms,Throughput,Memory_MB
resnet50,cnn,32,training,42.1,268.5,5200.3
```

### Text
```
================================================================================
                    PyTorch GPU Benchmark Report
================================================================================
...
```

### HTML
Interactive HTML report with tables and charts.

## Configuration File

```yaml
# configs/default_config.yaml
benchmark:
  iterations: 100
  warmup: 10
  precision: fp32
  device: auto

models:
  - resnet50
  - bert-tiny

batch_sizes:
  - 1
  - 8
  - 32

output:
  path: ./benchmark_report.json
  format: json
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision (for CNN models)
- transformers (optional, for HuggingFace models)

## Publishing to PyPI

```bash
# Build the package
pip install build
python -m build

# Upload to PyPI
pip install twine
twine upload dist/*
```

## License

MIT License
