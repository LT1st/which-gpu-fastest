# PyTorch GPU/CPU Benchmark Tool 使用文档

## 目录

- [简介](#简介)
- [安装](#安装)
- [快速开始](#快速开始)
- [命令详解](#命令详解)
  - [info - 设备信息](#info---设备信息)
  - [list-models - 模型列表](#list-models---模型列表)
  - [run - 运行测试](#run---运行测试)
  - [quick - 快速测试](#quick---快速测试)
  - [compare - 报告对比](#compare---报告对比)
- [支持的模型](#支持的模型)
- [输出格式](#输出格式)
- [配置文件](#配置文件)
- [进阶用法](#进阶用法)
- [常见问题](#常见问题)

---

## 简介

`which-gpu-fastest` 是一个 PyTorch 性能基准测试工具，用于评估不同硬件（GPU/CPU）上深度学习模型的训练和推理性能。

### 主要功能

- 自动检测设备（GPU/CPU）
- 测试多种主流深度学习模型
- 支持训练和推理性能测试
- 多种输出格式（JSON/CSV/TXT/HTML）
- 支持混合精度测试（FP32/FP16/BF16）
- 性能报告对比功能

---

## 安装

### 从 GitHub 安装

```bash
pip install git+https://github.com/LT1st/which-gpu-fastest.git
```

### 从源码安装

```bash
git clone https://github.com/LT1st/which-gpu-fastest.git
cd which-gpu-fastest
pip install -e .
```

### 验证安装

```bash
torch-gpu-benchmark --version
torch-gpu-benchmark --help
```

---

## 快速开始

### 1. 查看设备信息

```bash
torch-gpu-benchmark info
```

输出示例：
```
Device Information:
============================================================
Hostname: gpu-server-01
Platform: Linux-5.4.0-generic
Python: 3.10.12
PyTorch: 2.3.0

CPU:
  Name: Intel(R) Xeon(R) CPU
  Cores: 32
  Memory: 128.00 GB

GPU:
  [0] NVIDIA A100-SXM4-80GB
      Memory: 81920 MB
      Compute Capability: 8.0
```

### 2. 运行快速测试

```bash
torch-gpu-benchmark quick --light
```

### 3. 运行完整测试

```bash
torch-gpu-benchmark run -m resnet50,bert-tiny -b 1,8,32
```

---

## 命令详解

### info - 设备信息

显示当前系统的硬件配置信息。

```bash
torch-gpu-benchmark info [OPTIONS]
```

**选项：**

| 选项 | 说明 |
|------|------|
| `--json` | 以 JSON 格式输出 |

**示例：**

```bash
# 标准输出
torch-gpu-benchmark info

# JSON 格式（便于脚本处理）
torch-gpu-benchmark info --json
```

---

### list-models - 模型列表

列出所有可用的测试模型。

```bash
torch-gpu-benchmark list-models [OPTIONS]
```

**选项：**

| 选项 | 说明 |
|------|------|
| `--type TEXT` | 按类型筛选：`cnn`、`transformer`、`generative` |

**示例：**

```bash
# 列出所有模型
torch-gpu-benchmark list-models

# 只列出 CNN 模型
torch-gpu-benchmark list-models --type cnn

# 只列出 Transformer 模型
torch-gpu-benchmark list-models --type transformer
```

---

### run - 运行测试

运行自定义配置的性能测试。

```bash
torch-gpu-benchmark run [OPTIONS]
```

**选项：**

| 选项 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--models` | `-m` | `all` | 测试的模型，逗号分隔或 `all` |
| `--batch-sizes` | `-b` | `1,8,32` | 批次大小，逗号分隔 |
| `--precision` | `-p` | `fp32` | 精度模式：`fp32`/`fp16`/`bf16`/`tf32` |
| `--iterations` | `-i` | `100` | 测试迭代次数 |
| `--warmup` | `-w` | `10` | 预热迭代次数 |
| `--output` | `-o` | `benchmark_report.json` | 输出文件路径 |
| `--format` | `-f` | `json` | 输出格式：`json`/`csv`/`txt`/`html` |
| `--device` | | `auto` | 设备：`auto`/`cpu`/`cuda`/`cuda:0` |
| `--verbose` | `-v` | | 显示详细输出 |
| `--config` | | | YAML 配置文件路径 |

**示例：**

```bash
# 测试所有模型
torch-gpu-benchmark run

# 测试指定模型
torch-gpu-benchmark run -m resnet50,bert-tiny,unet

# 指定批次大小和精度
torch-gpu-benchmark run -m resnet50 -b 1,16,32,64 -p fp16

# 指定迭代次数
torch-gpu-benchmark run -m bert-base -i 200 -w 20

# 使用 CPU 测试
torch-gpu-benchmark run --device cpu

# 生成 HTML 报告
torch-gpu-benchmark run -o report.html -f html

# 使用配置文件
torch-gpu-benchmark run --config my_config.yaml

# 详细模式（用于调试）
torch-gpu-benchmark run -m resnet50 -v
```

---

### quick - 快速测试

使用预设配置快速运行测试。

```bash
torch-gpu-benchmark quick [OPTIONS]
```

**选项：**

| 选项 | 说明 |
|------|------|
| `--light` | 轻量级测试（约1分钟） |
| `--standard` | 标准测试（约3分钟），默认 |
| `--extensive` | 完整测试（约10分钟） |

**预设配置：**

| 模式 | 模型 | 批次大小 | 迭代次数 |
|------|------|----------|----------|
| `--light` | simple_cnn, bert-tiny | 1, 8 | 50 |
| `--standard` | resnet50, bert-tiny, unet | 1, 8, 32 | 100 |
| `--extensive` | resnet18, resnet50, bert-tiny, bert-mini, unet | 1, 8, 16, 32 | 200 |

**示例：**

```bash
# 轻量级测试（最快）
torch-gpu-benchmark quick --light

# 标准测试
torch-gpu-benchmark quick

# 完整测试
torch-gpu-benchmark quick --extensive

# 指定输出文件
torch-gpu-benchmark quick --light -o quick_test.json
```

---

### compare - 报告对比

对比两个测试报告的性能差异。

```bash
torch-gpu-benchmark compare <report1.json> <report2.json>
```

**示例：**

```bash
# 对比两次测试结果
torch-gpu-benchmark compare v100_report.json a100_report.json

# 对比 CPU 和 GPU 测试
torch-gpu-benchmark compare cpu_report.json gpu_report.json
```

**输出示例：**
```
Benchmark Comparison:
============================================================
Report 1: v100_report.json
Report 2: a100_report.json

Device Comparison:
  Report 1: NVIDIA V100-SXM2-32GB
  Report 2: NVIDIA A100-SXM4-80GB

Performance Comparison (Training Throughput):
------------------------------------------------------------
  resnet50 batch_size_32: 150.25 -> 268.50 (+78.7%)
  bert-tiny batch_size_32: 85.30 -> 136.25 (+59.7%)
```

---

## 支持的模型

### CNN 模型

| 模型名称 | 参数量 | 输入尺寸 | 说明 |
|----------|--------|----------|------|
| `resnet18` | 11.7M | 224×224 | 轻量级残差网络 |
| `resnet34` | 21.8M | 224×224 | 中型残差网络 |
| `resnet50` | 25.6M | 224×224 | 标准残差网络 |
| `resnet101` | 44.5M | 224×224 | 深层残差网络 |
| `resnet152` | 60.2M | 224×224 | 超深残差网络 |
| `vgg16` | 138M | 224×224 | VGG-16 |
| `vgg19` | 144M | 224×224 | VGG-19 |
| `alexnet` | 61M | 224×224 | AlexNet |
| `mobilenet_v2` | 3.5M | 224×224 | MobileNet V2 |
| `mobilenet_v3_small` | 2.5M | 224×224 | MobileNet V3 Small |
| `efficientnet_b0` | 5.3M | 224×224 | EfficientNet-B0 |
| `densenet121` | 8M | 224×224 | DenseNet-121 |
| `simple_cnn` | ~5M | 224×224 | 简单CNN（用于快速测试） |

### Transformer 模型

| 模型名称 | 参数量 | 序列长度 | 说明 |
|----------|--------|----------|------|
| `bert-tiny` | 4.4M | 128 | 超小BERT（快速测试） |
| `bert-mini` | 11.3M | 128 | 迷你BERT |
| `bert-small` | 29M | 128 | 小型BERT |
| `bert-base` | 110M | 128 | 标准BERT |
| `gpt2-small` | 124M | 128 | GPT-2 Small |
| `distilbert` | 66M | 128 | 蒸馏BERT |

### 生成模型

| 模型名称 | 参数量 | 输入尺寸 | 说明 |
|----------|--------|----------|------|
| `vae` | ~10M | 64×64 | 变分自编码器 |
| `unet` | ~30M | 128×128 | U-Net 分割网络 |
| `autoencoder` | ~5M | 64×64 | 卷积自编码器 |

---

## 输出格式

### JSON 格式

```bash
torch-gpu-benchmark run -o report.json -f json
```

结构：
```json
{
  "metadata": {
    "tool_version": "0.1.0",
    "pytorch_version": "2.3.0",
    "test_date": "2024-01-15T10:30:00"
  },
  "device_info": {
    "hostname": "gpu-server-01",
    "cpu_info": {...},
    "gpu_info": [...]
  },
  "test_configuration": {
    "precision": "fp32",
    "iterations": 100,
    "batch_sizes": [1, 8, 32]
  },
  "results": [
    {
      "model_name": "resnet50",
      "model_type": "cnn",
      "parameters_count": 25600000,
      "load_time_ms": 245.3,
      "inference_metrics": {...},
      "training_metrics": {...}
    }
  ],
  "summary": {
    "total_models_tested": 3,
    "best_model_throughput": {...}
  }
}
```

### CSV 格式

```bash
torch-gpu-benchmark run -o report.csv -f csv
```

输出：
```csv
Model,Type,Batch_Size,Mode,Forward_ms,Backward_ms,Throughput,Memory_MB
resnet50,cnn,1,inference,8.20,N/A,122.00,850.50
resnet50,cnn,1,training,8.50,15.20,45.20,1200.30
resnet50,cnn,32,inference,42.10,N/A,760.00,5200.30
```

### 文本格式

```bash
torch-gpu-benchmark run -o report.txt -f txt
```

输出：
```
================================================================================
                    PyTorch GPU Benchmark Report
================================================================================

Test Date: 2024-01-15T10:30:00
Duration: 180.5 seconds

Device Information:
  CPU: Intel(R) Xeon(R) CPU
  Cores: 32
  Memory: 128.00 GB
  GPU: NVIDIA A100-SXM4-80GB
  GPU Memory: 81920 MB

--------------------------------------------------------------------------------
                         Model Performance Summary
--------------------------------------------------------------------------------

ResNet-50 (cnn, 25.6M params)
  Training (batch_size=32, fp32):
    +---------------------------+-----------------+
    | Metric                    | Value           |
    +---------------------------+-----------------+
    | Forward Time              | 42.10 ms        |
    | Backward Time             | 95.30 ms        |
    | Total Iteration           | 145.90 ms       |
    | Throughput                | 219.35 smp/s    |
    | Peak Memory               | 5200.30 MB      |
    +---------------------------+-----------------+
```

### HTML 格式

```bash
torch-gpu-benchmark run -o report.html -f html
```

生成带有表格和样式美观的 HTML 页面，可在浏览器中查看。

---

## 配置文件

使用 YAML 配置文件可以保存常用的测试配置。

**示例配置文件 `my_config.yaml`：**

```yaml
# 基准测试配置
benchmark:
  iterations: 100        # 测试迭代次数
  warmup: 10             # 预热迭代次数
  precision: fp16        # 精度模式: fp32, fp16, bf16, tf32
  device: auto           # 设备: auto, cpu, cuda

# 测试的模型列表
models:
  - resnet50
  - resnet101
  - bert-tiny
  - bert-mini

# 批次大小列表
batch_sizes:
  - 1
  - 8
  - 16
  - 32

# 输出配置
output:
  path: ./my_benchmark.json
  format: json
  verbose: false
```

**使用配置文件：**

```bash
torch-gpu-benchmark run --config my_config.yaml
```

---

## 进阶用法

### 1. 多 GPU 测试

```bash
# 指定 GPU ID
torch-gpu-benchmark run --device cuda:0
torch-gpu-benchmark run --device cuda:1

# 对比不同 GPU
torch-gpu-benchmark run --device cuda:0 -o gpu0.json
torch-gpu-benchmark run --device cuda:1 -o gpu1.json
torch-gpu-benchmark compare gpu0.json gpu1.json
```

### 2. 精度对比测试

```bash
# FP32 测试
torch-gpu-benchmark run -p fp32 -o fp32_report.json

# FP16 测试
torch-gpu-benchmark run -p fp16 -o fp16_report.json

# 对比
torch-gpu-benchmark compare fp32_report.json fp16_report.json
```

### 3. 自动化脚本

```bash
#!/bin/bash
# benchmark_all.sh

DATE=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmarks_$DATE"

mkdir -p $OUTPUT_DIR

# 测试不同批次大小
for bs in "1" "8,16" "32,64"; do
    torch-gpu-benchmark run -b $bs -o "$OUTPUT_DIR/batch_${bs//,/_}.json"
done

# 测试不同精度
for prec in fp32 fp16; do
    torch-gpu-benchmark run -p $prec -o "$OUTPUT_DIR/${prec}_report.json"
done

echo "All benchmarks saved to $OUTPUT_DIR"
```

### 4. 性能调优建议

根据测试结果，可以做出以下优化决策：

1. **批次大小选择**
   - 选择吞吐量最高且不 OOM 的批次大小
   - 通常批次大小越大，GPU 利用率越高

2. **精度选择**
   - FP16/BF16 通常能提升 2x 性能
   - 注意检查精度损失是否可接受

3. **模型选择**
   - 根据吞吐量选择适合硬件的模型
   - 小模型可能受限于 CPU/数据传输

---

## 常见问题

### Q: 为什么显存显示为 0？

A: 在 CPU 模式下，显存统计不可用。需要使用 GPU 进行测试才能看到显存数据。

### Q: 测试结果波动很大怎么办？

A:
1. 增加预热迭代次数：`-w 20`
2. 增加测试迭代次数：`-i 200`
3. 关闭其他 GPU 进程
4. 使用 `--verbose` 查看详细时间分布

### Q: 如何测试自定义模型？

A: 目前工具支持预定义的模型列表。如需测试自定义模型，可以：
1. 在 `torch_gpu_benchmark/models/` 目录添加自定义模型
2. 在 `models/__init__.py` 中注册模型

### Q: 支持 macOS 吗？

A: 支持，但需要安装支持 Metal 的 PyTorch 版本。设备名称使用 `mps`。

### Q: 测试时报 OOM 错误？

A:
1. 减小批次大小：`-b 1,4,8`
2. 使用混合精度：`-p fp16`
3. 选择更小的模型

### Q: 如何解释测试结果？

A: 关键指标：
- **Throughput (samples/sec)**: 越高越好，表示每秒处理的样本数
- **Forward Time**: 前向传播时间，影响推理速度
- **Backward Time**: 反向传播时间，影响训练速度
- **Peak Memory**: 峰值显存占用

---

## 更新日志

### v0.1.0 (2024-01-15)
- 初始版本
- 支持 CNN、Transformer、生成模型
- 支持多种输出格式
- 支持报告对比功能

---

## 联系方式

- GitHub: https://github.com/LT1st/which-gpu-fastest
- Issues: https://github.com/LT1st/which-gpu-fastest/issues
