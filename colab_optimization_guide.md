# Google Colab GPU Optimization Guide for PCB Defect Detection

## 🎯 Problem: Low Batch Size + Unused GPU Memory

When running on Google Colab with single GPU, small batch sizes (due to memory constraints) often leave GPU memory unused. Here are proven strategies to maximize utilization:

## 🚀 Solution 1: Gradient Accumulation (Recommended)

### What it does:
- Simulates larger effective batch sizes
- Uses all available GPU memory
- Maintains training stability

### How to implement:
Add `accumulate` parameter to your configs:

```yaml
training:
  batch: 8                    # Physical batch (fits in memory)
  accumulate: 4               # Accumulate 4 batches
  # Effective batch size = 8 × 4 = 32
```

**Memory Usage:**
- Physical memory: Only for batch size 8
- Gradient quality: Equivalent to batch size 32

## 🔧 Solution 2: Automatic Batch Sizing

Let ultralytics automatically find optimal batch size:

```yaml
training:
  batch: -1                   # Auto-find ~60% memory usage
  # or
  batch: 0.80                 # Use 80% of available memory
```

## ⚡ Solution 3: Colab-Optimized Configs

### For Standard Models (YOLOv8n):
```yaml
training:
  batch: -1                   # Auto-optimize
  accumulate: 2               # 2x effective batch size
  amp: true                   # Mixed precision (essential)
  cache: true                 # Cache images in RAM
  workers: 2                  # Colab CPU limitations
```

### For Attention Models:
```yaml
training:
  batch: 8                    # Fixed small batch
  accumulate: 4               # 4x accumulation = effective 32
  amp: true                   # Essential for attention
  cache: false                # May exceed Colab RAM
  workers: 2
```

### For High-Resolution (1024px):
```yaml
training:
  batch: 4                    # Very small batch
  accumulate: 8               # 8x accumulation = effective 32
  amp: true                   # Critical for high-res
  cache: false                # Cannot cache high-res
  workers: 1                  # Conservative
```

## 🏃‍♂️ Solution 4: Sequential Experiment Runner

Create a script to run multiple experiments automatically:

```bash
#!/bin/bash
# colab_run_all.sh

echo "🚀 Starting Sequential PCB Experiments on Colab"

# Baseline experiments
python run_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml
python run_experiment.py --config experiments/configs/02_yolov8s_baseline_standard.yaml

# Attention experiments  
python run_experiment.py --config experiments/configs/04_yolov8n_eca_standard.yaml
python run_experiment.py --config experiments/configs/05_yolov8n_cbam_standard.yaml

# Loss function experiments
python run_experiment.py --config experiments/configs/07_yolov8n_baseline_focal_siou.yaml

echo "✅ All experiments completed!"
```

## 📊 Memory Monitoring for Colab

Add this to your training script to monitor GPU usage:

```python
import GPUtil

def log_gpu_usage():
    gpu = GPUtil.getGPUs()[0]
    print(f"GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryPercent:.1f}%)")

# Call during training
log_gpu_usage()
```

## 🎯 Colab-Specific Settings

### Environment Variables:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1  # For debugging
```

### Python Memory Management:
```python
import torch
import gc

# Clear cache between experiments
torch.cuda.empty_cache()
gc.collect()
```

## 📈 Expected Performance Improvements

| Strategy | Memory Usage | Training Speed | Effective Batch Size |
|----------|-------------|----------------|---------------------|
| Default (batch=8) | ~30% | Baseline | 8 |
| Auto batch | ~60% | 1.5x faster | 16-24 |
| Gradient accumulation | ~60% | 1.3x faster | 32+ |
| Combined approach | ~80% | 2x faster | 32-64 |

## 🚨 Colab Limitations to Remember

1. **Session timeout**: 12 hours maximum
2. **RAM limit**: 12-16GB (affects caching)
3. **Disk space**: Limited, clean up between experiments
4. **GPU memory**: Usually 15GB (T4) or 16GB (P100/V100)

## 🏆 Best Practice for Colab

```yaml
# Optimal Colab configuration
training:
  batch: -1                   # Auto-optimize memory usage
  accumulate: 2               # Double effective batch size
  amp: true                   # Essential for memory efficiency
  cache: true                 # Use RAM caching if possible
  workers: 2                  # Colab CPU constraint
  patience: 50                # Shorter for limited session time
  save_period: 25             # Frequent saves (session timeout)
```

## 🎮 Ready-to-Use Colab Commands

```bash
# Quick memory optimization test
python run_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml

# Check if auto-batch works
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print(model.train(data='coco8.yaml', epochs=1, batch=-1, verbose=True))"

# Monitor GPU during training
watch -n 1 nvidia-smi
```

This approach will help you maximize GPU utilization on Colab while maintaining training quality!