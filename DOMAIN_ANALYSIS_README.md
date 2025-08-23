# Domain Adaptation Analysis Script

## 📖 Overview

`run_domain_analysis.py` is a comprehensive end-to-end Python script that conducts domain adaptation studies using the Ultralytics YOLO framework. It evaluates how well a model pre-trained on the HRIPCB dataset generalizes to the "MIXED PCB DEFECT DATASET" and measures the performance improvement after fine-tuning.

## 🚀 Features

- **Complete End-to-End Pipeline**: From dataset preparation to final performance comparison
- **Zero-Shot Evaluation**: Establishes baseline performance on the target domain
- **Domain-Adaptive Fine-Tuning**: Optimized hyperparameters for domain adaptation
- **Comprehensive Reporting**: Detailed performance analysis and improvement metrics
- **Robust Error Handling**: Professional error handling and logging
- **Command-Line Interface**: Easy-to-use CLI with sensible defaults

## 📋 Requirements

```bash
pip install ultralytics
pip install pyyaml
```

## 🎯 Command-Line Usage

### Basic Usage
```bash
python run_domain_analysis.py --weights path/to/best.pt --dataset-dir path/to/mixed_pcb_dataset
```

### Advanced Usage
```bash
python run_domain_analysis.py \
  --weights experiments/results/E19_YOLOv10n_Ultimate/weights/best.pt \
  --dataset-dir datasets/mixed_pcb_defect_dataset \
  --epochs 30
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--weights` | ✅ Yes | - | Path to the best.pt file pre-trained on HRIPCB |
| `--dataset-dir` | ✅ Yes | - | Root directory of the MIXED PCB DEFECT DATASET |
| `--epochs` | ❌ No | 20 | Number of epochs for fine-tuning |

## 📁 Expected Dataset Structure

The script expects the following directory structure for the MIXED PCB DEFECT DATASET:

```
mixed_pcb_defect_dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 🔄 Analysis Pipeline

### Step 0: Dataset Preparation
- Creates `mixed_pcb_data.yaml` configuration file
- Defines six defect classes: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`
- Validates dataset structure

### Step 1: Zero-Shot Evaluation (Baseline)
- Loads pre-trained HRIPCB model
- Evaluates on MIXED PCB test set without any training
- Establishes baseline performance metrics
- Saves results to `runs/detect/domain_adaptation/[timestamp]/zeroshot_evaluation/`

### Step 2: Fine-Tuning on Target Domain
- Loads the same pre-trained model
- Fine-tunes on MIXED PCB training set
- Uses domain-adaptation optimized hyperparameters:
  - **Learning Rate**: `lr0=0.001` (low for fine-tuning)
  - **Batch Size**: `32` (smaller for stability)
  - **Patience**: Adaptive based on epoch count
  - **Mixed Precision**: Enabled for efficiency
- Saves fine-tuned model to `runs/detect/domain_adaptation/[timestamp]/finetune_on_mixed_pcb/`

### Step 3: Post-Fine-Tuning Evaluation
- Loads the fine-tuned model
- Evaluates on MIXED PCB test set
- Measures performance after domain adaptation
- Saves results to `runs/detect/domain_adaptation/[timestamp]/post_finetune_evaluation/`

### Step 4: Comprehensive Comparison Report
- Calculates absolute and percentage improvements
- Generates detailed console report
- Saves complete analysis to JSON files

## 📊 Output Structure

```
runs/detect/domain_adaptation/[timestamp]/
├── mixed_pcb_data.yaml                    # Dataset configuration
├── domain_adaptation_report.json          # Complete analysis results
├── zeroshot_evaluation/                   # Zero-shot results
│   ├── zeroshot_metrics.json
│   └── [ultralytics validation outputs]
├── finetune_on_mixed_pcb/                # Fine-tuning results
│   ├── weights/
│   │   ├── best.pt                       # Fine-tuned model
│   │   └── last.pt
│   └── [training outputs]
└── post_finetune_evaluation/             # Post-fine-tuning results
    ├── post_finetune_metrics.json
    └── [ultralytics validation outputs]
```

## 📈 Sample Output Report

```
================================================================================
🎯 DOMAIN ADAPTATION ANALYSIS - FINAL REPORT
================================================================================
📅 Analysis Date: 2025-08-23 15:30:45
🏷️  Source Model: best.pt
📊 Target Dataset: mixed_pcb_defect_dataset
🔄 Fine-tuning Epochs: 20
================================================================================

📈 PERFORMANCE COMPARISON:
------------------------------------------------------------
Metric               Zero-Shot    Fine-Tuned   Improvement    
------------------------------------------------------------
mAP@0.5             0.7234       0.8456       +0.1222 (+16.9%)
mAP@0.5:0.95        0.4123       0.5234       +0.1111 (+26.9%)
Precision           0.8012       0.8567       +0.0555 ( +6.9%)
Recall              0.6543       0.7891       +0.1348 (+20.6%)
F1-Score            0.7201       0.8201       +0.1000 (+13.9%)

================================================================================
🏆 DOMAIN ADAPTATION SUMMARY:
✅ Domain adaptation was SUCCESSFUL!
🚀 Achieved SIGNIFICANT performance improvement (>5%)

🎯 Key Result: mAP@0.5:0.95 improved by +0.1111 (+26.9%)
📁 Detailed results saved to: runs/detect/domain_adaptation/20250823_153045
================================================================================
```

## ⚙️ Fine-Tuning Hyperparameters

The script uses domain-adaptation optimized hyperparameters:

```python
# Fine-tuning configuration
epochs = args.epochs          # User-specified (default: 20)
patience = max(10, epochs//4) # Adaptive patience
batch = 32                    # Smaller batch for stability
lr0 = 0.001                   # Low learning rate for fine-tuning
lrf = 0.01                    # Final learning rate factor
momentum = 0.937              # Standard momentum
weight_decay = 0.0005         # Light regularization
warmup_epochs = 3             # Gentle warmup
```

## 🔧 Key Features

### Professional Error Handling
- Input validation for weights and dataset paths
- Graceful error messages with actionable suggestions
- Comprehensive logging to both console and file

### Performance Tracking
- Tracks all key YOLO metrics: mAP@0.5, mAP@0.5:0.95, precision, recall, F1
- Calculates both absolute and percentage improvements
- Per-class performance analysis

### Flexible Configuration
- Automatically detects dataset structure
- Creates properly formatted YAML configuration
- Handles different dataset sizes and class distributions

## 🚨 Troubleshooting

### Common Issues

1. **"Weights file not found"**
   ```bash
   # Ensure the path is correct
   ls -la path/to/your/best.pt
   ```

2. **"Dataset directory not found"**
   ```bash
   # Check dataset structure
   ls -la path/to/mixed_pcb_dataset/
   ```

3. **"CUDA out of memory"**
   ```bash
   # Reduce batch size in the script
   batch = 16  # Instead of 32
   ```

### Performance Expectations

- **Good Adaptation**: >10% improvement in mAP@0.5:0.95
- **Excellent Adaptation**: >20% improvement in mAP@0.5:0.95
- **Poor Adaptation**: <5% improvement (consider more epochs or data augmentation)

## 📝 Logging

The script creates detailed logs in `domain_adaptation_analysis.log` with:
- Timestamp for each operation
- Performance metrics at each step
- Error messages and debugging information
- Complete analysis timeline

## 🔄 Integration with Existing Experiments

This script is designed to work seamlessly with models trained using the comprehensive benchmark experiments (E01-E20). Simply use the `best.pt` file from any experiment as the `--weights` parameter.

Example with benchmark experiments:
```bash
# Using E19 (top performer) for domain adaptation
python run_domain_analysis.py \
  --weights experiments/results/E19_YOLOv10n_CBAM/weights/best.pt \
  --dataset-dir datasets/mixed_pcb_defect_dataset \
  --epochs 25
```

This comprehensive script provides a complete solution for domain adaptation analysis in PCB defect detection tasks.