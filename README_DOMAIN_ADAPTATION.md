# Cross-Domain PCB Defect Detection Evaluation

This module provides a clean, production-ready framework for evaluating domain adaptation performance in PCB defect detection systems.

## Overview

The `domain_adaptation_evaluation.py` script evaluates how well a model trained on one PCB dataset (source domain) generalizes to another PCB dataset (target domain), both with and without fine-tuning.

## Features

- **Zero-shot evaluation**: Tests pre-trained model on target domain without modification
- **Adaptive threshold optimization**: Automatically finds optimal confidence thresholds for target domain
- **Cross-domain fine-tuning**: Adapts model to target domain with optimized hyperparameters
- **Comprehensive reporting**: Generates detailed performance analysis and improvement metrics
- **Automatic preprocessing**: Handles grayscale-to-RGB conversion and class mapping
- **Research-backed optimizations**: Implements Test-Time Augmentation and relaxed NMS for better cross-domain performance

## Usage

### Basic Usage

```bash
python domain_adaptation_evaluation.py \
    --weights path/to/source_model.pt \
    --dataset path/to/target_dataset \
    --epochs 20
```

### Example

```bash
python domain_adaptation_evaluation.py \
    --weights experiments/results/best_model.pt \
    --dataset datasets/target_pcb_dataset \
    --epochs 25
```

## Dataset Structure

The target dataset should follow this structure:

```
target_dataset/
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

## Supported Class Mapping

The framework automatically handles mapping between different PCB defect class naming conventions:

- **Source classes (HRIPCB)**: Missing_hole, Mouse_bite, Open_circuit, Short, Spurious_copper, Spur
- **Target classes**: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper

## Output

The evaluation generates:

1. **Zero-shot results**: Performance on target domain without fine-tuning
2. **Fine-tuned results**: Performance after domain adaptation
3. **Improvement analysis**: Absolute and percentage improvements
4. **Comprehensive report**: JSON report with all metrics and analysis

### Sample Output

```
Starting domain adaptation analysis...
Preparing target dataset...
Evaluating zero-shot performance...
Zero-shot mAP@0.5: 0.3791
Fine-tuning model on target domain...
Evaluating fine-tuned model...
Fine-tuned mAP@0.5: 0.9503

Analysis completed in 542.3 seconds
Results saved to: domain_adaptation_results/20250827_031900

Domain Adaptation Results:
mAP@0.5:0.95 improvement: +0.3442 (+225.1%)
✓ Domain adaptation successful
```

## Key Metrics

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Cross-Domain Optimizations

The framework implements research-backed techniques for improved cross-domain performance:

- **Adaptive confidence thresholds**: Automatically optimized for target domain
- **Test-Time Augmentation**: Improves robustness to domain shift
- **Relaxed NMS**: Better generalization with iou=0.4
- **Increased detection limits**: Allows up to 1000 detections per image

## Requirements

- Python 3.8+
- ultralytics
- PIL (Pillow)
- numpy
- PyYAML

## Technical Details

- **Fine-tuning strategy**: Low learning rate (0.001) with adaptive patience
- **Batch size**: 32 (optimized for stability)
- **Image preprocessing**: Automatic grayscale-to-RGB conversion
- **Label remapping**: Automatic class index alignment between datasets
- **Evaluation**: Uses test split for unbiased performance assessment

This framework is designed for research reproducibility and can be easily integrated into larger experimental pipelines.