# 🚀 PCB Defect Detection Experiments - Ready to Run

All 36 experiment configurations have been updated and are ready for execution.

## 📋 Summary of Changes Made

✅ **Fixed Issues:**
- **Batch Size**: Changed from 64 to 32 for all experiments
- **WandB Project**: Unified to `pcb-defect-comprehensive-ablation`
- **Loss Function Spelling**: Fixed `verifocal` → `varifocal`
- **Hyperparameters**: Optimized for each loss function type
- **Experiment Runner**: Updated to handle corrected loss function names

## 🎯 Experiment Matrix (36 Total)

### Architecture + Loss + Attention Combinations
| Architecture | Loss Functions | Attention Mechanisms | Total |
|-------------|---------------|---------------------|-------|
| YOLOv8n | 4 types | 3 mechanisms | 12 |
| YOLOv10n | 4 types | 3 mechanisms | 12 |
| YOLO11n | 4 types | 3 mechanisms | 12 |

### Loss Functions (4 types):
1. **Standard** (CIoU + BCE)
2. **SIoU** (SIoU + BCE) 
3. **EIoU** (EIoU + BCE)
4. **VariFocal+EIoU** (EIoU + VariFocal)

### Attention Mechanisms (3 types):
1. **None** (baseline)
2. **CBAM** (Convolutional Block Attention)
3. **CoordAtt** (Coordinate Attention)

## 📂 Results Storage Structure

All results will be saved under:
```
experiments/pcb-defect-optimized-v2/
├── YOLOV8N_STANDARD_NONE_Experiment/
├── YOLOV8N_STANDARD_CBAM_Experiment/
├── YOLOV8N_STANDARD_COORDATT_Experiment/
└── ... (33 more experiments)
```

## 🔥 Individual Experiment Commands

### YOLOv8n Experiments (12 total)

#### Standard Loss (CIoU + BCE)
```bash
# Baseline
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_standard_none_config.yaml

# With attention mechanisms  
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_standard_cbam_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_standard_coordatt_config.yaml
```

#### SIoU Loss
```bash
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_siou_none_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_siou_cbam_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_siou_coordatt_config.yaml
```

#### EIoU Loss
```bash  
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_eiou_none_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_eiou_cbam_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_eiou_coordatt_config.yaml
```

#### VariFocal+EIoU Loss
```bash
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_verifocal_eiou_none_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_verifocal_eiou_cbam_config.yaml  
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_verifocal_eiou_coordatt_config.yaml
```

### YOLOv10n Experiments (12 total)

#### Standard Loss (CIoU + BCE)
```bash
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_standard_none_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_standard_cbam_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_standard_coordatt_config.yaml
```

#### SIoU Loss
```bash
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_siou_none_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_siou_cbam_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_siou_coordatt_config.yaml
```

#### EIoU Loss
```bash
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_eiou_none_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_eiou_cbam_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_eiou_coordatt_config.yaml
```

#### VariFocal+EIoU Loss
```bash
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_verifocal_eiou_none_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_verifocal_eiou_cbam_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov10n_verifocal_eiou_coordatt_config.yaml
```

### YOLO11n Experiments (12 total)

#### Standard Loss (CIoU + BCE)
```bash
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_standard_none_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_standard_cbam_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_standard_coordatt_config.yaml
```

#### SIoU Loss
```bash
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_siou_none_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_siou_cbam_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_siou_coordatt_config.yaml
```

#### EIoU Loss
```bash
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_eiou_none_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_eiou_cbam_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_eiou_coordatt_config.yaml
```

#### VariFocal+EIoU Loss
```bash
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_verifocal_eiou_none_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_verifocal_eiou_cbam_config.yaml
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolo11n_verifocal_eiou_coordatt_config.yaml
```

## 🛠️ Optimized Hyperparameters by Loss Type

### Standard Loss (CIoU + BCE)
- **Batch**: 32
- **LR**: 0.0005 → 0.005 (cosine decay)
- **Weight Decay**: 0.0002
- **Loss Weights**: box=7.5, cls=0.5, dfl=1.5

### SIoU Loss  
- **Batch**: 32
- **LR**: 0.0005 → 0.005 (cosine decay)
- **Weight Decay**: 0.0002
- **Loss Weights**: box=8.0, cls=0.8, dfl=1.5

### EIoU Loss
- **Batch**: 32  
- **LR**: 0.0005 → 0.005 (cosine decay)
- **Weight Decay**: 0.0002
- **Loss Weights**: box=7.8, cls=0.7, dfl=1.6

### VariFocal+EIoU Loss
- **Batch**: 32
- **LR**: 0.0005 → 0.005 (cosine decay) 
- **Weight Decay**: 0.0002
- **Loss Weights**: box=8.2, cls=1.0, dfl=1.8

## 📊 WandB Monitoring

All experiments log to the same WandB project:
- **Project**: `pcb-defect-comprehensive-ablation`
- **Unique Run Names**: Based on config file names
- **Metrics Tracked**: mAP@0.5, mAP@0.5:0.95, precision, recall, loss values
- **Model Artifacts**: Best weights automatically saved

## 🎯 Quick Start - Run Your First Experiment

```bash
# Start with YOLOv8n baseline
python scripts/experiments/comprehensive_experiment_runner.py --config experiments/configs/comprehensive_ablation/yolov8n_standard_none_config.yaml
```

## ⚡ Batch Execution (Optional)

To run multiple experiments in sequence, you can use the provided batch script:

```bash
cd experiments/configs/comprehensive_ablation
./run_all_experiments.sh
```

All experiments are ready to run with optimal configurations! 🚀