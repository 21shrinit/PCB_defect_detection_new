# 🔬 Systematic PCB Defect Detection Experiment Framework

## 📋 **Overview**

This framework provides a comprehensive experimental setup for evaluating YOLOv8 variants, attention mechanisms, and loss function combinations for PCB defect detection on edge devices.

## 🎯 **Experiment Matrix**

### **Phase 1: Baseline Models**
| Experiment | Model | Attention | Loss | Resolution | Purpose |
|------------|-------|-----------|------|------------|---------|
| 01 | YOLOv8n | None | Standard | 640px | Baseline reference |
| 02 | YOLOv8s | None | Standard | 640px | Model capacity study |
| 03 | YOLOv10s | None | Standard | 640px | Architecture comparison |

### **Phase 2: Attention Mechanisms**
| Experiment | Model | Attention | Loss | Resolution | Purpose |
|------------|-------|-----------|------|------------|---------|
| 04 | YOLOv8n | ECA | Standard | 640px | Ultra-efficient attention |
| 05 | YOLOv8n | CBAM | Standard | 640px | Dual attention |
| 06 | YOLOv8n | CoordAtt | Standard | 640px | Position-aware attention |

### **Phase 3: Loss Functions & Resolution Study**
| Experiment | Model | Loss Combination | Resolution | Purpose |
|------------|-------|------------------|------------|---------|
| 07 | YOLOv8n | Focal + SIoU | 640px | Hard examples + shape-aware |
| 08 | YOLOv8n | VeriFocal + EIoU | 640px | Quality-aware + enhanced IoU |
| 09 | YOLOv8n | VeriFocal + SIoU | 640px | Quality + shape optimization |
| 10 | YOLOv8n | Standard | 1024px | High-resolution study |
| 11 | YOLOv8s | Standard | 1024px | Model scaling + high-resolution |

## 📊 **Tracked Metrics**

All experiments automatically track:
- **Performance**: Precision, Recall, F1, mAP@0.5, mAP@0.5-0.95
- **Efficiency**: FPS, GFLOPs, Parameters, Memory usage
- **Per-class**: Individual defect type performance
- **Training**: Loss curves, convergence speed, stability

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Ensure you're in the project root
cd F:\PCB_defect

# Install dependencies (if not already done)
pip install ultralytics wandb

# Login to Weights & Biases
wandb login
```

### **2. Run Single Experiment**
```bash
# Run baseline YOLOv8n
python run_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml

# Run YOLOv8n with CBAM attention
python run_experiment.py --config experiments/configs/05_yolov8n_cbam_standard.yaml

# Run best loss combination
python run_experiment.py --config experiments/configs/08_yolov8n_verifocal_eiou.yaml
```

### **3. Run Systematic Study**
```bash
# List all available experiments
python experiments/run_systematic_study.py --list_experiments

# Run Phase 1 (baseline models)
python experiments/run_systematic_study.py --phase 1

# Run Phase 2 (attention mechanisms)
python experiments/run_systematic_study.py --phase 2

# Run Phase 3 (loss functions + resolution)
python experiments/run_systematic_study.py --phase 3

# Run complete study (all experiments)
python experiments/run_systematic_study.py --run_all
```

## 📁 **File Structure**
```
experiments/
├── README.md                          # This file
├── run_systematic_study.py           # Orchestrates all experiments
├── configs/                          # Experiment configurations
│   ├── templates/
│   │   └── base_config_template.yaml # Template for new experiments
│   ├── 01_yolov8n_baseline_standard.yaml
│   ├── 02_yolov8s_baseline_standard.yaml
│   ├── 03_yolov10s_baseline_standard.yaml
│   ├── 04_yolov8n_eca_standard.yaml
│   ├── 05_yolov8n_cbam_standard.yaml
│   ├── 06_yolov8n_coordatt_standard.yaml
│   ├── 07_yolov8n_baseline_focal_siou.yaml
│   ├── 08_yolov8n_verifocal_eiou.yaml
│   ├── 09_yolov8n_verifocal_siou.yaml
│   ├── 10_yolov8n_baseline_1024px.yaml
│   └── 11_yolov8s_baseline_1024px.yaml
├── results/                          # Training outputs (auto-created)
└── datasets/                         # Dataset configurations
    └── hripcb_data.yaml              # HRIPCB dataset config
```

## 🎯 **Loss Function Combinations Explained**

### **Standard (Baseline)**
- **Components**: DFL + BCE + CIoU
- **Use Case**: Baseline comparison

### **Focal + SIoU** ⭐ *Most Promising*
- **Components**: DFL + Focal + SIoU  
- **Benefits**: Hard example mining + shape-aware IoU
- **Best For**: Challenging defect cases and varied shapes

### **VeriFocal + EIoU** ⭐ *Most Promising*
- **Components**: DFL + VeriFocal + EIoU
- **Benefits**: Quality prediction + enhanced IoU optimization
- **Best For**: Precise localization and confidence calibration

### **VeriFocal + SIoU** ⭐ *Most Promising*
- **Components**: DFL + VeriFocal + SIoU
- **Benefits**: Quality awareness + shape optimization
- **Best For**: Balanced quality and shape understanding

## 📊 **Monitoring & Results**

### **Weights & Biases Integration**
All experiments automatically log to W&B:
- **Project**: `pcb-defect-systematic-study`
- **Real-time Metrics**: Training/validation curves
- **Model Artifacts**: Best checkpoints saved automatically
- **Experiment Comparison**: Side-by-side performance comparison

### **Local Results**
Training outputs saved to:
- **Checkpoints**: `experiments/results/{experiment_name}/weights/`
- **Logs**: `experiments/results/{experiment_name}/`
- **Plots**: `experiments/results/{experiment_name}/`

## 🔧 **Advanced Usage**

### **Custom Experiment Configuration**
1. Copy `templates/base_config_template.yaml`
2. Modify parameters as needed
3. Run with `python run_experiment.py --config your_config.yaml`

### **Domain Adaptation Study**
After completing the main study, use the best performing model for domain adaptation:
```bash
# Use best model for XD PCB dataset evaluation
python run_experiment.py --config experiments/configs/domain_adaptation/best_model_xd_pcb.yaml
```

### **Export Best Model**
```bash
# Export best model to ONNX for deployment
python run_experiment.py --config experiments/configs/export/best_model_onnx_export.yaml
```

## 📈 **Expected Timeline**

### **Phase 1 (1-2 days)**
- 3 experiments × 8-12 hours each
- Baseline model evaluation

### **Phase 2 (1-2 days)**
- 3 experiments × 8-12 hours each
- Attention mechanism evaluation

### **Phase 3 (2-3 days)**  
- 5 experiments × 8-12 hours each
- Loss function + resolution optimization

### **Total Study Duration: ~1 week**

## 🎯 **Success Criteria**

### **Breakthrough Indicators**
- **>5% mAP@0.5-0.95** improvement over baseline
- **>95% recall** for critical defects (Missing_hole, Open_circuit, Short)
- **>10 FPS** on Jetson Nano edge device
- **<2GB RAM** usage for inference pipeline

### **Expected Results**
- **Best Attention**: CBAM or CoordAtt (+3-5% mAP expected)
- **Best Loss**: VeriFocal + EIoU (+5-8% mAP expected)
- **Best Resolution**: 1024px for tiny defects (+10-15% small object AP)

## 🚨 **Troubleshooting**

### **Common Issues**
1. **CUDA OOM**: Reduce batch size in config
2. **W&B Login**: Run `wandb login` with your API key
3. **Config Not Found**: Check file paths and current directory
4. **Model Not Found**: Ensure attention mechanism configs exist

### **Performance Monitoring**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor system resources
htop

# Check experiment logs
tail -f systematic_study_*.log
```

## 📝 **Notes**

- All experiments use **seed=42** for reproducibility
- **Automatic Mixed Precision (AMP)** enabled for all experiments
- **Image caching enabled** for faster data loading
- **No early stopping** - full 200 epochs for complete training
- **Validation** runs every epoch with comprehensive metrics
- **Checkpoints** saved every 25 epochs + best model
- **Optimized settings**: 16 workers, GPU caching, validation batch=32, performance flags enabled

## 🔄 **Next Steps After Study**

1. **Analyze Results**: Compare all experiments in W&B
2. **Select Best Model**: Based on accuracy/efficiency trade-off
3. **Domain Adaptation**: Test on XD PCB dataset
4. **Production Deployment**: Export to ONNX/TensorRT
5. **Edge Optimization**: Further quantization and pruning