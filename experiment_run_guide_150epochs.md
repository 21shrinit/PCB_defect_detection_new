# 🚀 Experiment Run Guide - 150 Epochs Clean Setup

## 📋 **Quick Start Checklist**

### ✅ **Prerequisites**
- [x] WandB project name updated to: `pcb-defect-150epochs-v1`
- [x] All experiment configs updated with 150 epochs
- [x] Analysis notebook updated for new project
- [x] Clean separation from previous experiments

### 🎯 **Ready-to-Run Experiments**

All experiments are now configured for **150 epochs** and will log to the **clean new WandB project**: `pcb-defect-150epochs-v1`

## 🏃‍♂️ **Running Individual Experiments**

### **Baseline Experiments**
```bash
# HRIPCB Dataset
python run_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml
python run_experiment.py --config experiments/configs/02_yolov8s_baseline_standard.yaml
python run_experiment.py --config experiments/configs/03_yolov10s_baseline_standard.yaml

# Kaggle PCB Defect Dataset
python run_experiment.py --config experiments/configs/yolov8n_pcb_defect_baseline.yaml
```

### **Attention Mechanism Studies**
```bash
python run_experiment.py --config experiments/configs/04_yolov8n_eca_standard.yaml
python run_experiment.py --config experiments/configs/05_yolov8n_cbam_standard.yaml
python run_experiment.py --config experiments/configs/06_yolov8n_coordatt_standard.yaml
```

### **Loss Function Studies**
```bash
python run_experiment.py --config experiments/configs/02_yolov8n_siou_baseline_standard.yaml
python run_experiment.py --config experiments/configs/03_yolov8n_eiou_baseline_standard.yaml
python run_experiment.py --config experiments/configs/07_yolov8n_baseline_focal_siou.yaml
python run_experiment.py --config experiments/configs/08_yolov8n_verifocal_eiou.yaml
python run_experiment.py --config experiments/configs/09_yolov8n_verifocal_siou.yaml
```

### **Resolution Studies**
```bash
python run_experiment.py --config experiments/configs/10_yolov8n_baseline_1024px.yaml
python run_experiment.py --config experiments/configs/11_yolov8s_baseline_1024px.yaml
```

## 📊 **Monitoring & Analysis**

### **Real-time Monitoring**
1. **WandB Dashboard**: Monitor training progress in real-time
   - Project: `pcb-defect-150epochs-v1`
   - Track: Loss curves, mAP, precision, recall

2. **Local Logs**: Check experiment logs in `experiments/results/`

### **Post-Experiment Analysis**

#### **Individual Experiment Results**
```bash
# Check experiment summary
cat experiments/results/[experiment_name]/experiment_report.txt

# View detailed metrics
python -m json.tool experiments/results/[experiment_name]/experiment_summary.json
```

#### **Comprehensive Analysis**
```bash
# Analyze all experiments in the project
python scripts/analysis/analyze_wandb_results.py --project pcb-defect-150epochs-v1

# With CSV export
python scripts/analysis/analyze_wandb_results.py --project pcb-defect-150epochs-v1 --export-csv

# Detailed statistical analysis
python scripts/analysis/analyze_wandb_results.py --project pcb-defect-150epochs-v1 --detailed-analysis
```

#### **Jupyter Notebook Analysis**
```bash
jupyter notebook scripts/analysis/PCB_Defect_Detection_Results_Analysis.ipynb
```

## 🎛️ **Experiment Configuration Overview**

### **Common Settings (150 Epochs)**
- **Epochs**: 150 (reduced from 200 for faster training)
- **WandB Project**: `pcb-defect-150epochs-v1` (clean new project)
- **Test Evaluation**: Enabled for unbiased performance assessment
- **Comprehensive Logging**: Training + Validation + Test metrics

### **Dataset Configurations**
- **HRIPCB**: 1,386 images (1,109 train, 138 val, 139 test)
- **Kaggle PCB**: 10,668 images (8,534 train, 1,066 val, 1,068 test)

### **Model Variants**
- **YOLOv8n**: Lightweight, ~3M parameters
- **YOLOv8s**: Balanced, ~11M parameters  
- **YOLOv10s**: Latest architecture

### **Attention Mechanisms**
- **ECA**: Ultra-efficient channel attention (~5 parameters)
- **CBAM**: Dual channel + spatial attention
- **CoordAtt**: Position-aware attention

## 🔄 **Running Multiple Experiments**

### **Sequential Execution (Recommended)**
```bash
# Run experiments one by one for better tracking
python run_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml
# Wait for completion, check results
python run_experiment.py --config experiments/configs/04_yolov8n_eca_standard.yaml
# Continue...
```

### **Batch Execution (Advanced)**
```bash
# Use the example batch runner
python scripts/experiments/example_run_experiments.py
```

## 📈 **Expected Experiment Timeline**

### **Per Experiment (150 epochs)**
- **YOLOv8n (640px)**: ~2-3 hours on modern GPU
- **YOLOv8s (640px)**: ~3-4 hours on modern GPU
- **YOLOv8n (1024px)**: ~4-5 hours on modern GPU
- **YOLOv8s (1024px)**: ~6-8 hours on modern GPU

### **Total Study (14 experiments)**
- **Estimated**: 40-60 hours total
- **Recommend**: Run 2-3 experiments per day
- **Complete study**: 5-7 days

## 🎯 **Success Criteria**

### **Training Success Indicators**
- ✅ Loss curves converging
- ✅ mAP@0.5 > 0.60 on validation
- ✅ No overfitting signs
- ✅ Test metrics logged to WandB

### **Experiment Quality Checks**
- ✅ All metrics saved to WandB
- ✅ Model weights saved
- ✅ Test set evaluation completed
- ✅ Experiment summary generated

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config file
   # For 1024px experiments, use batch: 16 → 8
   ```

2. **WandB Authentication**
   ```bash
   wandb login
   # Or set environment variable
   export WANDB_API_KEY=your_key
   ```

3. **Dataset Path Issues**
   ```bash
   # Verify dataset exists
   ls datasets/HRIPCB/HRIPCB_UPDATE/
   ls datasets/PCB_Defect/pcb-defect-dataset/
   ```

### **Performance Optimization**
- **Reduce workers** if system becomes unresponsive
- **Disable cache** for large datasets if memory is limited
- **Use smaller batch sizes** for 1024px experiments

## 🎉 **Next Steps After Completion**

1. **Analyze Results**: Use the comprehensive analysis notebook
2. **Compare Performance**: Identify best-performing configurations  
3. **Select Best Model**: Based on test set metrics
4. **Deploy**: Use best model for production/edge deployment
5. **Domain Adaptation**: Experiment with XD-PCB dataset

---

## 🔗 **Quick Reference**

- **WandB Project**: `pcb-defect-150epochs-v1`
- **All epochs**: 150
- **Main dataset**: HRIPCB (experiments/configs/datasets/hripcb_data.yaml)
- **Kaggle dataset**: PCB_Defect (experiments/configs/datasets/pcb_defect_data.yaml)
- **Analysis**: `scripts/analysis/PCB_Defect_Detection_Results_Analysis.ipynb`

**Happy experimenting! 🚀**