# PCB Defect Detection Experiment Guide

## 🚀 New Simplified Experiment System

This guide explains the **new simplified experiment workflow** that runs **Training → Validation → Testing** in a single execution, providing complete and unbiased performance metrics.

## 🎯 Key Features

- ✅ **Complete Experiment in One Run**: Training, validation, and final test evaluation
- ✅ **Automatic Test Set Evaluation**: Unbiased performance assessment on held-out test data
- ✅ **WandB Integration**: Comprehensive logging of all phases
- ✅ **Individual Experiment Control**: Run experiments one at a time for better tracking
- ✅ **Comprehensive Results**: JSON summaries, human-readable reports, and detailed logs
- ✅ **No Phase Management**: Simplified workflow without complex phase dependencies

## 📁 File Structure

```
PCB_defect/
├── run_single_experiment.py           # 🆕 New single experiment runner
├── example_run_experiments.py         # 🆕 Example batch runner
├── experiments/
│   ├── configs/                       # Experiment configurations
│   │   ├── 01_yolov8n_baseline_standard.yaml
│   │   ├── 02_yolov8s_baseline_standard.yaml
│   │   └── ...
│   └── results/                       # Results directory
│       └── [experiment_name]/         # Individual experiment results
│           ├── experiment_summary.json
│           ├── experiment_report.txt
│           └── weights/
├── archived_run_experiment.py         # 📦 Old experiment runner (archived)
└── archived_run_systematic_study.py  # 📦 Old phase-based runner (archived)
```

## 🔧 How to Run Experiments

### Method 1: Single Experiment (Recommended)

```bash
# Run complete experiment (training + validation + testing)
python run_single_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml

# Run only test evaluation (if model already trained)
python run_single_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml --test_only
```

### Method 2: Multiple Experiments

```bash
# Use the example batch runner
python example_run_experiments.py

# Or run them manually one by one:
python run_single_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml
python run_single_experiment.py --config experiments/configs/02_yolov8s_baseline_standard.yaml
python run_single_experiment.py --config experiments/configs/04_yolov8n_eca_standard.yaml
```

## 📊 What Happens in Each Experiment

### 1. 🏋️ Training Phase
- Model trains on training set
- Validates periodically on validation set during training
- Saves best model based on validation performance
- Logs training metrics to WandB

### 2. 🔍 Validation Phase
- Evaluates trained model on validation set
- Generates validation metrics and visualizations
- Logs validation results to WandB

### 3. 🧪 **TEST Phase (NEW!)**
- **CRUCIAL**: Evaluates model on completely held-out test set
- Provides **unbiased performance assessment**
- These are the metrics you should report for model comparison
- Saves test predictions for detailed analysis
- Logs final test metrics to WandB

## 📈 Results and Outputs

Each experiment generates:

### 1. **Trained Model**
- `experiments/results/[experiment_name]/weights/best.pt`

### 2. **Comprehensive Summary**
- `experiment_summary.json`: Machine-readable results
- `experiment_report.txt`: Human-readable performance report

### 3. **WandB Logging**
- Training metrics: Loss curves, learning rates
- Validation metrics: mAP, precision, recall, F1, confusion matrix
- **Test metrics**: Final unbiased performance assessment
- Model artifacts and visualizations

### 4. **Detailed Logs**
- `experiment_[name]_[timestamp].log`: Complete execution log

## 🔧 Configuration Files

Each experiment is defined by a YAML configuration file:

```yaml
experiment:
  name: "01_yolov8n_baseline_standard_640px"
  type: "baseline_training"
  description: "YOLOv8n baseline with standard loss"
  tags: ["baseline", "yolov8n", "standard_loss"]

model:
  type: "yolov8n"
  attention_mechanism: "none"

data:
  path: "experiments/configs/datasets/hripcb_data.yaml"
  num_classes: 6

training:
  epochs: 200
  batch: 128
  imgsz: 640
  # ... training parameters

validation:
  batch: 32
  imgsz: 640
  conf_threshold: 0.001
  # ... validation parameters

testing:  # 🆕 NEW SECTION
  batch: 16
  imgsz: 640
  conf_threshold: 0.001
  split: "test"
  save_predictions: true
  # ... test parameters

wandb:
  project: "pcb-defect-systematic-study"
```

## 📊 Key Metrics to Track

### Training Metrics
- Loss curves (box, classification, DFL)
- Learning rate schedules
- Training time per epoch

### Validation Metrics (Training-time)
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall, F1 score
- Per-class performance
- Confusion matrix

### **Test Metrics (Final Assessment)** ⭐
- **mAP@0.5**: Primary detection performance metric
- **mAP@0.5:0.95**: Comprehensive accuracy across IoU thresholds  
- **Precision**: False positive rate assessment
- **Recall**: False negative rate assessment
- **F1 Score**: Balanced precision-recall metric
- **Per-class mAP**: Individual defect type performance

## 🎯 Best Practices

### 1. **One Experiment at a Time**
- Run experiments individually for better tracking
- Easier to debug and analyze results
- Clear separation of results

### 2. **Always Check Test Results**
- **Test metrics are your final performance assessment**
- Use test results for model comparison and selection
- Validation metrics are for training-time decisions only

### 3. **Experiment Naming**
- Use descriptive names: `yolov8n_eca_1024px_focal_loss`
- Include key parameters in the name
- Follow consistent naming convention

### 4. **WandB Organization**
- Tag experiments appropriately
- Use consistent project names
- Add detailed descriptions and notes

### 5. **Configuration Management**
- Keep one config file per experiment
- Document configuration changes
- Version control your config files

## 🔍 Analyzing Results

### 1. **Individual Experiment Analysis**
```bash
# Check experiment summary
cat experiments/results/01_yolov8n_baseline_standard_640px/experiment_report.txt

# View detailed metrics
python -m json.tool experiments/results/01_yolov8n_baseline_standard_640px/experiment_summary.json
```

### 2. **Cross-Experiment Comparison**
- Use the enhanced WandB analysis notebook
- Compare test metrics across experiments
- Generate Pareto plots for trade-off analysis

### 3. **Model Selection**
- **Primary criterion**: Test set mAP@0.5
- **Secondary criteria**: Test set precision/recall balance
- **Efficiency criteria**: FPS, model size, GFLOPs

## ⚠️ Important Notes

### ❗ **Test Set Usage**
- The test set is **only used for final evaluation**
- Never use test metrics for hyperparameter tuning
- Test results represent true generalization performance

### ❗ **Model Comparison**
- Always compare models using **test set metrics**
- Validation metrics can be biased due to model selection
- Report test metrics in papers/presentations

### ❗ **Reproducibility**
- Set random seeds in configurations
- Document environment setup
- Save complete configuration with results

## 🆘 Troubleshooting

### Common Issues

1. **Config File Not Found**
   ```bash
   # Make sure path is correct
   ls experiments/configs/01_yolov8n_baseline_standard.yaml
   ```

2. **CUDA Out of Memory**
   - Reduce batch size in config file
   - Use smaller image size (imgsz)
   - Enable gradient accumulation

3. **WandB Authentication**
   - Run `wandb login` first
   - Set WANDB_API_KEY environment variable
   - Check internet connection

4. **Dataset Path Issues**
   - Verify dataset YAML path is correct
   - Check that train/val/test splits exist
   - Ensure proper permissions

### Getting Help

1. Check the experiment logs for detailed error messages
2. Verify dataset setup and paths
3. Test with smaller configurations first
4. Check WandB dashboard for logged information

## 🎉 Next Steps

1. **Run your first experiment**:
   ```bash
   python run_single_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml
   ```

2. **Monitor progress** in WandB dashboard

3. **Analyze results** using the comprehensive analysis notebook

4. **Compare multiple experiments** to find the best model

5. **Deploy the best model** based on test set performance

---

**🎯 Remember**: The goal is to get **unbiased test set performance** for each model variant. This new system ensures you have complete, reliable metrics for model comparison and publication.