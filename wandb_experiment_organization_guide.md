# WandB Experiment Organization Guide

## 🎯 Project Organization Strategy

### **Option 1: Date-Based Projects (Recommended)**
```
pcb-defect-2025-01-21-v1    # Today's experiments with 150 epochs
pcb-defect-2025-01-21-v2    # If you need to re-run with different configs
```

### **Option 2: Version-Based Projects**
```
pcb-defect-hripcb-v2        # HRIPCB dataset, version 2 (150 epochs)
pcb-defect-kaggle-v1        # Kaggle PCB defect dataset, version 1
```

### **Option 3: Dataset + Config Based**
```
pcb-defect-hripcb-150epochs     # HRIPCB with 150 epochs
pcb-defect-kaggle-150epochs     # Kaggle dataset with 150 epochs
```

## 🚀 **Recommended Approach: Date + Version**

Use this naming pattern: `pcb-defect-YYYY-MM-DD-vN`

**Benefits:**
- Clear chronological order
- Easy to track experiment iterations
- Clean separation of experiment batches
- Simple analysis notebook updates

## 📋 **Implementation Steps**

### 1. **Choose New Project Name**
```bash
# For today's re-run with 150 epochs:
NEW_PROJECT="pcb-defect-2025-01-21-v1"
```

### 2. **Update All Config Files**
Replace all `wandb.project` values with the new project name

### 3. **Update Analysis Scripts**
Modify analysis notebook to point to new project

### 4. **Run Experiments**
Execute experiments - they'll go to the new clean project

### 5. **Compare with Previous**
You can still access old results for comparison if needed

## 🔧 **Experiment Naming Best Practices**

### **Experiment Names Should Include:**
- Model type: `yolov8n`, `yolov8s`, `yolov10s`
- Key feature: `baseline`, `eca`, `cbam`, `coordatt`
- Loss type: `standard`, `focal`, `siou`, `eiou`
- Resolution: `640px`, `1024px`
- Dataset: `hripcb`, `kaggle-pcb`

### **Example Good Names:**
```
yolov8n_baseline_standard_640px_hripcb_150ep
yolov8n_eca_standard_640px_hripcb_150ep
yolov8s_baseline_standard_1024px_hripcb_150ep
```

## 📊 **Analysis Strategy**

### **Single Project Analysis**
```python
# In analysis notebook
PROJECT_NAME = "pcb-defect-2025-01-21-v1"
runs = api.runs(f"your-username/{PROJECT_NAME}")
```

### **Cross-Project Comparison**
```python
# Compare multiple project versions
OLD_PROJECT = "pcb-defect-systematic-study"
NEW_PROJECT = "pcb-defect-2025-01-21-v1"

old_runs = api.runs(f"your-username/{OLD_PROJECT}")
new_runs = api.runs(f"your-username/{NEW_PROJECT}")
```

## ⚠️ **Important Notes**

1. **Don't Delete Old Projects** - Keep them for historical comparison
2. **Use Consistent Tagging** - Add tags like `150epochs`, `hripcb`, `baseline`
3. **Document Changes** - Note what changed between project versions
4. **Backup Results** - Export key metrics before major changes

## 🎯 **Immediate Action Items**

1. Choose your new project name
2. Update all experiment configs
3. Update analysis notebook
4. Run a test experiment to verify setup
5. Execute full experiment batch

This approach ensures clean organization and easy analysis!