# PCB Defect Detection - Organized Project Structure

## 📁 **Clean, Organized Codebase Structure**

The project has been reorganized for better maintainability, clear separation of concerns, and easier navigation.

## 🗂️ **Directory Structure**

```
PCB_defect/
├── 📄 README.md                           # Main project documentation
├── 📄 EXPERIMENT_GUIDE.md                 # How to run experiments
├── 📄 ORGANIZED_STRUCTURE.md             # This file - structure guide
├── 📄 requirements.txt                    # Python dependencies
├── 📄 pyproject.toml                      # Project configuration
├── ⚙️  yolov8*.pt                        # Pre-trained model weights
│
├── 📁 scripts/                           # 🆕 ORGANIZED SCRIPTS
│   ├── 📁 experiments/                   # Main experiment runners
│   │   ├── 🔬 run_single_experiment.py   # Primary experiment runner
│   │   └── 📊 example_run_experiments.py # Batch experiment example
│   ├── 📁 analysis/                      # Analysis and visualization
│   │   ├── 📈 PCB_Defect_Detection_Results_Analysis.ipynb
│   │   ├── 🔍 analyze_wandb_results.py   # WandB analysis script
│   │   └── 🧩 missing_functionality_additions.py
│   ├── 📁 setup/                         # Setup and configuration
│   │   ├── ⚙️  setup_wandb.py            # WandB setup
│   │   ├── 📥 download_deeppcb.py        # DeepPCB dataset downloader
│   │   ├── 📥 download_hripcb.py         # HRIPCB dataset downloader
│   │   └── ✅ verify_implementation.py   # Implementation verification
│   └── 📁 archived/                      # Archived/legacy scripts
│       ├── 🗂️ archived_run_experiment.py
│       ├── 🗂️ archived_run_systematic_study.py
│       ├── 🗂️ benchmark_all_attention.py
│       ├── 🗂️ custom_trainer.py
│       └── 🗂️ train_*.py                # Old training scripts
│
├── 📁 experiments/                       # Experiment configurations
│   ├── 📁 configs/                       # YAML configuration files
│   │   ├── 01_yolov8n_baseline_standard.yaml
│   │   ├── 02_yolov8s_baseline_standard.yaml
│   │   ├── ... (other experiment configs)
│   │   ├── 📁 datasets/                  # Dataset configurations
│   │   └── 📁 templates/                 # Configuration templates
│   └── 📁 results/                       # Experiment results (auto-created)
│
├── 📁 datasets/                          # Dataset storage
│   ├── 📁 HRIPCB/                        # Primary dataset for training
│   │   └── 📁 HRIPCB_UPDATE/
│   └── 📁 XD-PCB/                        # Domain adaptation dataset
│       ├── 📁 XD-Real/                   # Real PCB images
│       └── 📁 XD-Syn/                    # Synthetic PCB images
│
├── 📁 custom_modules/                    # Custom implementations
│   ├── attention.py                      # Attention mechanisms (CBAM, ECA, etc.)
│   ├── loss.py                          # Custom loss functions
│   └── mobilevit.py                     # MobileViT implementation
│
├── 📁 docs/                             # Documentation
│   └── 📁 guides/                       # Detailed guides and reports
│       ├── 📋 ATTENTION_MECHANISMS_DOCUMENTATION.md
│       ├── 📋 TRAINING_GUIDE.md
│       └── ... (other documentation files)
│
├── 📁 ultralytics/                      # Modified Ultralytics framework
│   └── ... (core YOLO implementation with custom modifications)
│
└── 📁 examples/                          # Usage examples and demos
    └── ... (various example implementations)
```

## 🚀 **How to Use the New Structure**

### **1. Running Experiments**
```bash
# Main experiment runner (recommended)
python scripts/experiments/run_single_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml

# Batch experiments
python scripts/experiments/example_run_experiments.py
```

### **2. Analysis and Visualization**
```bash
# Open the comprehensive analysis notebook
jupyter notebook scripts/analysis/PCB_Defect_Detection_Results_Analysis.ipynb

# Run WandB analysis
python scripts/analysis/analyze_wandb_results.py
```

### **3. Setup and Configuration**
```bash
# Setup WandB
python scripts/setup/setup_wandb.py

# Download datasets
python scripts/setup/download_hripcb.py
python scripts/setup/download_deeppcb.py

# Verify implementation
python scripts/setup/verify_implementation.py
```

## 🎯 **Key Benefits of New Structure**

### ✅ **Better Organization**
- **Clear separation** of experiments, analysis, setup, and archived code
- **Easy navigation** - find what you need quickly
- **Logical grouping** of related functionality

### ✅ **Improved Maintainability**
- **Single source of truth** for each type of functionality
- **Easier updates** and modifications
- **Clear dependencies** between components

### ✅ **Enhanced Usability**
- **Simple commands** to run experiments
- **Clear documentation** for each component
- **Easy onboarding** for new users

### ✅ **Future-Proof Structure**
- **Scalable organization** for additional experiments
- **Easy integration** of new features
- **Modular design** for component reuse

## 📊 **What's Changed**

### **🆕 New Main Entry Points**
- `scripts/experiments/run_single_experiment.py` - **Primary experiment runner**
- `scripts/analysis/PCB_Defect_Detection_Results_Analysis.ipynb` - **Main analysis notebook**
- `EXPERIMENT_GUIDE.md` - **Comprehensive usage guide**

### **🗂️ Organized Scripts**
- **Experiments**: Core experiment runners
- **Analysis**: WandB analysis and visualization tools
- **Setup**: Configuration and dataset setup scripts
- **Archived**: Legacy code for reference

### **🔧 Simplified Workflow**
- **No complex phase management** - run individual experiments
- **Automatic test evaluation** included in each experiment
- **Clear result organization** in `experiments/results/`

## 🎯 **Essential Files to Focus On**

### **For Running Experiments:**
1. `scripts/experiments/run_single_experiment.py` - Main experiment runner
2. `experiments/configs/*.yaml` - Experiment configurations
3. `EXPERIMENT_GUIDE.md` - How-to guide

### **For Analysis:**
1. `scripts/analysis/PCB_Defect_Detection_Results_Analysis.ipynb` - Comprehensive analysis
2. `scripts/analysis/analyze_wandb_results.py` - Automated analysis

### **For Setup:**
1. `scripts/setup/setup_wandb.py` - WandB configuration
2. `scripts/setup/download_*.py` - Dataset downloaders

## 🚮 **What Was Removed/Archived**

### **Archived (Not Deleted)**
- Old training scripts (`train_*.py`) - moved to `scripts/archived/`
- Legacy experiment runners - moved to `scripts/archived/`
- Benchmark scripts - moved to `scripts/archived/`

### **Removed**
- Duplicate configuration files
- Temporary log files
- Old config directory (duplicated functionality)
- DeepPCB dataset files (using HRIPCB as primary)

## 🎉 **Ready to Use!**

Your codebase is now **clean, organized, and ready for productive research**:

```bash
# Quick start - run your first organized experiment:
python scripts/experiments/run_single_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml
```

**All functionality is preserved** - just better organized and easier to use! 🚀