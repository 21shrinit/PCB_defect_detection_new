# 🔬 Comprehensive PCB Defect Detection Experiment Overview

## 📊 **Experiment Portfolio Summary**

We are conducting a systematic study of YOLO-based object detection for PCB defect detection with **6 defect types**: Missing_hole, Mouse_bite, Open_circuit, Short, Spurious_copper, Spur.

---

## 🎯 **Core Experiment Categories**

### **1. Baseline Model Comparison** 
**Objective**: Establish performance hierarchy across model sizes
- `01_yolov8n_baseline_standard.yaml` - **YOLOv8n** (3.2M params)
- `02_yolov8s_baseline_standard.yaml` - **YOLOv8s** (11M params) 
- `03_yolov10s_baseline_standard.yaml` - **YOLOv10s** (latest architecture)
- `yolov8n_pcb_defect_baseline.yaml` - **Large dataset** baseline (Kaggle PCB)

### **2. Attention Mechanism Study**
**Objective**: Evaluate attention's impact on tiny defect detection (2-16 pixels)
- `04_yolov8n_eca_standard.yaml` - **ECA-Net** (5 params, ultra-efficient)
- `05_yolov8n_cbam_standard.yaml` - **CBAM** (1K-10K params, balanced)
- `06_yolov8n_coordatt_standard.yaml` - **CoordAtt** (8-16K params, position-aware)

### **3. Loss Function Optimization**
**Objective**: Improve localization and classification for small objects
- `07_yolov8n_baseline_focal_siou.yaml` - **Focal + SIoU** (hard examples + shape-aware)
- `02_yolov8n_siou_baseline_standard.yaml` - **SIoU** (shape-aware IoU)
- `03_yolov8n_eiou_baseline_standard.yaml` - **EIoU** (efficient IoU)
- `08_yolov8n_verifocal_eiou.yaml` - **VeriFocal + EIoU**
- `09_yolov8n_verifocal_siou.yaml` - **VeriFocal + SIoU**

### **4. High-Resolution Analysis**
**Objective**: Maximize detection of tiny defects through increased input resolution
- `10_yolov8n_baseline_1024px.yaml` - **YOLOv8n @ 1024px**
- `11_yolov8s_baseline_1024px.yaml` - **YOLOv8s @ 1024px** (maximum capacity)

### **5. Specialized Configurations**
**Objective**: Environment-specific optimizations
- `colab_01_yolov8n_baseline_optimized.yaml` - **Colab-optimized** (gradient accumulation)

---

## ⚙️ **Critical Training Parameters & Their Impact**

### **🏗️ Model Architecture Parameters**
| Parameter | Impact | Experiment Dependency | Optimal Range |
|-----------|--------|---------------------|---------------|
| **Model Size** | Memory, accuracy, speed | All baseline experiments | nano < small < medium |
| **Input Resolution** | Small object detection | High-res experiments | 640px vs 1024px |
| **Attention Mechanism** | Feature focus, complexity | Attention experiments | ECA < CBAM < CoordAtt |

### **📈 Training Dynamics Parameters**
| Parameter | Impact | Critical For | Optimal Values |
|-----------|--------|--------------|----------------|
| **Epochs** | Convergence quality | All experiments | 150-400 (model dependent) |
| **Batch Size** | Gradient stability, BN stats | **CRITICAL for attention** | 16-128 (attention needs 64+) |
| **Learning Rate (lr0)** | Convergence speed/stability | Model size dependent | 0.0003-0.002 |
| **Weight Decay** | Regularization | All experiments | 0.0001-0.0005 |
| **Momentum** | Gradient smoothing | SGD/AdamW | 0.9-0.937 |

### **🎯 Optimization Strategy Parameters**
| Parameter | Impact | When to Use | Values |
|-----------|--------|-------------|--------|
| **Warmup Epochs** | Training stability | Complex models | 3-8 epochs |
| **Cosine Annealing** | Learning rate scheduling | Long training | True/False |
| **Gradient Accumulation** | Effective batch size | Memory constraints | 2-8x |
| **Mixed Precision (AMP)** | Memory efficiency | All experiments | True (always) |

### **🖼️ Data Augmentation Parameters**
| Parameter | Impact | PCB-Specific Consideration | Optimal Range |
|-----------|--------|---------------------------|---------------|
| **Mosaic** | Multi-object learning | Preserve small defects | 0.6-1.0 |
| **Mixup** | Regularization | Minimal for defects | 0.02-0.1 |
| **HSV Variations** | Color robustness | PCB consistency | Minimal |
| **Geometric Transforms** | Spatial robustness | Manufacturing variance | Conservative |

### **🔍 Loss Function Parameters**
| Parameter | Impact | Best For | Usage |
|-----------|--------|----------|-------|
| **Box Weight** | Localization emphasis | Small objects | 7.5-8.0 |
| **Class Weight** | Classification emphasis | Hard examples | 0.3-0.8 |
| **Focal Gamma** | Hard example focus | Class imbalance | 2.0 |
| **IoU Variants** | Shape-aware loss | Irregular defects | SIoU, EIoU |

---

## 🧪 **Experiment Dependencies & Interactions**

### **Attention Mechanism Sensitivities**
- **Batch Size**: Extremely sensitive (64+ required)
- **Learning Rate**: Prefers 0.001 (not lower)
- **Training Duration**: Needs 200+ epochs
- **Memory**: Higher overhead than baseline

### **High-Resolution Dependencies**
- **Batch Size**: Severely constrained (4-16 max)
- **Memory Management**: Cannot cache, needs AMP
- **Training Time**: 400+ epochs for convergence
- **Attention Compatibility**: Challenging combination

### **Loss Function Interactions**
- **SIoU**: Works well with higher LR (0.002)
- **Focal Loss**: Benefits from increased class weight
- **VeriFocal**: Specific to classification confidence

---

## 🎯 **Research Priority Matrix**

### **High Impact, Low Risk** (Priority 1)
1. **Baseline Model Comparison** - Establish hierarchy
2. **Simple Loss Functions** - SIoU, EIoU variants
3. **Standard Resolution Optimization** - 640px focus

### **High Impact, Medium Risk** (Priority 2)
1. **Attention Mechanisms** - ECA → CBAM → CoordAtt progression
2. **Learning Rate Scaling** - Model-specific optimization
3. **Advanced Loss Functions** - Focal + IoU combinations

### **High Impact, High Risk** (Priority 3)
1. **High-Resolution Training** - 1024px experiments
2. **Complex Attention + High-Res** - Resource intensive
3. **Multi-stage Training** - Curriculum learning

---

## 📊 **Current Performance Baseline**
- **YOLOv8n Baseline**: mAP50 ~95.7%, F1 ~94.5%
- **Target Improvement**: >96% mAP50, >95% F1
- **Acceptable Trade-offs**: 5% performance for 2x speed

---

## ⚠️ **Known Issues & Constraints**
1. **Attention + Small Batches**: Severe performance degradation
2. **High-Res + Large Models**: Memory limitations
3. **Complex Optimizations**: May hurt more than help
4. **Dataset Dependency**: Results may vary between HRIPCB and Kaggle PCB

---

## 🔄 **Experiment Workflow**
1. **Phase 1**: Baseline establishment (01-03)
2. **Phase 2**: Attention evaluation (04-06) 
3. **Phase 3**: Advanced techniques (07-11)
4. **Phase 4**: Hybrid optimization (best combinations)

This overview provides the foundation for systematic hyperparameter research and optimization strategies.