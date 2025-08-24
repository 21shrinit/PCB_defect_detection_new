# Ablation Study Plan for PCB Defect Detection

## Overview
Comprehensive ablation study to evaluate the effectiveness of different YOLO architectures, loss functions, and attention mechanisms on PCB defect detection using the Roboflow PCB dataset.

## Experimental Design

### Baseline Models
✅ **Ready to Use:**
- **RB00**: YOLOv8n Baseline (existing)
- **RB09**: YOLOv10n Baseline (new)
- **RB10**: YOLOv11n Baseline (new)

### Variables to Study

#### 1. Model Architectures
- **YOLOv8n**: Established architecture
- **YOLOv10n**: Latest architecture with dual assignments 
- **YOLOv11n**: Most recent architecture improvements

#### 2. Loss Functions (4 variants)
- **SIoU**: Shape-aware IoU loss
- **EIoU**: Efficient IoU loss  
- **Focal**: Focal classification loss (addresses class imbalance)
- **VariFocal**: Variable focal loss (quality-aware)

#### 3. Attention Mechanisms (3 variants)
- **Coordinate Attention (CA)**: Position-aware attention
- **ECA-Net**: Efficient channel attention
- **CBAM**: Convolutional block attention module

## Experiment Matrix

### Total Experiments: 48
```
3 Architectures × (1 Baseline + 4 Loss Functions + 3 Attention + 4×3 Loss+Attention)
= 3 × (1 + 4 + 3 + 12) = 3 × 20 = 60 experiments
```

### Experiment Categories per Architecture:

#### Category 1: Baselines (3 experiments)
- Architecture only, no modifications
- Files: RB00, RB09, RB10

#### Category 2: Loss Function Study (12 experiments) 
- Each architecture with each loss function
- **YOLOv8n**: RB01 (SIoU), RB04 (EIoU), RB07 (Focal), RB08 (VariFocal) 
- **YOLOv10n**: RB11-RB14 (to be created)
- **YOLOv11n**: RB15-RB18 (to be created)

#### Category 3: Attention Mechanism Study (9 experiments)
- Each architecture with each attention mechanism
- **YOLOv8n**: RB02 (ECA), RB03 (CBAM), RB?? (CA) - need to create CA
- **YOLOv10n**: RB19-RB21 (to be created) 
- **YOLOv11n**: RB22-RB24 (to be created)

#### Category 4: Combined Study (36 experiments)
- Each architecture with each loss+attention combination
- **YOLOv8n**: 12 combinations (some exist, need to complete)
- **YOLOv10n**: 12 combinations (all to be created)
- **YOLOv11n**: 12 combinations (all to be created)

## Configuration Naming Convention

### Pattern: `RB{XX}_{Architecture}_{Components}.yaml`

**Examples:**
- `RB09_YOLOv10n_Baseline.yaml` - YOLOv10n baseline
- `RB11_YOLOv10n_SIoU.yaml` - YOLOv10n with SIoU loss
- `RB19_YOLOv10n_ECA.yaml` - YOLOv10n with ECA attention
- `RB25_YOLOv10n_SIoU_ECA.yaml` - YOLOv10n with SIoU + ECA

## Evaluation Metrics

### Primary Metrics
- **mAP@0.5**: Primary detection metric
- **mAP@0.5-0.95**: Comprehensive detection metric  
- **Precision**: False positive rate
- **Recall**: False negative rate
- **F1 Score**: Now logged automatically! 

### Secondary Metrics
- **Training Speed**: FPS during training
- **Inference Speed**: FPS during validation
- **Model Size**: Parameter count
- **Memory Usage**: GPU memory consumption

## Expected Outcomes

### Hypothesis
1. **Architecture**: YOLOv11n > YOLOv10n > YOLOv8n (newer = better)
2. **Loss Functions**: Focal/VariFocal > SIoU/EIoU (handle class imbalance)
3. **Attention**: ECA > CBAM > CA (efficiency vs complexity trade-off)
4. **Combined**: Best loss + best attention should give optimal results

### Key Questions
1. Which architecture performs best on small PCB defects?
2. Do advanced loss functions help with class imbalance in PCB dataset?
3. Which attention mechanism is most effective for defect detection?
4. Is there synergy between loss functions and attention mechanisms?
5. What is the speed vs accuracy trade-off for each combination?

## Implementation Status

### ✅ Completed
- [x] Baseline configurations for all 3 architectures
- [x] Model loading verification
- [x] F1 score logging integration
- [x] Existing YOLOv8n variants (RB00-RB08)

### 🚧 To Create
- [ ] YOLOv8n + Coordinate Attention
- [ ] YOLOv10n loss function variants (4 configs)
- [ ] YOLOv10n attention variants (3 configs) 
- [ ] YOLOv10n combined variants (12 configs)
- [ ] YOLOv11n loss function variants (4 configs)
- [ ] YOLOv11n attention variants (3 configs)
- [ ] YOLOv11n combined variants (12 configs)

### Total: ~40 configurations still to create

## Next Steps
1. **Create remaining configurations** for systematic study
2. **Batch experiment runner** for efficient execution
3. **Results analysis framework** for comprehensive comparison
4. **Statistical significance testing** between approaches