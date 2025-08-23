# Attention Mechanisms Pretrained Weights Loading Troubleshooting

## Problem Description

When training attention-enhanced YOLOv8 models (ECA, CBAM, CoordAtt), the models fail to load pretrained weights properly, resulting in:

- ✅ Baseline models: Successfully load pretrained weights and achieve expected performance
- ❌ Attention models: Fail to load pretrained weights, poor performance due to random initialization

## Symptoms

### Error Logs
```
⚠️ Could not load pretrained weights: BaseModel.load() got an unexpected keyword argument 'strict'
Continuing with random initialization
```

### Performance Impact
- Attention models perform **worse than baseline** instead of showing improvements
- Missing "Transferred X/Y items from pretrained weights" message during training
- Poor convergence and low accuracy metrics

### Config Comparison
**Working (Baseline):**
```yaml
model:
  type: "yolov8n"
  config_path: ""                       # Empty = direct yolov8n.pt loading
  pretrained: true
```

**Failing (Attention):**
```yaml
model:
  type: "yolov8n"
  config_path: "ultralytics/cfg/models/v8/yolov8n-eca-final.yaml"  # Custom architecture
  pretrained: true
```

## Root Cause Analysis

### Primary Issue: Class Count Mismatch
- **Pretrained yolov8n.pt**: 80 classes (COCO dataset)
- **Custom attention models**: 6 classes (HRIPCB dataset)
- **Problem**: Manual weight loading can't adapt detection head (80 ≠ 6 classes)

### Secondary Issues
1. **Incorrect load() method usage**: Passing unsupported `strict` parameter
2. **Wrong loading approach**: Manual weight loading vs training-time adaptation
3. **Architecture-first loading**: Loading custom architecture before pretrained weights

## Technical Details

### Why Baseline Works
```python
# Baseline approach (CORRECT)
model = YOLO('yolov8n.pt')  # Loads 80-class pretrained model
# During training: Ultralytics automatically adapts 80→6 classes
```

### Why Attention Models Failed
```python
# Old attention approach (INCORRECT)
model = YOLO('custom-architecture.yaml')  # Loads 6-class architecture with random weights
model.model.load(pretrained_weights, strict=False)  # Fails due to class mismatch
```

### Architecture Verification
```python
# Check model parameters
baseline_params = 3,157,200  # Standard YOLOv8n
eca_params = 3,012,023      # ECA attention (fewer due to efficiency)

# Check class counts
baseline_nc = 80  # COCO classes
eca_nc = 6        # HRIPCB classes
```

## Solution Implementation

### Fixed Code Structure

**Model Creation (Fixed):**
```python
def create_model(self) -> YOLO:
    model_config = self.config['model']
    
    if model_config.get('config_path'):
        # Load custom architecture
        model = YOLO(str(model_path))
        
        # Don't manually load pretrained weights here
        # Let ultralytics handle it during training
        if model_config.get('pretrained', False):
            self.logger.info("Custom model will use pretrained weights during training")
            
    else:
        # Baseline approach
        model = YOLO(f'{model_type}.pt')
        
    return model
```

**Training Arguments (Fixed):**
```python
def run_training(self, model: YOLO) -> Dict[str, Any]:
    train_args = {
        'data': data_config['path'],
        'epochs': training_config.get('epochs', 100),
        # ... other args ...
    }
    
    # Add pretrained weights for attention models
    model_config = self.config['model']
    if model_config.get('pretrained', False) and model_config.get('config_path'):
        model_type = model_config.get('type', 'yolov8n')
        train_args['pretrained'] = f'{model_type}.pt'  # Key fix!
        
    results = model.train(**train_args)
```

## Verification Steps

### 1. Check Weight Transfer Message
**Expected during training:**
```
Transferred 319/356 items from pretrained weights
```

### 2. Verify Model Architecture
```python
from ultralytics import YOLO

# Load attention model
model = YOLO('ultralytics/cfg/models/v8/yolov8n-eca-final.yaml')

# Check architecture
print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
print(f"Has ECA modules: {any('eca' in name.lower() for name, _ in model.model.named_modules())}")

# Verify class count
head = model.model.model[-1]  # Detection head
print(f"Model classes: {getattr(head, 'nc', 'unknown')}")
```

### 3. Test Training Preparation
```python
# This should work without errors
model = YOLO('ultralytics/cfg/models/v8/yolov8n-eca-final.yaml')

training_args = {
    'data': 'experiments/configs/datasets/hripcb_data.yaml',
    'epochs': 1,
    'batch': 4,
    'pretrained': 'yolov8n.pt',  # Critical for attention models
    'exist_ok': True
}

print("✅ Training args prepared successfully")
```

## Expected Results After Fix

### Training Logs
```
Loading custom model from: ultralytics/cfg/models/v8/yolov8n-eca-final.yaml
Custom model will use pretrained yolov8n.pt during training
✅ Pretrained weights will be loaded with proper class adaptation during training
🎯 Will use pretrained weights during training: yolov8n.pt
```

### Weight Transfer
```
Transferred 319/356 items from pretrained weights
```

### Performance
- Attention models should **exceed baseline performance**
- ECA: +1-2% mAP improvement over baseline
- CBAM: +2-3% mAP improvement over baseline  
- CoordAtt: +2-3% mAP improvement over baseline

## Prevention Guidelines

### 1. Always Use Training-Time Pretrained Loading
```yaml
# CORRECT: Let ultralytics handle pretrained loading
training_args:
  pretrained: 'yolov8n.pt'
  
# INCORRECT: Manual weight loading in model creation
# model.load('yolov8n.pt')  # Don't do this for custom architectures
```

### 2. Verify Architecture Before Training
```python
# Check that attention modules are present
has_attention = any('eca' in name.lower() or 'cbam' in name.lower() 
                   for name, _ in model.model.named_modules())
assert has_attention, "Attention modules not found in model"
```

### 3. Monitor Training Logs
- Always check for "Transferred X/Y items from pretrained weights"
- Verify performance improvements over baseline
- Watch for class adaptation messages

## Common Mistakes to Avoid

### ❌ Manual Weight Loading
```python
# DON'T do this for custom architectures
model = YOLO('custom-config.yaml')
model.load('yolov8n.pt')  # Class mismatch causes issues
```

### ❌ Missing Pretrained in Training Args
```python
# Missing pretrained parameter
train_args = {
    'data': 'dataset.yaml',
    'epochs': 100,
    # 'pretrained': 'yolov8n.pt',  # ← Missing this!
}
```

### ❌ Wrong Config Structure
```yaml
# Don't manually load in model config
model:
  pretrained_weights: "yolov8n.pt"  # Wrong place
```

## File Locations

### Modified Files
- `scripts/experiments/run_single_experiment.py`: Main training script
- `experiments/configs/04_yolov8n_eca_standard.yaml`: ECA config
- `experiments/configs/05_yolov8n_cbam_standard.yaml`: CBAM config  
- `experiments/configs/06_yolov8n_coordatt_standard.yaml`: CoordAtt config

### Architecture Files
- `ultralytics/cfg/models/v8/yolov8n-eca-final.yaml`: ECA architecture
- `ultralytics/cfg/models/v8/yolov8-cbam.yaml`: CBAM architecture
- `ultralytics/cfg/models/v8/yolov8-ca.yaml`: CoordAtt architecture

## Related Documentation

- [Attention Mechanisms Documentation](ATTENTION_MECHANISMS_DOCUMENTATION.md)
- [Training Guide](TRAINING_GUIDE.md)
- [Unified Training Guide](UNIFIED_TRAINING_GUIDE.md)

## Troubleshooting Commands

### Quick Test
```bash
# Test if attention model loads correctly
python -c "
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v8/yolov8n-eca-final.yaml')
print('✅ Model loads successfully')
"
```

### Full Training Test
```bash
# Run one epoch to verify everything works
python run_experiment.py --config experiments/configs/04_yolov8n_eca_standard.yaml
```

### Architecture Verification
```bash
# Check if attention modules are present
python -c "
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v8/yolov8n-eca-final.yaml')
has_eca = any('eca' in name.lower() for name, _ in model.model.named_modules())
print(f'Has ECA attention: {has_eca}')
"
```

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Status**: Resolved ✅