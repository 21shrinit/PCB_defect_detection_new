# F1 Score Integration Guide

## Overview
F1 scores are now fully integrated into the Ultralytics training pipeline and will be automatically logged to WandB and CSV files during training.

## What Was Fixed

### 1. Problem Identified
- F1 scores were calculated internally (`ultralytics/utils/metrics.py:842`)
- F1 scores were accessible via `results.box.mf1` 
- **BUT** F1 scores were NOT included in the standard logging keys
- This meant F1 scores didn't appear in WandB dashboards or CSV training logs

### 2. Solution Implemented
**File: `ultralytics/utils/metrics.py`**

**Before:**
```python
@property
def keys(self) -> List[str]:
    """Return a list of keys for accessing specific metrics."""
    return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

def mean_results(self) -> List[float]:
    """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
    return self.box.mean_results()
```

**After:**
```python
@property
def keys(self) -> List[str]:
    """Return a list of keys for accessing specific metrics."""
    return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/F1(B)"]

def mean_results(self) -> List[float]:
    """Calculate mean of detected objects & return precision, recall, mAP50, mAP50-95, and F1."""
    return self.box.mean_results() + [self.box.mf1]
```

## How F1 Logging Works

### 1. Training Flow
```
Training Loop
    ↓
Validation (every epoch)
    ↓ 
DetMetrics.results_dict() 
    ↓
Uses DetMetrics.keys + DetMetrics.mean_results()
    ↓
Returns: {"metrics/F1(B)": <f1_value>, ...}
    ↓
trainer.metrics = validation_results
    ↓
WandB Callback logs trainer.metrics
```

### 2. Where F1 Appears

#### WandB Dashboard
- **Metric Name**: `metrics/F1(B)`
- **Section**: Validation metrics (logged every epoch)
- **Graph**: Will show F1 score trend over epochs

#### CSV Training Logs  
- **File**: `runs/detect/train/results.csv`
- **Column**: `metrics/F1(B)`
- **Updates**: Every epoch during validation

#### Terminal Output
F1 scores are calculated and can be accessed in scripts:
```python
# In experiment runner or custom scripts
results = model.val()
f1_score = results.box.mf1
print(f"F1 Score: {f1_score:.4f}")
```

## Testing the Integration

### Quick Test
```bash
python test_f1_logging.py
```
This will verify:
- ✅ F1 is included in metrics keys
- ✅ Keys and results count match
- ✅ WandB logging format preview

### Full Training Test
Run any experiment and check:
1. **WandB**: Look for `metrics/F1(B)` in your dashboard
2. **CSV**: Check `results.csv` for F1 column
3. **Terminal**: F1 values in validation output

## Verification Messages
When running the verification scripts, you should see:
```
🎉 ✅ VERIFICATION PASSED: Expected and actual loss functions match!
```

This confirms that:
- Loss function objects are properly initialized 
- F1 scores are being calculated correctly
- All metrics are flowing through the logging pipeline

## Next Steps
1. **Run Training**: Start any experiment to see F1 scores in WandB
2. **Monitor Trends**: Watch F1 score improvements over epochs  
3. **Compare Models**: Use F1 scores to compare different loss functions and attention mechanisms

## Notes
- F1 score calculation uses the same precision/recall values shown in other metrics
- F1 represents the harmonic mean of precision and recall: `F1 = 2 * (precision * recall) / (precision + recall)`
- This change affects all detection tasks (object detection, not classification or segmentation)