#!/usr/bin/env python3
"""
Training Stability Fix Script
Applies stability improvements to all experiment configurations to prevent:
- Validation loss NaN values
- Extreme loss fluctuations
- Training instability
"""

import os
import re
import glob

def fix_config_stability(config_path):
    """Apply stability fixes to a configuration file"""
    print(f"Fixing: {config_path}")
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Reduce high learning rates
    if 'lr0: 0.01' in content:
        content = re.sub(r'lr0: 0\.01', 'lr0: 0.005', content)
        content = re.sub(r'# Learning Rate \(Baseline\)', '# Learning Rate (Stable - Reduced)', content)
    elif 'lr0: 0.009' in content:
        content = re.sub(r'lr0: 0\.009', 'lr0: 0.005', content)
        content = re.sub(r'# Learning Rate \([^)]+\)', '# Learning Rate (Stability Optimized)', content)
    elif 'lr0: 0.008' in content:
        content = re.sub(r'lr0: 0\.008', 'lr0: 0.006', content)
        content = re.sub(r'# Learning Rate \([^)]+\)', '# Learning Rate (Advanced Loss Optimized)', content)
    elif 'lr0: 0.0075' in content:
        content = re.sub(r'lr0: 0\.0075', 'lr0: 0.006', content)  
    elif 'lr0: 0.007' in content:
        content = re.sub(r'lr0: 0\.007', 'lr0: 0.005', content)
    
    # Fix 2: Add gradient clipping if not present
    if 'max_norm:' not in content:
        # Find the lr section and add gradient clipping after
        lr_pattern = r'(lr0: [0-9.]+\n  lrf: [0-9.]+)\n(\s+)'
        replacement = r'\1\n\2\n\2# Numerical Stability\n\2max_norm: 10.0  # Gradient clipping\n\2'
        content = re.sub(lr_pattern, replacement, content)
    
    # Fix 3: Standardize validation batch to 64
    if 'batch: 128' in content and 'validation:' in content:
        content = re.sub(r'validation:\n  batch: 128', 'validation:\n  batch: 64  # Match training batch for consistency', content)
    
    # Fix 4: Reduce extremely high box weights
    if 'box_weight: 7.5' in content:
        content = re.sub(r'box_weight: 7\.5  # Ultralytics Default', 'box_weight: 5.5  # Reduced for stability', content)
        content = re.sub(r'# Loss Functions \(CIoU\+BCE - Default\)', '# Loss Functions (CIoU+BCE - Stabilized)', content)
    
    # Write back
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Fixed: {os.path.basename(config_path)}")

def main():
    """Apply stability fixes to all experiment configs"""
    config_dir = "experiments/configs/revised_benchmark_2025"
    config_pattern = os.path.join(config_dir, "*.yaml")
    
    configs = glob.glob(config_pattern)
    
    if not configs:
        print(f"No configs found in {config_dir}")
        return
    
    print(f"Found {len(configs)} configuration files")
    print("="*60)
    
    for config_path in sorted(configs):
        if 'README' not in config_path:  # Skip README
            fix_config_stability(config_path)
    
    print("="*60)
    print("🎉 All configurations stabilized!")
    print("\nKey fixes applied:")
    print("  • Learning rates reduced for numerical stability")  
    print("  • Gradient clipping (max_norm: 10.0) added")
    print("  • Validation batch size matched to training (64)")
    print("  • Box weights reduced where too high")
    print("\nThis should eliminate:")
    print("  ❌ NaN validation losses")
    print("  ❌ Extreme loss fluctuations") 
    print("  ❌ Training instability")

if __name__ == "__main__":
    main()