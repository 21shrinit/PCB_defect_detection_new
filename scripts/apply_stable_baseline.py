#!/usr/bin/env python3
"""
Apply Proven Stable Baseline Parameters
Based on working config: experiments/configs/01_yolov8n_baseline_standard.yaml

Key fixes:
- lr0: 0.001 (proven stable, was 5x higher)
- batch: 128 (larger batch = more stable)
- cache: false (avoid memory pressure)
- workers: 16 (optimal data loading)
- patience: 30 (proven working)
- Remove cos_lr, warmup_momentum, warmup_bias_lr (caused instability)
- Add mixup: 0.1 and copy_paste: 0.3 (regularization)
"""

import os
import re
import glob

def apply_stable_params(config_path):
    """Apply stable baseline parameters while preserving experiment-specific settings"""
    print(f"Applying stable params: {os.path.basename(config_path)}")
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # 1. Fix learning rate (critical - was 5x too high)
    content = re.sub(r'lr0: 0\.\d+', 'lr0: 0.001', content)
    
    # 2. Fix batch size and training params
    content = re.sub(r'batch: 64', 'batch: 128', content)
    content = re.sub(r'patience: 50', 'patience: 30', content)
    content = re.sub(r'workers: 8', 'workers: 16', content)
    content = re.sub(r'cache: "ram"', 'cache: false', content)
    
    # 3. Remove problematic optimizer params
    content = re.sub(r'  cos_lr: true\n', '', content)
    content = re.sub(r'  warmup_momentum: 0\.8\n', '', content)
    content = re.sub(r'  warmup_bias_lr: 0\.1\n', '', content)
    
    # 4. Remove gradient clipping (not in working config)
    content = re.sub(r'  # Numerical Stability\n  max_norm: \d+\.0  # Gradient clipping\n  \n', '', content)
    content = re.sub(r'  # .* Stability.*\n  max_norm: \d+\.0.*\n  \n', '', content)
    
    # 5. Add proven augmentations (mixup and copy_paste)
    if 'mixup: 0.0' in content:
        content = re.sub(r'mixup: 0\.0', 'mixup: 0.1', content)
    if 'copy_paste: 0.0' in content:
        content = re.sub(r'copy_paste: 0\.0', 'copy_paste: 0.3', content)
    
    # 6. Fix validation batch to match training
    content = re.sub(r'validation:\n  batch: 64', 'validation:\n  batch: 128', content)
    
    # 7. Restore standard box_weight for baselines only
    if 'baseline' in config_path.lower() and 'box_weight: 5.5' in content:
        content = re.sub(r'box_weight: 5\.5', 'box_weight: 7.5', content)
    
    # Write back
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Applied stable parameters to: {os.path.basename(config_path)}")

def main():
    """Apply stable parameters to all experiment configs"""
    config_dir = "experiments/configs/revised_benchmark_2025"
    config_pattern = os.path.join(config_dir, "*.yaml")
    
    configs = glob.glob(config_pattern)
    
    if not configs:
        print(f"No configs found in {config_dir}")
        return
    
    print(f"Applying proven stable parameters to {len(configs)} configurations")
    print("="*70)
    print("Based on working config: 01_yolov8n_baseline_standard.yaml")
    print("Key fixes:")
    print("  • lr0: 0.001 (was 5x higher)")
    print("  • batch: 128 (more stable gradients)")
    print("  • cache: false (avoid memory pressure)")
    print("  • Remove cos_lr and warmup extras")
    print("  • Add mixup + copy_paste regularization")
    print("="*70)
    
    for config_path in sorted(configs):
        if 'README' not in config_path:
            apply_stable_params(config_path)
    
    print("="*70)
    print("🎉 All configurations updated with proven stable parameters!")
    print("\n⚠️  CRITICAL FIXES APPLIED:")
    print("   • Learning rate reduced from 0.005+ to 0.001 (5x reduction)")
    print("   • Batch size increased to 128 for gradient stability")
    print("   • Removed cosine LR scheduler (was causing instability)")
    print("   • Added regularization (mixup + copy_paste)")
    print("\n✅ These should eliminate NaN values and validation fluctuations")

if __name__ == "__main__":
    main()