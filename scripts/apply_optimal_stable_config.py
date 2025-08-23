#!/usr/bin/env python3
"""
Apply Optimal Stable Configuration
Based on comprehensive analysis of working vs problematic configs

KEY FINDINGS:
- lr0: 0.005 with AdamW is TOO HIGH (causes NaN)
- Working configs use lr0: 0.001 (AdamW) or 0.0005 (attention)
- cache: "ram" causes memory pressure 
- cos_lr: true was not in working configs
- Missing regularization (mixup, copy_paste)
"""

import os
import re
import glob

def apply_optimal_config(config_path):
    """Apply optimal stable configuration based on experiment type"""
    filename = os.path.basename(config_path)
    print(f"Optimizing: {filename}")
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Determine experiment type for specific tuning
    is_attention = any(x in filename.lower() for x in ['cbam', 'eca', 'coord'])
    is_advanced_loss = any(x in filename.lower() for x in ['siou', 'eiou', 'focal', 'varifocal'])
    is_baseline = 'baseline' in filename.lower()
    
    # 1. CRITICAL: Fix learning rate based on type
    if is_attention:
        # Attention needs very low LR
        content = re.sub(r'lr0: 0\.\d+', 'lr0: 0.0005', content)
        print(f"  - Applied attention LR: 0.0005")
    elif is_advanced_loss:
        # Advanced losses need moderate LR
        content = re.sub(r'lr0: 0\.\d+', 'lr0: 0.0008', content)
        print(f"  - Applied advanced loss LR: 0.0008")
    else:
        # Baseline needs proven stable LR
        content = re.sub(r'lr0: 0\.\d+', 'lr0: 0.001', content)
        print(f"  - Applied baseline LR: 0.001")
    
    # 2. Fix batch size and core training params
    content = re.sub(r'batch: 64', 'batch: 128', content)
    content = re.sub(r'patience: 50', 'patience: 30', content)
    content = re.sub(r'workers: 8', 'workers: 16', content)
    content = re.sub(r'cache: "ram"', 'cache: false', content)
    
    # 3. Remove problematic scheduler settings
    content = re.sub(r'  cos_lr: true\n', '', content)
    content = re.sub(r'  warmup_momentum: 0\.8\n', '', content)
    content = re.sub(r'  warmup_bias_lr: 0\.1\n', '', content)
    content = re.sub(r'  # Numerical Stability\n  max_norm: \d+\.0.*\n  \n', '', content)
    content = re.sub(r'  # .* Stability.*\n  max_norm: \d+\.0.*\n  \n', '', content)
    
    # 4. Fix warmup for attention models
    if is_attention:
        content = re.sub(r'warmup_epochs: 3\.0', 'warmup_epochs: 5.0', content)
        print(f"  - Extended warmup for attention")
    
    # 5. Add regularization (critical for stability)
    if is_attention:
        # Attention models: disable mixup/copy_paste to maintain focus
        content = re.sub(r'mixup: 0\.\d+', 'mixup: 0.0', content)
        content = re.sub(r'copy_paste: 0\.\d+', 'copy_paste: 0.0', content)
        print(f"  - Disabled regularization for attention focus")
    else:
        # Non-attention: add proven regularization
        content = re.sub(r'mixup: 0\.0', 'mixup: 0.1', content)
        content = re.sub(r'copy_paste: 0\.0', 'copy_paste: 0.3', content)
        print(f"  - Added stabilizing regularization")
    
    # 6. Fix validation batch
    content = re.sub(r'validation:\n  batch: 64', 'validation:\n  batch: 128', content)
    
    # 7. Fix loss weights based on type
    if is_advanced_loss:
        # Advanced losses need lower box weights
        if 'box_weight: 7.5' in content:
            content = re.sub(r'box_weight: 7\.5', 'box_weight: 5.5', content)
        # Conservative focal parameters
        content = re.sub(r'focal_gamma: 2\.[0-9]+', 'focal_gamma: 1.8', content)
        content = re.sub(r'focal_alpha: 0\.[0-9]+', 'focal_alpha: 0.7', content)
        print(f"  - Applied advanced loss tuning")
    elif is_baseline:
        # Baselines use proven stable weights
        content = re.sub(r'box_weight: [0-9]\.[0-9]', 'box_weight: 7.5', content)
        print(f"  - Applied baseline loss weights")
    
    # Write back
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Optimized: {filename}\n")

def main():
    """Apply optimal configuration to all revised benchmark configs"""
    config_dir = "experiments/configs/revised_benchmark_2025"
    config_pattern = os.path.join(config_dir, "*.yaml")
    
    configs = glob.glob(config_pattern)
    
    if not configs:
        print(f"No configs found in {config_dir}")
        return
    
    print("=" * 80)
    print("APPLYING OPTIMAL STABLE CONFIGURATION")
    print("=" * 80)
    print("Based on analysis of working vs problematic configs:")
    print("  • CRITICAL: lr0 reduced to 0.001 (AdamW) / 0.0005 (attention)")
    print("  • batch: 128 (stable gradients)")
    print("  • cache: false (avoid memory pressure)")
    print("  • Remove cos_lr scheduler")
    print("  • Add mixup + copy_paste regularization")
    print("  • Architecture-specific tuning")
    print("=" * 80)
    
    for config_path in sorted(configs):
        if 'README' not in config_path:
            apply_optimal_config(config_path)
    
    print("=" * 80)
    print("🎉 ALL CONFIGURATIONS OPTIMIZED!")
    print("=" * 80)
    print("Expected results:")
    print("  ✅ No more NaN validation losses")
    print("  ✅ No more extreme loss spikes")
    print("  ✅ Smooth, stable training curves")
    print("  ✅ Proper convergence")
    print("=" * 80)

if __name__ == "__main__":
    main()