#!/usr/bin/env python3
"""
Apply optimized hyperparameters to all 36 config files based on analysis:
1. Learning Rate & Warmup Optimization: lr0=0.0008, warmup=6.0, lrf=0.01
2. Augmentation Intensity Reduction: hsv_s=0.4, scale=0.3, mosaic=0.8
3. Loss Weight Rebalancing: Different weights per loss function type
"""

import os
import yaml
import glob

def optimize_config_file(config_path):
    """Apply optimized hyperparameters to a single config file"""
    print(f"Optimizing: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. LEARNING RATE & WARMUP OPTIMIZATION
    if 'training' in config:
        config['training']['lr0'] = 0.0008  # Increased learning capacity
        config['training']['warmup_epochs'] = 6.0  # Reduced warmup for dataset size
        config['training']['lrf'] = 0.01  # Higher final LR for better convergence
        print(f"  LR optimization: lr0=0.0008, warmup=6.0, lrf=0.01")
    
    # 2. AUGMENTATION INTENSITY REDUCTION  
    if 'training' in config and 'augmentation' in config['training']:
        config['training']['augmentation']['hsv_s'] = 0.4  # Moderate saturation
        config['training']['augmentation']['scale'] = 0.3  # Conservative scaling
        config['training']['augmentation']['mosaic'] = 0.8  # Reduced mosaic intensity
        print(f"  Augmentation reduction: hsv_s=0.4, scale=0.3, mosaic=0.8")
    
    # 3. LOSS WEIGHT REBALANCING (by loss function type)
    if 'training' in config and 'loss' in config['training']:
        loss_type = config['training']['loss'].get('type', 'standard')
        
        if loss_type == 'standard':
            # Standard Loss: balanced weights
            config['training']['loss']['box_weight'] = 5.0
            config['training']['loss']['cls_weight'] = 1.0
            config['training']['loss']['dfl_weight'] = 1.5
            print(f"  Standard loss weights: box=5.0, cls=1.0, dfl=1.5")
            
        elif loss_type == 'siou':
            # SIoU Loss: slightly higher for IoU focus
            config['training']['loss']['box_weight'] = 6.0
            config['training']['loss']['cls_weight'] = 1.2
            config['training']['loss']['dfl_weight'] = 1.5
            print(f"  SIoU loss weights: box=6.0, cls=1.2, dfl=1.5")
            
        elif loss_type == 'eiou':
            # EIoU Loss: balanced with slight IoU emphasis
            config['training']['loss']['box_weight'] = 5.5
            config['training']['loss']['cls_weight'] = 1.1
            config['training']['loss']['dfl_weight'] = 1.6
            print(f"  EIoU loss weights: box=5.5, cls=1.1, dfl=1.6")
            
        elif 'varifocal' in loss_type:
            # VariFocal+EIoU: higher classification weight for focal loss
            config['training']['loss']['box_weight'] = 6.0
            config['training']['loss']['cls_weight'] = 1.5
            config['training']['loss']['dfl_weight'] = 1.8
            print(f"  VariFocal+EIoU loss weights: box=6.0, cls=1.5, dfl=1.8")
    
    # Write back to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

def main():
    config_dir = "experiments/configs/comprehensive_ablation"
    config_files = glob.glob(f"{config_dir}/*_config.yaml")
    
    print(f"Applying optimized hyperparameters to {len(config_files)} config files")
    print("Changes:")
    print("1. Learning Rate & Warmup: lr0=0.0008, warmup=6.0, lrf=0.01")
    print("2. Augmentation Reduction: hsv_s=0.4, scale=0.3, mosaic=0.8") 
    print("3. Loss Weight Rebalancing: by loss function type")
    print("=" * 60)
    
    success_count = 0
    for config_file in sorted(config_files):
        try:
            optimize_config_file(config_file)
            success_count += 1
        except Exception as e:
            print(f"ERROR optimizing {config_file}: {e}")
    
    print("=" * 60)
    print(f"Successfully optimized {success_count}/{len(config_files)} config files")
    print("\nExpected improvements:")
    print("- Break through 90-91% mAP plateau") 
    print("- Target: 93-95% mAP50, 50-55% mAP50-95")
    print("- Improved classification stability")
    print("- Coordinated training dynamics")

if __name__ == "__main__":
    main()