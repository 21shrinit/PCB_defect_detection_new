#!/usr/bin/env python3
"""
Update all comprehensive ablation config files:
1. Change batch size from 64 to 32
2. Keep wandb project as 'pcb-defect-comprehensive-ablation'
3. Fix verifocal spelling to varifocal
4. Ensure optimal hyperparameters
"""

import os
import yaml
import glob

def update_config_file(config_path):
    """Update a single config file"""
    print(f"Updating: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update batch sizes
    if 'training' in config and 'batch' in config['training']:
        config['training']['batch'] = 32
    
    if 'validation' in config and 'batch' in config['validation']:
        config['validation']['batch'] = 32
    
    # Keep wandb project consistent
    if 'wandb' in config:
        config['wandb']['project'] = 'pcb-defect-comprehensive-ablation'
    
    # Fix verifocal spelling to varifocal in loss type
    if 'training' in config and 'loss' in config['training']:
        loss_type = config['training']['loss'].get('type', '')
        if 'verifocal' in loss_type:
            # Fix spelling: verifocal -> varifocal
            new_loss_type = loss_type.replace('verifocal', 'varifocal')
            config['training']['loss']['type'] = new_loss_type
            print(f"  Fixed loss type: {loss_type} -> {new_loss_type}")
    
    # Ensure optimal hyperparameters by loss function
    if 'training' in config:
        loss_type = config['training'].get('loss', {}).get('type', 'standard')
        
        # Standard hyperparameters
        config['training']['optimizer'] = 'AdamW'
        config['training']['lr0'] = 0.0005
        config['training']['lrf'] = 0.005
        config['training']['weight_decay'] = 0.0002
        config['training']['momentum'] = 0.94
        config['training']['warmup_epochs'] = 15.0
        config['training']['scheduler'] = 'cosine'
        config['training']['patience'] = 50
        
        # Loss-specific hyperparameters
        if loss_type == 'standard':
            config['training']['loss']['box_weight'] = 7.5
            config['training']['loss']['cls_weight'] = 0.5
            config['training']['loss']['dfl_weight'] = 1.5
        elif loss_type == 'siou':
            config['training']['loss']['box_weight'] = 8.0
            config['training']['loss']['cls_weight'] = 0.8
            config['training']['loss']['dfl_weight'] = 1.5
        elif loss_type == 'eiou':
            config['training']['loss']['box_weight'] = 7.8
            config['training']['loss']['cls_weight'] = 0.7
            config['training']['loss']['dfl_weight'] = 1.6
        elif 'varifocal' in loss_type:
            config['training']['loss']['box_weight'] = 8.2
            config['training']['loss']['cls_weight'] = 1.0
            config['training']['loss']['dfl_weight'] = 1.8
    
    # Clean up metadata
    if 'metadata' in config:
        # Keep only essential metadata fields
        essential_fields = {
            'dataset_name': config['metadata'].get('dataset_name', 'HRIPCB'),
            'architecture': config['metadata'].get('architecture', ''),
            'loss_function': config['metadata'].get('loss_function', ''),
            'attention_mechanism': config['metadata'].get('attention_mechanism', 'none'),
            'experiment_phase': 'comprehensive_ablation_study'
        }
        config['metadata'] = essential_fields
    
    # Write back to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

def main():
    config_dir = "experiments/configs/comprehensive_ablation"
    config_files = glob.glob(f"{config_dir}/*_config.yaml")
    
    print(f"Found {len(config_files)} config files to update")
    
    for config_file in sorted(config_files):
        try:
            update_config_file(config_file)
        except Exception as e:
            print(f"ERROR updating {config_file}: {e}")
    
    print(f"\nUpdated {len(config_files)} config files:")
    print("- Batch size: 64 -> 32")
    print("- WandB project: pcb-defect-comprehensive-ablation")  
    print("- Loss spelling: verifocal -> varifocal")
    print("- Optimized hyperparameters by loss function")

if __name__ == "__main__":
    main()