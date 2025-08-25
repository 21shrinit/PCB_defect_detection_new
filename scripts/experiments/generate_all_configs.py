#!/usr/bin/env python3
"""
Comprehensive Configuration Generator for PCB Defect Detection Experiments
=========================================================================

Generates all 48 configuration files for comprehensive ablation study:
- 3 Architectures (YOLOv8n, YOLOv10n, YOLOv11n)
- 4 Loss Functions (Standard, SIoU, EIoU, VeriFocal+EIoU)  
- 4 Attention Mechanisms (None, CBAM, CoordAtt, ECA)

Uses standardized format compatible with run_single_experiment_FIXED.py
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Configuration mappings for proper integration
ARCHITECTURE_CONFIG_MAP = {
    'yolov8n': {
        'none': 'yolov8n.pt',
        'cbam': 'ultralytics/cfg/models/v8/yolov8n-cbam-neck-optimal.yaml',
        'coordatt': 'ultralytics/cfg/models/v8/yolov8n-ca-dual-placement.yaml',
        'eca': 'ultralytics/cfg/models/v8/yolov8n-eca-final.yaml'
    },
    'yolov10n': {
        'none': 'yolov10n.pt',
        'cbam': 'ultralytics/cfg/models/v10/yolov10n-cbam-research-optimal.yaml',
        'coordatt': 'ultralytics/cfg/models/v10/yolov10n-ca-dual-placement.yaml',
        'eca': 'ultralytics/cfg/models/v10/yolov10n-eca-final.yaml'
    },
    'yolo11n': {
        'none': 'yolo11n.pt',
        'cbam': 'ultralytics/cfg/models/11/yolo11n-cbam-neck-optimal.yaml',
        'coordatt': 'ultralytics/cfg/models/11/yolo11n-ca-dual-placement.yaml',
        'eca': 'ultralytics/cfg/models/11/yolo11n-eca-final.yaml'
    }
}

LOSS_FUNCTION_CONFIG = {
    'standard': {
        'type': 'standard',
        'description': 'CIoU + BCE',
        'box_weight': 7.5,
        'cls_weight': 0.5,
        'dfl_weight': 1.5
    },
    'siou': {
        'type': 'siou',
        'description': 'SIoU + BCE',
        'box_weight': 8.0,
        'cls_weight': 0.8,
        'dfl_weight': 1.5
    },
    'eiou': {
        'type': 'eiou', 
        'description': 'EIoU + BCE',
        'box_weight': 8.2,
        'cls_weight': 0.7,
        'dfl_weight': 1.6
    },
    'verifocal_eiou': {
        'type': 'verifocal_eiou',
        'description': 'EIoU + VeriFocal',
        'box_weight': 8.5,
        'cls_weight': 0.8,
        'dfl_weight': 1.9
    }
}

ATTENTION_MECHANISM_INFO = {
    'none': {
        'name': 'None',
        'description': 'Standard baseline without attention',
        'expected_improvement': '0% (baseline)'
    },
    'cbam': {
        'name': 'CBAM',
        'description': 'Convolutional Block Attention Module',
        'expected_improvement': '+4-7% mAP over baseline'
    },
    'coordatt': {
        'name': 'CoordAtt',
        'description': 'Coordinate Attention',
        'expected_improvement': '+2-4% mAP over baseline'
    },
    'eca': {
        'name': 'ECA',
        'description': 'Efficient Channel Attention',
        'expected_improvement': '+1-3% mAP over baseline'
    }
}

def generate_config(architecture: str, loss_function: str, attention: str, 
                   output_dir: Path, base_config: Dict[str, Any]) -> str:
    """Generate a single configuration file."""
    
    # Get configuration details
    attention_info = ATTENTION_MECHANISM_INFO[attention]
    loss_info = LOSS_FUNCTION_CONFIG[loss_function]
    
    # Determine model configuration path
    if attention == 'none':
        model_config = ARCHITECTURE_CONFIG_MAP[architecture][attention]
        config_path = None  # Use pretrained model
    else:
        config_path = ARCHITECTURE_CONFIG_MAP[architecture][attention]
        model_config = config_path
    
    # Create experiment name
    exp_name = f"{architecture.upper()}_{loss_function.upper()}_{attention.upper()}_Experiment"
    
    # Build configuration
    config = {
        'experiment': {
            'name': exp_name,
            'type': 'comprehensive_ablation_study',
            'mode': 'train',
            'description': f"{architecture.upper()} with {loss_info['description']} and {attention_info['description']}",
            'tags': [
                'comprehensive_ablation',
                architecture,
                f"loss_{loss_function}",
                f"attention_{attention}",
                'pcb_defect_detection'
            ]
        },
        
        'model': {
            'type': architecture,
            'pretrained': True,
            'attention_mechanism': attention
        },
        
        'data': {
            'num_classes': 6
        },
        
        'training': {
            'dataset': {
                'path': base_config['dataset_path']
            },
            'epochs': base_config['epochs'],
            'batch': base_config['batch_size'],
            'imgsz': 640,
            'device': "0",
            'workers': 16,
            'seed': 42,
            
            # Optimizer settings (all AdamW for consistency)
            'optimizer': 'AdamW',
            'lr0': 0.0008 if attention != 'none' else 0.001,
            'lrf': 0.002 if attention != 'none' else 0.01,
            'weight_decay': 0.0001 if attention != 'none' else 0.0005,
            'momentum': 0.95 if attention != 'none' else 0.937,
            'warmup_epochs': 20.0 if attention != 'none' else 3.0,
            'patience': 50,
            
            # Performance optimizations
            'save_period': 25,
            'validate': True,
            'cache': 'disk',
            'amp': True,
            'name': f"{architecture}_{loss_function}_{attention}_experiment",
            
            # Loss configuration - CRITICAL for proper integration
            'loss': {
                'type': loss_info['type'],
                'box_weight': loss_info['box_weight'],
                'cls_weight': loss_info['cls_weight'],
                'dfl_weight': loss_info['dfl_weight']
            },
            
            # Augmentation (attention-aware)
            'augmentation': {
                'mosaic': 0.8 if attention != 'none' else 1.0,
                'mixup': 0.1 if attention != 'none' else 0.0,
                'copy_paste': 0.2 if attention != 'none' else 0.0,
                'hsv_h': 0.01 if attention != 'none' else 0.015,
                'hsv_s': 0.4 if attention != 'none' else 0.7,
                'hsv_v': 0.2 if attention != 'none' else 0.4,
                'degrees': 0.0,
                'translate': 0.05 if attention != 'none' else 0.1,
                'scale': 0.3 if attention != 'none' else 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.4 if attention != 'none' else 0.5
            }
        },
        
        'validation': {
            'batch': base_config['batch_size'],
            'imgsz': 640,
            'conf_threshold': 0.001,
            'iou_threshold': 0.6,
            'max_detections': 300,
            'split': 'val'
        },
        
        'wandb': {
            'project': "pcb-defect-comprehensive-ablation",
            'name': f"{architecture}_{loss_function}_{attention}_experiment",
            'save_code': True,
            'dir': "./wandb_logs"
        },
        
        'metadata': {
            'dataset_name': 'HRIPCB',
            'architecture': architecture,
            'loss_function': loss_function,
            'attention_mechanism': attention,
            'expected_improvement': attention_info['expected_improvement'],
            'experiment_phase': 'comprehensive_ablation_study',
            'notes': f"Systematic evaluation of {architecture.upper()} + {loss_info['description']} + {attention_info['description']}"
        }
    }
    
    # Add model config path if using custom architecture
    if config_path:
        config['model']['config_path'] = config_path
    
    # Generate filename
    filename = f"{architecture}_{loss_function}_{attention}_config.yaml"
    filepath = output_dir / filename
    
    # Write configuration file
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    return str(filepath)

def generate_all_configs(output_dir: str, dataset_path: str, epochs: int = 150, 
                        batch_size: int = 64) -> List[str]:
    """Generate all 48 configuration files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_config = {
        'dataset_path': dataset_path,
        'epochs': epochs,
        'batch_size': batch_size
    }
    
    generated_configs = []
    total_configs = 0
    
    print(f"Generating comprehensive experiment configurations...")
    print(f"   Output directory: {output_path}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print()
    
    # Generate all combinations
    for architecture in ARCHITECTURE_CONFIG_MAP.keys():
        for loss_function in LOSS_FUNCTION_CONFIG.keys():
            for attention in ATTENTION_MECHANISM_INFO.keys():
                try:
                    config_file = generate_config(
                        architecture, loss_function, attention, 
                        output_path, base_config
                    )
                    generated_configs.append(config_file)
                    total_configs += 1
                    
                    print(f"Generated: {Path(config_file).name}")
                    
                except Exception as e:
                    print(f"Failed to generate {architecture}_{loss_function}_{attention}: {e}")
    
    print(f"\nSuccessfully generated {total_configs}/48 configuration files")
    print(f"All configs saved to: {output_path}")
    
    return generated_configs

def generate_experiment_summary(configs: List[str], output_dir: Path):
    """Generate experiment summary and execution guide."""
    
    summary = {
        'total_experiments': len(configs),
        'architectures': list(ARCHITECTURE_CONFIG_MAP.keys()),
        'loss_functions': list(LOSS_FUNCTION_CONFIG.keys()),
        'attention_mechanisms': list(ATTENTION_MECHANISM_INFO.keys()),
        'execution_order': {
            'phase_1_baselines': [
                config for config in configs 
                if '_none_' in Path(config).name
            ],
            'phase_2_cbam': [
                config for config in configs 
                if '_cbam_' in Path(config).name
            ],
            'phase_3_alternative_attention': [
                config for config in configs 
                if '_coordatt_' in Path(config).name or '_eca_' in Path(config).name
            ]
        },
        'estimated_total_time_hours': len(configs) * 2.5,  # ~2.5 hours per experiment
        'execution_guide': {
            'single_experiment': 'python scripts/experiments/run_single_experiment_FIXED.py --config <config_file>',
            'comprehensive_runner': 'python scripts/experiments/comprehensive_experiment_runner.py --config <config_file>',
            'batch_execution': 'Use the comprehensive experiment runner for automated execution'
        }
    }
    
    # Save summary
    summary_file = output_dir / 'experiment_summary.yaml'
    with open(summary_file, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False, indent=2)
    
    # Create execution script
    execution_script = output_dir / 'run_all_experiments.sh'
    with open(execution_script, 'w', encoding='utf-8') as f:
        f.write('#!/bin/bash\n')
        f.write('# Comprehensive PCB Defect Detection Experiments\n')
        f.write('# Generated automatically - execute in phases\n\n')
        
        f.write('# Phase 1: Baseline Experiments (12 experiments)\n')
        for config in summary['execution_order']['phase_1_baselines']:
            f.write(f'python scripts/experiments/comprehensive_experiment_runner.py --config {config}\n')
        
        f.write('\n# Phase 2: CBAM Attention Experiments (12 experiments)\n')
        for config in summary['execution_order']['phase_2_cbam']:
            f.write(f'python scripts/experiments/comprehensive_experiment_runner.py --config {config}\n')
        
        f.write('\n# Phase 3: Alternative Attention Experiments (24 experiments)\n')
        for config in summary['execution_order']['phase_3_alternative_attention']:
            f.write(f'python scripts/experiments/comprehensive_experiment_runner.py --config {config}\n')
    
    # Make script executable
    os.chmod(execution_script, 0o755)
    
    print(f"\nExperiment summary saved: {summary_file}")
    print(f"Execution script created: {execution_script}")

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive experiment configurations')
    parser.add_argument('--output_dir', type=str, default='experiments/configs/comprehensive_ablation',
                       help='Output directory for configuration files')
    parser.add_argument('--dataset_path', type=str, default='experiments/configs/datasets/hripcb_data.yaml',
                       help='Path to dataset configuration')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size')
    
    args = parser.parse_args()
    
    # Generate all configurations
    configs = generate_all_configs(
        args.output_dir, 
        args.dataset_path,
        args.epochs,
        args.batch_size
    )
    
    # Generate execution summary
    generate_experiment_summary(configs, Path(args.output_dir))
    
    print(f"\nReady to run comprehensive PCB defect detection experiments!")
    print(f"   Total experiments: {len(configs)}")
    print(f"   Estimated time: {len(configs) * 2.5:.1f} hours")
    print(f"\nStart with: python scripts/experiments/comprehensive_experiment_runner.py --config <config_file>")

if __name__ == '__main__':
    main()