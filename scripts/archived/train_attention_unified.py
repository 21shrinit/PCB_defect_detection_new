#!/usr/bin/env python3
"""
Unified Attention Mechanism Training Script for PCB Defect Detection
===================================================================

This script provides a unified training interface for all attention mechanisms in YOLOv8:
- ECA-Net (Ultra-efficient channel attention)
- CBAM (Convolutional Block Attention Module) 
- CoordAtt (Coordinate Attention)
- Baseline YOLOv8n (no attention)

The script automatically detects the attention mechanism from the config file and
applies the appropriate training strategy. Uses two-stage training for optimal performance.

Key Features:
- Single script for all attention mechanisms
- Configuration-driven via YAML
- Two-stage training (warmup + fine-tuning)
- F1 score logging enabled
- Automatic attention mechanism detection
- Optimized for 15GB GPU utilization

Supported Attention Mechanisms:
- ECA_Final_Backbone: Ultra-efficient (5 parameters)
- CBAM_Neck_Only: Balanced efficiency (1K-10K parameters)  
- CoordAtt_Position7: Maximum accuracy (8-16K parameters)
- Baseline: No attention mechanism

Usage:
    # ECA-Net Final Backbone
    python train_attention_unified.py --config configs/config_eca_final.yaml
    
    # CBAM Neck Only
    python train_attention_unified.py --config configs/config_cbam_neck.yaml
    
    # CoordAtt Position 7
    python train_attention_unified.py --config configs/config_ca_position7.yaml
    
    # Resume any training
    python train_attention_unified.py --config configs/config_eca_final.yaml --resume

Author: MLOps Engineering Team
Date: 2025-01-20
Version: 2.0.0
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import YOLOv8
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    print("❌ Error: ultralytics package not found!")
    print("📦 Please install: pip install ultralytics")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedAttentionTrainer:
    """Unified trainer for all attention mechanisms in YOLOv8."""
    
    # Supported attention mechanisms and their characteristics
    ATTENTION_MECHANISMS = {
        # Optimized configurations (recommended)
        'ECA_Final_Backbone': {
            'name': 'ECA-Net Final Backbone',
            'description': 'Ultra-efficient channel attention (5 parameters)',
            'efficiency': 'highest',
            'parameters': '5',
            'target_use': 'real-time applications'
        },
        'CBAM_Neck_Only': {
            'name': 'CBAM Neck Feature Fusion',
            'description': 'Dual attention in neck layers (1K-10K parameters)', 
            'efficiency': 'balanced',
            'parameters': '1K-10K',
            'target_use': 'balanced accuracy-efficiency'
        },
        'CoordAtt_Position7': {
            'name': 'Coordinate Attention Position 7',
            'description': 'Position-aware attention (8-16K parameters)',
            'efficiency': 'moderate',
            'parameters': '8-16K', 
            'target_use': 'maximum accuracy'
        },
        'Baseline': {
            'name': 'YOLOv8n Baseline',
            'description': 'No attention mechanism',
            'efficiency': 'baseline',
            'parameters': '0',
            'target_use': 'baseline comparison'
        },
        # Legacy configurations (backward compatibility)
        'ECA': {
            'name': 'ECA-Net (Legacy)',
            'description': 'ECA attention mechanism (legacy config)',
            'efficiency': 'high',
            'parameters': '5-13',
            'target_use': 'backward compatibility'
        },
        'CBAM': {
            'name': 'CBAM (Legacy)',
            'description': 'CBAM attention mechanism (legacy config)',
            'efficiency': 'balanced',
            'parameters': '1K-10K',
            'target_use': 'backward compatibility'
        },
        'CoordAtt': {
            'name': 'CoordAtt (Legacy)',
            'description': 'Coordinate attention (legacy config)',
            'efficiency': 'moderate',
            'parameters': '8-16K',
            'target_use': 'backward compatibility'
        },
        'none': {
            'name': 'No Attention (Legacy)',
            'description': 'Baseline without attention (legacy config)',
            'efficiency': 'baseline',
            'parameters': '0',
            'target_use': 'backward compatibility'
        }
    }
    
    def __init__(self, config_path: str):
        """
        Initialize unified trainer with configuration.
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.attention_mechanism = self.detect_attention_mechanism()
        self.validate_config()
        
        # Set up paths
        self.setup_paths()
        
        logger.info(f"🚀 Unified Attention Trainer initialized")
        logger.info(f"📁 Config: {self.config_path}")
        logger.info(f"🎯 Attention Mechanism: {self.attention_mechanism}")
        logger.info(f"🔬 Description: {self.ATTENTION_MECHANISMS[self.attention_mechanism]['description']}")
        logger.info(f"⚡ Efficiency: {self.ATTENTION_MECHANISMS[self.attention_mechanism]['efficiency']}")
        logger.info(f"📊 Parameters: {self.ATTENTION_MECHANISMS[self.attention_mechanism]['parameters']}")
        
    def load_config(self) -> Dict[str, Any]:
        """Load and parse YAML configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Configuration file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error: {e}")
            sys.exit(1)
            
    def detect_attention_mechanism(self) -> str:
        """Detect attention mechanism from configuration."""
        try:
            attention_mechanism = self.config['model']['attention_mechanism']
            
            if attention_mechanism in self.ATTENTION_MECHANISMS:
                return attention_mechanism
            else:
                logger.error(f"❌ Unsupported attention mechanism: {attention_mechanism}")
                logger.info(f"📋 Supported mechanisms: {list(self.ATTENTION_MECHANISMS.keys())}")
                sys.exit(1)
                
        except KeyError:
            logger.error("❌ Attention mechanism not specified in config['model']['attention_mechanism']")
            sys.exit(1)
            
    def validate_config(self):
        """Validate configuration structure."""
        required_sections = ['project', 'model', 'data', 'training_strategy', 'environment']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"❌ Missing required config section: {section}")
                sys.exit(1)
                
        # Validate model config exists
        model_config_path = Path(self.config['model']['config_path'])
        if not model_config_path.exists():
            logger.error(f"❌ Model config not found: {model_config_path}")
            sys.exit(1)
            
        # Validate data config exists  
        data_config_path = Path(self.config['data']['config_path'])
        if not data_config_path.exists():
            logger.error(f"❌ Data config not found: {data_config_path}")
            sys.exit(1)
            
        logger.info("✅ Configuration validation passed")
        
    def setup_paths(self):
        """Set up training paths."""
        self.project_name = self.config['project']['name']
        self.experiment_name = self.config['project']['experiment_name']
        
        # Create experiment directory
        self.experiment_dir = Path("experiments") / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage-specific directories
        self.warmup_dir = self.experiment_dir / "warmup"
        self.finetune_dir = self.experiment_dir / "finetune"
        
        logger.info(f"📁 Experiment directory: {self.experiment_dir}")
        
    def create_yolo_model(self) -> YOLO:
        """Create YOLO model with appropriate configuration."""
        try:
            model_config_path = self.config['model']['config_path']
            pretrained_weights = self.config['model']['pretrained_weights']
            
            logger.info(f"🏗️  Creating YOLO model...")
            logger.info(f"   Model config: {model_config_path}")
            logger.info(f"   Pretrained: {pretrained_weights}")
            
            # Load custom model configuration first (to get correct nc)
            model = YOLO(model_config_path)
            
            # Load pretrained weights (only compatible layers will be loaded)
            if pretrained_weights and pretrained_weights != "":
                try:
                    logger.info(f"🔄 Loading compatible pretrained weights...")
                    model.load(pretrained_weights)
                    logger.info(f"✅ Pretrained weights loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️  Could not load all pretrained weights: {e}")
                    logger.info("📝 This is normal when class count differs - continuing with random initialization for incompatible layers")
            
            logger.info(f"✅ YOLO model created successfully")
            logger.info(f"🔬 Model architecture: {self.config['model']['architecture']}")
            logger.info(f"🎯 Attention mechanism: {self.attention_mechanism}")
            logger.info(f"📊 Classes: {self.config['model']['num_classes']}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create YOLO model: {e}")
            sys.exit(1)
            
    def get_training_args(self, stage: str = 'single_stage') -> Dict[str, Any]:
        """
        Get training arguments for specified stage.
        
        Args:
            stage (str): Training stage ('single_stage', 'warmup', or 'finetune' for legacy)
            
        Returns:
            Dict[str, Any]: Training arguments dictionary
        """
        # Support both single-stage and legacy two-stage configurations
        if stage == 'single_stage' and 'single_stage' in self.config['training_strategy']:
            stage_config = self.config['training_strategy']['single_stage']
        elif stage in self.config['training_strategy']:
            stage_config = self.config['training_strategy'][stage]
        else:
            # Fallback to first available stage
            available_stages = list(self.config['training_strategy'].keys())
            stage_config = self.config['training_strategy'][available_stages[0]]
        env_config = self.config['environment']
        
        # Base training arguments
        train_args = {
            'data': self.config['data']['config_path'],
            'epochs': stage_config['epochs'],
            'imgsz': env_config['imgsz'],
            'batch': env_config['batch_size'],
            'device': env_config['device'],
            'workers': env_config['workers'],
            'exist_ok': True,
            'optimizer': self.config['optimizer']['type'],
            'lr0': stage_config['learning_rate'],
            'weight_decay': stage_config['weight_decay'],
            'momentum': stage_config['momentum'],
            'patience': stage_config['patience'],
            'save_period': stage_config['save_period'],
            'cache': env_config['cache_images'],
            'amp': env_config['mixed_precision'],
            'deterministic': env_config['deterministic'],
            'seed': env_config['seed'],
            'verbose': True,
            'plots': True,
        }
        
        # Stage-specific settings
        if stage == 'warmup':
            train_args.update({
                'project': str(self.warmup_dir.parent),
                'name': self.warmup_dir.name,
                'pretrained': True,
                'freeze': list(range(stage_config['freeze_layers'])),
            })
        else:  # finetune
            train_args.update({
                'project': str(self.finetune_dir.parent),
                'name': self.finetune_dir.name,
                'pretrained': False,
                'freeze': [],  # No frozen layers in fine-tuning
            })
        
        # Add scheduler configuration
        if 'scheduler' in self.config:
            scheduler_config = self.config['scheduler']
            train_args.update({
                'warmup_epochs': scheduler_config['warmup_epochs'],
                'warmup_momentum': scheduler_config['warmup_momentum'],
                'warmup_bias_lr': scheduler_config['warmup_bias_lr'],
                'lrf': scheduler_config['final_lr_ratio'],
            })
        
        # Add loss configuration
        if 'loss' in self.config and 'standard' in self.config['loss']:
            loss_config = self.config['loss']['standard']
            train_args.update({
                'box': loss_config['box_weight'],
                'cls': loss_config['cls_weight'],
                'dfl': loss_config['dfl_weight'],
            })
        
        # Add augmentation configuration
        if 'augmentation' in self.config and self.config['augmentation']['enabled']:
            aug_config = self.config['augmentation']
            train_args.update({
                'mosaic': aug_config['mosaic'],
                'mixup': aug_config['mixup'],
                'copy_paste': aug_config['copy_paste'],
                'hsv_h': aug_config['hsv_h'],
                'hsv_s': aug_config['hsv_s'],
                'hsv_v': aug_config['hsv_v'],
                'degrees': aug_config['degrees'],
                'translate': aug_config['translate'],
                'scale': aug_config['scale'],
                'shear': aug_config['shear'],
                'perspective': aug_config['perspective'],
                'flipud': aug_config['flipud'],
                'fliplr': aug_config['fliplr'],
            })
            
        return train_args
        
    def train_stage_warmup(self, model: YOLO) -> Path:
        """
        Stage 1: Warmup training with frozen backbone.
        
        Args:
            model (YOLO): YOLO model instance
            
        Returns:
            Path: Path to best checkpoint from warmup
        """
        logger.info("=" * 80)
        logger.info("🔥 STAGE 1: WARMUP TRAINING")
        logger.info("=" * 80)
        logger.info(f"🎯 Attention: {self.ATTENTION_MECHANISMS[self.attention_mechanism]['name']}")
        
        # Get training arguments
        train_args = self.get_training_args('warmup')
            
        logger.info(f"🔧 Warmup training parameters:")
        for key, value in train_args.items():
            logger.info(f"   {key}: {value}")
            
        # Start warmup training
        try:
            logger.info(f"🚀 Starting warmup training...")
            results = model.train(**train_args)
            
            # Find best checkpoint
            best_checkpoint = self.warmup_dir / "weights" / "best.pt"
            
            if best_checkpoint.exists():
                logger.info(f"✅ Warmup training completed successfully!")
                logger.info(f"🏆 Best checkpoint: {best_checkpoint}")
                return best_checkpoint
            else:
                logger.error(f"❌ Best checkpoint not found: {best_checkpoint}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"❌ Warmup training failed: {e}")
            sys.exit(1)
            
    def train_stage_finetune(self, best_warmup_checkpoint: Path) -> Path:
        """
        Stage 2: Fine-tuning with full network trainable.
        
        Args:
            best_warmup_checkpoint (Path): Path to best warmup checkpoint
            
        Returns:
            Path: Path to final best checkpoint
        """
        logger.info("=" * 80)
        logger.info("⚡ STAGE 2: FINE-TUNING")
        logger.info("=" * 80)
        logger.info(f"🎯 Attention: {self.ATTENTION_MECHANISMS[self.attention_mechanism]['name']}")
        
        # Load model from warmup checkpoint
        model = YOLO(str(best_warmup_checkpoint))
        
        # Get training arguments
        train_args = self.get_training_args('finetune')
            
        logger.info(f"🔧 Fine-tuning parameters:")
        for key, value in train_args.items():
            logger.info(f"   {key}: {value}")
            
        # Start fine-tuning
        try:
            logger.info(f"🚀 Starting fine-tuning from {best_warmup_checkpoint}")
            results = model.train(**train_args)
            
            # Find final best checkpoint
            final_best_checkpoint = self.finetune_dir / "weights" / "best.pt"
            
            if final_best_checkpoint.exists():
                logger.info(f"✅ Fine-tuning completed successfully!")
                logger.info(f"🏆 Final best checkpoint: {final_best_checkpoint}")
                return final_best_checkpoint
            else:
                logger.error(f"❌ Final best checkpoint not found: {final_best_checkpoint}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            sys.exit(1)
            
    def validate_final_model(self, model_path: Path):
        """Validate final model and display results with F1 score."""
        logger.info("=" * 80)
        logger.info("📊 FINAL MODEL VALIDATION")
        logger.info("=" * 80)
        
        try:
            model = YOLO(str(model_path))
            
            # Validation arguments
            val_args = {
                'data': self.config['data']['config_path'],
                'imgsz': self.config['environment']['imgsz'],
                'batch': self.config['environment']['batch_size'],
                'device': self.config['environment']['device'],
                'verbose': True,
                'plots': True,
            }
            
            # Add validation configuration if present
            if 'validation' in self.config:
                val_config = self.config['validation']
                val_args.update({
                    'conf': val_config.get('conf_threshold', 0.001),
                    'iou': val_config.get('iou_threshold', 0.6),
                    'max_det': val_config.get('max_detections', 300),
                })
            
            logger.info("🔍 Running final validation...")
            results = model.val(**val_args)
            
            # Extract and display key metrics including F1 score
            logger.info("📊 Final Model Performance:")
            logger.info(f"   mAP@0.5: {results.box.map50:.4f}")
            logger.info(f"   mAP@0.5-0.95: {results.box.map:.4f}")
            logger.info(f"   Precision: {results.box.mp:.4f}")
            logger.info(f"   Recall: {results.box.mr:.4f}")
            logger.info(f"   F1 Score: {results.box.mf1:.4f}")  # F1 score logging confirmed
            
            # Display attention mechanism performance summary
            mechanism_info = self.ATTENTION_MECHANISMS[self.attention_mechanism]
            logger.info("🎯 Attention Mechanism Summary:")
            logger.info(f"   Mechanism: {mechanism_info['name']}")
            logger.info(f"   Parameters: {mechanism_info['parameters']} additional")
            logger.info(f"   Efficiency: {mechanism_info['efficiency']}")
            logger.info(f"   Target Use: {mechanism_info['target_use']}")
            
            logger.info("✅ Final validation completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Final validation failed: {e}")
            
    def train_single_stage(self, model: YOLO) -> Path:
        """
        Single-stage training with mechanism-specific parameters.
        
        Args:
            model (YOLO): YOLO model instance
            
        Returns:
            Path: Path to best checkpoint
        """
        logger.info("=" * 80)
        logger.info("🚀 SINGLE-STAGE TRAINING")
        logger.info("=" * 80)
        logger.info(f"🎯 Attention: {self.ATTENTION_MECHANISMS[self.attention_mechanism]['name']}")
        logger.info("📊 Research shows strategic single placement doesn't require two-stage training")
        
        # Get training arguments for single stage
        train_args = self.get_training_args('single_stage')
        
        # Set up training directory
        single_stage_dir = self.experiment_dir / "single_stage"
        train_args.update({
            'project': str(single_stage_dir.parent),
            'name': single_stage_dir.name,
            'pretrained': True,
        })
            
        logger.info(f"🔧 Single-stage training parameters:")
        for key, value in train_args.items():
            logger.info(f"   {key}: {value}")
            
        # Start single-stage training
        try:
            logger.info(f"🚀 Starting single-stage training...")
            results = model.train(**train_args)
            
            # Find best checkpoint
            best_checkpoint = single_stage_dir / "weights" / "best.pt"
            
            if best_checkpoint.exists():
                logger.info(f"✅ Single-stage training completed successfully!")
                logger.info(f"🏆 Best checkpoint: {best_checkpoint}")
                return best_checkpoint
            else:
                logger.error(f"❌ Best checkpoint not found: {best_checkpoint}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"❌ Single-stage training failed: {e}")
            sys.exit(1)

    def run_training(self, resume: bool = False):
        """
        Run training process (single-stage or two-stage based on config).
        
        Args:
            resume (bool): Whether to resume from existing checkpoints
        """
        mechanism_info = self.ATTENTION_MECHANISMS[self.attention_mechanism]
        
        # Determine training strategy from config
        training_strategy = "single-stage" if "single_stage" in self.config['training_strategy'] else "two-stage"
        
        logger.info("🚀 Starting Unified Attention Mechanism Training")
        logger.info("=" * 80)
        logger.info(f"📋 Experiment: {self.config['project']['experiment_name']}")
        logger.info(f"🔬 Model: {self.config['model']['architecture']}")
        logger.info(f"🎯 Attention: {mechanism_info['name']}")
        logger.info(f"📊 Parameters: {mechanism_info['parameters']} additional")
        logger.info(f"⚡ Efficiency: {mechanism_info['efficiency']}")
        logger.info(f"🎲 Target Use: {mechanism_info['target_use']}")
        logger.info(f"📚 Dataset: {self.config['data']['dataset_name']}")
        logger.info(f"🔄 Training Strategy: {training_strategy}")
        logger.info(f"📈 Batch Size: {self.config['environment']['batch_size']}")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            if training_strategy == "single-stage":
                # Single-stage training path
                single_stage_checkpoint = self.experiment_dir / "single_stage" / "weights" / "best.pt"
                
                if resume and single_stage_checkpoint.exists():
                    logger.info("🔄 Resuming: Single-stage training already completed")
                    self.validate_final_model(single_stage_checkpoint)
                    return
                else:
                    model = self.create_yolo_model()
                    best_checkpoint = self.train_single_stage(model)
            else:
                # Legacy two-stage training path
                warmup_checkpoint = self.warmup_dir / "weights" / "best.pt"
                finetune_checkpoint = self.finetune_dir / "weights" / "best.pt"
                
                if resume:
                    if finetune_checkpoint.exists():
                        logger.info("🔄 Resuming: Fine-tuning already completed")
                        self.validate_final_model(finetune_checkpoint)
                        return
                    elif warmup_checkpoint.exists():
                        logger.info("🔄 Resuming: Starting fine-tuning from warmup checkpoint")
                        best_checkpoint = self.train_stage_finetune(warmup_checkpoint)
                    else:
                        logger.info("🔄 Resuming: No checkpoints found, starting from scratch")
                        model = self.create_yolo_model()
                        warmup_checkpoint = self.train_stage_warmup(model)
                        best_checkpoint = self.train_stage_finetune(warmup_checkpoint)
                else:
                    # Fresh two-stage training
                    model = self.create_yolo_model()
                    warmup_checkpoint = self.train_stage_warmup(model)
                    best_checkpoint = self.train_stage_finetune(warmup_checkpoint)
            
            # Final validation
            self.validate_final_model(best_checkpoint)
            
            # Training completion
            end_time = datetime.now()
            total_time = end_time - start_time
            
            logger.info("=" * 80)
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"⏱️  Total training time: {total_time}")
            logger.info(f"📁 Experiment directory: {self.experiment_dir}")
            logger.info(f"🏆 Final model: {best_checkpoint}")
            logger.info(f"🎯 {mechanism_info['name']} successfully trained!")
            
        except KeyboardInterrupt:
            logger.info("⚠️  Training interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main function with comprehensive help."""
    parser = argparse.ArgumentParser(
        description="Unified Attention Mechanism Training for YOLOv8 PCB Defect Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ECA-Net (ultra-efficient, 5 parameters)
  python train_attention_unified.py --config configs/config_eca_final.yaml
  
  # Train CBAM (balanced, 1K-10K parameters)  
  python train_attention_unified.py --config configs/config_cbam_neck.yaml
  
  # Train CoordAtt (maximum accuracy, 8-16K parameters)
  python train_attention_unified.py --config configs/config_ca_position7.yaml
  
  # Resume any training
  python train_attention_unified.py --config configs/config_eca_final.yaml --resume

Supported Attention Mechanisms:
  - ECA_Final_Backbone: Ultra-efficient channel attention
  - CBAM_Neck_Only: Dual attention in neck feature fusion  
  - CoordAtt_Position7: Position-aware attention
  - Baseline: No attention mechanism
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from existing checkpoints')
    parser.add_argument('--list-mechanisms', action='store_true',
                        help='List all supported attention mechanisms')
    
    args = parser.parse_args()
    
    # List mechanisms if requested
    if args.list_mechanisms:
        print("🎯 Supported Attention Mechanisms:")
        print("=" * 60)
        for key, info in UnifiedAttentionTrainer.ATTENTION_MECHANISMS.items():
            print(f"Mechanism: {key}")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Efficiency: {info['efficiency']}")
            print(f"  Parameters: {info['parameters']}")
            print(f"  Target Use: {info['target_use']}")
            print()
        return
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Create and run trainer
    trainer = UnifiedAttentionTrainer(str(config_path))
    trainer.run_training(resume=args.resume)


if __name__ == "__main__":
    main()