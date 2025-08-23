#!/usr/bin/env python3
"""
Production-Ready YOLOv8 Two-Stage Training Script
================================================

This script automates YOLOv8 two-stage training to prevent destructive learning dynamics
when integrating attention mechanisms. The training is fully configuration-driven via YAML.

Two-Stage Process:
1. Stage 1 (Warm-up): Train with frozen backbone layers to initialize attention modules
2. Stage 2 (Fine-tune): Resume from best checkpoint with all layers trainable

Key Features:
- Fully configuration-driven via YAML
- Automatic checkpoint detection and resumption
- Robust error handling and validation
- Continuous experiment logging (W&B, TensorBoard)
- Production-ready code quality

Author: MLOps Engineering Team
Version: 2.0.0
Created: 2025-01-20
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import YOLOv8
try:
    from ultralytics import YOLO
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


class TwoStageTrainer:
    """
    Production-ready two-stage YOLOv8 trainer to prevent destructive learning dynamics.
    
    This trainer implements the critical two-stage approach:
    1. Stage 1: Warm-up with frozen backbone layers
    2. Stage 2: Fine-tuning with all layers trainable
    
    All configuration is loaded from YAML file with comprehensive validation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the two-stage trainer with configuration validation.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_and_validate_config()
        
        # Initialize tracking variables
        self.stage1_results = None
        self.stage2_results = None
        self.stage1_model = None
        self.stage2_model = None
        
        logger.info("🚀 Two-Stage Trainer Initialized")
        logger.info(f"📋 Configuration: {self.config_path}")
        
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """
        Load and validate the YAML configuration file.
        
        Returns:
            Dict[str, Any]: Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required configuration keys are missing
            yaml.YAMLError: If YAML parsing fails
        """
        # Check if config file exists
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            # Load YAML configuration
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")
        
        # Define required configuration keys
        required_keys = {
            'model': ['config_path'],
            'data': ['config_path'],
            'training_strategy': ['warmup', 'finetune'],
            'environment': ['imgsz', 'batch_size']
        }
        
        # Validate required sections exist
        for section in required_keys.keys():
            if section not in config:
                raise ValueError(f"Missing required configuration section: '{section}'")
        
        # Validate required keys within sections
        for section, keys in required_keys.items():
            for key in keys:
                if key not in config[section]:
                    raise ValueError(f"Missing required key '{key}' in section '{section}'")
        
        # Validate warmup section
        warmup_required = ['epochs', 'freeze_layers', 'learning_rate']
        for key in warmup_required:
            if key not in config['training_strategy']['warmup']:
                raise ValueError(f"Missing required warmup key: '{key}'")
        
        # Validate finetune section
        finetune_required = ['epochs', 'learning_rate']
        for key in finetune_required:
            if key not in config['training_strategy']['finetune']:
                raise ValueError(f"Missing required finetune key: '{key}'")
        
        # Validate file paths exist
        model_config = Path(config['model']['config_path'])
        if not model_config.exists():
            raise FileNotFoundError(f"Model configuration not found: {model_config}")
        
        data_config = Path(config['data']['config_path'])
        if not data_config.exists():
            raise FileNotFoundError(f"Data configuration not found: {data_config}")
        
        logger.info("✅ Configuration validation completed")
        return config
    
    def _prepare_stage_config(self, stage: str) -> Dict[str, Any]:
        """
        Prepare training configuration for a specific stage.
        
        Args:
            stage (str): Training stage ('warmup' or 'finetune')
            
        Returns:
            Dict[str, Any]: Training configuration dictionary
        """
        stage_config = self.config['training_strategy'][stage]
        env_config = self.config['environment']
        
        # Base training configuration
        train_config = {
            # Dataset and model parameters
            'data': self.config['data']['config_path'],
            'epochs': stage_config['epochs'],
            'imgsz': env_config['imgsz'],
            'batch': env_config['batch_size'],
            
            # Learning parameters
            'lr0': stage_config['learning_rate'],
            'momentum': stage_config.get('momentum', 0.937),
            'weight_decay': stage_config.get('weight_decay', 0.0005),
            
            # Hardware configuration
            'device': env_config.get('device', '0'),
            'workers': env_config.get('workers', 8),
            'amp': env_config.get('mixed_precision', True),
            
            # Training behavior
            'patience': stage_config.get('patience', 30),
            'save_period': stage_config.get('save_period', 10),
            'save': True,
            'exist_ok': True,
            'verbose': True,
            
            # Optimizer
            'optimizer': self.config.get('optimizer', {}).get('type', 'SGD'),
            
            # Advanced settings
            'cache': env_config.get('cache_images', False),
            'deterministic': env_config.get('deterministic', True),
            'seed': env_config.get('seed', 42),
        }
        
        # Add stage-specific parameters
        if 'project' in self.config:
            train_config['project'] = self.config['project'].get('name', 'runs/train')
            train_config['name'] = f"{stage}_{self.config['project'].get('experiment_name', 'experiment')}"
        
        # Add optional parameters if present in config
        optional_params = ['warmup_epochs', 'warmup_momentum', 'warmup_bias_lr', 'lrf']
        for param in optional_params:
            if param in stage_config:
                train_config[param] = stage_config[param]
        
        return train_config
    
    def stage_1_warmup(self) -> Any:
        """
        Execute Stage 1: Warm-up training with frozen backbone layers.
        
        This critical stage prevents destructive learning dynamics by freezing
        pretrained backbone weights while training new attention mechanisms.
        
        Returns:
            Training results from Stage 1
            
        Raises:
            Exception: If Stage 1 training fails
        """
        logger.info("=" * 80)
        logger.info("🔥 STAGE 1: WARM-UP TRAINING WITH FROZEN BACKBONE")
        logger.info("=" * 80)
        
        warmup_config = self.config['training_strategy']['warmup']
        freeze_layers = warmup_config['freeze_layers']
        
        logger.info(f"📋 Stage: Warm-up Phase")
        logger.info(f"🎯 Epochs: {warmup_config['epochs']}")
        logger.info(f"❄️  Frozen Layers: {freeze_layers}")
        logger.info(f"📊 Learning Rate: {warmup_config['learning_rate']}")
        logger.info(f"📖 Purpose: Initialize attention layers while preserving backbone weights")
        
        try:
            # Load model from configuration
            model_config_path = self.config['model']['config_path']
            logger.info(f"🤖 Loading model from: {model_config_path}")
            
            self.stage1_model = YOLO(model_config_path)
            logger.info("✅ Model loaded successfully")
            
            # Prepare training configuration
            train_config = self._prepare_stage_config('warmup')
            
            # Add freeze parameter for Stage 1
            train_config['freeze'] = freeze_layers
            
            logger.info("🚀 Starting Stage 1 warm-up training...")
            logger.info(f"❄️  Freezing first {freeze_layers} layers of backbone")
            logger.info("⏳ This may take some time...")
            
            # Execute Stage 1 training
            self.stage1_results = self.stage1_model.train(**train_config)
            
            # Validate Stage 1 completion
            if self.stage1_results is None:
                raise RuntimeError("Stage 1 training returned no results")
            
            # Log Stage 1 completion
            stage1_map = self.stage1_results.results_dict.get('metrics/mAP50(B)', 'N/A')
            logger.info("✅ Stage 1 (Warm-up) completed successfully!")
            logger.info(f"📊 Warm-up mAP@0.5: {stage1_map}")
            logger.info(f"📁 Results saved to: {self.stage1_results.save_dir}")
            
            return self.stage1_results
            
        except Exception as e:
            logger.error(f"❌ Stage 1 (Warm-up) training failed: {str(e)}")
            raise
    
    def _find_best_checkpoint(self) -> Path:
        """
        Automatically locate the best checkpoint from Stage 1 training.
        
        Returns:
            Path: Path to the best.pt checkpoint file
            
        Raises:
            FileNotFoundError: If best.pt checkpoint cannot be found
        """
        if self.stage1_results is None:
            raise RuntimeError("Stage 1 must be completed before finding checkpoint")
        
        # Primary location: Stage 1 results directory
        best_checkpoint = self.stage1_results.save_dir / 'weights' / 'best.pt'
        
        if best_checkpoint.exists():
            logger.info(f"📂 Found Stage 1 checkpoint: {best_checkpoint}")
            return best_checkpoint
        
        # Fallback: Search in common locations
        possible_locations = [
            self.stage1_results.save_dir / 'best.pt',
            Path('runs/train') / 'exp' / 'weights' / 'best.pt',
            Path('runs/train') / 'exp1' / 'weights' / 'best.pt',
        ]
        
        for location in possible_locations:
            if location.exists():
                logger.info(f"📂 Found Stage 1 checkpoint (fallback): {location}")
                return location
        
        raise FileNotFoundError("Could not locate best.pt checkpoint from Stage 1 training")
    
    def stage_2_finetune(self) -> Any:
        """
        Execute Stage 2: Fine-tuning with all layers trainable.
        
        This stage loads the best checkpoint from Stage 1 and continues training
        with reduced learning rate and all layers unfrozen.
        
        Returns:
            Training results from Stage 2
            
        Raises:
            Exception: If Stage 2 training fails
        """
        logger.info("=" * 80)
        logger.info("🎯 STAGE 2: FINE-TUNING WITH ALL LAYERS TRAINABLE")
        logger.info("=" * 80)
        
        if self.stage1_results is None:
            raise RuntimeError("Stage 1 must be completed before Stage 2")
        
        finetune_config = self.config['training_strategy']['finetune']
        
        logger.info(f"📋 Stage: Fine-tuning Phase")
        logger.info(f"🎯 Epochs: {finetune_config['epochs']}")
        logger.info(f"🔥 Frozen Layers: 0 (all layers trainable)")
        logger.info(f"📊 Learning Rate: {finetune_config['learning_rate']} (reduced)")
        logger.info(f"📖 Purpose: Fine-tune entire network with stable learning rate")
        
        try:
            # Find and load the best checkpoint from Stage 1
            best_checkpoint = self._find_best_checkpoint()
            
            logger.info(f"📂 Loading Stage 1 checkpoint: {best_checkpoint}")
            self.stage2_model = YOLO(str(best_checkpoint))
            logger.info("✅ Checkpoint loaded successfully")
            
            # Prepare training configuration for Stage 2
            train_config = self._prepare_stage_config('finetune')
            
            # Note: Not using resume=True since we're starting fresh fine-tuning from Stage 1 checkpoint
            
            logger.info("🚀 Starting Stage 2 fine-tuning...")
            logger.info("🔥 All layers now trainable")
            logger.info("📈 Using reduced learning rate for stability")
            logger.info("⏳ This may take longer than Stage 1...")
            
            # Execute Stage 2 training
            self.stage2_results = self.stage2_model.train(**train_config)
            
            # Validate Stage 2 completion
            if self.stage2_results is None:
                raise RuntimeError("Stage 2 training returned no results")
            
            # Log Stage 2 completion
            stage2_map = self.stage2_results.results_dict.get('metrics/mAP50(B)', 'N/A')
            stage1_map = self.stage1_results.results_dict.get('metrics/mAP50(B)', 0)
            
            try:
                improvement = float(stage2_map) - float(stage1_map)
                improvement_str = f"{improvement:+.4f}"
            except (ValueError, TypeError):
                improvement_str = "N/A"
            
            logger.info("✅ Stage 2 (Fine-tuning) completed successfully!")
            logger.info(f"📊 Final mAP@0.5: {stage2_map}")
            logger.info(f"📈 Stage 1→2 Improvement: {improvement_str}")
            logger.info(f"📁 Results saved to: {self.stage2_results.save_dir}")
            
            return self.stage2_results
            
        except Exception as e:
            logger.error(f"❌ Stage 2 (Fine-tuning) training failed: {str(e)}")
            raise
    
    def run_complete_training(self) -> tuple:
        """
        Execute the complete two-stage training pipeline.
        
        Returns:
            tuple: (stage1_results, stage2_results)
            
        Raises:
            Exception: If any stage fails
        """
        start_time = datetime.now()
        
        logger.info("🚀 STARTING COMPLETE TWO-STAGE TRAINING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"⏰ Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📋 Configuration: {self.config_path}")
        
        # Display training strategy
        warmup_config = self.config['training_strategy']['warmup']
        finetune_config = self.config['training_strategy']['finetune']
        
        logger.info("📋 TRAINING STRATEGY:")
        logger.info(f"   Stage 1: {warmup_config['epochs']} epochs, "
                   f"{warmup_config['freeze_layers']} frozen layers, "
                   f"LR={warmup_config['learning_rate']}")
        logger.info(f"   Stage 2: {finetune_config['epochs']} epochs, "
                   f"0 frozen layers, "
                   f"LR={finetune_config['learning_rate']}")
        logger.info("=" * 80)
        
        try:
            # Execute Stage 1: Warm-up
            stage1_results = self.stage_1_warmup()
            
            # Execute Stage 2: Fine-tuning
            stage2_results = self.stage_2_finetune()
            
            # Calculate total training time
            end_time = datetime.now()
            total_time = end_time - start_time
            
            # Final success message with results
            logger.info("\n" + "=" * 80)
            logger.info("🎉 COMPLETE TWO-STAGE TRAINING FINISHED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"⏰ Total Training Time: {total_time}")
            logger.info(f"📊 Stage 1 Results: {stage1_results.save_dir}")
            logger.info(f"📊 Stage 2 Results: {stage2_results.save_dir}")
            
            # Display performance summary
            try:
                stage1_map = float(stage1_results.results_dict.get('metrics/mAP50(B)', 0))
                stage2_map = float(stage2_results.results_dict.get('metrics/mAP50(B)', 0))
                improvement = stage2_map - stage1_map
                
                logger.info("📈 PERFORMANCE SUMMARY:")
                logger.info(f"   Stage 1 mAP@0.5: {stage1_map:.4f}")
                logger.info(f"   Stage 2 mAP@0.5: {stage2_map:.4f}")
                logger.info(f"   Overall Improvement: {improvement:+.4f}")
            except (ValueError, TypeError):
                logger.info("📈 Performance metrics available in results directories")
            
            logger.info("=" * 80)
            
            return stage1_results, stage2_results
            
        except Exception as e:
            logger.error(f"❌ Two-stage training pipeline failed: {str(e)}")
            raise


def main():
    """
    Main function with command-line argument parsing and execution.
    """
    parser = argparse.ArgumentParser(
        description="Production-Ready YOLOv8 Two-Stage Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config.yaml
  python train_unified.py
  
  # Use custom configuration
  python train_unified.py --config my_config.yaml
  
  # Use specific attention mechanism config
  python train_unified.py --config configs/config_cbam.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without training'
    )
    
    args = parser.parse_args()
    
    # Header
    print("=" * 80)
    print("🚀 YOLOv8 Two-Stage Training Script v2.0")
    print("   Production-Ready Configuration-Driven Training")
    print("=" * 80)
    print(f"📋 Configuration: {args.config}")
    print("=" * 80)
    
    try:
        # Initialize trainer
        trainer = TwoStageTrainer(args.config)
        
        if args.validate_only:
            print("✅ Configuration validation completed successfully!")
            print("🎯 Ready for training. Remove --validate-only to start.")
            return
        
        # Execute complete training pipeline
        stage1_results, stage2_results = trainer.run_complete_training()
        
        # Final success message
        print("\n🎉 TWO-STAGE TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Access your results at:")
        print(f"   Stage 1: {stage1_results.save_dir}")
        print(f"   Stage 2: {stage2_results.save_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        logger.error(f"Detailed error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()