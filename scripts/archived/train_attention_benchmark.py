#!/usr/bin/env python3
"""
Production-Ready Two-Stage YOLOv8 Attention Mechanism Training Framework
=========================================================================

This framework implements a systematic approach to training YOLOv8 models with custom
attention mechanisms using a two-stage strategy that prevents destructive learning dynamics.

Key Features:
- Two-stage training: warmup with frozen backbone + fine-tuning
- Comprehensive configuration management via single YAML file
- Robust checkpoint management and recovery
- Production-ready logging and monitoring
- Systematic benchmarking capabilities
- MLOps best practices implementation

Author: MLOps Engineering Team
Version: 2.0.0
Created: 2025-01-20
"""

import os
import sys
import yaml
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import shutil
import json

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

# Configure comprehensive logging
def setup_logging(log_dir: Path, experiment_name: str) -> logging.Logger:
    """Setup comprehensive logging for the training pipeline."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_dir / f'{experiment_name}_training.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class AttentionTrainingPipeline:
    """
    Production-ready two-stage training pipeline for attention-enhanced YOLOv8 models.
    
    This pipeline implements the critical two-stage training strategy:
    1. Stage 1 (Warmup): Train new attention layers with frozen pretrained backbone
    2. Stage 2 (Fine-tune): Train entire network with reduced learning rate
    
    This prevents "destructive learning dynamics" where randomly initialized attention
    layers damage valuable pretrained backbone representations.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the attention training pipeline.
        
        Args:
            config_path (str): Path to the master YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_and_validate_config()
        
        # Setup experiment directory structure
        self.experiment_dir = self._setup_experiment_directory()
        
        # Setup logging
        self.logger = setup_logging(
            self.experiment_dir / 'logs',
            self.config['project']['experiment_name']
        )
        
        # Initialize tracking variables
        self.warmup_results = None
        self.finetune_results = None
        self.model = None
        
        self.logger.info("🚀 Attention Training Pipeline Initialized")
        self.logger.info(f"📁 Experiment Directory: {self.experiment_dir}")
        
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate the configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate critical configuration sections
        required_sections = ['project', 'model', 'data', 'training_strategy', 'environment']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate two-stage strategy
        if 'warmup' not in config['training_strategy']:
            raise ValueError("Missing warmup configuration in training_strategy")
        if 'finetune' not in config['training_strategy']:
            raise ValueError("Missing finetune configuration in training_strategy")
            
        return config
    
    def _setup_experiment_directory(self) -> Path:
        """Setup the experiment directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config['project']['experiment_name']
        
        # Create main experiment directory
        experiment_dir = Path(f"runs/experiments/{exp_name}_{timestamp}")
        
        # Create subdirectories
        subdirs = ['logs', 'checkpoints', 'results', 'configs', 'models', 'tensorboard']
        for subdir in subdirs:
            (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Save configuration for reproducibility
        config_backup = experiment_dir / 'configs' / 'config.yaml'
        shutil.copy2(self.config_path, config_backup)
        
        # Save environment information
        env_info = {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'timestamp': timestamp,
            'config_path': str(self.config_path.absolute()),
            'experiment_directory': str(experiment_dir.absolute())
        }
        
        with open(experiment_dir / 'configs' / 'environment.json', 'w') as f:
            json.dump(env_info, f, indent=2)
        
        return experiment_dir
    
    def _validate_environment(self) -> None:
        """Validate the training environment."""
        self.logger.info("🔍 Validating training environment...")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            self.logger.error("❌ CUDA not available! Attention mechanisms require GPU training.")
            raise RuntimeError("CUDA-capable GPU required for attention mechanism training")
        
        # Log GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.logger.info(f"🎯 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Validate model configuration file
        model_config_path = Path(self.config['model']['config_path'])
        if not model_config_path.exists():
            raise FileNotFoundError(f"Model configuration not found: {model_config_path}")
        
        # Validate dataset configuration file  
        data_config_path = Path(self.config['data']['config_path'])
        if not data_config_path.exists():
            raise FileNotFoundError(f"Dataset configuration not found: {data_config_path}")
        
        self.logger.info("✅ Environment validation completed")
    
    def _prepare_training_config(self, stage: str) -> Dict[str, Any]:
        """
        Prepare training configuration for a specific stage.
        
        Args:
            stage (str): Training stage ('warmup' or 'finetune')
            
        Returns:
            Dict[str, Any]: Training configuration for the stage
        """
        stage_config = self.config['training_strategy'][stage]
        env_config = self.config['environment']
        
        # Base configuration
        training_config = {
            # Dataset and model
            'data': self.config['data']['config_path'],
            'epochs': stage_config['epochs'],
            'imgsz': env_config['imgsz'],
            'batch': env_config['batch_size'],
            
            # Hardware configuration
            'device': env_config['device'],
            'workers': env_config.get('workers', 8),
            'amp': env_config.get('mixed_precision', True),
            
            # Learning parameters
            'lr0': stage_config['learning_rate'],
            'momentum': stage_config.get('momentum', 0.937),
            'weight_decay': stage_config.get('weight_decay', 0.0005),
            
            # Training behavior
            'patience': stage_config['patience'],
            'save_period': stage_config['save_period'],
            'save': True,
            'exist_ok': True,
            'verbose': True,
            
            # Optimizer configuration
            'optimizer': self.config.get('optimizer', {}).get('type', 'SGD'),
            
            # Project organization
            'project': str(self.experiment_dir / 'runs'),
            'name': f"{stage}_{stage_config['name']}",
            
            # Advanced settings
            'cache': env_config.get('cache_images', False),
            'deterministic': env_config.get('deterministic', True),
            'seed': env_config.get('seed', 42),
        }
        
        # Add stage-specific configurations
        if stage == 'warmup':
            # Warmup-specific settings
            training_config['freeze'] = [i for i in range(stage_config['freeze_layers'])]
            training_config['warmup_epochs'] = 3.0
            training_config['warmup_momentum'] = 0.8
            training_config['warmup_bias_lr'] = 0.1
            
        elif stage == 'finetune':
            # Fine-tuning specific settings (no frozen layers)
            training_config['warmup_epochs'] = 1.0  # Minimal warmup for fine-tuning
            training_config['lrf'] = 0.01  # Final learning rate factor
        
        return training_config
    
    def freeze_backbone_layers(self, model: YOLO, num_layers: int) -> None:
        """
        Freeze the first N backbone layers to preserve pretrained features.
        
        Args:
            model (YOLO): The YOLO model
            num_layers (int): Number of layers to freeze
        """
        if num_layers == 0:
            self.logger.info("🔥 No layers frozen - full model training")
            return
        
        frozen_count = 0
        for i, (name, param) in enumerate(model.model.named_parameters()):
            if i < num_layers:
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
        
        self.logger.info(f"❄️  Frozen {frozen_count} backbone parameters")
        self.logger.info(f"🔥 Trainable parameters: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}")
    
    def stage_1_warmup_training(self) -> Any:
        """
        Stage 1: Warmup training with frozen backbone layers.
        
        This critical phase allows new attention layers to learn meaningful 
        representations without damaging pretrained backbone features.
        
        Returns:
            Training results from the warmup phase
        """
        self.logger.info("=" * 80)
        self.logger.info("🔥 STAGE 1: WARMUP TRAINING WITH FROZEN BACKBONE")
        self.logger.info("=" * 80)
        
        warmup_config = self.config['training_strategy']['warmup']
        self.logger.info(f"📋 Stage: {warmup_config['name']}")
        self.logger.info(f"📖 Description: {warmup_config['description']}")
        self.logger.info(f"🎯 Epochs: {warmup_config['epochs']}")
        self.logger.info(f"❄️  Frozen Layers: {warmup_config['freeze_layers']}")
        self.logger.info(f"📊 Learning Rate: {warmup_config['learning_rate']}")
        
        try:
            # Initialize model
            self.logger.info("🤖 Loading model with attention mechanisms...")
            model_config_path = self.config['model']['config_path']
            self.model = YOLO(model_config_path)
            
            # Log model information
            self.logger.info("📊 Model Architecture Summary:")
            model_info(self.model.model)
            
            # Prepare training configuration
            training_config = self._prepare_training_config('warmup')
            
            self.logger.info("🚀 Starting warmup training...")
            self.logger.info(f"📁 Results will be saved to: {training_config['project']}/{training_config['name']}")
            
            # Execute warmup training
            self.warmup_results = self.model.train(**training_config)
            
            # Save checkpoint for stage 2
            warmup_checkpoint = self.experiment_dir / 'checkpoints' / 'warmup_complete.pt'
            best_warmup = self.warmup_results.save_dir / 'weights' / 'best.pt'
            if best_warmup.exists():
                shutil.copy2(best_warmup, warmup_checkpoint)
                self.logger.info(f"💾 Warmup checkpoint saved: {warmup_checkpoint}")
            
            self.logger.info("✅ Stage 1 (Warmup) completed successfully!")
            self.logger.info(f"📊 Best mAP@0.5: {self.warmup_results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
            
            return self.warmup_results
            
        except Exception as e:
            self.logger.error(f"❌ Stage 1 (Warmup) failed: {str(e)}")
            self.logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            raise
    
    def stage_2_finetuning(self) -> Any:
        """
        Stage 2: Fine-tuning with unfrozen layers and reduced learning rate.
        
        This phase fine-tunes the entire network including attention mechanisms
        with a carefully reduced learning rate to prevent overfitting.
        
        Returns:
            Training results from the fine-tuning phase
        """
        self.logger.info("=" * 80)
        self.logger.info("🎯 STAGE 2: FINE-TUNING WITH REDUCED LEARNING RATE")
        self.logger.info("=" * 80)
        
        if self.warmup_results is None:
            raise RuntimeError("Stage 1 (Warmup) must be completed before Stage 2")
        
        finetune_config = self.config['training_strategy']['finetune']
        self.logger.info(f"📋 Stage: {finetune_config['name']}")
        self.logger.info(f"📖 Description: {finetune_config['description']}")
        self.logger.info(f"🎯 Epochs: {finetune_config['epochs']}")
        self.logger.info(f"🔥 Frozen Layers: {finetune_config['freeze_layers']} (all unfrozen)")
        self.logger.info(f"📊 Learning Rate: {finetune_config['learning_rate']} (reduced)")
        
        try:
            # Load the best warmup checkpoint
            warmup_checkpoint = self.experiment_dir / 'checkpoints' / 'warmup_complete.pt'
            if warmup_checkpoint.exists():
                self.logger.info(f"📂 Loading warmup checkpoint: {warmup_checkpoint}")
                self.model = YOLO(warmup_checkpoint)
            else:
                # Fallback to warmup results
                best_warmup = self.warmup_results.save_dir / 'weights' / 'best.pt'
                self.logger.info(f"📂 Loading from warmup results: {best_warmup}")
                self.model = YOLO(best_warmup)
            
            # Prepare fine-tuning configuration
            training_config = self._prepare_training_config('finetune')
            
            self.logger.info("🚀 Starting fine-tuning phase...")
            self.logger.info(f"📁 Results will be saved to: {training_config['project']}/{training_config['name']}")
            
            # Execute fine-tuning
            self.finetune_results = self.model.train(**training_config)
            
            # Save final checkpoint
            final_checkpoint = self.experiment_dir / 'checkpoints' / 'final_model.pt'
            best_finetune = self.finetune_results.save_dir / 'weights' / 'best.pt'
            if best_finetune.exists():
                shutil.copy2(best_finetune, final_checkpoint)
                self.logger.info(f"💾 Final checkpoint saved: {final_checkpoint}")
            
            self.logger.info("✅ Stage 2 (Fine-tuning) completed successfully!")
            self.logger.info(f"📊 Final mAP@0.5: {self.finetune_results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
            
            return self.finetune_results
            
        except Exception as e:
            self.logger.error(f"❌ Stage 2 (Fine-tuning) failed: {str(e)}")
            self.logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            raise
    
    def run_complete_pipeline(self) -> Tuple[Any, Any]:
        """
        Execute the complete two-stage training pipeline.
        
        Returns:
            Tuple[Any, Any]: (warmup_results, finetune_results)
        """
        start_time = datetime.now()
        
        self.logger.info("🚀 Starting Complete Two-Stage Attention Training Pipeline")
        self.logger.info(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Validate environment
            self._validate_environment()
            
            # Execute Stage 1: Warmup
            warmup_results = self.stage_1_warmup_training()
            
            # Execute Stage 2: Fine-tuning  
            finetune_results = self.stage_2_finetuning()
            
            # Calculate total training time
            end_time = datetime.now()
            total_time = end_time - start_time
            
            # Generate final summary
            self._generate_training_summary(warmup_results, finetune_results, total_time)
            
            self.logger.info("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
            self.logger.info(f"⏰ Total Training Time: {total_time}")
            
            return warmup_results, finetune_results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            self.logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise
    
    def _generate_training_summary(self, warmup_results: Any, finetune_results: Any, 
                                 total_time: Any) -> None:
        """Generate a comprehensive training summary."""
        summary = {
            'experiment_info': {
                'name': self.config['project']['experiment_name'],
                'description': self.config['project']['description'],
                'model_architecture': self.config['model']['architecture'],
                'attention_mechanism': self.config['model']['attention_mechanism'],
                'total_training_time': str(total_time),
                'completed_at': datetime.now().isoformat()
            },
            'stage_1_warmup': {
                'epochs': self.config['training_strategy']['warmup']['epochs'],
                'frozen_layers': self.config['training_strategy']['warmup']['freeze_layers'],
                'learning_rate': self.config['training_strategy']['warmup']['learning_rate'],
                'final_map50': warmup_results.results_dict.get('metrics/mAP50(B)', 'N/A'),
                'save_dir': str(warmup_results.save_dir)
            },
            'stage_2_finetune': {
                'epochs': self.config['training_strategy']['finetune']['epochs'], 
                'learning_rate': self.config['training_strategy']['finetune']['learning_rate'],
                'final_map50': finetune_results.results_dict.get('metrics/mAP50(B)', 'N/A'),
                'save_dir': str(finetune_results.save_dir)
            }
        }
        
        # Save summary
        summary_path = self.experiment_dir / 'results' / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"📋 Training summary saved: {summary_path}")
    
    def export_final_model(self, formats: List[str] = ['onnx', 'torchscript']) -> None:
        """
        Export the final trained model to specified formats.
        
        Args:
            formats (List[str]): List of export formats
        """
        if self.finetune_results is None:
            self.logger.warning("⚠️  No fine-tuning results available for export")
            return
        
        self.logger.info(f"📦 Exporting final model to formats: {formats}")
        
        try:
            final_model_path = self.finetune_results.save_dir / 'weights' / 'best.pt'
            model = YOLO(final_model_path)
            
            export_dir = self.experiment_dir / 'models'
            
            for fmt in formats:
                self.logger.info(f"📤 Exporting to {fmt.upper()}...")
                exported_path = model.export(format=fmt, optimize=True, simplify=True)
                
                # Copy to experiment directory
                exported_file = Path(exported_path)
                target_path = export_dir / exported_file.name
                shutil.copy2(exported_file, target_path)
                self.logger.info(f"✅ {fmt.upper()} export saved: {target_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Model export failed: {str(e)}")


def main():
    """Main function to run the attention training pipeline."""
    parser = argparse.ArgumentParser(
        description="Production-Ready Two-Stage YOLOv8 Attention Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config.yaml
  python train_attention_benchmark.py
  
  # Run with custom configuration
  python train_attention_benchmark.py --config custom_config.yaml
  
  # Run with model export
  python train_attention_benchmark.py --export onnx torchscript
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the master configuration YAML file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--export',
        nargs='*',
        default=['onnx'],
        help='Export formats for the final model (default: onnx)'
    )
    
    parser.add_argument(
        '--stage',
        choices=['warmup', 'finetune', 'complete'],
        default='complete',
        help='Training stage to run (default: complete)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 YOLOv8 Attention Mechanism Training Pipeline v2.0")
    print("   Production-Ready Two-Stage Training Framework")
    print("=" * 80)
    print(f"📋 Configuration: {args.config}")
    print(f"📦 Export Formats: {args.export}")
    print(f"🎯 Stage: {args.stage}")
    print("=" * 80)
    
    try:
        # Initialize pipeline
        pipeline = AttentionTrainingPipeline(args.config)
        
        if args.stage == 'complete':
            # Run complete pipeline
            warmup_results, finetune_results = pipeline.run_complete_pipeline()
            
        elif args.stage == 'warmup':
            # Run only warmup stage
            warmup_results = pipeline.stage_1_warmup_training()
            
        elif args.stage == 'finetune':
            # Run only fine-tuning stage (requires warmup checkpoint)
            finetune_results = pipeline.stage_2_finetuning()
        
        # Export final model if requested
        if args.export and args.stage in ['complete', 'finetune']:
            pipeline.export_final_model(args.export)
        
        print("\n" + "=" * 80)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 Experiment Directory: {pipeline.experiment_dir}")
        print(f"📋 Training Summary: {pipeline.experiment_dir / 'results' / 'training_summary.json'}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        print(f"🔍 Check logs for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()