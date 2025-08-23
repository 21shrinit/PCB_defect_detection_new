#!/usr/bin/env python3
"""
Universal Experiment Runner for PCB Defect Detection
===================================================

This script serves as the single, universal entry point for all experiments.
It is entirely configuration-driven, accepting only a YAML configuration file path.

Features:
- Universal experiment execution (train/val/export)
- Automatic Weights & Biases integration
- Comprehensive metric tracking (P, R, F1, mAP50, mAP50-95, FPS, GFLOPs, Params)
- Support for all model variants (YOLOv8n, YOLOv8s, YOLOv10s)
- Attention mechanism integration
- Loss function customization
- Resolution scaling studies

Usage:
    python run_experiment.py --config experiments/configs/experiment_001.yaml
    python run_experiment.py --config experiments/configs/yolov8s_cbam_focal.yaml
    python run_experiment.py --config experiments/configs/high_res_study.yaml

Author: PCB Defect Detection Team
Date: 2025-01-20
Version: 1.0.0
"""

import os
import sys
import yaml
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# Third-party imports
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# WandB import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('experiment_log.log')
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Universal experiment runner for PCB defect detection experiments.
    
    This class handles all aspects of experiment execution including:
    - Configuration parsing and validation
    - Model instantiation and setup
    - Weights & Biases integration
    - Experiment execution (train/val/export)
    - Comprehensive metric tracking and logging
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the experiment runner with configuration.
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.load_and_validate_config()
        self.experiment_info = self.extract_experiment_info()
        # WandB integration handled via environment variables
        self.model = None
        
        logger.info(f"🚀 ExperimentRunner initialized")
        logger.info(f"📁 Config: {self.config_path}")
        logger.info(f"🔬 Experiment: {self.experiment_info['name']}")
        logger.info(f"🎯 Mode: {self.experiment_info['mode']}")
        
    def load_and_validate_config(self) -> Dict[str, Any]:
        """
        Load and validate the YAML configuration file.
        
        Returns:
            Dict[str, Any]: Parsed and validated configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If required sections are missing
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Validate required sections
            required_sections = ['experiment', 'model', 'data', 'training', 'wandb']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                raise ValueError(f"Missing required config sections: {missing_sections}")
                
            logger.info("✅ Configuration loaded and validated successfully")
            return config
            
        except FileNotFoundError as e:
            logger.error(f"❌ Configuration file error: {e}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error: {e}")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"❌ Configuration validation error: {e}")
            sys.exit(1)
            
    def extract_experiment_info(self) -> Dict[str, Any]:
        """
        Extract key experiment information for logging and tracking.
        
        Returns:
            Dict[str, Any]: Experiment metadata
        """
        exp_config = self.config['experiment']
        
        # Generate dynamic run name if not specified
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{exp_config.get('type', 'experiment')}_{timestamp}"
        
        experiment_info = {
            'name': exp_config.get('name', default_name),
            'type': exp_config.get('type', 'training'),
            'mode': exp_config.get('mode', 'train'),
            'description': exp_config.get('description', 'PCB defect detection experiment'),
            'tags': exp_config.get('tags', []),
            'timestamp': timestamp,
            'config_path': str(self.config_path)
        }
        
        return experiment_info
        
    def setup_wandb_environment(self):
        """
        Set up WandB environment variables for ultralytics integration.
        """
        if not WANDB_AVAILABLE:
            logger.info("WandB not available, skipping integration")
            return
            
        try:
            if 'wandb' not in self.config:
                logger.info("No W&B configuration found, skipping W&B integration")
                return
                
            # Check if WandB is disabled via environment variable
            if os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1', 'yes'):
                logger.info("W&B disabled via environment variable")
                return
                
            wandb_config = self.config['wandb']
            
            # Set environment variables for ultralytics WandB integration
            os.environ['WANDB_PROJECT'] = wandb_config.get('project', 'pcb-defect-detection')
            os.environ['WANDB_NAME'] = self.experiment_info['name']
            os.environ['WANDB_NOTES'] = self.experiment_info['description']
            os.environ['WANDB_DIR'] = wandb_config.get('dir', './wandb_logs')
            
            # Tags need to be joined as comma-separated string
            if self.experiment_info['tags']:
                os.environ['WANDB_TAGS'] = ','.join(self.experiment_info['tags'])
            
            logger.info(f"✅ WandB environment configured for ultralytics")
            logger.info(f"📊 Project: {wandb_config.get('project', 'pcb-defect-detection')}")
            logger.info(f"🏃 Run: {self.experiment_info['name']}")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to setup WandB environment: {e}")
            logger.info("Continuing without W&B logging")
            
    def create_model(self) -> YOLO:
        """
        Create and configure YOLO model based on configuration.
        
        Returns:
            YOLO: Configured YOLO model instance
        """
        try:
            model_config = self.config['model']
            
            # Determine model configuration path
            model_type = model_config.get('type', 'yolov8n')
            custom_config = model_config.get('config_path')
            pretrained = model_config.get('pretrained', True)
            
            logger.info(f"🏗️  Creating {model_type} model...")
            
            if custom_config:
                # Use custom model configuration (with attention mechanisms)
                logger.info(f"📁 Custom config: {custom_config}")
                model = YOLO(custom_config)
                
                # Load pretrained weights if specified
                if pretrained and isinstance(pretrained, str):
                    logger.info(f"⚡ Loading pretrained weights: {pretrained}")
                    model.load(pretrained)
                elif pretrained and pretrained is True:
                    # Load default pretrained weights
                    base_weights = f"{model_type}.pt"
                    if Path(base_weights).exists() or model_type in ['yolov8n', 'yolov8s', 'yolov10s']:
                        logger.info(f"⚡ Loading default pretrained weights: {base_weights}")
                        try:
                            model.load(base_weights)
                        except Exception as e:
                            logger.warning(f"⚠️  Could not load pretrained weights: {e}")
                            logger.info("📝 Continuing with random initialization")
            else:
                # Use standard ultralytics model
                model_path = f"{model_type}.pt" if pretrained else f"{model_type}.yaml"
                logger.info(f"📁 Standard model: {model_path}")
                model = YOLO(model_path)
            
            # Log model information
            if hasattr(model.model, 'model'):
                try:
                    # Count parameters
                    total_params = sum(p.numel() for p in model.model.parameters())
                    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                    
                    logger.info(f"📊 Model Statistics:")
                    logger.info(f"   Total parameters: {total_params:,}")
                    logger.info(f"   Trainable parameters: {trainable_params:,}")
                    logger.info(f"   Model type: {model_type}")
                    
                    # Model stats will be logged automatically by ultralytics WandB integration
                    pass
                        
                except Exception as e:
                    logger.warning(f"⚠️  Could not extract model statistics: {e}")
            
            logger.info("✅ Model created successfully")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create model: {e}")
            sys.exit(1)
            
    def verify_wandb_setup(self):
        """
        Verify WandB environment is properly configured for ultralytics.
        """
        if not WANDB_AVAILABLE:
            return
            
        if 'WANDB_PROJECT' in os.environ:
            logger.info("✅ WandB integration ready - using built-in ultralytics support")
        else:
            logger.info("No WandB configuration detected")
                
    def execute_training(self, model: YOLO) -> Dict[str, Any]:
        """
        Execute model training with specified configuration.
        
        Args:
            model (YOLO): YOLO model instance
            
        Returns:
            Dict[str, Any]: Training results
        """
        try:
            training_config = self.config['training']
            data_config = self.config['data']
            
            # Prepare training arguments
            train_args = {
                'data': data_config['path'],
                'epochs': training_config.get('epochs', 100),
                'imgsz': training_config.get('imgsz', 640),
                'batch': training_config.get('batch', 16),
                'device': training_config.get('device', '0'),
                'workers': training_config.get('workers', 16),
                'optimizer': training_config.get('optimizer', 'AdamW'),
                'lr0': training_config.get('lr0', 0.001),
                'weight_decay': training_config.get('weight_decay', 0.0005),
                'momentum': training_config.get('momentum', 0.937),
                'patience': training_config.get('patience', 0),  # No early stopping
                'save_period': training_config.get('save_period', 10),
                'cache': training_config.get('cache', True),
                'amp': training_config.get('amp', True),
                'project': training_config.get('project', 'experiments'),
                'name': self.experiment_info['name'],
                'exist_ok': True,
                'verbose': True,
                'plots': True,
                'val': training_config.get('validate', True),
                'seed': training_config.get('seed', 42)
            }
            
            # WandB integration is handled automatically by ultralytics
            # when environment variables are set properly
            
            # Add performance optimization flags
            train_args.update({
                'deterministic': False,  # Allow non-deterministic operations for speed
                'single_cls': False,     # Multi-class detection
                'rect': False,           # Rectangular training (can be memory intensive)
                'cos_lr': False,         # Keep linear LR schedule
                'close_mosaic': 10,      # Disable mosaic in last 10 epochs for stability
                'overlap_mask': True,    # Enable mask overlap for better augmentation
                'mask_ratio': 4,         # Mask ratio for segmentation (if applicable)
                'dropout': 0.0,          # No dropout for faster training
                'save_txt': False,       # Don't save txt files for speed
                'save_conf': False,      # Don't save confidence files for speed
            })
            
            # Add loss function configuration if specified
            if 'loss' in training_config:
                loss_config = training_config['loss']
                train_args.update({
                    'box': loss_config.get('box_weight', 7.5),
                    'cls': loss_config.get('cls_weight', 0.5),
                    'dfl': loss_config.get('dfl_weight', 1.5)
                })
                
                # Log loss configuration
                logger.info(f"🎯 Loss Configuration:")
                logger.info(f"   Box weight: {train_args['box']}")
                logger.info(f"   Classification weight: {train_args['cls']}")
                logger.info(f"   DFL weight: {train_args['dfl']}")
                
            # Add augmentation configuration if specified
            if 'augmentation' in training_config:
                aug_config = training_config['augmentation']
                train_args.update({
                    'mosaic': aug_config.get('mosaic', 1.0),
                    'mixup': aug_config.get('mixup', 0.1),
                    'copy_paste': aug_config.get('copy_paste', 0.3),
                    'hsv_h': aug_config.get('hsv_h', 0.015),
                    'hsv_s': aug_config.get('hsv_s', 0.7),
                    'hsv_v': aug_config.get('hsv_v', 0.4),
                    'degrees': aug_config.get('degrees', 0.0),
                    'translate': aug_config.get('translate', 0.1),
                    'scale': aug_config.get('scale', 0.5),
                    'shear': aug_config.get('shear', 0.0),
                    'perspective': aug_config.get('perspective', 0.0),
                    'flipud': aug_config.get('flipud', 0.0),
                    'fliplr': aug_config.get('fliplr', 0.5)
                })
                
            logger.info(f"🚀 Starting training with configuration:")
            for key, value in train_args.items():
                logger.info(f"   {key}: {value}")
                
            # Start training
            start_time = time.time()
            results = model.train(**train_args)
            training_time = time.time() - start_time
            
            # Log training completion
            logger.info(f"✅ Training completed successfully!")
            logger.info(f"⏱️  Total training time: {training_time:.2f} seconds")
            
            # Final metrics are logged automatically by ultralytics WandB integration
            logger.info("Training completed - metrics logged automatically by ultralytics")
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
            
    def execute_validation(self, model: YOLO) -> Dict[str, Any]:
        """
        Execute model validation with specified configuration.
        
        Args:
            model (YOLO): YOLO model instance
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            validation_config = self.config.get('validation', {})
            data_config = self.config['data']
            
            # Prepare validation arguments
            val_args = {
                'data': data_config['path'],
                'imgsz': validation_config.get('imgsz', 640),
                'batch': validation_config.get('batch', 1),
                'device': validation_config.get('device', 'auto'),
                'workers': validation_config.get('workers', 8),
                'conf': validation_config.get('conf_threshold', 0.001),
                'iou': validation_config.get('iou_threshold', 0.6),
                'max_det': validation_config.get('max_detections', 300),
                'split': validation_config.get('split', 'val'),
                'save_json': validation_config.get('save_json', False),
                'save_hybrid': validation_config.get('save_hybrid', False),
                'verbose': True,
                'plots': True
            }
            
            logger.info(f"🔍 Starting validation with configuration:")
            for key, value in val_args.items():
                logger.info(f"   {key}: {value}")
                
            # Start validation
            start_time = time.time()
            results = model.val(**val_args)
            validation_time = time.time() - start_time
            
            # Extract and log metrics
            metrics = self.extract_validation_metrics(results)
            metrics['validation_time_seconds'] = validation_time
            
            logger.info(f"✅ Validation completed successfully!")
            logger.info(f"⏱️  Validation time: {validation_time:.2f} seconds")
            logger.info(f"📊 Validation Results:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"   {key}: {value:.4f}")
                    
            # Validation metrics are logged automatically by ultralytics
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise
            
    def execute_export(self, model: YOLO) -> Dict[str, Any]:
        """
        Execute model export with specified configuration.
        
        Args:
            model (YOLO): YOLO model instance
            
        Returns:
            Dict[str, Any]: Export results
        """
        try:
            export_config = self.config.get('export', {})
            
            # Prepare export arguments
            export_args = {
                'format': export_config.get('format', 'onnx'),
                'imgsz': export_config.get('imgsz', 640),
                'optimize': export_config.get('optimize', True),
                'half': export_config.get('half', False),
                'int8': export_config.get('int8', False),
                'dynamic': export_config.get('dynamic', False),
                'simplify': export_config.get('simplify', True),
                'opset': export_config.get('opset', 11)
            }
            
            logger.info(f"📦 Starting export with configuration:")
            for key, value in export_args.items():
                logger.info(f"   {key}: {value}")
                
            # Start export
            start_time = time.time()
            results = model.export(**export_args)
            export_time = time.time() - start_time
            
            logger.info(f"✅ Export completed successfully!")
            logger.info(f"⏱️  Export time: {export_time:.2f} seconds")
            logger.info(f"📁 Exported model: {results}")
            
            # Export metrics would be logged by ultralytics if supported
            logger.info(f"Export format: {export_args['format']}, Time: {export_time:.2f}s")
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            raise
            
    def extract_validation_metrics(self, results) -> Dict[str, float]:
        """
        Extract comprehensive metrics from validation results.
        
        Args:
            results: Validation results object
            
        Returns:
            Dict[str, float]: Extracted metrics
        """
        metrics = {}
        
        try:
            # Extract box metrics if available
            if hasattr(results, 'box') and results.box is not None:
                box_metrics = results.box
                
                # Primary metrics
                if hasattr(box_metrics, 'map'):
                    metrics['mAP50_95'] = float(box_metrics.map)
                if hasattr(box_metrics, 'map50'):
                    metrics['mAP50'] = float(box_metrics.map50)
                if hasattr(box_metrics, 'map75'):
                    metrics['mAP75'] = float(box_metrics.map75)
                if hasattr(box_metrics, 'mp'):
                    metrics['precision'] = float(box_metrics.mp)
                if hasattr(box_metrics, 'mr'):
                    metrics['recall'] = float(box_metrics.mr)
                if hasattr(box_metrics, 'mf1'):
                    metrics['f1'] = float(box_metrics.mf1)
                    
                # Per-class metrics
                if hasattr(box_metrics, 'ap') and box_metrics.ap is not None:
                    for i, ap in enumerate(box_metrics.ap):
                        metrics[f'class_{i}_mAP50_95'] = float(ap)
                        
                if hasattr(box_metrics, 'ap50') and box_metrics.ap50 is not None:
                    for i, ap50 in enumerate(box_metrics.ap50):
                        metrics[f'class_{i}_mAP50'] = float(ap50)
                        
                # Speed metrics (if available)
                if hasattr(results, 'speed') and results.speed is not None:
                    speed = results.speed
                    if 'preprocess' in speed:
                        metrics['speed_preprocess_ms'] = float(speed['preprocess'])
                    if 'inference' in speed:
                        metrics['speed_inference_ms'] = float(speed['inference'])
                    if 'postprocess' in speed:
                        metrics['speed_postprocess_ms'] = float(speed['postprocess'])
                        
                    # Calculate FPS if inference time is available
                    if 'inference' in speed and speed['inference'] > 0:
                        fps = 1000.0 / speed['inference']  # Convert ms to FPS
                        metrics['fps'] = fps
                        
        except Exception as e:
            logger.warning(f"⚠️  Could not extract all metrics: {e}")
            
        return metrics
        
    def get_file_size_mb(self, file_path: str) -> float:
        """
        Get file size in megabytes.
        
        Args:
            file_path (str): Path to file
            
        Returns:
            float: File size in MB
        """
        try:
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                size_mb = size_bytes / (1024 * 1024)
                return size_mb
        except Exception:
            pass
        return 0.0
        
    def finalize_experiment(self):
        """
        Finalize the experiment and cleanup resources.
        """
        # With ultralytics WandB integration, finalization is handled automatically
        logger.info("✅ Experiment finalization complete")
            
    def run(self):
        """
        Execute the complete experiment pipeline.
        """
        try:
            logger.info("🚀 Starting experiment execution pipeline")
            
            # Setup W&B environment for ultralytics integration
            self.setup_wandb_environment()
            
            # Create model
            self.model = self.create_model()
            
            # Verify W&B setup
            self.verify_wandb_setup()
            
            # Execute based on experiment mode
            mode = self.experiment_info['mode'].lower()
            
            if mode == 'train':
                logger.info("🏋️  Executing training mode")
                results = self.execute_training(self.model)
                
            elif mode == 'val' or mode == 'validate':
                logger.info("🔍 Executing validation mode")
                results = self.execute_validation(self.model)
                
            elif mode == 'export':
                logger.info("📦 Executing export mode")
                results = self.execute_export(self.model)
                
            else:
                raise ValueError(f"Unsupported experiment mode: {mode}")
                
            logger.info(f"✅ Experiment '{self.experiment_info['name']}' completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            # Attempt to finalize even on failure, but don't let it cause additional errors
            try:
                self.finalize_experiment()
            except Exception as finalize_error:
                logger.warning(f"⚠️  Additional error during finalization: {finalize_error}")
            raise
        else:
            # Only run finalization if experiment succeeded
            try:
                self.finalize_experiment()
            except Exception as finalize_error:
                logger.warning(f"⚠️  Finalization failed but experiment succeeded: {finalize_error}")
                # Don't re-raise finalization errors for successful experiments


def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Universal Experiment Runner for PCB Defect Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run training experiment
  python run_experiment.py --config experiments/configs/yolov8n_baseline_train.yaml
  
  # Run validation experiment
  python run_experiment.py --config experiments/configs/yolov8s_cbam_validate.yaml
  
  # Run export experiment
  python run_experiment.py --config experiments/configs/yolov10s_export.yaml
  
  # High-resolution training study
  python run_experiment.py --config experiments/configs/high_res_1024px_study.yaml
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
        
    try:
        # Create and run experiment
        runner = ExperimentRunner(str(config_path))
        runner.run()
        
    except KeyboardInterrupt:
        logger.info("⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Experiment execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()