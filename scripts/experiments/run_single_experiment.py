#!/usr/bin/env python3
"""
Single Experiment Runner for PCB Defect Detection
=================================================

This script runs a complete experiment (training + validation + testing) from a single YAML config file.
Simplified workflow without phase management - just run individual experiments.

Features:
- Complete experiment in one run: Train → Validate → Test
- Comprehensive test set evaluation
- Automatic WandB logging for all phases
- Model performance analysis
- Export capabilities (optional)
- Clean, trackable results

Usage:
    # From project root:
    python scripts/experiments/run_single_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml
    python scripts/experiments/run_single_experiment.py --config experiments/configs/custom_config.yaml --test_only

Author: PCB Defect Detection Team
Date: 2025-01-21
Version: 2.0.0 (Simplified)
"""

import os
import sys
import yaml
import argparse
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# Add project root to path for imports FIRST
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up from scripts/experiments/
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure ultralytics uses local version with custom modules
ultralytics_path = PROJECT_ROOT / "ultralytics"
if ultralytics_path.exists():
    print(f"Using local ultralytics with custom modules: {ultralytics_path}")

# Third-party imports (after path setup)
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
def setup_logging(experiment_name: str):
    """Setup logging for the experiment."""
    log_filename = f"experiment_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filename)
        ]
    )
    return logging.getLogger(__name__)


class SingleExperimentRunner:
    """
    Complete experiment runner for individual PCB defect detection experiments.
    
    Handles training, validation, and testing in a single streamlined workflow.
    """
    
    def __init__(self, config_path: str):
        """Initialize the experiment runner."""
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.experiment_info = self.extract_experiment_info()
        
        # Setup logging
        self.logger = setup_logging(self.experiment_info['name'])
        
        # Initialize experiment tracking
        self.results = {
            'experiment_info': self.experiment_info,
            'training_results': None,
            'validation_results': None,
            'test_results': None,
            'model_path': None,
            'export_results': None,
            'timing': {},
            'metrics_summary': {}
        }
        
        self.logger.info("SingleExperimentRunner initialized")
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Experiment: {self.experiment_info['name']}")
        
    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Validate required sections
            required_sections = ['experiment', 'model', 'data', 'training']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                raise ValueError(f"Missing required config sections: {missing_sections}")
                
            return config
            
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            sys.exit(1)
            
    def extract_experiment_info(self) -> Dict[str, Any]:
        """Extract experiment metadata."""
        exp_config = self.config['experiment']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            'name': exp_config.get('name', f'experiment_{timestamp}'),
            'type': exp_config.get('type', 'detection'),
            'description': exp_config.get('description', 'PCB defect detection experiment'),
            'tags': exp_config.get('tags', []),
            'timestamp': timestamp,
            'config_path': str(self.config_path),
            'model_type': self.config['model'].get('type', 'yolov8n'),
            'attention_mechanism': self.config['model'].get('attention_mechanism', 'none'),
            'image_size': self.config['training'].get('imgsz', 640)
        }
        
    def setup_wandb(self):
        """Setup WandB integration."""
        if not WANDB_AVAILABLE:
            self.logger.info("WandB not available, skipping integration")
            return
            
        try:
            if 'wandb' not in self.config:
                self.logger.info("No W&B config found, skipping integration")
                return
                
            wandb_config = self.config['wandb']
            
            # Set environment variables for ultralytics integration
            # Note: project name is now handled via training args, not env vars
            os.environ['WANDB_NAME'] = self.experiment_info['name']
            os.environ['WANDB_NOTES'] = self.experiment_info['description']
            
            if self.experiment_info['tags']:
                os.environ['WANDB_TAGS'] = ','.join(self.experiment_info['tags'])
            
            self.logger.info(f"✅ WandB configured: {wandb_config.get('project')}")
            self.logger.info(f"🎯 Using project path for WandB: {wandb_config.get('project', 'pcb-defect-150epochs-v1')}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  WandB setup failed: {e}")
            
    def create_model(self) -> YOLO:
        """Create YOLO model based on configuration."""
        try:
            model_config = self.config['model']
            model_type = model_config.get('type', 'yolov8n')
            
            # Handle custom model configurations
            if model_config.get('config_path'):
                model_path = model_config['config_path']
                if not os.path.exists(model_path):
                    # Try relative to project root
                    model_path = PROJECT_ROOT / model_path
                
                if os.path.exists(model_path):
                    self.logger.info(f"Loading custom model from: {model_path}")
                    
                    # Debug: Check if C2f_CoordAtt is available
                    try:
                        from ultralytics.nn.modules.block import C2f_CoordAtt
                        self.logger.info("C2f_CoordAtt module is available")
                    except ImportError as ie:
                        self.logger.error(f"C2f_CoordAtt import failed: {ie}")
                        raise
                    
                    # Create model with custom architecture
                    # For custom architectures, let ultralytics handle pretrained loading during training
                    # This ensures proper class adaptation (80 COCO classes -> 6 HRIPCB classes)
                    if model_config.get('pretrained', False):
                        # Use the pretrained model as base, then override with custom config during training
                        self.logger.info(f"Custom model will use pretrained {model_type}.pt during training")
                        self.logger.info("✅ Pretrained weights will be loaded with proper class adaptation during training")
                        model = YOLO(str(model_path))
                    else:
                        model = YOLO(str(model_path))
                    
                    self.logger.info(f"Custom model loaded successfully: {model_path}")
                else:
                    self.logger.warning(f"Custom model not found at {model_path}, using pretrained: {model_type}")
                    model = YOLO(f'{model_type}.pt')
            else:
                # Use pretrained model for baseline configs
                model = YOLO(f'{model_type}.pt')
                self.logger.info(f"Pretrained model loaded: {model_type}")
                
            return model
            
        except Exception as e:
            self.logger.error(f"Model creation failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
            
    def run_training(self, model: YOLO) -> Dict[str, Any]:
        """Execute model training."""
        try:
            self.logger.info("🏋️  Starting training phase...")
            training_config = self.config['training']
            data_config = self.config['data']
            
            # Prepare training arguments
            train_args = {
                'data': data_config['path'],
                'epochs': training_config.get('epochs', 100),
                'imgsz': training_config.get('imgsz', 640),
                'batch': training_config.get('batch', 16),
                'device': training_config.get('device', 'auto'),
                'workers': training_config.get('workers', 8),
                'cache': training_config.get('cache', True),
                'amp': training_config.get('amp', True),
                'project': training_config.get('project', 'experiments/pcb-defect-150epochs-v1'),  # This becomes WandB project name
                'name': self.experiment_info['name'],
                'exist_ok': True,
                'save_period': training_config.get('save_period', 50),
                'val': training_config.get('validate', True)
            }
            
            # Add pretrained weights to training args if specified in model config
            model_config = self.config['model']
            if model_config.get('pretrained', False) and model_config.get('config_path'):
                model_type = model_config.get('type', 'yolov8n')
                train_args['pretrained'] = f'{model_type}.pt'
                self.logger.info(f"🎯 Will use pretrained weights during training: {train_args['pretrained']}")
            
            # Add optimizer settings
            if 'optimizer' in training_config:
                train_args['optimizer'] = training_config['optimizer']
            if 'lr0' in training_config:
                train_args['lr0'] = training_config['lr0']
            if 'lrf' in training_config:
                train_args['lrf'] = training_config['lrf']
            if 'weight_decay' in training_config:
                train_args['weight_decay'] = training_config['weight_decay']
            if 'momentum' in training_config:
                train_args['momentum'] = training_config['momentum']
            if 'warmup_epochs' in training_config:
                train_args['warmup_epochs'] = training_config['warmup_epochs']
            if 'patience' in training_config:
                train_args['patience'] = training_config['patience']
                
            # Add augmentation settings
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
            
            self.logger.info("Training configuration:")
            for key, value in train_args.items():
                self.logger.info(f"   {key}: {value}")
            
            # Execute training
            start_time = time.time()
            results = model.train(**train_args)
            training_time = time.time() - start_time
            
            # Store results
            self.results['training_results'] = results
            self.results['timing']['training_time'] = training_time
            self.results['model_path'] = str(results.save_dir / 'weights' / 'best.pt')
            
            self.logger.info(f"✅ Training completed in {training_time:.2f} seconds")
            self.logger.info(f"📁 Model saved to: {self.results['model_path']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            raise
            
    def run_validation(self, model: YOLO) -> Dict[str, Any]:
        """Execute validation on validation set."""
        try:
            self.logger.info("🔍 Starting validation phase...")
            validation_config = self.config.get('validation', {})
            data_config = self.config['data']
            
            val_args = {
                'data': data_config['path'],
                'split': 'val',  # Validation set
                'imgsz': validation_config.get('imgsz', 640),
                'batch': validation_config.get('batch', 1),
                'device': validation_config.get('device', 'auto'),
                'conf': validation_config.get('conf_threshold', 0.001),
                'iou': validation_config.get('iou_threshold', 0.6),
                'max_det': validation_config.get('max_detections', 300),
                'save_json': True,
                'verbose': True,
                'plots': True
            }
            
            start_time = time.time()
            results = model.val(**val_args)
            validation_time = time.time() - start_time
            
            # Store results
            self.results['validation_results'] = results
            self.results['timing']['validation_time'] = validation_time
            
            # Extract key metrics directly from results.box (like the working old code)
            if hasattr(results, 'box'):
                precision = float(results.box.mp) if hasattr(results.box, 'mp') else 0.0
                recall = float(results.box.mr) if hasattr(results.box, 'mr') else 0.0
                
                # Debug F1 score availability
                self.logger.info(f"🔍 Debug - Available box attributes: {[attr for attr in dir(results.box) if not attr.startswith('_')]}")
                
                # Try multiple F1 score sources
                f1_score = 0.0
                if hasattr(results.box, 'mf1') and results.box.mf1 is not None:
                    f1_score = float(results.box.mf1)
                    self.logger.info(f"✅ F1 from results.box.mf1: {f1_score}")
                elif hasattr(results.box, 'f1') and results.box.f1 is not None:
                    f1_score = float(results.box.f1)
                    self.logger.info(f"✅ F1 from results.box.f1: {f1_score}")
                else:
                    # Manual F1 calculation: F1 = 2 * (precision * recall) / (precision + recall)
                    if precision > 0 and recall > 0:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                        self.logger.info(f"🔧 F1 calculated manually: {f1_score:.4f} (P={precision:.4f}, R={recall:.4f})")
                    else:
                        self.logger.warning("⚠️  Cannot calculate F1: precision or recall is 0")
                
                self.results['metrics_summary']['validation'] = {
                    'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
                    'mAP50-95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1_score  # ✅ Now uses fallback calculation if needed
                }
            
            self.logger.info(f"✅ Validation completed in {validation_time:.2f} seconds")
            self.logger.info("📊 Validation metrics:")
            for key, value in self.results['metrics_summary'].get('validation', {}).items():
                self.logger.info(f"   {key}: {value:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            raise
            
    def run_test_evaluation(self, model: YOLO) -> Dict[str, Any]:
        """Execute comprehensive test set evaluation."""
        try:
            self.logger.info("🧪 Starting TEST SET evaluation...")
            self.logger.info("=" * 60)
            self.logger.info("IMPORTANT: This is the FINAL TEST EVALUATION")
            self.logger.info("These metrics represent unbiased performance assessment")
            self.logger.info("=" * 60)
            
            data_config = self.config['data']
            validation_config = self.config.get('validation', {})
            
            # Test evaluation arguments
            test_args = {
                'data': data_config['path'],
                'split': 'test',  # TEST SET - This is the key difference
                'imgsz': validation_config.get('imgsz', 640),
                'batch': validation_config.get('batch', 1),
                'device': validation_config.get('device', 'auto'),
                'conf': validation_config.get('conf_threshold', 0.001),
                'iou': validation_config.get('iou_threshold', 0.6),
                'max_det': validation_config.get('max_detections', 300),
                'save_json': True,
                'verbose': True,
                'plots': True,
                'save_txt': True,  # Save predictions for analysis
                'save_conf': True  # Save confidence scores
            }
            
            self.logger.info("Test evaluation configuration:")
            for key, value in test_args.items():
                self.logger.info(f"   {key}: {value}")
            
            start_time = time.time()
            results = model.val(**test_args)  # Using .val() but with split='test'
            test_time = time.time() - start_time
            
            # Store results
            self.results['test_results'] = results
            self.results['timing']['test_time'] = test_time
            
            # Extract comprehensive test metrics directly from results.box (like the working old code)
            test_metrics = {}
            if hasattr(results, 'box'):
                precision = float(results.box.mp) if hasattr(results.box, 'mp') else 0.0
                recall = float(results.box.mr) if hasattr(results.box, 'mr') else 0.0
                
                # Try multiple F1 score sources for test metrics too
                f1_score = 0.0
                if hasattr(results.box, 'mf1') and results.box.mf1 is not None:
                    f1_score = float(results.box.mf1)
                elif hasattr(results.box, 'f1') and results.box.f1 is not None:
                    f1_score = float(results.box.f1)
                else:
                    # Manual F1 calculation for test metrics
                    if precision > 0 and recall > 0:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                        self.logger.info(f"🔧 Test F1 calculated manually: {f1_score:.4f}")
                
                test_metrics = {
                    'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
                    'mAP50-95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1_score  # ✅ Now uses fallback calculation
                }
                
                # Per-class metrics available in results.box.maps if needed
                # Note: Per-class breakdown can be accessed via results.box.maps if required
            
            self.results['metrics_summary']['test'] = test_metrics
            
            # Log comprehensive test results
            self.logger.info(f"✅ TEST EVALUATION COMPLETED in {test_time:.2f} seconds")
            self.logger.info("🏆 FINAL TEST METRICS (Unbiased Performance):")
            self.logger.info("-" * 50)
            for metric_name, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"   📊 {metric_name}: {value:.4f}")
            self.logger.info("-" * 50)
            
            # Log to WandB if available
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    'test/mAP50': test_metrics.get('mAP50', 0.0),
                    'test/mAP50-95': test_metrics.get('mAP50-95', 0.0),
                    'test/precision': test_metrics.get('precision', 0.0),
                    'test/recall': test_metrics.get('recall', 0.0),
                    'test/f1': test_metrics.get('f1', 0.0),
                    'test/evaluation_time': test_time
                })
                self.logger.info("📊 Test metrics logged to WandB")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Test evaluation failed: {e}")
            raise
            
    def save_experiment_summary(self):
        """Save comprehensive experiment summary."""
        try:
            summary_dir = Path('experiments/results') / self.experiment_info['name']
            summary_dir.mkdir(parents=True, exist_ok=True)
            
            # Create comprehensive summary
            summary = {
                'experiment_info': self.experiment_info,
                'configuration': self.config,
                'results_summary': {
                    'training_completed': self.results['training_results'] is not None,
                    'validation_completed': self.results['validation_results'] is not None,
                    'test_completed': self.results['test_results'] is not None,
                    'model_path': self.results['model_path'],
                    'timing': self.results['timing'],
                    'metrics': self.results['metrics_summary']
                },
                'completion_timestamp': datetime.now().isoformat()
            }
            
            # Save summary as JSON
            summary_file = summary_dir / 'experiment_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"📄 Experiment summary saved: {summary_file}")
            
            # Also save as human-readable text
            report_file = summary_dir / 'experiment_report.txt'
            with open(report_file, 'w') as f:
                f.write(f"PCB DEFECT DETECTION EXPERIMENT REPORT\n")
                f.write(f"=" * 60 + "\n\n")
                f.write(f"Experiment: {self.experiment_info['name']}\n")
                f.write(f"Type: {self.experiment_info['type']}\n")
                f.write(f"Model: {self.experiment_info['model_type']}\n")
                f.write(f"Attention: {self.experiment_info['attention_mechanism']}\n")
                f.write(f"Image Size: {self.experiment_info['image_size']}\n")
                f.write(f"Timestamp: {self.experiment_info['timestamp']}\n\n")
                
                f.write(f"TIMING SUMMARY\n")
                f.write(f"-" * 30 + "\n")
                for phase, time_taken in self.results['timing'].items():
                    f.write(f"{phase}: {time_taken:.2f} seconds\n")
                
                if self.results['metrics_summary']:
                    f.write(f"\nPERFORMANCE METRICS\n")
                    f.write(f"-" * 30 + "\n")
                    
                    for phase, metrics in self.results['metrics_summary'].items():
                        f.write(f"\n{phase.upper()} METRICS:\n")
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                f.write(f"  {metric}: {value:.4f}\n")
                
            self.logger.info(f"📊 Human-readable report saved: {report_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to save experiment summary: {e}")
            
    def run_complete_experiment(self, test_only: bool = False):
        """Execute complete experiment workflow."""
        try:
            start_time = time.time()
            self.logger.info("🚀 Starting complete experiment workflow...")
            
            # Setup
            self.setup_wandb()
            
            if not test_only:
                # 1. Create and train model
                model = self.create_model()
                self.run_training(model)
                
                # Load best model for evaluation
                if self.results['model_path'] and os.path.exists(self.results['model_path']):
                    model = YOLO(self.results['model_path'])
                    self.logger.info(f"✅ Best model loaded for evaluation: {self.results['model_path']}")
                
                # 2. Validation (on validation set)
                self.run_validation(model)
            else:
                # Test-only mode: load existing model
                if not self.results['model_path']:
                    raise ValueError("No model path specified for test-only mode")
                model = YOLO(self.results['model_path'])
                self.logger.info("🧪 Running in TEST-ONLY mode")
            
            # 3. CRUCIAL: Test evaluation (on test set)
            self.run_test_evaluation(model)
            
            # 4. Save comprehensive results
            self.save_experiment_summary()
            
            total_time = time.time() - start_time
            self.results['timing']['total_experiment_time'] = total_time
            
            # Final summary
            self.logger.info("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 60)
            self.logger.info(f"📊 Total experiment time: {total_time:.2f} seconds")
            self.logger.info(f"🏆 Final Test Results:")
            
            if 'test' in self.results['metrics_summary']:
                for metric, value in self.results['metrics_summary']['test'].items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"   {metric}: {value:.4f}")
            
            self.logger.info("=" * 60)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Experiment failed: {e}")
            raise


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description='Run single PCB defect detection experiment')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to experiment configuration YAML file')
    parser.add_argument('--test_only', action='store_true',
                       help='Run only test evaluation (requires trained model)')
    
    args = parser.parse_args()
    
    try:
        # Create and run experiment
        runner = SingleExperimentRunner(args.config)
        results = runner.run_complete_experiment(test_only=args.test_only)
        
        print("\n✅ Experiment completed successfully!")
        print(f"📁 Results saved in: experiments/results/{runner.experiment_info['name']}/")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()