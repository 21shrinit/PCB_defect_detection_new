#!/usr/bin/env python3
"""
Example script demonstrating how to use the unified training pipeline.
This script shows different ways to run the two-stage training process.
"""

import os
import sys
from pathlib import Path
from train_unified import UnifiedTrainer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """
    Example 1: Basic usage with default config.yaml
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    try:
        # Initialize trainer with default config
        trainer = UnifiedTrainer('config.yaml')
        
        # Run complete training pipeline
        warmup_results, finetune_results = trainer.run_complete_training()
        
        # Export the final model
        trainer.export_final_model(['onnx', 'torchscript'])
        
        print("✅ Basic training completed successfully!")
        
    except Exception as e:
        print(f"❌ Basic training failed: {e}")


def example_programmatic_config():
    """
    Example 2: Programmatic usage with custom configuration
    """
    print("=" * 60)
    print("EXAMPLE 2: Programmatic Configuration")
    print("=" * 60)
    
    try:
        # You could modify config programmatically before training
        trainer = UnifiedTrainer('config.yaml')
        
        # Example: Modify some parameters programmatically
        # trainer.config['training_strategy']['warmup']['epochs'] = 30
        # trainer.config['training_strategy']['finetune']['learning_rate'] = 0.0005
        
        # Run training
        warmup_results, finetune_results = trainer.run_complete_training()
        
        print("✅ Programmatic training completed!")
        
    except Exception as e:
        print(f"❌ Programmatic training failed: {e}")


def example_cbam_training():
    """
    Example 3: Training CBAM model using the unified pipeline
    """
    print("=" * 60)
    print("EXAMPLE 3: CBAM Model Training")
    print("=" * 60)
    
    try:
        # For CBAM training, you would use a config that specifies the CBAM model
        config_path = 'config.yaml'  # Make sure this points to CBAM model config
        
        trainer = UnifiedTrainer(config_path)
        
        # Verify we're using the CBAM model
        model_config = trainer.config['model']['config_path']
        if 'cbam' in model_config.lower():
            print(f"✅ Using CBAM model: {model_config}")
        else:
            print(f"⚠️  Model config: {model_config}")
        
        # Run the two-stage training
        warmup_results, finetune_results = trainer.run_complete_training()
        
        print("✅ CBAM training completed!")
        
    except Exception as e:
        print(f"❌ CBAM training failed: {e}")


def example_monitoring_training():
    """
    Example 4: Training with detailed monitoring
    """
    print("=" * 60)
    print("EXAMPLE 4: Training with Monitoring")
    print("=" * 60)
    
    try:
        trainer = UnifiedTrainer('config.yaml')
        
        # Stage 1: Warmup with monitoring
        print("Starting Stage 1: Warmup...")
        warmup_checkpoint = trainer.stage1_warmup_training()
        print(f"Warmup checkpoint saved to: {warmup_checkpoint}")
        
        # You could add custom validation or analysis here
        print("Analyzing warmup results...")
        
        # Stage 2: Fine-tuning with monitoring
        print("Starting Stage 2: Fine-tuning...")
        trainer.stage2_finetune_training(warmup_checkpoint)
        
        # Custom analysis of final results
        print("Analyzing fine-tuning results...")
        
        print("✅ Monitored training completed!")
        
    except Exception as e:
        print(f"❌ Monitored training failed: {e}")


def validate_environment():
    """
    Validate that the environment is set up correctly for training.
    """
    print("🔍 Validating environment...")
    
    # Check if config file exists
    config_path = Path('config.yaml')
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    # Check if model config exists
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config_path = Path(config['model']['config_path'])
        if not model_config_path.exists():
            print(f"❌ Model config not found: {model_config_path}")
            return False
        
        data_config_path = Path(config['data']['config_path'])
        if not data_config_path.exists():
            print(f"❌ Data config not found: {data_config_path}")
            return False
        
    except Exception as e:
        print(f"❌ Error validating config: {e}")
        return False
    
    # Check if CUDA is available (optional but recommended)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, training will use CPU")
    except ImportError:
        print("⚠️  PyTorch not available")
    
    print("✅ Environment validation passed!")
    return True


def main():
    """
    Main function to run examples.
    """
    print("🚀 Unified Training Pipeline Examples")
    print("=" * 60)
    
    # Validate environment first
    if not validate_environment():
        print("❌ Environment validation failed. Please check your setup.")
        sys.exit(1)
    
    # Run examples (uncomment the one you want to try)
    
    # Example 1: Basic usage
    # example_basic_usage()
    
    # Example 2: Programmatic configuration
    # example_programmatic_config()
    
    # Example 3: CBAM model training
    example_cbam_training()
    
    # Example 4: Training with monitoring
    # example_monitoring_training()
    
    print("\n🎯 Examples completed!")


if __name__ == "__main__":
    main()