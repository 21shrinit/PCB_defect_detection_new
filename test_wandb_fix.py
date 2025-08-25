#!/usr/bin/env python3
"""
Quick test to verify WandB and CBAM fixes
"""
import os
import sys
import yaml
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_wandb_config():
    """Test WandB configuration in sample configs."""
    print("🧪 Testing WandB configuration...")
    
    configs = [
        'experiments/configs/comprehensive_ablation/yolov8n_standard_none_config.yaml',
        'experiments/configs/comprehensive_ablation/yolov8n_verifocal_eiou_cbam_config.yaml'
    ]
    
    for config_path in configs:
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"\n✅ Config: {Path(config_path).name}")
            print(f"   WandB Project: {config['wandb']['project']}")
            print(f"   WandB Name: {config['wandb']['name']}")
            print(f"   Training Name: {config['training']['name']}")
            print(f"   Attention: {config['model'].get('attention_mechanism', 'none')}")
        else:
            print(f"❌ Config not found: {config_path}")

def test_deterministic_config():
    """Test deterministic algorithms configuration."""
    print("\n🔧 Testing PyTorch deterministic algorithms...")
    
    # Test baseline configuration
    print("Baseline (none) configuration:")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"   Deterministic: {torch.are_deterministic_algorithms_enabled()}")
    print(f"   CUDNN Deterministic: {torch.backends.cudnn.deterministic}")
    print(f"   CUDNN Benchmark: {torch.backends.cudnn.benchmark}")
    
    # Test attention configuration
    print("\nAttention (CBAM) configuration:")
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"   Deterministic: {torch.are_deterministic_algorithms_enabled()}")
    print(f"   CUDNN Deterministic: {torch.backends.cudnn.deterministic}")
    print(f"   CUDNN Benchmark: {torch.backends.cudnn.benchmark}")

def test_imports():
    """Test critical imports."""
    print("\n📦 Testing imports...")
    
    try:
        import wandb
        print("✅ WandB import successful")
    except ImportError as e:
        print(f"❌ WandB import failed: {e}")
    
    try:
        from scripts.experiments.run_single_experiment_FIXED import FixedExperimentRunner
        print("✅ FixedExperimentRunner import successful")
    except ImportError as e:
        print(f"❌ FixedExperimentRunner import failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing WandB and CBAM fixes...\n")
    
    test_imports()
    test_wandb_config()
    test_deterministic_config()
    
    print("\n✅ All tests completed!")