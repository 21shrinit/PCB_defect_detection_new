#!/usr/bin/env python3
"""
Integration Test Script
======================
Tests if the training pipeline can handle all our experiment configurations.

This script performs dry-run testing of:
✅ Model loading (YOLOv8n, YOLOv10n, YOLOv11n)
✅ Config parsing
✅ Parameter extraction
✅ Loss function detection
✅ Attention mechanism verification

Usage: python test_integration.py
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_config_parsing(config_path: str) -> Dict[str, Any]:
    """Test config file parsing and parameter extraction."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract key parameters
        model_type = config['model'].get('type', 'unknown')
        attention = config['model'].get('attention_mechanism', 'none')
        loss_type = config['training'].get('loss', {}).get('type', 'standard')
        loss_weights = {
            'box_weight': config['training'].get('loss', {}).get('box_weight', 'not_set'),
            'cls_weight': config['training'].get('loss', {}).get('cls_weight', 'not_set'),
            'dfl_weight': config['training'].get('loss', {}).get('dfl_weight', 'not_set')
        }
        
        return {
            'status': 'success',
            'model_type': model_type,
            'attention': attention,
            'loss_type': loss_type,
            'loss_weights': loss_weights,
            'config_path': config_path
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'config_path': config_path
        }

def test_model_availability():
    """Test if required YOLO models are available."""
    models_to_test = ['yolov8n.pt', 'yolov10n.pt', 'yolo11n.pt']
    model_status = {}
    
    for model in models_to_test:
        try:
            from ultralytics import YOLO
            # This will try to download if not available
            YOLO(model)
            model_status[model] = 'available'
        except Exception as e:
            model_status[model] = f'error: {str(e)}'
    
    return model_status

def test_attention_modules():
    """Test if custom attention modules are available."""
    attention_status = {}
    
    try:
        from ultralytics.nn.modules.block import C2f_CBAM
        attention_status['C2f_CBAM'] = 'available'
    except ImportError as e:
        attention_status['C2f_CBAM'] = f'missing: {str(e)}'
    
    try:
        from ultralytics.nn.modules.block import C2f_ECA
        attention_status['C2f_ECA'] = 'available'
    except ImportError as e:
        attention_status['C2f_ECA'] = f'missing: {str(e)}'
    
    try:
        from ultralytics.nn.modules.block import C2f_CoordAtt
        attention_status['C2f_CoordAtt'] = 'available'
    except ImportError as e:
        attention_status['C2f_CoordAtt'] = f'missing: {str(e)}'
    
    return attention_status

def main():
    """Run comprehensive integration tests."""
    print("🔍 PCB Defect Detection - Integration Test")
    print("=" * 50)
    
    # Test 1: Model availability
    print("\n1️⃣  Testing YOLO model availability...")
    model_status = test_model_availability()
    for model, status in model_status.items():
        status_icon = "✅" if status == 'available' else "❌"
        print(f"   {status_icon} {model}: {status}")
    
    # Test 2: Attention modules
    print("\n2️⃣  Testing attention module availability...")
    attention_status = test_attention_modules()
    for module, status in attention_status.items():
        status_icon = "✅" if status == 'available' else "❌"
        print(f"   {status_icon} {module}: {status}")
    
    # Test 3: Config parsing
    print("\n3️⃣  Testing experiment config parsing...")
    
    # Test key experiment configs
    test_configs = [
        'experiments/configs/19_yolov8n_eca_verifocal_siou.yaml',
        'experiments/configs/14_yolov10n_eca_focal_eiou_STABLE.yaml',
        'experiments/configs/16_yolov11n_baseline_standard.yaml',
        'experiments/configs/13_yolov10n_cbam_verifocal_siou_STABLE.yaml'
    ]
    
    config_results = []
    for config_path in test_configs:
        if os.path.exists(config_path):
            result = test_config_parsing(config_path)
            config_results.append(result)
            
            status_icon = "✅" if result['status'] == 'success' else "❌"
            config_name = os.path.basename(config_path)
            print(f"   {status_icon} {config_name}")
            
            if result['status'] == 'success':
                print(f"      Model: {result['model_type']}")
                print(f"      Attention: {result['attention']}")
                print(f"      Loss Type: {result['loss_type']}")
                print(f"      Box Weight: {result['loss_weights']['box_weight']}")
            else:
                print(f"      Error: {result['error']}")
        else:
            print(f"   ❌ {config_path} - File not found")
    
    # Summary
    print("\n📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    total_models = len(model_status)
    available_models = sum(1 for status in model_status.values() if status == 'available')
    print(f"Models Available: {available_models}/{total_models}")
    
    total_attention = len(attention_status)
    available_attention = sum(1 for status in attention_status.values() if status == 'available')
    print(f"Attention Modules: {available_attention}/{total_attention}")
    
    total_configs = len(config_results)
    valid_configs = sum(1 for result in config_results if result['status'] == 'success')
    print(f"Valid Configs: {valid_configs}/{total_configs}")
    
    # Overall status
    if available_models == total_models and available_attention == total_attention and valid_configs == total_configs:
        print("\n🎉 INTEGRATION TEST PASSED - All systems ready!")
        return True
    else:
        print("\n⚠️  INTEGRATION TEST ISSUES - Some components need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)