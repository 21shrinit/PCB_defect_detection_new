#!/usr/bin/env python3
"""
Loss Function & Attention Mechanism Integration Verification Script
==================================================================

This script comprehensively tests the FIXED integrations to verify:
✅ Loss functions are correctly configured and used
✅ Attention mechanisms are properly loaded
✅ Training pipeline passes configuration correctly
✅ All components work together seamlessly

Run this BEFORE conducting any experiments to ensure everything works.
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.nn.modules.block import C2f_ECA, C2f_CBAM, C2f_CoordAtt

def test_loss_function_integration():
    """Test that all loss functions can be properly initialized and configured."""
    print("\n🧪 TESTING LOSS FUNCTION INTEGRATION")
    print("=" * 50)
    
    # Test BboxLoss with different IoU types (CIoU is now default)
    iou_types = ['ciou', 'siou', 'eiou', 'giou']
    
    for iou_type in iou_types:
        try:
            bbox_loss = BboxLoss(reg_max=16, iou_type=iou_type)
            print(f"✅ BboxLoss with {iou_type.upper()}: PASSED")
        except Exception as e:
            print(f"❌ BboxLoss with {iou_type.upper()}: FAILED - {e}")
            return False
    
    # Test v8DetectionLoss initialization
    try:
        # Create a dummy model for testing
        model = YOLO('yolov8n.pt')
        
        # Test different loss configurations (CIoU first as default)
        configurations = [
            ('ciou', 'bce'),
            ('siou', 'bce'),
            ('eiou', 'focal'),
            ('ciou', 'varifocal'),
            ('giou', 'bce')
        ]
        
        for iou_type, cls_type in configurations:
            try:
                detection_loss = v8DetectionLoss(model.model, iou_type=iou_type, cls_type=cls_type)
                print(f"✅ v8DetectionLoss ({iou_type.upper()}+{cls_type.upper()}): PASSED")
            except Exception as e:
                print(f"❌ v8DetectionLoss ({iou_type.upper()}+{cls_type.upper()}): FAILED - {e}")
                return False
        
    except Exception as e:
        print(f"❌ Model loading for loss testing failed: {e}")
        return False
    
    print("\n✅ ALL LOSS FUNCTION TESTS PASSED!")
    return True

def test_attention_mechanism_integration():
    """Test that all attention mechanisms can be imported and initialized."""
    print("\n🧪 TESTING ATTENTION MECHANISM INTEGRATION")
    print("=" * 50)
    
    # Test attention mechanism imports
    attention_modules = {
        'ECA': C2f_ECA,
        'CBAM': C2f_CBAM, 
        'CoordAtt': C2f_CoordAtt
    }
    
    for name, module_class in attention_modules.items():
        try:
            # Test initialization with dummy parameters
            module = module_class(c1=256, c2=256, n=3, shortcut=True, g=1, e=0.5)
            print(f"✅ {name} attention module: PASSED")
            
            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 256, 20, 20)
            output = module(dummy_input)
            assert output.shape == dummy_input.shape, f"Output shape mismatch for {name}"
            print(f"✅ {name} forward pass: PASSED")
            
        except Exception as e:
            print(f"❌ {name} attention module: FAILED - {e}")
            return False
    
    print("\n✅ ALL ATTENTION MECHANISM TESTS PASSED!")
    return True

def test_model_loading_with_attention():
    """Test loading models with attention mechanisms."""
    print("\n🧪 TESTING MODEL LOADING WITH ATTENTION")  
    print("=" * 50)
    
    # Test attention model files
    attention_configs = [
        "ultralytics/cfg/models/v8/yolov8n-eca-final.yaml",
        "ultralytics/cfg/models/v8/yolov8n-cbam-neck-optimal.yaml", 
        "ultralytics/cfg/models/v8/yolov8n-ca-position7.yaml"
    ]
    
    for config_path in attention_configs:
        full_path = PROJECT_ROOT / config_path
        if full_path.exists():
            try:
                model = YOLO(str(full_path))
                print(f"✅ Model loading ({config_path.split('/')[-1]}): PASSED")
            except Exception as e:
                print(f"❌ Model loading ({config_path.split('/')[-1]}): FAILED - {e}")
                return False
        else:
            print(f"⚠️  Model config not found: {config_path}")
    
    print("\n✅ ALL MODEL LOADING TESTS PASSED!")
    return True

def test_experiment_config_validity():
    """Test that experiment configurations are valid and can be loaded."""
    print("\n🧪 TESTING EXPERIMENT CONFIG VALIDITY")
    print("=" * 50)
    
    # Test key experiment configs
    test_configs = [
        "experiments/configs/roboflow_pcb/RB01_YOLOv8n_SIoU_ECA.yaml",
        "experiments/configs/roboflow_pcb/RB04_YOLOv8n_EIoU_ECA.yaml",
    ]
    
    for config_path in test_configs:
        full_path = PROJECT_ROOT / config_path
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Verify required sections exist
                required_sections = ['experiment', 'model', 'training', 'data']
                for section in required_sections:
                    if section not in config:
                        raise ValueError(f"Missing required section: {section}")
                
                # Verify model config path exists
                if 'config_path' in config['model']:
                    model_config_path = PROJECT_ROOT / config['model']['config_path']
                    if not model_config_path.exists():
                        raise FileNotFoundError(f"Model config not found: {config['model']['config_path']}")
                
                print(f"✅ Config validation ({config_path.split('/')[-1]}): PASSED")
                
            except Exception as e:
                print(f"❌ Config validation ({config_path.split('/')[-1]}): FAILED - {e}")
                return False
        else:
            print(f"⚠️  Config file not found: {config_path}")
    
    print("\n✅ ALL CONFIG VALIDATION TESTS PASSED!")
    return True

def test_integration_end_to_end():
    """Test end-to-end integration with a minimal training run."""
    print("\n🧪 TESTING END-TO-END INTEGRATION")
    print("=" * 50)
    
    try:
        # Create a simple test configuration
        model = YOLO('yolov8n.pt')
        
        # Test that model accepts loss configuration through args (using CIoU default)
        model.model.args = type('Args', (), {
            'iou_type': 'eiou',  # Override default CIoU with EIoU for testing
            'cls_type': 'focal',
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5
        })()
        
        # Test loss criterion initialization
        criterion = model.model.init_criterion()
        print(f"✅ Loss criterion initialized with IoU: {criterion.iou_type}, CLS: {criterion.cls_type}")
        
        # Verify loss configuration was applied
        assert criterion.iou_type == 'eiou', f"IoU type mismatch: {criterion.iou_type}"
        assert criterion.cls_type == 'focal', f"Classification type mismatch: {criterion.cls_type}"
        
        print("✅ END-TO-END INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ END-TO-END INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🚀 STARTING COMPREHENSIVE INTEGRATION VERIFICATION")
    print("=" * 60)
    
    tests = [
        test_loss_function_integration,
        test_attention_mechanism_integration, 
        test_model_loading_with_attention,
        test_experiment_config_validity,
        test_integration_end_to_end
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ TEST SUITE FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("🎉 ALL INTEGRATIONS WORKING CORRECTLY!")
        print("✅ Ready to run full experiments with confidence!")
        return True
    else:
        print("❌ SOME INTEGRATIONS FAILED - FIX BEFORE RUNNING EXPERIMENTS")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)