#!/usr/bin/env python3
"""
Test script to verify YOLOv8 models with attention can be instantiated.
"""

import torch
import sys
import os

# Add the ultralytics directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
ultralytics_path = os.path.join(current_dir, '..', 'ultralytics')
sys.path.insert(0, ultralytics_path)
sys.path.insert(0, os.path.join(current_dir, '..'))

def test_model_instantiation():
    """Test YOLOv8 model instantiation with attention modules."""
    print("🏗️  Testing Model Instantiation...")
    
    try:
        # Import YOLO class from the local ultralytics
        from ultralytics import YOLO
        
        # Test ECA model
        print("  📍 Testing YOLOv8-ECA...")
        model_eca = YOLO('ultralytics/cfg/models/v8/yolov8-eca.yaml')
        print("    ✅ YOLOv8-ECA: PASS")
        
        # Test CBAM model
        print("  📍 Testing YOLOv8-CBAM...")
        model_cbam = YOLO('ultralytics/cfg/models/v8/yolov8-cbam.yaml')
        print("    ✅ YOLOv8-CBAM: PASS")
        
        # Test CoordAtt model
        print("  📍 Testing YOLOv8-CA...")
        model_ca = YOLO('ultralytics/cfg/models/v8/yolov8-ca.yaml')
        print("    ✅ YOLOv8-CA: PASS")
        
        print("  🎉 All models instantiated successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Model instantiation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test forward pass through attention-enhanced models."""
    print("\n🚀 Testing Forward Pass...")
    
    try:
        # Import YOLO class from the local ultralytics
        from ultralytics import YOLO
        
        # Test with a small input to avoid memory issues
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Test ECA model forward pass
        print("  📍 Testing YOLOv8-ECA forward pass...")
        model_eca = YOLO('ultralytics/cfg/models/v8/yolov8-eca.yaml')
        with torch.no_grad():
            output_eca = model_eca(dummy_input)
        print("    ✅ YOLOv8-ECA forward pass: PASS")
        
        # Test CBAM model forward pass
        print("  📍 Testing YOLOv8-CBAM forward pass...")
        model_cbam = YOLO('ultralytics/cfg/models/v8/yolov8-cbam.yaml')
        with torch.no_grad():
            output_cbam = model_cbam(dummy_input)
        print("    ✅ YOLOv8-CBAM forward pass: PASS")
        
        # Test CoordAtt model forward pass
        print("  📍 Testing YOLOv8-CA forward pass...")
        model_ca = YOLO('ultralytics/cfg/models/v8/yolov8-ca.yaml')
        with torch.no_grad():
            output_ca = model_ca(dummy_input)
        print("    ✅ YOLOv8-CA forward pass: PASS")
        
        print("  🎉 All forward passes completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run model instantiation and forward pass tests."""
    print("🎯 YOLOv8 Attention Model Integration Test")
    print("=" * 50)
    
    tests = [
        ("Model Instantiation", test_model_instantiation),
        ("Forward Pass", test_forward_pass),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Attention models are ready for training.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
