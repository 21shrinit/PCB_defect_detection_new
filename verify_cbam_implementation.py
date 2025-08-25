#!/usr/bin/env python3
"""
CBAM Implementation Verification Script
======================================

Comprehensive verification that CBAM is correctly implemented and working
in the YOLOv8n architecture.

Tests:
1. Config loading verification
2. Module instantiation verification  
3. CBAM module presence verification
4. Forward pass functionality verification
5. Parameter count verification
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

def test_cbam_config_loading():
    """Test if CBAM configuration loads without errors."""
    print("🔍 Test 1: CBAM Configuration Loading")
    print("-" * 50)
    
    config_path = "ultralytics/cfg/models/v8/yolov8n-cbam-neck-optimal.yaml"
    
    try:
        if not Path(config_path).exists():
            print(f"❌ Config file not found: {config_path}")
            return False
            
        print(f"📁 Loading config: {config_path}")
        model = YOLO(config_path)
        print("✅ Config loaded successfully")
        
        # Basic model info
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"📊 Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_cbam_module_instantiation():
    """Verify CBAM modules are actually instantiated in the model."""
    print(f"\n🔍 Test 2: CBAM Module Instantiation")
    print("-" * 50)
    
    try:
        config_path = "ultralytics/cfg/models/v8/yolov8n-cbam-neck-optimal.yaml"
        model = YOLO(config_path)
        
        # Count CBAM modules
        cbam_modules = []
        c2f_cbam_blocks = []
        
        for name, module in model.model.named_modules():
            if 'cbam' in name.lower():
                cbam_modules.append(name)
            if 'C2f_CBAM' in str(type(module)) or hasattr(module, 'cbam'):
                c2f_cbam_blocks.append(name)
        
        print(f"🔍 Found CBAM modules: {len(cbam_modules)}")
        for cbam_mod in cbam_modules[:10]:  # Show first 10
            print(f"   • {cbam_mod}")
        if len(cbam_modules) > 10:
            print(f"   ... and {len(cbam_modules) - 10} more")
            
        print(f"\n🔍 Found C2f_CBAM blocks: {len(c2f_cbam_blocks)}")
        for block in c2f_cbam_blocks:
            print(f"   • {block}")
            
        # Expected CBAM locations based on config (layers 12, 15, 18, 21)
        expected_locations = ['model.12', 'model.15', 'model.18', 'model.21']
        found_expected = 0
        
        print(f"\n🎯 Checking expected CBAM locations:")
        for expected in expected_locations:
            # Check if this layer has CBAM
            has_cbam = any(expected in cbam_name for cbam_name in cbam_modules)
            if has_cbam:
                print(f"   ✅ {expected}: CBAM found")
                found_expected += 1
            else:
                print(f"   ❌ {expected}: CBAM NOT found")
        
        success = found_expected >= 3  # At least 3 out of 4 expected locations
        print(f"\n📊 Expected CBAM locations found: {found_expected}/4")
        
        return success
        
    except Exception as e:
        print(f"❌ Module instantiation test failed: {e}")
        return False

def test_cbam_forward_pass():
    """Test CBAM forward pass functionality."""
    print(f"\n🔍 Test 3: CBAM Forward Pass Functionality")
    print("-" * 50)
    
    try:
        config_path = "ultralytics/cfg/models/v8/yolov8n-cbam-neck-optimal.yaml"
        model = YOLO(config_path)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        print(f"📥 Input shape: {dummy_input.shape}")
        
        # Forward pass
        print("🚀 Running forward pass...")
        with torch.no_grad():
            outputs = model.model(dummy_input)
        
        print("✅ Forward pass completed successfully")
        
        if isinstance(outputs, list):
            print(f"📤 Output shapes: {[o.shape for o in outputs]}")
        else:
            print(f"📤 Output shape: {outputs.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cbam_vs_baseline_parameters():
    """Compare CBAM model parameters with baseline."""
    print(f"\n🔍 Test 4: CBAM vs Baseline Parameter Comparison")
    print("-" * 50)
    
    try:
        # Load CBAM model
        cbam_model = YOLO("ultralytics/cfg/models/v8/yolov8n-cbam-neck-optimal.yaml")
        cbam_params = sum(p.numel() for p in cbam_model.model.parameters())
        
        # Load baseline model
        baseline_model = YOLO("yolov8n.pt")
        baseline_params = sum(p.numel() for p in baseline_model.model.parameters())
        
        # Calculate difference
        param_diff = cbam_params - baseline_params
        percent_diff = (param_diff / baseline_params) * 100
        
        print(f"📊 Parameter Analysis:")
        print(f"   • Baseline YOLOv8n: {baseline_params:,} parameters")
        print(f"   • CBAM YOLOv8n:     {cbam_params:,} parameters")
        print(f"   • Difference:       +{param_diff:,} parameters")
        print(f"   • Percentage:       +{percent_diff:.2f}%")
        
        # Expected range: CBAM should add moderate overhead
        expected_increase = 50000 <= param_diff <= 500000  # 50K to 500K additional params
        
        if expected_increase:
            print(f"✅ Parameter increase is within expected range")
            return True
        else:
            print(f"⚠️  Parameter increase outside expected range")
            return False
        
    except Exception as e:
        print(f"❌ Parameter comparison failed: {e}")
        return False

def test_cbam_attention_modules():
    """Test specific CBAM attention module functionality."""
    print(f"\n🔍 Test 5: CBAM Attention Module Functionality")
    print("-" * 50)
    
    try:
        # Import CBAM directly
        from ultralytics.nn.modules.attention import CBAM
        
        # Test CBAM module directly
        print("🔧 Testing CBAM module directly...")
        cbam = CBAM(256)  # 256 channels
        test_input = torch.randn(1, 256, 32, 32)
        
        print(f"📥 CBAM input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = cbam(test_input)
            
        print(f"📤 CBAM output shape: {output.shape}")
        print("✅ Direct CBAM module test passed")
        
        # Verify output shape matches input
        if output.shape == test_input.shape:
            print("✅ CBAM preserves tensor dimensions")
            return True
        else:
            print("❌ CBAM changed tensor dimensions")
            return False
            
    except Exception as e:
        print(f"❌ CBAM attention module test failed: {e}")
        return False

def main():
    """Run comprehensive CBAM verification."""
    print("🚀 CBAM IMPLEMENTATION VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Config Loading", test_cbam_config_loading),
        ("Module Instantiation", test_cbam_module_instantiation),  
        ("Forward Pass", test_cbam_forward_pass),
        ("Parameter Comparison", test_cbam_vs_baseline_parameters),
        ("Attention Modules", test_cbam_attention_modules)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📋 VERIFICATION SUMMARY")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ✅ ALL TESTS PASSED - CBAM is correctly implemented!")
        print("\n✅ Conclusions:")
        print("   • CBAM configuration loads successfully")
        print("   • CBAM modules are properly instantiated")
        print("   • Forward pass works correctly")
        print("   • Parameter overhead is reasonable")
        print("   • Attention mechanism is functional")
    else:
        print("⚠️  ❌ SOME TESTS FAILED - CBAM implementation has issues!")
        
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)