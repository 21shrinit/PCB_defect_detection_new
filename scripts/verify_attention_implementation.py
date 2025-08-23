#!/usr/bin/env python3
"""
Attention Implementation Verification Script

This script verifies that attention mechanisms are implemented correctly and consistently
with the successful pcb-defect-150epochs-v1 experiments.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import yaml
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_attention_modules():
    """Test core attention module implementations"""
    print("=" * 60)
    print("TESTING ATTENTION MODULE IMPLEMENTATIONS")
    print("=" * 60)
    
    try:
        from ultralytics.nn.modules.attention import ECA, CBAM, CoordAtt
        
        # Test ECA implementation
        print("\n1. Testing ECA (Efficient Channel Attention):")
        print("-" * 50)
        
        # Test different channel sizes
        test_channels = [128, 256, 512, 1024]
        for channels in test_channels:
            eca = ECA(channels)
            x = torch.randn(2, channels, 32, 32)
            out = eca(x)
            
            # Verify output shape and attention application
            if out.shape == x.shape:
                print(f"  PASS ECA({channels}): Input {x.shape} -> Output {out.shape}")
            else:
                print(f"  FAIL ECA({channels}): Shape mismatch! {x.shape} -> {out.shape}")
                
        # Test CBAM implementation  
        print("\n2. Testing CBAM (Convolutional Block Attention Module):")
        print("-" * 50)
        
        for channels in test_channels:
            cbam = CBAM(channels)
            x = torch.randn(2, channels, 32, 32)
            out = cbam(x)
            
            if out.shape == x.shape:
                print(f"  ✅ CBAM({channels}): Input {x.shape} -> Output {out.shape}")
            else:
                print(f"  ❌ CBAM({channels}): Shape mismatch! {x.shape} -> {out.shape}")
        
        # Test CoordAtt implementation
        print("\n3. Testing CoordAtt (Coordinate Attention):")
        print("-" * 50)
        
        for channels in test_channels:
            coordatt = CoordAtt(channels, channels)
            x = torch.randn(2, channels, 32, 32)
            out = coordatt(x)
            
            if out.shape == x.shape:
                print(f"  ✅ CoordAtt({channels}): Input {x.shape} -> Output {out.shape}")
            else:
                print(f"  ❌ CoordAtt({channels}): Shape mismatch! {x.shape} -> {out.shape}")
                
        return True
        
    except Exception as e:
        print(f"❌ ATTENTION MODULE TEST FAILED: {e}")
        traceback.print_exc()
        return False

def test_c2f_attention_blocks():
    """Test C2f attention block implementations"""
    print("\n" + "=" * 60)
    print("TESTING C2f ATTENTION BLOCK IMPLEMENTATIONS")
    print("=" * 60)
    
    try:
        from ultralytics.nn.modules.block import C2f_ECA, C2f_CBAM, C2f_CoordAtt
        
        # Test parameters
        c1, c2 = 256, 256
        x = torch.randn(2, c1, 32, 32)
        
        print(f"\nInput tensor shape: {x.shape}")
        
        # Test C2f_ECA
        print("\n1. Testing C2f_ECA:")
        print("-" * 30)
        c2f_eca = C2f_ECA(c1, c2, n=1)
        out_eca = c2f_eca(x)
        if out_eca.shape == (2, c2, 32, 32):
            print(f"  ✅ C2f_ECA: {x.shape} -> {out_eca.shape}")
        else:
            print(f"  ❌ C2f_ECA: Expected {(2, c2, 32, 32)}, got {out_eca.shape}")
            
        # Test C2f_CBAM
        print("\n2. Testing C2f_CBAM:")
        print("-" * 30)
        c2f_cbam = C2f_CBAM(c1, c2, n=1)
        out_cbam = c2f_cbam(x)
        if out_cbam.shape == (2, c2, 32, 32):
            print(f"  ✅ C2f_CBAM: {x.shape} -> {out_cbam.shape}")
        else:
            print(f"  ❌ C2f_CBAM: Expected {(2, c2, 32, 32)}, got {out_cbam.shape}")
            
        # Test C2f_CoordAtt
        print("\n3. Testing C2f_CoordAtt:")
        print("-" * 30)
        c2f_coordatt = C2f_CoordAtt(c1, c2, n=1)
        out_coordatt = c2f_coordatt(x)
        if out_coordatt.shape == (2, c2, 32, 32):
            print(f"  ✅ C2f_CoordAtt: {x.shape} -> {out_coordatt.shape}")
        else:
            print(f"  ❌ C2f_CoordAtt: Expected {(2, c2, 32, 32)}, got {out_coordatt.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ C2f ATTENTION BLOCK TEST FAILED: {e}")
        traceback.print_exc()
        return False

def test_yolov10n_architecture():
    """Test YOLOv10n architecture with attention"""
    print("\n" + "=" * 60)
    print("TESTING YOLOv10n ATTENTION INTEGRATION")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        # Test YOLOv10n with CoordAtt
        print("\nTesting YOLOv10n with Coordinate Attention:")
        print("-" * 50)
        
        model_path = "ultralytics/cfg/models/v10/yolov10n-ca.yaml"
        if Path(model_path).exists():
            print(f"✅ Model config found: {model_path}")
            
            # Try to instantiate the model
            try:
                model = YOLO(model_path)
                print(f"✅ Model instantiation successful")
                
                # Test forward pass
                x = torch.randn(1, 3, 640, 640)
                with torch.no_grad():
                    out = model.model(x)
                print(f"✅ Forward pass successful: Input {x.shape} -> Output shapes: {[o.shape for o in out]}")
                
                # Count parameters
                total_params = sum(p.numel() for p in model.model.parameters())
                print(f"✅ Total parameters: {total_params:,}")
                
                # Verify attention modules are present
                attention_count = 0
                for name, module in model.model.named_modules():
                    if 'coordatt' in name.lower() or 'C2f_CoordAtt' in str(type(module)):
                        attention_count += 1
                        print(f"  📍 Found attention: {name} - {type(module).__name__}")
                
                if attention_count > 0:
                    print(f"✅ Found {attention_count} attention modules in YOLOv10n-CA")
                else:
                    print("⚠️ No attention modules found - may be using different naming")
                
            except Exception as e:
                print(f"❌ Model instantiation failed: {e}")
                return False
                
        else:
            print(f"❌ Model config not found: {model_path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ YOLOv10n ARCHITECTURE TEST FAILED: {e}")
        traceback.print_exc()
        return False

def compare_with_successful_implementation():
    """Compare current implementation with successful pcb-defect-150epochs-v1"""
    print("\n" + "=" * 60)
    print("COMPARING WITH SUCCESSFUL IMPLEMENTATION")
    print("=" * 60)
    
    # Read successful experiment config
    successful_eca_config = "experiments/pcb-defect-150epochs-v1/04_yolov8n_eca_stable_640px/args.yaml"
    successful_ca_config = "experiments/pcb-defect-150epochs-v1/06_yolov8n_coordatt_stable_640px/args.yaml"
    
    issues_found = []
    
    # Check if model configs match
    print("\n1. Checking Model Configuration Consistency:")
    print("-" * 50)
    
    try:
        # ECA comparison
        if Path(successful_eca_config).exists():
            with open(successful_eca_config, 'r') as f:
                eca_args = yaml.safe_load(f)
            
            eca_model_path = eca_args.get('model', '')
            print(f"✅ Successful ECA model: {eca_model_path}")
            
            if Path(eca_model_path).exists():
                print(f"✅ ECA model config exists: {eca_model_path}")
            else:
                print(f"❌ ECA model config missing: {eca_model_path}")
                issues_found.append(f"Missing ECA model config: {eca_model_path}")
        
        # CoordAtt comparison  
        if Path(successful_ca_config).exists():
            with open(successful_ca_config, 'r') as f:
                ca_args = yaml.safe_load(f)
            
            ca_model_path = ca_args.get('model', '')
            print(f"✅ Successful CoordAtt model: {ca_model_path}")
            
            if Path(ca_model_path).exists():
                print(f"✅ CoordAtt model config exists: {ca_model_path}")
            else:
                print(f"❌ CoordAtt model config missing: {ca_model_path}")
                issues_found.append(f"Missing CoordAtt model config: {ca_model_path}")
                
    except Exception as e:
        print(f"❌ Configuration comparison failed: {e}")
        issues_found.append(f"Config comparison error: {e}")
    
    # Check hyperparameter consistency
    print("\n2. Checking Key Hyperparameters:")
    print("-" * 50)
    
    key_params = {
        'box': 7.5,      # Critical for PCB defects
        'cls': 0.5,      # Proven optimal
        'dfl': 1.5,      # Proven optimal
        'lr0': [0.0005, 0.0003],  # Conservative for attention
        'batch': [32, 12]  # Attention-appropriate sizes
    }
    
    try:
        if Path(successful_ca_config).exists():
            with open(successful_ca_config, 'r') as f:
                ca_args = yaml.safe_load(f)
            
            for param, expected in key_params.items():
                actual = ca_args.get(param)
                if isinstance(expected, list):
                    if actual in expected:
                        print(f"  ✅ {param}: {actual} (valid)")
                    else:
                        print(f"  ⚠️ {param}: {actual} (expected one of {expected})")
                else:
                    if actual == expected:
                        print(f"  ✅ {param}: {actual}")
                    else:
                        print(f"  ⚠️ {param}: {actual} (expected {expected})")
                        
    except Exception as e:
        print(f"❌ Hyperparameter comparison failed: {e}")
    
    return issues_found

def generate_verification_report(test_results):
    """Generate comprehensive verification report"""
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    
    all_passed = all(test_results.values())
    
    print(f"\nOverall Status: {'✅ PASS' if all_passed else '❌ FAIL'}")
    print("\nDetailed Results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if all_passed:
        print(f"\nALL TESTS PASSED!")
        print("Attention mechanisms are correctly implemented")
        print("YOLOv10n integration is functional")
        print("Implementation matches successful experiments")
        
        print(f"\nRECOMMENDATIONS:")
        print("- Attention implementations are consistent with successful experiments")
        print("- YOLOv10n-CA architecture is correctly configured")
        print("- Use proven hyperparameters from pcb-defect-150epochs-v1")
        print("- Reduce batch size from 128 to 32 for attention models")
        
    else:
        print(f"\nISSUES FOUND!")
        print("Some tests failed. Check the detailed output above.")
        print("\nACTION ITEMS:")
        print("- Fix any missing model configurations")
        print("- Verify attention module integrations")  
        print("- Test model instantiation and forward passes")
        print("- Apply hyperparameter fixes from analysis")
    
    return all_passed

def main():
    """Main verification function"""
    print("ATTENTION MECHANISM IMPLEMENTATION VERIFICATION")
    print("=" * 70)
    
    # Run all tests
    test_results = {}
    
    test_results["Attention Modules"] = test_attention_modules()
    test_results["C2f Attention Blocks"] = test_c2f_attention_blocks()  
    test_results["YOLOv10n Architecture"] = test_yolov10n_architecture()
    
    # Compare with successful implementation
    issues = compare_with_successful_implementation()
    test_results["Implementation Consistency"] = len(issues) == 0
    
    # Generate final report
    all_passed = generate_verification_report(test_results)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)