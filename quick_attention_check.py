#!/usr/bin/env python3
"""
Quick Attention Mechanism Check
===============================
Fast verification script for Colab to check if attention mechanisms are working.

This performs essential checks in <2 minutes:
✅ Module imports
✅ Model loading with attention  
✅ Parameter count verification
✅ Basic forward pass test

Usage: python quick_attention_check.py
"""

import torch
import os
from pathlib import Path

def quick_import_check():
    """Quick check of attention module imports."""
    print("🔍 Quick Import Check...")
    
    modules = {
        'C2f_CBAM': 'ultralytics.nn.modules.block',
        'C2f_ECA': 'ultralytics.nn.modules.block',
        'C2f_CoordAtt': 'ultralytics.nn.modules.block'
    }
    
    results = {}
    for module_name, module_path in modules.items():
        try:
            exec(f"from {module_path} import {module_name}")
            results[module_name] = True
            print(f"   ✅ {module_name}: Available")
        except ImportError:
            results[module_name] = False
            print(f"   ❌ {module_name}: Missing")
    
    return results

def quick_model_check():
    """Quick check of model loading with attention."""
    print("\n🏗️  Quick Model Loading Check...")
    
    from ultralytics import YOLO
    
    # Test basic models first
    models_to_test = [
        ('yolov8n', 'yolov8n.pt'),
        ('yolov10n', 'yolov10n.pt'),
        ('yolo11n', 'yolo11n.pt')
    ]
    
    model_results = {}
    for model_name, model_file in models_to_test:
        try:
            model = YOLO(model_file)
            param_count = sum(p.numel() for p in model.model.parameters())
            model_results[model_name] = {
                'loaded': True,
                'parameters': param_count
            }
            print(f"   ✅ {model_name}: Loaded ({param_count:,} params)")
        except Exception as e:
            model_results[model_name] = {
                'loaded': False,
                'error': str(e)
            }
            print(f"   ❌ {model_name}: Failed - {e}")
    
    return model_results

def quick_attention_model_check():
    """Quick check of attention model configs."""
    print("\n🎯 Quick Attention Model Check...")
    
    attention_configs = [
        ('YOLOv8n-ECA', 'ultralytics/cfg/models/v8/yolov8n-eca-final.yaml'),
        ('YOLOv8n-CBAM', 'ultralytics/cfg/models/v8/yolov8n-cbam-neck-optimal.yaml'),
        ('YOLOv10n-ECA', 'ultralytics/cfg/models/v10/yolov10n-eca-research-optimal.yaml')
    ]
    
    attention_results = {}
    from ultralytics import YOLO
    
    for model_name, config_path in attention_configs:
        try:
            if os.path.exists(config_path):
                model = YOLO(config_path)
                param_count = sum(p.numel() for p in model.model.parameters())
                
                # Quick attention detection
                attention_modules = 0
                for name, module in model.model.named_modules():
                    if any(att in module.__class__.__name__ for att in ['CBAM', 'ECA', 'CoordAtt']):
                        attention_modules += 1
                
                attention_results[model_name] = {
                    'loaded': True,
                    'parameters': param_count,
                    'attention_modules': attention_modules,
                    'config_exists': True
                }
                print(f"   ✅ {model_name}: Loaded ({param_count:,} params, {attention_modules} attention modules)")
            else:
                attention_results[model_name] = {
                    'loaded': False,
                    'config_exists': False,
                    'error': f'Config missing: {config_path}'
                }
                print(f"   ❌ {model_name}: Config missing - {config_path}")
                
        except Exception as e:
            attention_results[model_name] = {
                'loaded': False,
                'config_exists': os.path.exists(config_path),
                'error': str(e)
            }
            print(f"   ❌ {model_name}: Failed - {e}")
    
    return attention_results

def quick_forward_pass_test():
    """Quick forward pass test with attention."""
    print("\n⚡ Quick Forward Pass Test...")
    
    try:
        from ultralytics import YOLO
        
        # Test with YOLOv8n + ECA if available
        eca_config = 'ultralytics/cfg/models/v8/yolov8n-eca-final.yaml'
        
        if os.path.exists(eca_config):
            model = YOLO(eca_config)
            model.model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Forward pass
            with torch.no_grad():
                output = model.model(dummy_input)
            
            print(f"   ✅ Forward pass successful: Input {list(dummy_input.shape)}")
            if isinstance(output, (list, tuple)):
                print(f"      Output shapes: {[list(o.shape) for o in output]}")
            else:
                print(f"      Output shape: {list(output.shape)}")
            
            return True
        else:
            print(f"   ⚠️  ECA config not available for testing")
            return False
            
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False

def main():
    """Run quick attention mechanism check."""
    print("⚡ PCB Defect Detection - Quick Attention Check")
    print("="*50)
    
    # Run all quick checks
    import_results = quick_import_check()
    model_results = quick_model_check()
    attention_results = quick_attention_model_check()
    forward_pass_ok = quick_forward_pass_test()
    
    # Quick summary
    print("\n📊 QUICK CHECK SUMMARY")
    print("="*30)
    
    imports_ok = all(import_results.values())
    models_ok = all(r['loaded'] for r in model_results.values())
    attention_ok = any(r.get('loaded', False) and r.get('attention_modules', 0) > 0 
                      for r in attention_results.values())
    
    print(f"✅ Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"✅ Basic Models: {'PASS' if models_ok else 'FAIL'}")
    print(f"✅ Attention Models: {'PASS' if attention_ok else 'FAIL'}")
    print(f"✅ Forward Pass: {'PASS' if forward_pass_ok else 'FAIL'}")
    
    if imports_ok and models_ok and attention_ok and forward_pass_ok:
        print(f"\n🎉 QUICK CHECK PASSED - Attention mechanisms working!")
        return True
    else:
        print(f"\n⚠️  QUICK CHECK ISSUES - Some components need attention")
        
        if not imports_ok:
            print("   • Fix attention module imports")
        if not models_ok:
            print("   • Fix basic model loading")
        if not attention_ok:
            print("   • Fix attention model configurations")
        if not forward_pass_ok:
            print("   • Fix forward pass execution")
        
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)