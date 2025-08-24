#!/usr/bin/env python3
"""
Test YOLOv10n and YOLOv11n Baseline Model Loading
================================================

Verify that the new baseline configurations can load models correctly.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

def test_model_loading():
    """Test that YOLOv10n and YOLOv11n models can be loaded."""
    print("🧪 Testing Baseline Model Loading")
    print("=" * 40)
    
    models_to_test = [
        ("yolov8n", "YOLOv8n - Existing Baseline"),
        ("yolov10n", "YOLOv10n - New Baseline"),
        ("yolo11n", "YOLOv11n - New Baseline")
    ]
    
    results = {}
    
    for model_name, description in models_to_test:
        print(f"\n🔍 Testing {description}")
        try:
            # Try to load the model
            model = YOLO(f'{model_name}.pt')
            
            # Get model info
            model_info = model.info(detailed=False, verbose=False)
            
            print(f"✅ {model_name}: Model loaded successfully")
            print(f"   └─ Model type: {type(model.model).__name__}")
            
            results[model_name] = {
                'status': 'success',
                'model_type': type(model.model).__name__
            }
            
        except Exception as e:
            print(f"❌ {model_name}: Failed to load - {e}")
            results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results

def test_config_loading():
    """Test that the new config files are valid."""
    print("\n🔍 Testing Configuration Files")
    print("-" * 40)
    
    configs_to_test = [
        "experiments/configs/roboflow_pcb/RB00_YOLOv8n_Baseline.yaml",
        "experiments/configs/roboflow_pcb/RB09_YOLOv10n_Baseline.yaml", 
        "experiments/configs/roboflow_pcb/RB10_YOLOv11n_Baseline.yaml"
    ]
    
    for config_path in configs_to_test:
        config_file = Path(config_path)
        if config_file.exists():
            print(f"✅ {config_file.name}: Configuration file exists")
        else:
            print(f"❌ {config_file.name}: Configuration file missing")

if __name__ == "__main__":
    print("🎯 YOLO Baseline Models Test")
    print("=" * 50)
    
    # Test model loading
    results = test_model_loading()
    
    # Test config files
    test_config_loading()
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("-" * 20)
    
    successful_models = [name for name, info in results.items() if info['status'] == 'success']
    failed_models = [name for name, info in results.items() if info['status'] == 'failed']
    
    print(f"✅ Successfully loaded: {', '.join(successful_models)}")
    if failed_models:
        print(f"❌ Failed to load: {', '.join(failed_models)}")
        print("\n🔧 Next steps for failed models:")
        for model in failed_models:
            error = results[model]['error']
            print(f"   - {model}: {error}")
    
    if len(successful_models) == 3:
        print(f"\n🎉 ALL BASELINE MODELS READY FOR ABLATION STUDY!")
        print("   You can now run experiments with:")
        print("   - RB00_YOLOv8n_Baseline.yaml")
        print("   - RB09_YOLOv10n_Baseline.yaml") 
        print("   - RB10_YOLOv11n_Baseline.yaml")
    else:
        print(f"\n⚠️  Some models need attention before starting ablation study")