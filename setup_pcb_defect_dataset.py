#!/usr/bin/env python3
"""
Quick setup and test script for PCB Defect dataset training
This script helps you get started with the new PCB Defect dataset
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 PCB Defect Dataset Setup Guide")
    print("=" * 50)
    
    print("\n📋 Step-by-step setup instructions:")
    print("\n1. 📥 Download the dataset:")
    print("   python scripts/setup/download_pcb_defect.py")
    
    print("\n2. 🔍 Inspect the downloaded dataset:")
    print("   - Check datasets/PCB_Defect/ directory")
    print("   - Note the actual class names and count")
    print("   - Verify the data splits (train/val/test)")
    
    print("\n3. ⚙️ Update the dataset configuration:")
    print("   - Edit experiments/configs/datasets/pcb_defect_data.yaml")
    print("   - Update class names and statistics")
    print("   - Verify the data paths")
    
    print("\n4. 🏋️ Run the baseline training:")
    print("   python run_experiment.py --config experiments/configs/yolov8n_pcb_defect_baseline.yaml")
    
    print("\n📁 Files created for you:")
    files_created = [
        "scripts/setup/download_pcb_defect.py",
        "experiments/configs/datasets/pcb_defect_data.yaml", 
        "experiments/configs/yolov8n_pcb_defect_baseline.yaml"
    ]
    
    for file_path in files_created:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (not found)")
    
    print("\n⚠️  Important notes:")
    print("   - Update the dataset YAML after downloading to set correct class names")
    print("   - The config uses conservative settings (batch=64) for initial testing")
    print("   - Adjust batch size and other parameters based on your GPU memory")
    print("   - Set your Kaggle credentials before downloading:")
    print("     $env:KAGGLE_USERNAME='your_username'")
    print("     $env:KAGGLE_KEY='your_api_key'")
    
    print("\n🎯 Next steps after download:")
    print("   1. Inspect the dataset structure")
    print("   2. Update pcb_defect_data.yaml with actual class names")
    print("   3. Test with a small training run first")
    print("   4. Scale up batch size and epochs once stable")

if __name__ == "__main__":
    main()