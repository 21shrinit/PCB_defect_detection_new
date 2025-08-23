#!/usr/bin/env python3
"""
MobileViT Hybrid Training Script for PCB Defect Detection
Model C: YOLOv8n + MobileViT Hybrid Backbone

This script trains the MobileViT-enhanced model using the custom trainer.
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
from custom_trainer import MyCustomTrainer

def setup_environment():
    """Setup the training environment and paths."""
    # Add current directory to Python path
    sys.path.append(str(Path(__file__).parent))
    
    # Set up W&B project
    os.environ['WANDB_PROJECT'] = 'PCB_Defect_Detection'
    os.environ['WANDB_NAME'] = 'YOLOv8n_MobileViT_Hybrid'
    
    print("🚀 Setting up MobileViT training environment...")
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"🔧 Custom trainer: {MyCustomTrainer.__name__}")

def train_mobilevit_model():
    """Train the MobileViT-enhanced YOLOv8n model."""
    try:
        # Build a new model from the standard yolov8n configuration
        print("📦 Loading YOLOv8n model with MobileViT hybrid backbone...")
        model = YOLO("yolov8n.yaml")
        
        # Absolute path to dataset YAML
        repo_dir = Path(__file__).parent
        data_yaml = repo_dir / 'experiments' / 'configs' / 'datasets' / 'pcb_data.yaml'
        
        # Configure training parameters
        training_config = {
            'data': str(data_yaml),
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'device': '0' if torch.cuda.is_available() else 'cpu',
            'workers': 8,
            'project': 'runs/train',
            'name': 'yolov8n_MobileViT_Hybrid',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'SGD',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'label_smoothing': 0.0,
            'patience': 50,
            'save_period': 10,
            'plots': True,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'conf': 0.001,
            'iou': 0.6,
            'max_det': 300,
            'half': True,
            'dnn': False,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'boxes': True,
            'format': 'torchscript',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': False,
            'opset': 17,
            'workspace': 4,
            'nms': False,
        }
        
        print("🎯 Starting MobileViT training with custom Focal-SIoU loss...")
        print("=" * 60)
        
        # Train the model using the custom trainer
        results = model.train(
            trainer=MyCustomTrainer,
            **training_config
        )
        
        print("✅ MobileViT training completed successfully!")
        print(f"📊 Results saved to: {results.save_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during MobileViT training: {e}")
        raise

def validate_model(model_path):
    """Validate the trained MobileViT model on test set."""
    try:
        print("🔍 Validating trained MobileViT model...")
        
        # Load the trained model
        model = YOLO(model_path)
        
        # Validate on test set
        results = model.val(
            data='experiments/configs/datasets/pcb_data.yaml',
            split='test',
            imgsz=640,
            batch=16,
            device='auto',
            plots=True,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=True,
            dnn=False,
            visualize=False,
            augment=False,
            agnostic_nms=False,
            classes=None,
            retina_masks=False,
            boxes=True,
            format='torchscript',
            keras=False,
            optimize=False,
            int8=False,
            dynamic=False,
            simplify=False,
            opset=17,
            workspace=4,
            nms=False,
        )
        
        print("✅ MobileViT model validation completed!")
        print(f"📊 Validation results: {results}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during MobileViT model validation: {e}")
        raise

def main():
    """Main training function."""
    print("🎯 PCB Defect Detection - MobileViT Training")
    print("=" * 60)
    print("Model C: YOLOv8n + MobileViT Hybrid Backbone")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Train MobileViT model
    training_results = train_mobilevit_model()
    
    # Validate the model
    model_path = training_results.save_dir / 'weights' / 'best.pt'
    if model_path.exists():
        validation_results = validate_model(str(model_path))
    else:
        print("⚠️  Best model weights not found, skipping validation")
    
    print("\n🎉 MobileViT training pipeline completed!")
    print("📝 Next steps:")
    print("1. Review training metrics in W&B dashboard")
    print("2. Analyze model performance on test set")
    print("3. Compare with baseline and CBAM models")
    print("4. Generate final benchmark report")

if __name__ == "__main__":
    main()
