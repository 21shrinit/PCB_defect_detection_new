#!/usr/bin/env python3
"""
Baseline Training Script for PCB Defect Detection
Model A: YOLOv8n with Custom Focal-SIoU Loss

This script trains the baseline model using the custom trainer with enhanced loss functions.
Ensures proper loss backpropagation and smooth training without augmentations.
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
    os.environ['WANDB_NAME'] = 'YOLOv8n_FocalSIoU_Baseline'
    
    print("🚀 Setting up baseline training environment...")
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"🔧 Custom trainer: {MyCustomTrainer.__name__}")


def get_device_str() -> str:
    """Return '0' if CUDA available (GPU 0), else 'cpu'."""
    return '0' if torch.cuda.is_available() else 'cpu'


def train_baseline_model():
    """Train the baseline YOLOv8n model with custom loss."""
    try:
        # Load YOLOv8n model with pretrained weights
        print("📦 Loading YOLOv8n model...")
        model = YOLO("yolov8n.pt")
        
        # Absolute path to dataset YAML
        repo_dir = Path(__file__).parent
        data_yaml = repo_dir / 'experiments' / 'configs' / 'datasets' / 'pcb_data.yaml'
        
        # Configure training parameters - NO AUGMENTATIONS, 150 EPOCHS, 30 PATIENCE
        training_config = {
            'data': str(data_yaml),
            'epochs': 150,  # Full 150 epochs
            'imgsz': 640,
            'batch': 16,
            'device': get_device_str(),
            'workers': 8,
            'project': 'runs/train',
            'name': 'yolov8n_FocalSIoU_Baseline',
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
            'cls': 2.0,  # Increased for Focal Loss
            'dfl': 1.5,  # Enable DFL for full YOLOv8 benefits
            'label_smoothing': 0.0,
            'patience': 30,  # Early stopping patience
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
            
            # DISABLE ALL AUGMENTATIONS - Data is already augmented
            'augment': False,
            'mosaic': 0.0,  # Disable mosaic
            'mixup': 0.0,   # Disable mixup
            'copy_paste': 0.0,  # Disable copy-paste
            'erasing': 0.0,  # Disable random erasing
            'auto_augment': None,  # Disable auto augmentation
            'hsv_h': 0.0,   # Disable HSV augmentation
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'degrees': 0.0,  # Disable rotation
            'translate': 0.0,  # Disable translation
            'scale': 0.0,   # Disable scaling
            'shear': 0.0,   # Disable shearing
            'perspective': 0.0,  # Disable perspective
            'flipud': 0.0,  # Disable vertical flip
            'fliplr': 0.0,  # Disable horizontal flip
            
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
            'verbose': True,
            'seed': 42,
            'deterministic': True,
        }
        
        print("🎯 Starting baseline training with Custom Focal-SIoU loss...")
        print("=" * 60)
        print("🔧 Custom Losses Implemented:")
        print("   📦 Box Loss: SIoU Loss (instead of CIoU)")
        print("   🎯 Classification Loss: Focal Loss (instead of BCE)")
        print("   🔥 DFL Loss: Distribution Focal Loss (re-enabled)")
        print("   ⚖️  Dynamic Weight Balancing: Progressive cls_weight increase")
        print("   🎯 Full YOLOv8 Architecture: All three losses working together")
        print("   🚫 No Augmentations: Data is already augmented")
        print("   ⏱️  Training: 150 epochs with 30 patience")
        print("=" * 60)
        
        # Train the model using the custom trainer
        results = model.train(
            trainer=MyCustomTrainer,
            **training_config
        )
        
        print("✅ Baseline training completed successfully!")
        print(f"📊 Results saved to: {results.save_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during baseline training: {e}")
        import traceback
        traceback.print_exc()
        raise


def validate_model(model_path):
    """Validate the trained model on test set."""
    try:
        print("🔍 Validating trained model...")
        
        # Load the trained model
        model = YOLO(model_path)
        
        # Absolute path to dataset YAML
        repo_dir = Path(__file__).parent
        data_yaml = repo_dir / 'experiments' / 'configs' / 'datasets' / 'pcb_data.yaml'
        
        # Validate on test set
        results = model.val(
            data=str(data_yaml),
            split='test',
            imgsz=640,
            batch=16,
            device=get_device_str(),
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
            augment=False,  # No augmentation during validation
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
        
        print("✅ Model validation completed!")
        print(f"📊 Validation results: {results}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during model validation: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main training function."""
    print("🎯 PCB Defect Detection - Baseline Training")
    print("=" * 60)
    print("Model A: YOLOv8n + Custom Focal-SIoU Loss")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Train baseline model
    training_results = train_baseline_model()
    
    # Validate the model
    model_path = training_results.save_dir / 'weights' / 'best.pt'
    if model_path.exists():
        validation_results = validate_model(str(model_path))
    else:
        print("⚠️  Best model weights not found, skipping validation")
    
    print("\n🎉 Baseline training pipeline completed!")
    print("📝 Next steps:")
    print("1. Review training metrics in W&B dashboard")
    print("2. Analyze model performance on test set")
    print("3. Proceed to Model B (YOLOv8n + CBAM Attention)")
    print("4. Proceed to Model C (YOLOv8n + MobileViT Hybrid)")


if __name__ == "__main__":
    main()
