#!/usr/bin/env python3
"""
Standard YOLOv8n Baseline Training Script
Trains YOLOv8n with default Ultralytics loss functions (no custom modifications)
Ensures proper loss backpropagation and smooth training without augmentations.
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import wandb

def setup_environment():
    """Setup training environment and verify GPU availability."""
    print("🎯 Standard YOLOv8n Baseline Training")
    print("=" * 60)
    print("Model: YOLOv8n (Standard Ultralytics)")
    print("Loss: Default CIoU + BCE + DFL")
    print("Training: 150 epochs with 30 patience")
    print("Augmentation: None (data already augmented)")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = '0'
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("⚠️  No GPU detected, using CPU")
    
    return device

def train_standard_model(device):
    """Train standard YOLOv8n model with default settings."""
    print("\n🚀 Starting standard YOLOv8n training...")
    
    # Get the absolute path to the dataset config
    script_dir = Path(__file__).parent
    dataset_config = script_dir / "experiments" / "configs" / "datasets" / "pcb_data.yaml"
    
    if not dataset_config.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config}")
    
    print(f"📁 Dataset config: {dataset_config}")
    
    # Initialize W&B
    try:
        wandb.init(
            project="PCB_defect_detection",
            name="yolov8n_standard_baseline",
            config={
                "model": "yolov8n",
                "loss": "standard_ultralytics",
                "dataset": "HRIPCB",
                "epochs": 150,
                "batch_size": 16,
                "img_size": 640,
                "device": device,
                "augmentation": "none"
            }
        )
        print("✅ W&B initialized successfully")
    except Exception as e:
        print(f"⚠️  W&B initialization failed: {e}")
        wandb = None
    
    # Load YOLOv8n model
    print("📦 Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")  # Use pretrained weights
    
    # Training configuration - NO AUGMENTATIONS, 150 EPOCHS, 30 PATIENCE
    training_config = {
        'data': str(dataset_config),
        'epochs': 150,  # Full 150 epochs
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'name': 'yolov8n_standard_baseline',
        'patience': 30,  # Early stopping patience
        'save': True,
        'save_period': 10,
        'plots': True,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'cache': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save_json': False,
        'save_hybrid': False,
        'conf': 0.001,
        'iou': 0.6,
        'max_det': 300,
        'half': True,
        'dnn': False,
        'source': None,
        'show': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'vid_stride': 1,
        'line_thickness': 3,
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
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64
    }
    
    print("🎯 Training configuration:")
    print("   📊 Epochs: 150")
    print("   ⏱️  Patience: 30")
    print("   🚫 Augmentation: None")
    print("   📦 Model: YOLOv8n (pretrained)")
    print("   🎯 Loss: Standard Ultralytics (CIoU + BCE + DFL)")
    
    # Start training
    try:
        print("\n🔥 Starting training...")
        results = model.train(**training_config)
        
        print("✅ Standard training completed successfully!")
        print(f"📊 Results saved to: {results.save_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def validate_model(model_path, device):
    """Validate the trained model on test set."""
    print(f"\n🔍 Validating model: {model_path}")
    
    # Get the absolute path to the dataset config
    script_dir = Path(__file__).parent
    dataset_config = script_dir / "experiments" / "configs" / "datasets" / "pcb_data.yaml"
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Validation configuration
    val_config = {
        'data': str(dataset_config),
        'split': 'test',  # Use test split
        'device': device,
        'verbose': True,
        'save_txt': False,
        'save_conf': False,
        'save_json': False,
        'plots': True,
        'conf': 0.001,
        'iou': 0.6,
        'max_det': 300,
        'half': True,
        'dnn': False,
        'source': None,
        'show': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'vid_stride': 1,
        'line_thickness': 3,
        'visualize': False,
        'augment': False,  # No augmentation during validation
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
    
    try:
        print("🔍 Starting validation...")
        results = model.val(**val_config)
        
        print("✅ Validation completed successfully!")
        print(f"📊 Validation results: {results}")
        
        return results
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """Main training function."""
    print("🎯 PCB Defect Detection - Standard Baseline Training")
    print("=" * 60)
    print("Model: YOLOv8n (Standard Ultralytics)")
    print("Loss: Default CIoU + BCE + DFL")
    print("=" * 60)
    
    # Setup environment
    device = setup_environment()
    
    # Train standard model
    training_results = train_standard_model(device)
    
    # Validate the model
    model_path = training_results.save_dir / 'weights' / 'best.pt'
    if model_path.exists():
        validation_results = validate_model(str(model_path), device)
    else:
        print("⚠️  Best model weights not found, skipping validation")
    
    print("\n🎉 Standard baseline training pipeline completed!")
    print("📝 Next steps:")
    print("1. Review training metrics in W&B dashboard")
    print("2. Analyze model performance on test set")
    print("3. Compare with custom loss baseline")
    print("4. Proceed to attention-enhanced models")


if __name__ == "__main__":
    main()
