#!/usr/bin/env python3
"""
Simple Domain Adaptation Script for PCB Defect Detection
=======================================================

Fine-tunes a HRIPCB-trained YOLO model on the XD-PCB dataset.

Usage:
    python run_simple_domain_adaptation.py --weights path/to/hripcb_model.pt

The script performs:
1. Creates xd_pcb.yaml dataset configuration for XD-Real subset
2. Fine-tunes the provided model on XD-PCB dataset
3. Saves results with descriptive naming

XD-Real Dataset:
- Classes: 2 (open_circuit=1, short_circuit=2) 
- Total images: ~246 images
- Class distribution: open_circuit=208, short_circuit=118

Author: PCB Defect Detection Team
Date: January 2025
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


def create_xd_pcb_yaml(output_path="xd_pcb.yaml"):
    """
    Create XD-PCB dataset configuration file for the XD-Real subset.
    
    Args:
        output_path (str): Path where to save the YAML configuration
    """
    # Get absolute paths for the XD-Real dataset
    current_dir = Path(__file__).parent.absolute()
    xd_real_path = current_dir / "datasets" / "XD-PCB" / "XD-Real"
    
    # XD-PCB dataset configuration for XD-Real subset
    xd_pcb_config = {
        'path': str(xd_real_path),
        'train': 'images',  # All images are in single directory for now
        'val': 'images',    # Using same directory for validation (will split during training)
        'test': 'images',   # Using same directory for testing
        
        # XD-Real only has 2 defect classes (YOLO requires 0-based indexing)
        'nc': 2,
        'names': {
            0: 'open_circuit',     # Class ID 0 (mapped from XD-Real class 1)
            1: 'short_circuit'     # Class ID 1 (mapped from XD-Real class 2)
        }
    }
    
    # Write configuration to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(xd_pcb_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created XD-PCB dataset configuration: {output_path}")
    print(f"   Dataset path: {xd_real_path}")
    print(f"   Classes: {xd_pcb_config['nc']} - {list(xd_pcb_config['names'].values())}")
    print(f"⚠️  NOTE: XD-PCB labels use 1-based indexing (1,2) but YOLO requires 0-based (0,1)")
    print(f"   Make sure label files are converted: class 1→0, class 2→1")
    
    return output_path


def validate_xd_pcb_dataset(dataset_path):
    """
    Validate XD-PCB dataset structure and content before training.
    
    Args:
        dataset_path (Path): Path to XD-PCB dataset directory
        
    Returns:
        dict: Dataset statistics and validation results
    """
    print(f"🔍 Validating XD-PCB dataset at: {dataset_path}")
    
    # Check directory structure
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Get file lists
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    label_files = list(labels_dir.glob("*.txt"))
    
    # Basic statistics
    stats = {
        'total_images': len(image_files),
        'total_labels': len(label_files),
        'class_counts': {0: 0, 1: 0},  # open_circuit, short_circuit
        'total_instances': 0,
        'images_without_labels': [],
        'labels_without_images': [],
        'invalid_labels': []
    }
    
    # Check image-label matching
    image_stems = {f.stem for f in image_files}
    label_stems = {f.stem for f in label_files}
    
    stats['images_without_labels'] = list(image_stems - label_stems)
    stats['labels_without_images'] = list(label_stems - image_stems)
    
    # Analyze label content
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    stats['invalid_labels'].append(f"{label_file.name}:{line_num} - Invalid format")
                    continue
                
                try:
                    class_id = int(parts[0])
                    if class_id in [0, 1]:  # Valid classes after conversion
                        stats['class_counts'][class_id] += 1
                        stats['total_instances'] += 1
                    elif class_id in [1, 2]:  # Original XD-PCB format (needs conversion)
                        mapped_class = class_id - 1  # 1->0, 2->1
                        stats['class_counts'][mapped_class] += 1
                        stats['total_instances'] += 1
                    else:
                        stats['invalid_labels'].append(f"{label_file.name}:{line_num} - Invalid class {class_id}")
                except ValueError:
                    stats['invalid_labels'].append(f"{label_file.name}:{line_num} - Non-numeric class")
                    
        except Exception as e:
            stats['invalid_labels'].append(f"{label_file.name} - Read error: {str(e)}")
    
    # Print validation results
    print(f"📊 Dataset Validation Results:")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Total labels: {stats['total_labels']}")
    print(f"   Total instances: {stats['total_instances']}")
    print(f"   Class distribution:")
    print(f"     open_circuit (0): {stats['class_counts'][0]} instances")
    print(f"     short_circuit (1): {stats['class_counts'][1]} instances")
    
    # Report issues
    if stats['images_without_labels']:
        print(f"⚠️  {len(stats['images_without_labels'])} images without labels")
    if stats['labels_without_images']:
        print(f"⚠️  {len(stats['labels_without_images'])} labels without images")
    if stats['invalid_labels']:
        print(f"⚠️  {len(stats['invalid_labels'])} invalid label entries:")
        for issue in stats['invalid_labels'][:5]:  # Show first 5 issues
            print(f"      {issue}")
        if len(stats['invalid_labels']) > 5:
            print(f"      ... and {len(stats['invalid_labels']) - 5} more")
    
    # Validation summary
    if stats['total_images'] == 0:
        raise ValueError("No images found in dataset")
    if stats['total_instances'] == 0:
        raise ValueError("No valid instances found in labels")
    if stats['class_counts'][0] == 0 and stats['class_counts'][1] == 0:
        raise ValueError("No valid class instances found")
    
    print("✅ Dataset validation passed")
    return stats


def convert_xd_pcb_labels_to_yolo_format(dataset_path):
    """
    Convert XD-PCB label files from 1-based indexing to 0-based indexing for YOLO.
    
    Args:
        dataset_path (Path): Path to XD-PCB dataset directory
    """
    labels_dir = dataset_path / "labels"
    if not labels_dir.exists():
        print(f"⚠️  Labels directory not found: {labels_dir}")
        return
    
    converted_count = 0
    label_files = list(labels_dir.glob("*.txt"))
    
    print(f"🔄 Converting {len(label_files)} label files from 1-based to 0-based indexing...")
    
    for label_file in label_files:
        try:
            # Read original content
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Convert class indices: 1->0, 2->1
            converted_lines = []
            file_modified = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 5:  # Valid YOLO format: class x y w h
                    class_id = int(parts[0])
                    if class_id == 1:
                        parts[0] = '0'  # open_circuit: 1 -> 0
                        file_modified = True
                    elif class_id == 2:
                        parts[0] = '1'  # short_circuit: 2 -> 1
                        file_modified = True
                    converted_lines.append(' '.join(parts) + '\n')
                else:
                    converted_lines.append(line + '\n')
            
            # Write back if modified
            if file_modified:
                with open(label_file, 'w') as f:
                    f.writelines(converted_lines)
                converted_count += 1
                
        except Exception as e:
            print(f"⚠️  Error converting {label_file}: {e}")
    
    print(f"✅ Converted {converted_count} label files to 0-based indexing")


def extract_model_name(weights_path):
    """
    Extract a clean model name from the weights path for naming the output directory.
    
    Args:
        weights_path (str): Path to the model weights file
        
    Returns:
        str: Clean model name for directory naming
    """
    path = Path(weights_path)
    
    # Remove .pt extension and extract meaningful name
    model_name = path.stem
    
    # Clean up common patterns in model names
    model_name = model_name.replace('best', '').replace('last', '').replace('weights', '')
    model_name = model_name.replace('_', '').replace('-', '').strip()
    
    # If name is empty after cleaning, use parent directory name
    if not model_name or model_name in ['', 'best', 'last']:
        parent_dirs = path.parts[-3:-1]  # Get experiment directory names
        model_name = '_'.join([d for d in parent_dirs if d not in ['weights', 'runs', 'train']])
        
    # Fallback to generic name if still empty
    if not model_name:
        model_name = 'hripcb_model'
        
    return model_name


def fine_tune_model(weights_path, dataset_config_path, baseline_mode=False):
    """
    Fine-tune a model on XD-PCB dataset.
    
    Args:
        weights_path (str): Path to the pre-trained model weights (None for baseline)
        dataset_config_path (str): Path to the XD-PCB dataset configuration
        baseline_mode (bool): Whether using default YOLOv8n (baseline test)
        
    Returns:
        object: Training results from YOLO
    """
    if baseline_mode:
        print(f"🤖 Loading default YOLOv8n model (baseline test)")
        model = YOLO('yolov8n.pt')  # Default YOLOv8n
        base_model_name = "yolov8n_default"
        output_name = f"{base_model_name}-baseline-xdpcb"
    else:
        print(f"🤖 Loading pre-trained model from: {weights_path}")
        model = YOLO(weights_path)
        base_model_name = extract_model_name(weights_path)
        output_name = f"{base_model_name}-finetune-xdpcb"
    
    if baseline_mode:
        print(f"🎯 Starting baseline training (default YOLOv8n)...")
        print(f"   Source: Default COCO weights (80 classes)")
        print(f"   Target domain: XD-PCB (2 classes)")
    else:
        print(f"🎯 Starting domain adaptation fine-tuning...")
        print(f"   Source domain: HRIPCB (6 classes)")
        print(f"   Target domain: XD-PCB (2 classes)")
    print(f"   Output directory: runs/train/{output_name}")
    
    # Fine-tuning hyperparameters optimized for domain adaptation
    training_args = {
        'data': dataset_config_path,
        'epochs': 50,                    # Reduced epochs for fine-tuning
        'batch': 16,                     # Conservative batch size
        'imgsz': 640,                    # Standard input size
        'lr0': 0.001,                    # Lower learning rate for fine-tuning
        'lrf': 0.01,                     # Final learning rate factor
        'momentum': 0.937,               # Standard momentum
        'weight_decay': 0.0005,          # Light regularization
        'warmup_epochs': 3,              # Short warmup for fine-tuning
        'patience': 15,                  # Early stopping patience
        'device': '0',                   # Use GPU 0
        'workers': 8,                    # Data loading workers
        'project': 'runs/train',         # Project directory
        'name': output_name,             # Experiment name
        'exist_ok': True,                # Allow overwriting existing runs
        'pretrained': False,             # Already using pre-trained weights
        'optimizer': 'AdamW',            # Modern optimizer for fine-tuning
        'verbose': True,                 # Detailed logging
        'save_period': 10,               # Save checkpoint every 10 epochs
        'val': True,                     # Enable validation
        'plots': True,                   # Generate training plots
        'save_json': True,               # Save results in JSON format
        
        # Conservative augmentation for domain adaptation
        'hsv_h': 0.010,                  # Minimal hue variation
        'hsv_s': 0.3,                    # Reduced saturation variation  
        'hsv_v': 0.2,                    # Reduced brightness variation
        'degrees': 0,                    # No rotation
        'translate': 0.05,               # Minimal translation
        'scale': 0.2,                    # Minimal scaling
        'shear': 0,                      # No shearing
        'perspective': 0,                # No perspective transform
        'flipud': 0,                     # No vertical flip
        'fliplr': 0.5,                   # 50% horizontal flip
        'mixup': 0,                      # No mixup (can confuse domain adaptation)
        'copy_paste': 0,                 # No copy-paste augmentation
    }
    
    print(f"📊 Training configuration:")
    print(f"   Epochs: {training_args['epochs']}")
    print(f"   Learning rate: {training_args['lr0']}")
    print(f"   Batch size: {training_args['batch']}")
    print(f"   Optimizer: {training_args['optimizer']}")
    print(f"   Patience: {training_args['patience']}")
    
    # Start fine-tuning
    results = model.train(**training_args)
    
    print(f"✅ Fine-tuning completed!")
    print(f"   Results saved to: {results.save_dir}")
    print(f"   Best weights: {results.save_dir}/weights/best.pt")
    print(f"   Last weights: {results.save_dir}/weights/last.pt")
    
    return results


def main():
    """Main function with argument parsing and execution."""
    parser = argparse.ArgumentParser(
        description="Fine-tune HRIPCB-trained YOLO model on XD-PCB dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Domain adaptation from HRIPCB model:
  python run_simple_domain_adaptation.py --weights runs/train/exp/weights/best.pt
  python run_simple_domain_adaptation.py --weights models/hripcb_eca_best.pt
  
  # Baseline test with default YOLOv8n:
  python run_simple_domain_adaptation.py --baseline
  
  # Validate XD-PCB dataset only (no training):
  python run_simple_domain_adaptation.py --validate-only
        """
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        required=False,
        default=None,
        help='Path to the pre-trained model weights (.pt file) trained on HRIPCB dataset. If not provided, uses default YOLOv8n'
    )
    
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Use default YOLOv8n model without any pre-training (baseline test)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate the XD-PCB dataset without training'
    )
    
    args = parser.parse_args()
    
    # Handle validation-only mode
    if args.validate_only:
        print("=" * 80)
        print("🔍 XD-PCB DATASET VALIDATION")
        print("=" * 80)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        try:
            # Just validate the dataset
            xd_real_path = Path(__file__).parent.absolute() / "datasets" / "XD-PCB" / "XD-Real"
            dataset_stats = validate_xd_pcb_dataset(xd_real_path)
            
            print("=" * 80)
            print("✅ DATASET VALIDATION COMPLETED")
            print("=" * 80)
            print(f"📊 Summary: {dataset_stats['total_images']} images, {dataset_stats['total_instances']} instances")
            print(f"🏷️  Classes: open_circuit={dataset_stats['class_counts'][0]}, short_circuit={dataset_stats['class_counts'][1]}")
            
        except Exception as e:
            print(f"❌ Dataset validation failed: {str(e)}")
            sys.exit(1)
        
        sys.exit(0)
    
    # Determine if using baseline or pre-trained model
    if args.baseline:
        weights_path = None
        print("🔍 Using default YOLOv8n model (baseline test)")
    elif args.weights:
        # Validate weights file exists
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"❌ Error: Weights file not found: {weights_path}")
            print(f"Please check the path and ensure the .pt file exists.")
            sys.exit(1)
        
        if not str(weights_path).endswith('.pt'):
            print(f"❌ Error: Weights file must be a .pt file, got: {weights_path}")
            sys.exit(1)
    else:
        print(f"❌ Error: Must provide either --weights path, --baseline flag, or --validate-only")
        print(f"   Use --weights for domain adaptation from HRIPCB model")
        print(f"   Use --baseline for default YOLOv8n comparison")
        print(f"   Use --validate-only to check dataset without training")
        sys.exit(1)
    
    print("=" * 80)
    if args.baseline:
        print("🚀 PCB DEFECT DETECTION - BASELINE TEST")
        print("   Training default YOLOv8n on XD-PCB dataset")
        print("=" * 80)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Source model: Default YOLOv8n (COCO pretrained)")
    else:
        print("🚀 PCB DEFECT DETECTION - DOMAIN ADAPTATION")
        print("   Fine-tuning HRIPCB model on XD-PCB dataset")
        print("=" * 80)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Source model: {weights_path}")
    print("")
    
    try:
        # Step 1: Create XD-PCB dataset configuration
        print("📋 Step 1: Creating XD-PCB dataset configuration...")
        yaml_config_path = create_xd_pcb_yaml("xd_pcb.yaml")
        print("")
        
        # Step 2: Validate XD-PCB dataset
        print("🔍 Step 2: Validating XD-PCB dataset...")
        xd_real_path = Path(__file__).parent.absolute() / "datasets" / "XD-PCB" / "XD-Real"
        dataset_stats = validate_xd_pcb_dataset(xd_real_path)
        print("")
        
        # Step 3: Convert XD-PCB labels to YOLO format (0-based indexing)
        print("🔄 Step 3: Converting XD-PCB labels to YOLO format...")
        convert_xd_pcb_labels_to_yolo_format(xd_real_path)
        print("")
        
        # Step 4: Fine-tune the model
        if args.baseline:
            print("🎯 Step 4: Training default YOLOv8n on XD-PCB dataset...")
            results = fine_tune_model(None, yaml_config_path, baseline_mode=True)
        else:
            print("🎯 Step 4: Fine-tuning model on XD-PCB dataset...")
            results = fine_tune_model(str(weights_path), yaml_config_path, baseline_mode=False)
        print("")
        
        # Success summary
        print("=" * 80)
        if args.baseline:
            print("🎉 BASELINE TRAINING COMPLETED SUCCESSFULLY!")
        else:
            print("🎉 DOMAIN ADAPTATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📊 Training Results:")
        print(f"   Experiment directory: {results.save_dir}")
        print(f"   Best model: {results.save_dir}/weights/best.pt")
        print(f"   Training plots: {results.save_dir}/*.png")
        print(f"   Results JSON: {results.save_dir}/results.json")
        print("")
        if args.baseline:
            print(f"🔄 Baseline Training Summary:")
            print(f"   Source: Default COCO weights (80 classes)")
            print(f"   Target: XD-PCB dataset (2 classes: open_circuit, short_circuit)")
            print(f"   Training: 50 epochs from scratch with conservative hyperparameters")
        else:
            print(f"🔄 Domain Adaptation Summary:")
            print(f"   Source: HRIPCB dataset (6 classes)")
            print(f"   Target: XD-PCB dataset (2 classes: open_circuit, short_circuit)")
            print(f"   Fine-tuning: 50 epochs with conservative hyperparameters")
        print("")
        print(f"📈 Next Steps:")
        print(f"   1. Evaluate the fine-tuned model: python -c \"from ultralytics import YOLO; model = YOLO('{results.save_dir}/weights/best.pt'); model.val()\"")
        print(f"   2. Test inference: python -c \"from ultralytics import YOLO; model = YOLO('{results.save_dir}/weights/best.pt'); model.predict('path/to/test/image.jpg')\"")
        print(f"   3. Export for deployment: python -c \"from ultralytics import YOLO; model = YOLO('{results.save_dir}/weights/best.pt'); model.export(format='onnx')\"")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\\n⚠️  Training interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Domain adaptation failed: {str(e)}")
        print(f"Please check the error message above and ensure:")
        print(f"   1. The weights file is a valid YOLO model")
        print(f"   2. The XD-PCB dataset exists in datasets/XD-PCB/XD-Real/")
        print(f"   3. You have sufficient GPU memory/disk space")
        sys.exit(1)


if __name__ == "__main__":
    main()