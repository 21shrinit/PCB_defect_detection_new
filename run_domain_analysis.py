#!/usr/bin/env python3
"""
Domain Adaptation Analysis Script for PCB Defect Detection
===========================================================

This script conducts a comprehensive domain adaptation study using the Ultralytics YOLO framework.
It evaluates how well a model pre-trained on the HRIPCB dataset generalizes to the "MIXED PCB DEFECT DATASET"
and measures the performance improvement after fine-tuning.

Author: Claude Code
Date: 2025-08-23
"""

import os
import sys
import argparse
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: Ultralytics not installed. Please install with: pip install ultralytics")
    sys.exit(1)

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('domain_adaptation_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


class DomainAdaptationAnalyzer:
    """
    Comprehensive domain adaptation analyzer for PCB defect detection models
    """
    
    def __init__(self, weights_path: str, dataset_dir: str, epochs: int = 20, output_dir: str = None):
        """
        Initialize the domain adaptation analyzer
        
        Args:
            weights_path: Path to the pre-trained model weights (best.pt)
            dataset_dir: Root directory of the MIXED PCB DEFECT DATASET
            epochs: Number of epochs for fine-tuning
            output_dir: Custom output directory (optional, defaults to runs/detect/domain_adaptation)
        """
        self.weights_path = Path(weights_path)
        self.dataset_dir = Path(dataset_dir)
        self.epochs = epochs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Performance tracking
        self.zeroshot_results = {}
        self.finetuned_results = {}
        
        # Validate inputs
        self._validate_inputs()
        
        # Create output directories
        if output_dir:
            self.output_dir = Path(output_dir) / self.timestamp
        else:
            self.output_dir = Path("runs/detect/domain_adaptation") / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Domain Adaptation Analysis initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _validate_inputs(self):
        """Validate input paths and requirements"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"❌ Weights file not found: {self.weights_path}")
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"❌ Dataset directory not found: {self.dataset_dir}")
        
        # Check for required dataset structure
        required_dirs = ['train', 'val', 'test']
        for dir_name in required_dirs:
            if not (self.dataset_dir / dir_name).exists():
                logger.warning(f"⚠️ Directory not found: {self.dataset_dir / dir_name}")
        
        logger.info(f"✅ Input validation completed")
    
    def prepare_target_dataset(self) -> str:
        """
        Prepare the MIXED PCB DEFECT DATASET for Ultralytics
        
        Returns:
            Path to the created mixed_pcb_data.yaml file
        """
        logger.info("📋 Step: Preparing target dataset configuration")
        
        # CRITICAL: Map DeepPCB classes to HRIPCB training classes
        # DeepPCB has: [copper, mousebite, open, pin-hole, short, spur]
        # HRIPCB has:  [Missing_hole, Mouse_bite, Open_circuit, Short, Spurious_copper, Spur]
        
        logger.info("🔧 CRITICAL: Creating DeepPCB to HRIPCB class mapping")
        logger.info("📊 DeepPCB original classes: copper, mousebite, open, pin-hole, short, spur")
        logger.info("📊 HRIPCB training classes: Missing_hole, Mouse_bite, Open_circuit, Short, Spurious_copper, Spur")
        
        # Map DeepPCB classes to HRIPCB equivalents with proper indices
        deeppcb_to_hripcb_mapping = {
            0: ('copper', 'Spurious_copper'),      # copper -> Spurious_copper  
            1: ('mousebite', 'Mouse_bite'),        # mousebite -> Mouse_bite
            2: ('open', 'Open_circuit'),           # open -> Open_circuit  
            3: ('pin-hole', 'Missing_hole'),       # pin-hole -> Missing_hole
            4: ('short', 'Short'),                 # short -> Short (exact match)
            5: ('spur', 'Spur')                    # spur -> Spur (exact match)
        }
        
        # Create HRIPCB-compatible class names in correct order for model compatibility
        class_names = [
            'Missing_hole',      # 0: HRIPCB index 0 (mapped from DeepPCB pin-hole)
            'Mouse_bite',        # 1: HRIPCB index 1 (mapped from DeepPCB mousebite)  
            'Open_circuit',      # 2: HRIPCB index 2 (mapped from DeepPCB open)
            'Short',             # 3: HRIPCB index 3 (mapped from DeepPCB short)
            'Spurious_copper',   # 4: HRIPCB index 4 (mapped from DeepPCB copper)
            'Spur'               # 5: HRIPCB index 5 (mapped from DeepPCB spur)
        ]
        
        logger.info("✅ Class mapping created:")
        for i, class_name in enumerate(class_names):
            original_deeppcb = [k for k, v in deeppcb_to_hripcb_mapping.items() if v[1] == class_name]
            if original_deeppcb:
                deeppcb_name = deeppcb_to_hripcb_mapping[original_deeppcb[0]][0]
                logger.info(f"   {i}: {deeppcb_name} -> {class_name}")
            else:
                logger.info(f"   {i}: {class_name}")
        
        # CRITICAL: We need to remap DeepPCB label indices to HRIPCB indices
        # DeepPCB -> HRIPCB index mapping:
        self.class_index_mapping = {
            0: 4,  # copper (DeepPCB 0) -> Spurious_copper (HRIPCB 4)
            1: 1,  # mousebite (DeepPCB 1) -> Mouse_bite (HRIPCB 1)  
            2: 2,  # open (DeepPCB 2) -> Open_circuit (HRIPCB 2)
            3: 0,  # pin-hole (DeepPCB 3) -> Missing_hole (HRIPCB 0)
            4: 3,  # short (DeepPCB 4) -> Short (HRIPCB 3)
            5: 5   # spur (DeepPCB 5) -> Spur (HRIPCB 5)
        }
        
        logger.info("🔧 Index remapping required:")
        for deeppcb_idx, hripcb_idx in self.class_index_mapping.items():
            logger.info(f"   DeepPCB {deeppcb_idx} -> HRIPCB {hripcb_idx}")
        
        logger.info("⚠️  WARNING: Label files need index remapping before training!")
        
        # Create dataset configuration
        dataset_config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        # Save configuration file
        config_path = self.output_dir / "mixed_pcb_data.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✅ Dataset configuration created: {config_path}")
        logger.info(f"📊 Classes defined: {list(dataset_config['names'].values())}")
        logger.info(f"🔍 Class mapping verification:")
        for idx, name in dataset_config['names'].items():
            logger.info(f"   {idx}: {name}")
        
        # Validate dataset structure
        self._validate_dataset_structure()
        
        # Check for grayscale images and preprocess if needed
        self._preprocess_grayscale_images()
        
        # Remap label indices to match HRIPCB training
        self._remap_label_indices()
        
        return str(config_path)
    
    def _validate_dataset_structure(self):
        """Validate the dataset has the expected structure"""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            images_dir = self.dataset_dir / split / 'images'
            labels_dir = self.dataset_dir / split / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                image_count = len(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
                label_count = len(list(labels_dir.glob('*.txt')))
                logger.info(f"📂 {split}: {image_count} images, {label_count} labels")
                
                # Validate label format (check a few sample labels)
                if label_count > 0:
                    sample_labels = list(labels_dir.glob('*.txt'))[:3]  # Check first 3 labels
                    for label_file in sample_labels:
                        try:
                            with open(label_file, 'r') as f:
                                lines = f.readlines()
                                if lines:
                                    first_line = lines[0].strip().split()
                                    if len(first_line) >= 5:
                                        class_id = int(first_line[0])
                                        if 0 <= class_id <= 5:
                                            logger.info(f"✅ {split} labels format validated - class_id range: 0-5")
                                        else:
                                            logger.warning(f"⚠️ {split} has class_id {class_id} outside expected range 0-5")
                                        break
                        except Exception as e:
                            logger.warning(f"⚠️ Error reading label {label_file}: {e}")
            else:
                logger.warning(f"⚠️ Missing directories for {split} split")
    
    def _preprocess_grayscale_images(self):
        """Check for and convert grayscale images to RGB format"""
        logger.info("🔍 Checking for grayscale images in dataset...")
        
        # Check if preprocessing already done
        preprocessing_marker = self.output_dir / ".grayscale_preprocessed"
        if preprocessing_marker.exists():
            logger.info("✅ Grayscale preprocessing already completed (marker found)")
            return
        
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            logger.error("❌ PIL (Pillow) required for image preprocessing. Install with: pip install Pillow")
            return
        
        splits = ['train', 'val', 'test']
        total_images_processed = 0
        total_grayscale_converted = 0
        
        for split in splits:
            images_dir = self.dataset_dir / split / 'images'
            
            if not images_dir.exists():
                logger.warning(f"⚠️ Images directory not found: {images_dir}")
                continue
                
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpeg'))
            split_processed = 0
            split_converted = 0
            
            logger.info(f"🔍 Checking {len(image_files)} images in {split} split...")
            
            # Sample a few images to determine if dataset is grayscale
            sample_size = min(10, len(image_files))
            grayscale_count = 0
            
            for img_file in image_files[:sample_size]:
                try:
                    with Image.open(img_file) as img:
                        if img.mode in ['L', 'LA']:  # Grayscale modes
                            grayscale_count += 1
                        elif img.mode == 'RGB':
                            # Check if RGB image is actually grayscale (all channels equal)
                            img_array = np.array(img)
                            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                                if np.array_equal(img_array[:,:,0], img_array[:,:,1]) and np.array_equal(img_array[:,:,1], img_array[:,:,2]):
                                    grayscale_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error checking image {img_file}: {e}")
            
            # Determine if dataset is predominantly grayscale
            is_grayscale_dataset = grayscale_count > (sample_size * 0.7)  # 70% threshold
            
            if is_grayscale_dataset:
                logger.info(f"📊 {split} split detected as GRAYSCALE dataset ({grayscale_count}/{sample_size} samples)")
                logger.info(f"🔄 Converting grayscale images to RGB format for HRIPCB model compatibility...")
                
                # Convert all images in this split
                for img_file in image_files:
                    try:
                        with Image.open(img_file) as img:
                            original_mode = img.mode
                            
                            # Convert grayscale to RGB
                            if img.mode in ['L', 'LA']:
                                rgb_img = img.convert('RGB')
                                rgb_img.save(img_file, quality=95)  # High quality to preserve details
                                split_converted += 1
                            elif img.mode == 'RGB':
                                # Check if RGB is actually grayscale
                                img_array = np.array(img)
                                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                                    if np.array_equal(img_array[:,:,0], img_array[:,:,1]) and np.array_equal(img_array[:,:,1], img_array[:,:,2]):
                                        # Already RGB format but grayscale content - keep as is
                                        pass
                                    
                            split_processed += 1
                            
                    except Exception as e:
                        logger.error(f"❌ Error converting image {img_file}: {e}")
                        
            else:
                logger.info(f"📊 {split} split detected as RGB dataset ({sample_size - grayscale_count}/{sample_size} samples)")
                split_processed = len(image_files)
            
            logger.info(f"✅ {split}: Processed {split_processed} images, converted {split_converted} from grayscale")
            total_images_processed += split_processed
            total_grayscale_converted += split_converted
        
        logger.info(f"🎯 Image preprocessing completed:")
        logger.info(f"   📁 Total images processed: {total_images_processed}")
        logger.info(f"   🔄 Grayscale images converted: {total_grayscale_converted}")
        
        if total_grayscale_converted > 0:
            logger.info(f"   ✅ Dataset converted from GRAYSCALE to RGB for HRIPCB model compatibility")
        else:
            logger.info(f"   ✅ Dataset already in RGB format")
        
        # Create marker file
        with open(preprocessing_marker, 'w') as f:
            f.write(f"Grayscale preprocessing completed at {datetime.now().isoformat()}\n")
            f.write(f"Images processed: {total_images_processed}\n")
            f.write(f"Grayscale converted: {total_grayscale_converted}\n")
    
    def _remap_label_indices(self):
        """Remap DeepPCB label indices to match HRIPCB training indices"""
        
        # Check if remapping has already been done (safety check)
        remapping_marker = self.output_dir / ".labels_remapped"
        if remapping_marker.exists():
            logger.info("✅ Label remapping already completed (marker found)")
            return
            
        logger.info("🔄 Starting label index remapping (DeepPCB -> HRIPCB)")
        
        splits = ['train', 'val', 'test']
        total_files_processed = 0
        total_labels_remapped = 0
        
        for split in splits:
            labels_dir = self.dataset_dir / split / 'labels'
            
            if not labels_dir.exists():
                logger.warning(f"⚠️ Labels directory not found: {labels_dir}")
                continue
                
            label_files = list(labels_dir.glob('*.txt'))
            split_processed = 0
            split_remapped = 0
            
            logger.info(f"🔄 Processing {len(label_files)} label files in {split} split...")
            
            for label_file in label_files:
                try:
                    # Read original labels
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    if not lines:
                        continue
                        
                    # Remap indices and write back
                    remapped_lines = []
                    file_had_remapping = False
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            original_class_id = int(parts[0])
                            
                            if original_class_id in self.class_index_mapping:
                                new_class_id = self.class_index_mapping[original_class_id]
                                parts[0] = str(new_class_id)
                                if original_class_id != new_class_id:
                                    file_had_remapping = True
                                    split_remapped += 1
                                    
                            remapped_lines.append(' '.join(parts) + '\n')
                        else:
                            remapped_lines.append(line)  # Keep invalid lines as-is
                    
                    # Write back remapped labels
                    with open(label_file, 'w') as f:
                        f.writelines(remapped_lines)
                    
                    split_processed += 1
                    if file_had_remapping:
                        total_labels_remapped += split_remapped
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {label_file}: {e}")
            
            logger.info(f"✅ {split}: Processed {split_processed} files, remapped {split_remapped} labels")
            total_files_processed += split_processed
        
        logger.info(f"🎯 Label remapping completed:")
        logger.info(f"   📁 Total files processed: {total_files_processed}")
        logger.info(f"   🔄 Total labels remapped: {total_labels_remapped}")
        logger.info(f"   ✅ Dataset ready for HRIPCB-trained model evaluation")
        
        # Create marker file to indicate remapping is done
        with open(remapping_marker, 'w') as f:
            f.write(f"Label remapping completed at {datetime.now().isoformat()}\n")
            f.write(f"Files processed: {total_files_processed}\n")
            f.write(f"Labels remapped: {total_labels_remapped}\n")
    
    def run_zeroshot_evaluation(self, dataset_config: str) -> Dict[str, Any]:
        """
        Step 1: Zero-shot evaluation (baseline performance)
        
        Args:
            dataset_config: Path to the dataset YAML configuration
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("🎯 Step 1: Zero-Shot Evaluation (Baseline)")
        
        try:
            # Load pre-trained model
            model = YOLO(str(self.weights_path))
            logger.info(f"✅ Loaded pre-trained model: {self.weights_path}")
            
            # Create zeroshot evaluation directory
            zeroshot_dir = self.output_dir / "zeroshot_evaluation"
            zeroshot_dir.mkdir(exist_ok=True)
            
            # Run validation on test set
            logger.info("🔍 Running zero-shot evaluation on MIXED PCB test set...")
            results = model.val(
                data=dataset_config,
                split='test',
                save=True,
                save_json=True,
                project=str(zeroshot_dir.parent),
                name=zeroshot_dir.name,
                exist_ok=True,
                verbose=True
            )
            
            # Extract key metrics with validation
            logger.info("🔍 Extracting zero-shot evaluation metrics...")
            logger.info(f"📊 Raw results type: {type(results)}")
            logger.info(f"📊 Has box attribute: {hasattr(results, 'box')}")
            
            if hasattr(results, 'box') and results.box is not None:
                metrics = {
                    'mAP50': float(results.box.map50),
                    'mAP50_95': float(results.box.map),
                    'precision': float(results.box.mp),
                    'recall': float(results.box.mr),
                    'f1': float(results.box.f1.mean() if hasattr(results.box, 'f1') else 0.0),
                    'class_maps': results.box.maps.tolist() if hasattr(results.box, 'maps') else []
                }
                logger.info(f"📊 Successful metric extraction - mAP50: {metrics['mAP50']:.4f}")
            else:
                logger.error("❌ No box results found in zero-shot evaluation!")
                metrics = {
                    'mAP50': 0.0,
                    'mAP50_95': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'class_maps': []
                }
            
            self.zeroshot_results = metrics
            
            # Save detailed results
            results_file = zeroshot_dir / "zeroshot_metrics.json"
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Print results
            logger.info("📊 Zero-Shot Evaluation Results:")
            logger.info(f"   mAP@0.5: {metrics['mAP50']:.4f}")
            logger.info(f"   mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
            logger.info(f"   Precision: {metrics['precision']:.4f}")
            logger.info(f"   Recall: {metrics['recall']:.4f}")
            logger.info(f"   F1-Score: {metrics['f1']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error during zero-shot evaluation: {str(e)}")
            raise
    
    def run_finetuning(self, dataset_config: str) -> str:
        """
        Step 2: Fine-tuning on the new domain
        
        Args:
            dataset_config: Path to the dataset YAML configuration
            
        Returns:
            Path to the fine-tuned model weights
        """
        logger.info("🎯 Step 2: Fine-Tuning on MIXED PCB Dataset")
        
        try:
            # Load pre-trained model for fine-tuning
            model = YOLO(str(self.weights_path))
            logger.info(f"✅ Loaded model for fine-tuning: {self.weights_path}")
            
            # Create fine-tuning directory
            finetune_dir = self.output_dir / "finetune_on_mixed_pcb"
            finetune_dir.mkdir(exist_ok=True)
            
            # Fine-tuning hyperparameters optimized for domain adaptation
            logger.info(f"🔧 Starting fine-tuning for {self.epochs} epochs...")
            results = model.train(
                data=dataset_config,
                epochs=self.epochs,
                patience=max(10, self.epochs // 4),  # Adaptive patience
                batch=32,  # Smaller batch for fine-tuning stability
                imgsz=640,
                device='0',
                workers=8,
                
                # Fine-tuning optimized learning parameters
                lr0=0.001,  # Low learning rate for fine-tuning
                lrf=0.01,   # Final learning rate factor
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                
                # Training configuration
                save=True,
                save_period=5,  # Save checkpoints every 5 epochs
                cache=False,    # Avoid memory issues
                amp=True,       # Mixed precision
                
                # Output configuration
                project=str(finetune_dir.parent),
                name=finetune_dir.name,
                exist_ok=True,
                verbose=True
            )
            
            # Get path to best fine-tuned model
            best_model_path = finetune_dir / "weights" / "best.pt"
            
            if best_model_path.exists():
                logger.info(f"✅ Fine-tuning completed successfully")
                logger.info(f"💾 Best model saved to: {best_model_path}")
                return str(best_model_path)
            else:
                raise FileNotFoundError(f"Fine-tuned model not found at: {best_model_path}")
                
        except Exception as e:
            logger.error(f"❌ Error during fine-tuning: {str(e)}")
            raise
    
    def run_post_finetune_evaluation(self, finetuned_weights: str, dataset_config: str) -> Dict[str, Any]:
        """
        Step 3: Post-fine-tuning evaluation
        
        Args:
            finetuned_weights: Path to fine-tuned model weights
            dataset_config: Path to the dataset YAML configuration
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("🎯 Step 3: Post-Fine-Tuning Evaluation")
        
        try:
            # Load fine-tuned model
            model = YOLO(finetuned_weights)
            logger.info(f"✅ Loaded fine-tuned model: {finetuned_weights}")
            
            # Create post-finetune evaluation directory
            postfinetune_dir = self.output_dir / "post_finetune_evaluation"
            postfinetune_dir.mkdir(exist_ok=True)
            
            # Run validation on test set
            logger.info("🔍 Running post-fine-tuning evaluation on MIXED PCB test set...")
            results = model.val(
                data=dataset_config,
                split='test',
                save=True,
                save_json=True,
                project=str(postfinetune_dir.parent),
                name=postfinetune_dir.name,
                exist_ok=True,
                verbose=True
            )
            
            # Extract key metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1': float(results.box.f1.mean() if hasattr(results.box, 'f1') else 0.0),
                'class_maps': results.box.maps.tolist() if hasattr(results.box, 'maps') else []
            }
            
            self.finetuned_results = metrics
            
            # Save detailed results
            results_file = postfinetune_dir / "post_finetune_metrics.json"
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Print results
            logger.info("📊 Post-Fine-Tuning Evaluation Results:")
            logger.info(f"   mAP@0.5: {metrics['mAP50']:.4f}")
            logger.info(f"   mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
            logger.info(f"   Precision: {metrics['precision']:.4f}")
            logger.info(f"   Recall: {metrics['recall']:.4f}")
            logger.info(f"   F1-Score: {metrics['f1']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error during post-fine-tuning evaluation: {str(e)}")
            raise
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate final comparison report
        
        Returns:
            Dictionary containing the complete analysis results
        """
        logger.info("📋 Generating Final Comparison Report")
        
        if not self.zeroshot_results or not self.finetuned_results:
            raise ValueError("❌ Missing evaluation results. Cannot generate comparison report.")
        
        # Calculate improvements
        mAP50_improvement = self.finetuned_results['mAP50'] - self.zeroshot_results['mAP50']
        mAP50_95_improvement = self.finetuned_results['mAP50_95'] - self.zeroshot_results['mAP50_95']
        
        mAP50_improvement_pct = (mAP50_improvement / self.zeroshot_results['mAP50']) * 100
        mAP50_95_improvement_pct = (mAP50_95_improvement / self.zeroshot_results['mAP50_95']) * 100
        
        # Create comprehensive report
        report = {
            'analysis_timestamp': self.timestamp,
            'configuration': {
                'source_weights': str(self.weights_path),
                'target_dataset': str(self.dataset_dir),
                'finetuning_epochs': self.epochs
            },
            'zeroshot_performance': self.zeroshot_results,
            'finetuned_performance': self.finetuned_results,
            'improvements': {
                'mAP50_absolute': mAP50_improvement,
                'mAP50_95_absolute': mAP50_95_improvement,
                'mAP50_percentage': mAP50_improvement_pct,
                'mAP50_95_percentage': mAP50_95_improvement_pct
            },
            'summary': {
                'domain_adaptation_success': mAP50_95_improvement > 0,
                'significant_improvement': abs(mAP50_95_improvement_pct) > 5.0
            }
        }
        
        # Save complete report
        report_file = self.output_dir / "domain_adaptation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print formatted console report
        self._print_console_report(report)
        
        return report
    
    def _print_console_report(self, report: Dict[str, Any]):
        """Print formatted console report"""
        
        print("\n" + "="*80)
        print("🎯 DOMAIN ADAPTATION ANALYSIS - FINAL REPORT")
        print("="*80)
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏷️  Source Model: {self.weights_path.name}")
        print(f"📊 Target Dataset: {self.dataset_dir.name}")
        print(f"🔄 Fine-tuning Epochs: {self.epochs}")
        print("="*80)
        
        print("\n📈 PERFORMANCE COMPARISON:")
        print("-" * 60)
        print(f"{'Metric':<20} {'Zero-Shot':<12} {'Fine-Tuned':<12} {'Improvement':<15}")
        print("-" * 60)
        
        zs = report['zeroshot_performance']
        ft = report['finetuned_performance']
        imp = report['improvements']
        
        print(f"{'mAP@0.5':<20} {zs['mAP50']:<12.4f} {ft['mAP50']:<12.4f} {imp['mAP50_absolute']:>+7.4f} ({imp['mAP50_percentage']:>+6.1f}%)")
        print(f"{'mAP@0.5:0.95':<20} {zs['mAP50_95']:<12.4f} {ft['mAP50_95']:<12.4f} {imp['mAP50_95_absolute']:>+7.4f} ({imp['mAP50_95_percentage']:>+6.1f}%)")
        print(f"{'Precision':<20} {zs['precision']:<12.4f} {ft['precision']:<12.4f} {ft['precision']-zs['precision']:>+7.4f} ({((ft['precision']-zs['precision'])/zs['precision']*100):>+6.1f}%)")
        print(f"{'Recall':<20} {zs['recall']:<12.4f} {ft['recall']:<12.4f} {ft['recall']-zs['recall']:>+7.4f} ({((ft['recall']-zs['recall'])/zs['recall']*100):>+6.1f}%)")
        print(f"{'F1-Score':<20} {zs['f1']:<12.4f} {ft['f1']:<12.4f} {ft['f1']-zs['f1']:>+7.4f} ({((ft['f1']-zs['f1'])/zs['f1']*100):>+6.1f}%)")
        
        print("\n" + "="*80)
        print("🏆 DOMAIN ADAPTATION SUMMARY:")
        
        if report['summary']['domain_adaptation_success']:
            print("✅ Domain adaptation was SUCCESSFUL!")
            if report['summary']['significant_improvement']:
                print("🚀 Achieved SIGNIFICANT performance improvement (>5%)")
            else:
                print("📈 Achieved modest performance improvement")
        else:
            print("❌ Domain adaptation did not improve performance")
            print("💡 Consider: More epochs, different learning rates, or data augmentation")
        
        print(f"\n🎯 Key Result: mAP@0.5:0.95 improved by {imp['mAP50_95_absolute']:+.4f} ({imp['mAP50_95_percentage']:+.1f}%)")
        print(f"📁 Detailed results saved to: {self.output_dir}")
        print("="*80)
    
    def run_complete_analysis(self):
        """
        Run the complete domain adaptation analysis pipeline
        """
        try:
            logger.info("🚀 Starting Complete Domain Adaptation Analysis")
            
            # Step 0: Prepare dataset
            dataset_config = self.prepare_target_dataset()
            
            # Step 1: Zero-shot evaluation
            zeroshot_results = self.run_zeroshot_evaluation(dataset_config)
            
            # Step 2: Fine-tuning
            finetuned_weights = self.run_finetuning(dataset_config)
            
            # Step 3: Post-fine-tuning evaluation
            finetuned_results = self.run_post_finetune_evaluation(finetuned_weights, dataset_config)
            
            # Step 4: Generate comparison report
            final_report = self.generate_comparison_report()
            
            logger.info("🎉 Domain adaptation analysis completed successfully!")
            return final_report
            
        except Exception as e:
            logger.error(f"❌ Domain adaptation analysis failed: {str(e)}")
            raise


def main():
    """Main function with command-line interface"""
    
    parser = argparse.ArgumentParser(
        description="Domain Adaptation Analysis for PCB Defect Detection Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_domain_analysis.py --weights path/to/best.pt --dataset-dir path/to/mixed_pcb_dataset
    python run_domain_analysis.py --weights models/hripcb_best.pt --dataset-dir datasets/mixed_pcb --epochs 30
    python run_domain_analysis.py --weights best.pt --dataset-dir deeppcb_dataset --output-dir domain_results --epochs 30
        """
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to the best.pt file of the model pre-trained on HRIPCB'
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True,
        help='Path to the root directory of the unzipped "MIXED PCB DEFECT DATASET"'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs for fine-tuning (default: 20)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory for results (default: runs/detect/domain_adaptation)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize and run domain adaptation analyzer
        analyzer = DomainAdaptationAnalyzer(
            weights_path=args.weights,
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            output_dir=args.output_dir
        )
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Results saved to: {analyzer.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())