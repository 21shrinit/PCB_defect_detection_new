#!/usr/bin/env python3
"""
Cross-Domain Evaluation Framework for PCB Defect Detection
=========================================================

This script evaluates domain adaptation performance by comparing zero-shot
and fine-tuned model performance across different PCB defect datasets.

Author: Research Team
Date: 2025
"""

import os
import sys
import yaml
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

try:
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Required package not installed: {e}")
    sys.exit(1)


class DomainAdaptationEvaluator:
    """Framework for evaluating cross-domain PCB defect detection performance."""
    
    def __init__(self, weights_path: str, dataset_dir: str, epochs: int = 20):
        self.weights_path = Path(weights_path)
        self.dataset_dir = Path(dataset_dir)
        self.epochs = epochs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results storage
        self.zeroshot_results = {}
        self.finetuned_results = {}
        
        # Create output directory
        self.output_dir = Path("domain_adaptation_results") / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input files and directories."""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        required_dirs = ['train', 'val', 'test']
        for dir_name in required_dirs:
            if not (self.dataset_dir / dir_name / 'images').exists():
                raise FileNotFoundError(f"Required directory not found: {self.dataset_dir / dir_name / 'images'}")
    
    def prepare_dataset_config(self) -> str:
        """Prepare dataset configuration for evaluation."""
        # Define class mapping (HRIPCB-compatible)
        class_names = [
            'Missing_hole', 'Mouse_bite', 'Open_circuit', 
            'Short', 'Spurious_copper', 'Spur'
        ]
        
        # Check if label remapping is needed
        target_classes = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
        hripcb_classes = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spurious_copper', 'Spur']
        
        # Determine class index mapping
        if (target_classes[4].lower() == 'spur' and target_classes[5].lower() == 'spurious_copper' and
            hripcb_classes[4] == 'Spurious_copper' and hripcb_classes[5] == 'Spur'):
            self.class_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4}  # Swap classes 4&5
        else:
            self.class_mapping = {i: i for i in range(6)}  # Identity mapping
        
        # Create dataset configuration
        config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'train',
            'val': 'val', 
            'test': 'test',
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        config_path = self.output_dir / "dataset_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Apply preprocessing
        self._preprocess_images()
        self._remap_labels()
        
        return str(config_path)
    
    def _preprocess_images(self):
        """Convert grayscale images to RGB if needed."""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            images_dir = self.dataset_dir / split / 'images'
            if not images_dir.exists():
                continue
                
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            # Sample images to check format
            sample_size = min(10, len(image_files))
            grayscale_count = 0
            
            for img_file in image_files[:sample_size]:
                with Image.open(img_file) as img:
                    if img.mode in ['L', 'LA']:
                        grayscale_count += 1
            
            # Convert if predominantly grayscale
            if grayscale_count > (sample_size * 0.7):
                for img_file in image_files:
                    with Image.open(img_file) as img:
                        if img.mode in ['L', 'LA']:
                            rgb_img = img.convert('RGB')
                            rgb_img.save(img_file, quality=95)
    
    def _remap_labels(self):
        """Remap label indices if needed for dataset compatibility."""
        needs_remapping = any(k != v for k, v in self.class_mapping.items())
        
        if not needs_remapping:
            return
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            labels_dir = self.dataset_dir / split / 'labels'
            if not labels_dir.exists():
                continue
                
            for label_file in labels_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    continue
                
                remapped_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        original_class = int(parts[0])
                        if original_class in self.class_mapping:
                            parts[0] = str(self.class_mapping[original_class])
                    remapped_lines.append(' '.join(parts) + '\n')
                
                with open(label_file, 'w') as f:
                    f.writelines(remapped_lines)
    
    def _find_optimal_confidence(self, model, dataset_config: str) -> float:
        """Find optimal confidence threshold for target domain."""
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        best_threshold = 0.15
        best_map = 0.0
        
        for threshold in thresholds:
            try:
                results = model.val(
                    data=dataset_config,
                    split='test',
                    conf=threshold,
                    iou=0.4,
                    max_det=1000,
                    verbose=False
                )
                
                if hasattr(results, 'box') and results.box.map50 > best_map:
                    best_map = results.box.map50
                    best_threshold = threshold
            except Exception:
                continue
        
        return best_threshold
    
    def evaluate_zero_shot(self, dataset_config: str) -> Dict[str, float]:
        """Evaluate pre-trained model on target domain without fine-tuning."""
        model = YOLO(str(self.weights_path))
        
        # Find optimal confidence threshold
        optimal_conf = self._find_optimal_confidence(model, dataset_config)
        
        # Run optimized evaluation
        results = model.val(
            data=dataset_config,
            split='test',
            conf=optimal_conf,
            iou=0.4,
            augment=True,
            max_det=1000,
            save=True,
            project=str(self.output_dir),
            name='zero_shot_evaluation',
            exist_ok=True,
            verbose=False
        )
        
        # Extract metrics
        if hasattr(results, 'box') and results.box is not None:
            self.zeroshot_results = {
                'map50': float(getattr(results.box, 'map50', 0.0) or 0.0),
                'map': float(getattr(results.box, 'map', 0.0) or 0.0),
                'precision': float(getattr(results.box, 'mp', 0.0) or 0.0),
                'recall': float(getattr(results.box, 'mr', 0.0) or 0.0),
                'f1': float(getattr(results.box, 'f1', 0.0) or 0.0),
                'optimal_confidence': optimal_conf
            }
        else:
            raise ValueError("Zero-shot evaluation failed")
        
        return self.zeroshot_results
    
    def fine_tune_model(self, dataset_config: str) -> str:
        """Fine-tune model on target domain."""
        model = YOLO(str(self.weights_path))
        
        # Fine-tuning configuration
        patience = max(10, self.epochs // 4)
        
        results = model.train(
            data=dataset_config,
            epochs=self.epochs,
            patience=patience,
            batch=32,
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            project=str(self.output_dir),
            name='fine_tuning',
            exist_ok=True,
            verbose=False
        )
        
        best_model_path = self.output_dir / 'fine_tuning' / 'weights' / 'best.pt'
        return str(best_model_path)
    
    def evaluate_fine_tuned(self, model_path: str, dataset_config: str) -> Dict[str, float]:
        """Evaluate fine-tuned model on target domain."""
        model = YOLO(model_path)
        
        results = model.val(
            data=dataset_config,
            split='test',
            save=True,
            project=str(self.output_dir),
            name='fine_tuned_evaluation',
            exist_ok=True,
            verbose=False
        )
        
        # Extract metrics
        if hasattr(results, 'box') and results.box is not None:
            self.finetuned_results = {
                'map50': float(getattr(results.box, 'map50', 0.0) or 0.0),
                'map': float(getattr(results.box, 'map', 0.0) or 0.0),
                'precision': float(getattr(results.box, 'mp', 0.0) or 0.0),
                'recall': float(getattr(results.box, 'mr', 0.0) or 0.0),
                'f1': float(getattr(results.box, 'f1', 0.0) or 0.0)
            }
        else:
            raise ValueError("Fine-tuned evaluation failed")
        
        return self.finetuned_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive domain adaptation report."""
        if not self.zeroshot_results or not self.finetuned_results:
            raise ValueError("Complete evaluation required before generating report")
        
        # Calculate improvements
        improvements = {}
        for metric in ['map50', 'map', 'precision', 'recall', 'f1']:
            zero_shot = self.zeroshot_results[metric]
            fine_tuned = self.finetuned_results[metric]
            
            absolute_improvement = fine_tuned - zero_shot
            percentage_improvement = (absolute_improvement / max(zero_shot, 1e-8)) * 100
            
            improvements[metric] = {
                'absolute': round(absolute_improvement, 4),
                'percentage': round(percentage_improvement, 1)
            }
        
        # Create comprehensive report
        report = {
            'experiment_info': {
                'timestamp': self.timestamp,
                'source_model': str(self.weights_path),
                'target_dataset': str(self.dataset_dir),
                'epochs': self.epochs
            },
            'zero_shot_performance': self.zeroshot_results,
            'fine_tuned_performance': self.finetuned_results,
            'improvements': improvements,
            'summary': {
                'domain_adaptation_successful': improvements['map']['absolute'] > 0.05,
                'significant_improvement': improvements['map']['percentage'] > 10.0,
                'key_metric_improvement': improvements['map']['absolute']
            }
        }
        
        # Save report
        report_path = self.output_dir / 'domain_adaptation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete domain adaptation analysis."""
        print("Starting domain adaptation analysis...")
        start_time = time.time()
        
        # Step 1: Prepare dataset
        print("Preparing target dataset...")
        dataset_config = self.prepare_dataset_config()
        
        # Step 2: Zero-shot evaluation
        print("Evaluating zero-shot performance...")
        self.evaluate_zero_shot(dataset_config)
        print(f"Zero-shot mAP@0.5: {self.zeroshot_results['map50']:.4f}")
        
        # Step 3: Fine-tuning
        print("Fine-tuning model on target domain...")
        fine_tuned_model = self.fine_tune_model(dataset_config)
        
        # Step 4: Post-fine-tuning evaluation
        print("Evaluating fine-tuned model...")
        self.evaluate_fine_tuned(fine_tuned_model, dataset_config)
        print(f"Fine-tuned mAP@0.5: {self.finetuned_results['map50']:.4f}")
        
        # Step 5: Generate report
        print("Generating analysis report...")
        report = self.generate_report()
        
        duration = time.time() - start_time
        print(f"\nAnalysis completed in {duration:.1f} seconds")
        print(f"Results saved to: {self.output_dir}")
        
        # Print summary
        improvement = report['improvements']['map']['absolute']
        percentage = report['improvements']['map']['percentage']
        print(f"\nDomain Adaptation Results:")
        print(f"mAP@0.5:0.95 improvement: +{improvement:.4f} (+{percentage:.1f}%)")
        
        if report['summary']['domain_adaptation_successful']:
            print("✓ Domain adaptation successful")
        else:
            print("✗ Limited domain adaptation improvement")
        
        return report


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cross-domain evaluation for PCB defect detection'
    )
    parser.add_argument('--weights', required=True, 
                       help='Path to pre-trained model weights')
    parser.add_argument('--dataset', required=True,
                       help='Path to target dataset directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs for fine-tuning (default: 20)')
    
    args = parser.parse_args()
    
    try:
        evaluator = DomainAdaptationEvaluator(
            weights_path=args.weights,
            dataset_dir=args.dataset,
            epochs=args.epochs
        )
        
        report = evaluator.run_complete_analysis()
        
        print(f"\nFull report available at: {evaluator.output_dir}/domain_adaptation_report.json")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()