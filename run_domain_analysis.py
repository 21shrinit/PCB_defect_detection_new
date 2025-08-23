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
    
    def __init__(self, weights_path: str, dataset_dir: str, epochs: int = 20):
        """
        Initialize the domain adaptation analyzer
        
        Args:
            weights_path: Path to the pre-trained model weights (best.pt)
            dataset_dir: Root directory of the MIXED PCB DEFECT DATASET
            epochs: Number of epochs for fine-tuning
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
        
        # Define the six defect classes common with HRIPCB
        class_names = [
            'missing_hole',
            'mouse_bite', 
            'open_circuit',
            'short',
            'spur',
            'spurious_copper'
        ]
        
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
        
        # Validate dataset structure
        self._validate_dataset_structure()
        
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
            else:
                logger.warning(f"⚠️ Missing directories for {split} split")
    
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
            
            # Extract key metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1': float(results.box.f1.mean() if hasattr(results.box, 'f1') else 0.0),
                'class_maps': results.box.maps.tolist() if hasattr(results.box, 'maps') else []
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
    
    args = parser.parse_args()
    
    try:
        # Initialize and run domain adaptation analyzer
        analyzer = DomainAdaptationAnalyzer(
            weights_path=args.weights,
            dataset_dir=args.dataset_dir,
            epochs=args.epochs
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