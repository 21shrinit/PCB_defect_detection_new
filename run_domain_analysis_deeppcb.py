#!/usr/bin/env python3
"""
Domain Adaptation Analysis Script: HRIPCB → DeepPCB
=================================================

This script conducts a rigorous domain adaptation study by:
1. Zero-shot evaluation of HRIPCB-trained model on DeepPCB
2. Fine-tuning the model on DeepPCB dataset
3. Post-tuning evaluation to measure performance improvement
4. Comprehensive analysis and reporting

Usage:
    python run_domain_analysis_deeppcb.py --weights path/to/best.pt --data-yaml path/to/deeppcb_data.yaml --epochs 30

Author: Research Team
Date: 2025
"""

import argparse
import os
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml

try:
    from ultralytics import YOLO
    import torch
    import numpy as np
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Please install: pip install ultralytics torch numpy")
    exit(1)


class DomainAdaptationAnalyzer:
    """
    Comprehensive domain adaptation analyzer for PCB defect detection models.
    """
    
    def __init__(self, weights_path: str, data_yaml: str, epochs: int = 30, output_dir: str = "domain_analysis_results"):
        """
        Initialize the domain adaptation analyzer.
        
        Args:
            weights_path: Path to pre-trained HRIPCB model weights
            data_yaml: Path to DeepPCB dataset YAML configuration
            epochs: Number of epochs for fine-tuning
            output_dir: Base directory for saving results
        """
        self.weights_path = Path(weights_path)
        self.data_yaml = Path(data_yaml)
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"domain_adaptation_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Validate inputs
        self.validate_inputs()
        
        # Results storage
        self.results = {
            "experiment_info": {
                "weights_path": str(self.weights_path),
                "data_yaml": str(self.data_yaml),
                "epochs": self.epochs,
                "timestamp": timestamp,
                "pytorch_version": torch.__version__,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "zero_shot": {},
            "fine_tuning": {},
            "post_tuning": {},
            "analysis": {}
        }
        
        self.logger.info(f"🚀 Domain Adaptation Analysis initialized")
        self.logger.info(f"📁 Results will be saved to: {self.run_dir}")

    def setup_logging(self):
        """Setup comprehensive logging system."""
        log_file = self.run_dir / "domain_analysis.log"
        
        # Create logger
        self.logger = logging.getLogger("DomainAdaptation")
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def validate_inputs(self):
        """Validate all input paths and configurations."""
        # Check weights file
        if not self.weights_path.exists():
            raise FileNotFoundError(f"❌ Weights file not found: {self.weights_path}")
        
        # Check data YAML
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"❌ Data YAML not found: {self.data_yaml}")
        
        # Load and validate YAML structure
        try:
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
                
            required_keys = ['train', 'val', 'test', 'nc', 'names']
            missing_keys = [key for key in required_keys if key not in data_config]
            
            if missing_keys:
                raise ValueError(f"❌ Missing required keys in YAML: {missing_keys}")
                
            self.logger.info(f"✅ Dataset configuration validated")
            self.logger.info(f"📊 Dataset info: {data_config['nc']} classes - {data_config['names']}")
            
            # Store dataset info
            self.results["experiment_info"]["dataset_info"] = {
                "num_classes": data_config['nc'],
                "class_names": data_config['names'],
                "train_path": data_config.get('train', 'N/A'),
                "val_path": data_config.get('val', 'N/A'),
                "test_path": data_config.get('test', 'N/A')
            }
            
        except Exception as e:
            raise ValueError(f"❌ Invalid YAML configuration: {e}")

    def load_model(self, weights_path: Optional[str] = None) -> YOLO:
        """
        Load YOLO model from weights.
        
        Args:
            weights_path: Optional custom weights path
            
        Returns:
            Loaded YOLO model
        """
        weights = weights_path or str(self.weights_path)
        try:
            model = YOLO(weights)
            self.logger.info(f"✅ Model loaded successfully from: {weights}")
            return model
        except Exception as e:
            self.logger.error(f"❌ Failed to load model from {weights}: {e}")
            raise

    def run_zero_shot_evaluation(self) -> Dict[str, Any]:
        """
        Step 1: Evaluate pre-trained model on DeepPCB without fine-tuning.
        
        Returns:
            Zero-shot evaluation results
        """
        self.logger.info("🔍 Step 1: Starting Zero-Shot Evaluation")
        
        # Load pre-trained model
        model = self.load_model()
        
        # Create zero-shot results directory
        zeroshot_dir = self.run_dir / "zeroshot_evaluation"
        zeroshot_dir.mkdir(exist_ok=True)
        
        try:
            # Run validation on test set
            self.logger.info(f"📝 Running validation on DeepPCB test set...")
            
            validation_results = model.val(
                data=str(self.data_yaml),
                split='test',
                save=True,
                save_json=True,
                plots=True,
                verbose=True,
                project=str(zeroshot_dir),
                name="zero_shot_results"
            )
            
            # Extract key metrics
            metrics = self.extract_validation_metrics(validation_results)
            
            self.logger.info("✅ Zero-shot evaluation completed")
            self.logger.info(f"📊 Key Results:")
            self.logger.info(f"   mAP@0.5: {metrics.get('mAP_50', 'N/A'):.4f}")
            self.logger.info(f"   mAP@0.5:0.95: {metrics.get('mAP_50_95', 'N/A'):.4f}")
            self.logger.info(f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
            self.logger.info(f"   Recall: {metrics.get('recall', 'N/A'):.4f}")
            
            # Store results
            self.results["zero_shot"] = {
                "metrics": metrics,
                "results_path": str(zeroshot_dir / "zero_shot_results"),
                "status": "completed"
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Zero-shot evaluation failed: {e}")
            self.results["zero_shot"]["status"] = "failed"
            self.results["zero_shot"]["error"] = str(e)
            raise

    def run_fine_tuning(self) -> Dict[str, Any]:
        """
        Step 2: Fine-tune the model on DeepPCB dataset.
        
        Returns:
            Fine-tuning results and metrics
        """
        self.logger.info("🔧 Step 2: Starting Fine-Tuning Process")
        
        # Load fresh pre-trained model
        model = self.load_model()
        
        # Create fine-tuning results directory
        finetune_dir = self.run_dir / "fine_tuning"
        finetune_dir.mkdir(exist_ok=True)
        
        try:
            # Fine-tuning hyperparameters
            finetune_params = {
                'data': str(self.data_yaml),
                'epochs': self.epochs,
                'lr0': 0.001,  # Low learning rate for fine-tuning
                'lrf': 0.01,   # Final learning rate factor
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'optimizer': 'AdamW',
                'patience': 10,
                'save': True,
                'save_period': 5,
                'cache': True,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'workers': 8,
                'project': str(finetune_dir),
                'name': 'domain_finetune',
                'exist_ok': True,
                'plots': True,
                'val': True
            }
            
            self.logger.info(f"🎯 Fine-tuning parameters:")
            for key, value in finetune_params.items():
                if key not in ['data', 'project', 'name']:  # Skip long paths
                    self.logger.info(f"   {key}: {value}")
            
            # Start fine-tuning
            self.logger.info(f"🚀 Starting fine-tuning for {self.epochs} epochs...")
            
            training_results = model.train(**finetune_params)
            
            # Extract training metrics
            train_metrics = self.extract_training_metrics(training_results)
            
            self.logger.info("✅ Fine-tuning completed successfully")
            self.logger.info(f"📊 Final Training Results:")
            self.logger.info(f"   Best mAP@0.5: {train_metrics.get('best_mAP_50', 'N/A')}")
            self.logger.info(f"   Best mAP@0.5:0.95: {train_metrics.get('best_mAP_50_95', 'N/A')}")
            
            # Store results
            best_weights_path = finetune_dir / "domain_finetune" / "weights" / "best.pt"
            
            self.results["fine_tuning"] = {
                "metrics": train_metrics,
                "best_weights": str(best_weights_path),
                "training_path": str(finetune_dir / "domain_finetune"),
                "hyperparameters": finetune_params,
                "status": "completed"
            }
            
            return train_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Fine-tuning failed: {e}")
            self.results["fine_tuning"]["status"] = "failed"
            self.results["fine_tuning"]["error"] = str(e)
            raise

    def run_post_tuning_evaluation(self) -> Dict[str, Any]:
        """
        Step 3: Evaluate fine-tuned model on test set.
        
        Returns:
            Post-tuning evaluation results
        """
        self.logger.info("📈 Step 3: Starting Post-Tuning Evaluation")
        
        # Get fine-tuned model path
        if "best_weights" not in self.results["fine_tuning"]:
            raise ValueError("❌ Fine-tuning must be completed before post-tuning evaluation")
        
        finetuned_weights = self.results["fine_tuning"]["best_weights"]
        
        # Load fine-tuned model
        model = self.load_model(finetuned_weights)
        
        # Create post-tuning results directory
        posttuning_dir = self.run_dir / "post_tuning_evaluation"
        posttuning_dir.mkdir(exist_ok=True)
        
        try:
            # Run validation on test set
            self.logger.info(f"📝 Evaluating fine-tuned model on DeepPCB test set...")
            
            validation_results = model.val(
                data=str(self.data_yaml),
                split='test',
                save=True,
                save_json=True,
                plots=True,
                verbose=True,
                project=str(posttuning_dir),
                name="post_tuning_results"
            )
            
            # Extract key metrics
            metrics = self.extract_validation_metrics(validation_results)
            
            self.logger.info("✅ Post-tuning evaluation completed")
            self.logger.info(f"📊 Fine-Tuned Model Results:")
            self.logger.info(f"   mAP@0.5: {metrics.get('mAP_50', 'N/A'):.4f}")
            self.logger.info(f"   mAP@0.5:0.95: {metrics.get('mAP_50_95', 'N/A'):.4f}")
            self.logger.info(f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
            self.logger.info(f"   Recall: {metrics.get('recall', 'N/A'):.4f}")
            
            # Store results
            self.results["post_tuning"] = {
                "metrics": metrics,
                "results_path": str(posttuning_dir / "post_tuning_results"),
                "status": "completed"
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Post-tuning evaluation failed: {e}")
            self.results["post_tuning"]["status"] = "failed"
            self.results["post_tuning"]["error"] = str(e)
            raise

    def extract_validation_metrics(self, results) -> Dict[str, float]:
        """Extract key metrics from YOLO validation results."""
        try:
            if hasattr(results, 'results_dict'):
                # Method 1: Use results_dict
                results_dict = results.results_dict
                return {
                    'mAP_50': float(results_dict.get('metrics/mAP50(B)', 0)),
                    'mAP_50_95': float(results_dict.get('metrics/mAP50-95(B)', 0)),
                    'precision': float(results_dict.get('metrics/precision(B)', 0)),
                    'recall': float(results_dict.get('metrics/recall(B)', 0))
                }
            elif hasattr(results, 'box'):
                # Method 2: Use box metrics
                box = results.box
                return {
                    'mAP_50': float(box.map50) if hasattr(box, 'map50') else 0,
                    'mAP_50_95': float(box.map) if hasattr(box, 'map') else 0,
                    'precision': float(box.mp) if hasattr(box, 'mp') else 0,
                    'recall': float(box.mr) if hasattr(box, 'mr') else 0
                }
            else:
                # Fallback
                return {'mAP_50': 0, 'mAP_50_95': 0, 'precision': 0, 'recall': 0}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract validation metrics: {e}")
            return {'mAP_50': 0, 'mAP_50_95': 0, 'precision': 0, 'recall': 0}

    def extract_training_metrics(self, results) -> Dict[str, float]:
        """Extract key metrics from YOLO training results."""
        try:
            # Try to get metrics from trainer
            if hasattr(results, 'trainer') and hasattr(results.trainer, 'best_fitness'):
                trainer = results.trainer
                return {
                    'best_fitness': float(trainer.best_fitness) if trainer.best_fitness else 0,
                    'best_mAP_50': float(getattr(trainer, 'best_map50', 0)),
                    'best_mAP_50_95': float(getattr(trainer, 'best_map', 0)),
                    'final_loss': float(getattr(trainer, 'loss', 0))
                }
            else:
                # Fallback metrics
                return {
                    'best_fitness': 0,
                    'best_mAP_50': 0,
                    'best_mAP_50_95': 0,
                    'final_loss': 0
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not extract training metrics: {e}")
            return {'best_fitness': 0, 'best_mAP_50': 0, 'best_mAP_50_95': 0, 'final_loss': 0}

    def analyze_domain_adaptation(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of domain adaptation results.
        
        Returns:
            Analysis results and insights
        """
        self.logger.info("🔬 Step 4: Conducting Domain Adaptation Analysis")
        
        try:
            zero_shot = self.results["zero_shot"]["metrics"]
            post_tuning = self.results["post_tuning"]["metrics"]
            
            # Calculate improvements
            improvements = {}
            for metric in ['mAP_50', 'mAP_50_95', 'precision', 'recall']:
                zero_val = zero_shot.get(metric, 0)
                post_val = post_tuning.get(metric, 0)
                
                # Absolute improvement
                abs_improvement = post_val - zero_val
                
                # Relative improvement
                rel_improvement = (abs_improvement / zero_val * 100) if zero_val > 0 else 0
                
                improvements[metric] = {
                    'zero_shot': zero_val,
                    'post_tuning': post_val,
                    'absolute_improvement': abs_improvement,
                    'relative_improvement': rel_improvement
                }
            
            # Overall assessment
            overall_mAP_improvement = improvements['mAP_50_95']['relative_improvement']
            
            if overall_mAP_improvement > 20:
                adaptation_quality = "Excellent"
            elif overall_mAP_improvement > 10:
                adaptation_quality = "Good"
            elif overall_mAP_improvement > 5:
                adaptation_quality = "Moderate"
            elif overall_mAP_improvement > 0:
                adaptation_quality = "Minimal"
            else:
                adaptation_quality = "Poor"
            
            analysis = {
                "improvements": improvements,
                "overall_assessment": {
                    "adaptation_quality": adaptation_quality,
                    "primary_improvement": overall_mAP_improvement,
                    "best_performing_metric": max(improvements.keys(), 
                                                key=lambda k: improvements[k]['relative_improvement'])
                },
                "recommendations": self.generate_recommendations(improvements)
            }
            
            # Log analysis
            self.logger.info("🎯 Domain Adaptation Analysis Complete")
            self.logger.info(f"📊 Overall Assessment: {adaptation_quality}")
            self.logger.info(f"🚀 mAP@0.5:0.95 Improvement: {overall_mAP_improvement:.2f}%")
            
            for metric, data in improvements.items():
                self.logger.info(f"   {metric}: {data['zero_shot']:.4f} → {data['post_tuning']:.4f} "
                               f"({data['relative_improvement']:+.2f}%)")
            
            self.results["analysis"] = analysis
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {e}")
            raise

    def generate_recommendations(self, improvements: Dict) -> list:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        mAP_improvement = improvements['mAP_50_95']['relative_improvement']
        precision_improvement = improvements['precision']['relative_improvement']
        recall_improvement = improvements['recall']['relative_improvement']
        
        if mAP_improvement < 5:
            recommendations.append(
                "Consider longer fine-tuning (50+ epochs) or different learning rate schedule"
            )
            recommendations.append(
                "Investigate domain gap: analyze dataset differences in detail"
            )
        
        if precision_improvement < 0:
            recommendations.append(
                "Increase classification loss weight or use focal loss to improve precision"
            )
        
        if recall_improvement < 0:
            recommendations.append(
                "Consider data augmentation or increase detection confidence threshold"
            )
        
        if mAP_improvement > 15:
            recommendations.append(
                "Excellent adaptation! Consider this as production-ready model"
            )
        
        return recommendations

    def save_comprehensive_report(self):
        """Save detailed analysis report and results."""
        # Save JSON results
        results_file = self.run_dir / "domain_adaptation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate markdown report
        self.generate_markdown_report()
        
        # Copy important files
        self.organize_output_files()
        
        self.logger.info(f"📄 Comprehensive report saved to: {self.run_dir}")

    def generate_markdown_report(self):
        """Generate detailed markdown report."""
        report_path = self.run_dir / "DOMAIN_ADAPTATION_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write(f"""# Domain Adaptation Analysis Report
## HRIPCB → DeepPCB Transfer Learning Study

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Experiment Configuration
- **Source Model:** {self.weights_path.name}
- **Target Dataset:** DeepPCB
- **Fine-tuning Epochs:** {self.epochs}
- **Device:** {self.results['experiment_info']['device']}

### Dataset Information
- **Classes:** {self.results['experiment_info']['dataset_info']['num_classes']}
- **Class Names:** {', '.join(self.results['experiment_info']['dataset_info']['class_names'])}

### Results Summary

#### Zero-Shot Performance (Baseline)
""")
            
            if "metrics" in self.results["zero_shot"]:
                zero_metrics = self.results["zero_shot"]["metrics"]
                f.write(f"""
| Metric | Value |
|--------|-------|
| mAP@0.5 | {zero_metrics['mAP_50']:.4f} |
| mAP@0.5:0.95 | {zero_metrics['mAP_50_95']:.4f} |
| Precision | {zero_metrics['precision']:.4f} |
| Recall | {zero_metrics['recall']:.4f} |
""")
            
            f.write(f"""
#### Post Fine-Tuning Performance
""")
            
            if "metrics" in self.results["post_tuning"]:
                post_metrics = self.results["post_tuning"]["metrics"]
                f.write(f"""
| Metric | Value |
|--------|-------|
| mAP@0.5 | {post_metrics['mAP_50']:.4f} |
| mAP@0.5:0.95 | {post_metrics['mAP_50_95']:.4f} |
| Precision | {post_metrics['precision']:.4f} |
| Recall | {post_metrics['recall']:.4f} |
""")
            
            if "analysis" in self.results:
                analysis = self.results["analysis"]
                f.write(f"""
#### Performance Improvements

| Metric | Before | After | Absolute Δ | Relative Δ |
|--------|--------|-------|------------|------------|
""")
                
                for metric, data in analysis["improvements"].items():
                    f.write(f"| {metric} | {data['zero_shot']:.4f} | {data['post_tuning']:.4f} | "
                           f"{data['absolute_improvement']:+.4f} | {data['relative_improvement']:+.2f}% |\n")
                
                f.write(f"""
### Analysis Summary
- **Adaptation Quality:** {analysis['overall_assessment']['adaptation_quality']}
- **Best Improvement:** {analysis['overall_assessment']['primary_improvement']:.2f}% in mAP@0.5:0.95
- **Top Performing Metric:** {analysis['overall_assessment']['best_performing_metric']}

### Recommendations
""")
                
                for i, rec in enumerate(analysis["recommendations"], 1):
                    f.write(f"{i}. {rec}\n")
            
            f.write(f"""
### File Organization
```
{self.run_dir.name}/
├── domain_adaptation_results.json          # Complete results data
├── DOMAIN_ADAPTATION_REPORT.md            # This report
├── domain_analysis.log                    # Detailed execution log
├── zeroshot_evaluation/                   # Zero-shot validation results
├── fine_tuning/                          # Training logs and weights
├── post_tuning_evaluation/               # Fine-tuned model validation
└── summary_plots/                        # Key visualizations
```

### Technical Details
- **PyTorch Version:** {self.results['experiment_info']['pytorch_version']}
- **Execution Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Hardware:** {self.results['experiment_info']['device'].upper()}
""")
        
        self.logger.info(f"📝 Markdown report generated: {report_path}")

    def organize_output_files(self):
        """Organize and copy important output files."""
        summary_dir = self.run_dir / "summary_plots"
        summary_dir.mkdir(exist_ok=True)
        
        # Copy key plots if they exist
        plots_to_copy = [
            "confusion_matrix.png",
            "results.png", 
            "val_batch0_labels.jpg",
            "val_batch0_pred.jpg"
        ]
        
        for phase in ["zeroshot_evaluation", "post_tuning_evaluation"]:
            phase_dir = self.run_dir / phase
            if phase_dir.exists():
                for subdir in phase_dir.rglob("*"):
                    if subdir.is_dir():
                        for plot_name in plots_to_copy:
                            plot_path = subdir / plot_name
                            if plot_path.exists():
                                dest_path = summary_dir / f"{phase}_{plot_name}"
                                shutil.copy2(plot_path, dest_path)

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Execute the complete domain adaptation analysis pipeline.
        
        Returns:
            Complete results dictionary
        """
        try:
            self.logger.info("🎯 Starting Complete Domain Adaptation Analysis")
            self.logger.info("=" * 60)
            
            # Step 1: Zero-shot evaluation
            zero_shot_metrics = self.run_zero_shot_evaluation()
            
            # Step 2: Fine-tuning
            fine_tuning_metrics = self.run_fine_tuning()
            
            # Step 3: Post-tuning evaluation
            post_tuning_metrics = self.run_post_tuning_evaluation()
            
            # Step 4: Analysis
            analysis_results = self.analyze_domain_adaptation()
            
            # Step 5: Generate comprehensive report
            self.save_comprehensive_report()
            
            self.logger.info("=" * 60)
            self.logger.info("🎉 Domain Adaptation Analysis COMPLETED Successfully!")
            self.logger.info(f"📁 All results saved to: {self.run_dir}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Analysis pipeline failed: {e}")
            self.results["status"] = "failed"
            self.results["error"] = str(e)
            raise


def main():
    """Main execution function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Domain Adaptation Analysis: HRIPCB → DeepPCB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_domain_analysis_deeppcb.py --weights models/hripcb_best.pt --data-yaml configs/deeppcb_data.yaml

    python run_domain_analysis_deeppcb.py --weights runs/train/exp1/weights/best.pt --data-yaml data/deeppcb_data.yaml --epochs 50
        """
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to pre-trained HRIPCB model weights (best.pt file)'
    )
    
    parser.add_argument(
        '--data-yaml',
        type=str,
        required=True,
        help='Path to DeepPCB dataset YAML configuration file'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs for fine-tuning (default: 30)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='domain_analysis_results',
        help='Base directory for saving results (default: domain_analysis_results)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = DomainAdaptationAnalyzer(
            weights_path=args.weights,
            data_yaml=args.data_yaml,
            epochs=args.epochs,
            output_dir=args.output_dir
        )
        
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 DOMAIN ADAPTATION ANALYSIS SUMMARY")
        print("="*80)
        
        if "analysis" in results and "improvements" in results["analysis"]:
            improvements = results["analysis"]["improvements"]
            print(f"📊 Performance Improvements:")
            print(f"   mAP@0.5:     {improvements['mAP_50']['zero_shot']:.4f} → "
                  f"{improvements['mAP_50']['post_tuning']:.4f} "
                  f"({improvements['mAP_50']['relative_improvement']:+.2f}%)")
            print(f"   mAP@0.5:0.95: {improvements['mAP_50_95']['zero_shot']:.4f} → "
                  f"{improvements['mAP_50_95']['post_tuning']:.4f} "
                  f"({improvements['mAP_50_95']['relative_improvement']:+.2f}%)")
            
            assessment = results["analysis"]["overall_assessment"]
            print(f"\n🎯 Overall Assessment: {assessment['adaptation_quality']}")
            print(f"📈 Primary Improvement: {assessment['primary_improvement']:.2f}% in mAP@0.5:0.95")
        
        print(f"\n📁 Detailed results available in: {analyzer.run_dir}")
        print(f"📄 See DOMAIN_ADAPTATION_REPORT.md for comprehensive analysis")
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())