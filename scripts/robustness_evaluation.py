#!/usr/bin/env python3
"""
Robustness Evaluation Script for PCB Defect Detection
Tests trained models against various image degradations to evaluate robustness.
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustnessEvaluator:
    def __init__(self, test_images_dir: str, results_dir: str = "robustness_results"):
        self.test_images_dir = Path(test_images_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Store original images for processing
        self.original_images = []
        self.load_test_images()
        
    def load_test_images(self):
        """Load test images for degradation testing."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for ext in image_extensions:
            self.original_images.extend(list(self.test_images_dir.glob(f"**/*{ext}")))
            
        logger.info(f"📷 Loaded {len(self.original_images)} test images")
        
    def add_gaussian_noise(self, image: np.ndarray, noise_level: float = 25.0) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
        
    def add_gaussian_blur(self, image: np.ndarray, blur_kernel: int = 5, sigma: float = 1.5) -> np.ndarray:
        """Add Gaussian blur to image."""
        return cv2.GaussianBlur(image, (blur_kernel, blur_kernel), sigma)
        
    def add_motion_blur(self, image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Add motion blur to simulate camera shake."""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)
        
    def adjust_brightness_contrast(self, image: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
        """Adjust brightness and contrast."""
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted
        
    def create_degraded_dataset(self, degradation_type: str, intensity: str = "medium"):
        """Create degraded version of test dataset."""
        degraded_dir = self.results_dir / f"degraded_{degradation_type}_{intensity}"
        degraded_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔧 Creating degraded dataset: {degradation_type} ({intensity})")
        
        # Set degradation parameters based on intensity
        if degradation_type == "gaussian_noise":
            noise_levels = {"light": 15.0, "medium": 25.0, "heavy": 40.0}
            noise_level = noise_levels[intensity]
            
        elif degradation_type == "gaussian_blur":
            blur_params = {"light": (3, 0.8), "medium": (5, 1.5), "heavy": (7, 2.5)}
            kernel_size, sigma = blur_params[intensity]
            
        elif degradation_type == "motion_blur":
            kernel_sizes = {"light": 7, "medium": 15, "heavy": 23}
            kernel_size = kernel_sizes[intensity]
            
        elif degradation_type == "low_light":
            brightness_levels = {"light": -20, "medium": -40, "heavy": -60}
            contrast_levels = {"light": 0.8, "medium": 0.6, "heavy": 0.4}
            brightness = brightness_levels[intensity]
            contrast = contrast_levels[intensity]
            
        # Process each image
        for img_path in self.original_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            # Apply degradation
            if degradation_type == "gaussian_noise":
                degraded_image = self.add_gaussian_noise(image, noise_level)
            elif degradation_type == "gaussian_blur":
                degraded_image = self.add_gaussian_blur(image, kernel_size, sigma)  
            elif degradation_type == "motion_blur":
                degraded_image = self.add_motion_blur(image, kernel_size)
            elif degradation_type == "low_light":
                degraded_image = self.adjust_brightness_contrast(image, brightness, contrast)
            else:
                degraded_image = image
                
            # Save degraded image
            output_path = degraded_dir / img_path.name
            cv2.imwrite(str(output_path), degraded_image)
            
        logger.info(f"✅ Created {len(self.original_images)} degraded images")
        return str(degraded_dir)
        
    def evaluate_model(self, model_path: str, test_data_path: str) -> Dict:
        """Evaluate model on test data and return metrics."""
        try:
            model = YOLO(model_path)
            results = model.val(data=test_data_path, verbose=False)
            
            # Extract metrics
            if hasattr(results, 'box') and hasattr(results.box, 'map50'):
                return {
                    'mAP50': round(results.box.map50, 4),
                    'mAP50_95': round(results.box.map, 4), 
                    'precision': round(results.box.mp, 4),
                    'recall': round(results.box.mr, 4)
                }
            else:
                return {'mAP50': 0.0, 'mAP50_95': 0.0, 'precision': 0.0, 'recall': 0.0}
                
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {'mAP50': 0.0, 'mAP50_95': 0.0, 'precision': 0.0, 'recall': 0.0}
            
    def run_robustness_study(self, model_configs: Dict[str, str], original_data_config: str):
        """Run complete robustness evaluation."""
        degradation_types = ["gaussian_noise", "gaussian_blur", "motion_blur", "low_light"]
        intensities = ["light", "medium", "heavy"]
        
        results = {
            'original': {},
            'degraded': {}
        }
        
        # Test on original (clean) data
        logger.info("🧪 Testing on original clean data...")
        for model_name, model_path in model_configs.items():
            logger.info(f"  Testing {model_name}...")
            metrics = self.evaluate_model(model_path, original_data_config)
            results['original'][model_name] = metrics
            
        # Test on degraded data
        for degradation_type in degradation_types:
            results['degraded'][degradation_type] = {}
            
            for intensity in intensities:
                logger.info(f"🔧 Testing {degradation_type} ({intensity})...")
                results['degraded'][degradation_type][intensity] = {}
                
                # Create degraded dataset
                degraded_dataset_dir = self.create_degraded_dataset(degradation_type, intensity)
                
                # Create temporary data config for degraded images
                degraded_data_config = self.create_degraded_data_config(degraded_dataset_dir, original_data_config)
                
                # Test each model
                for model_name, model_path in model_configs.items():
                    logger.info(f"  Testing {model_name} on {degradation_type}_{intensity}...")
                    metrics = self.evaluate_model(model_path, degraded_data_config)
                    results['degraded'][degradation_type][intensity][model_name] = metrics
                    
        # Save results
        results_path = self.results_dir / "robustness_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"💾 Results saved to: {results_path}")
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
        
    def create_degraded_data_config(self, degraded_images_dir: str, original_config: str) -> str:
        """Create data config file pointing to degraded images."""
        # Read original config
        import yaml
        with open(original_config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update paths to point to degraded images
        config['test'] = f"{degraded_images_dir}"
        
        # Save temporary config
        temp_config_path = self.results_dir / "temp_degraded_config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        return str(temp_config_path)
        
    def generate_summary_report(self, results: Dict):
        """Generate human-readable summary report."""
        report_path = self.results_dir / "robustness_summary.md"
        
        with open(report_path, 'w') as f:
            f.write("# PCB Defect Detection - Robustness Evaluation Report\\n\\n")
            
            # Original performance
            f.write("## Performance on Clean Data\\n\\n")
            f.write("| Model | mAP50 | mAP50-95 | Precision | Recall |\\n")
            f.write("|-------|-------|----------|-----------|--------|\\n")
            
            for model_name, metrics in results['original'].items():
                f.write(f"| {model_name} | {metrics['mAP50']:.3f} | {metrics['mAP50_95']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} |\\n")
                
            # Degradation analysis
            f.write("\\n## Robustness Under Degradation\\n\\n")
            
            for degradation_type in results['degraded'].keys():
                f.write(f"### {degradation_type.replace('_', ' ').title()}\\n\\n")
                
                for intensity in ['light', 'medium', 'heavy']:
                    if intensity in results['degraded'][degradation_type]:
                        f.write(f"#### {intensity.title()} {degradation_type.replace('_', ' ')}\\n\\n")
                        f.write("| Model | mAP50 | Drop | mAP50-95 | Drop |\\n")
                        f.write("|-------|-------|------|----------|------|\\n")
                        
                        for model_name in results['original'].keys():
                            if model_name in results['degraded'][degradation_type][intensity]:
                                original_map50 = results['original'][model_name]['mAP50']
                                degraded_map50 = results['degraded'][degradation_type][intensity][model_name]['mAP50']
                                map50_drop = ((original_map50 - degraded_map50) / original_map50) * 100
                                
                                original_map50_95 = results['original'][model_name]['mAP50_95']
                                degraded_map50_95 = results['degraded'][degradation_type][intensity][model_name]['mAP50_95']
                                map50_95_drop = ((original_map50_95 - degraded_map50_95) / original_map50_95) * 100
                                
                                f.write(f"| {model_name} | {degraded_map50:.3f} | {map50_drop:.1f}% | {degraded_map50_95:.3f} | {map50_95_drop:.1f}% |\\n")
                        f.write("\\n")
            
            # Best model analysis
            f.write("\\n## Best Performing Models by Scenario\\n\\n")
            
            f.write("| Scenario | Best Model | mAP50 |\\n")
            f.write("|----------|------------|-------|\\n")
            
            # Clean data winner
            clean_winner = max(results['original'].items(), key=lambda x: x[1]['mAP50'])
            f.write(f"| Clean Data | {clean_winner[0]} | {clean_winner[1]['mAP50']:.3f} |\\n")
            
            # Degraded data winners
            for degradation_type in results['degraded'].keys():
                for intensity in ['heavy']:  # Focus on most challenging
                    if intensity in results['degraded'][degradation_type]:
                        scenario_results = results['degraded'][degradation_type][intensity]
                        if scenario_results:
                            winner = max(scenario_results.items(), key=lambda x: x[1]['mAP50'])
                            scenario_name = f"{degradation_type.replace('_', ' ').title()} ({intensity})"
                            f.write(f"| {scenario_name} | {winner[0]} | {winner[1]['mAP50']:.3f} |\\n")
                            
        logger.info(f"📊 Summary report generated: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="PCB Defect Detection Robustness Evaluation")
    parser.add_argument("--test_images", required=True, help="Path to test images directory")
    parser.add_argument("--data_config", required=True, help="Path to original data config YAML")
    parser.add_argument("--results_dir", default="robustness_results", help="Results output directory")
    
    args = parser.parse_args()
    
    # Define model configurations to test
    model_configs = {
        "RB00_Baseline": "experiments/roboflow-pcb-training/RB00_YOLOv8n_Baseline_Roboflow_640px/weights/best.pt",
        "RB01_SIoU_ECA": "experiments/roboflow-pcb-training/RB01_YOLOv8n_SIoU_ECA_Roboflow_640px/weights/best.pt",
        "RB04_EIoU_ECA": "experiments/roboflow-pcb-training/RB04_YOLOv8n_EIoU_ECA_Roboflow_640px/weights/best.pt", 
        "RB06_SIoU": "experiments/roboflow-pcb-training/RB06_YOLOv8n_SIoU_Roboflow_640px/weights/best.pt"
    }
    
    # Run robustness evaluation
    evaluator = RobustnessEvaluator(args.test_images, args.results_dir)
    results = evaluator.run_robustness_study(model_configs, args.data_config)
    
    logger.info("🎉 Robustness evaluation completed!")

if __name__ == "__main__":
    main()