#!/usr/bin/env python3
"""
Example: How to Run PCB Defect Detection Experiments
===================================================

This script demonstrates how to run individual experiments with the new simplified workflow.
Each experiment runs training, validation, and testing automatically.

Usage Examples:
    # From project root - run a single experiment
    python scripts/experiments/run_single_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml
    
    # Run test evaluation only (if model already trained)
    python scripts/experiments/run_single_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml --test_only
    
    # Run multiple experiments sequentially (this script)
    python scripts/experiments/example_run_experiments.py

Author: PCB Defect Detection Team
Date: 2025-01-21
"""

import subprocess
import sys
import time
from pathlib import Path

def run_experiment(config_path: str):
    """Run a single experiment and return success status."""
    print(f"\n🚀 Starting experiment: {config_path}")
    print("=" * 60)
    
    try:
        # Run the experiment
        result = subprocess.run([
            sys.executable, 
            'scripts/experiments/run_single_experiment.py',
            '--config', config_path
        ], check=True, capture_output=True, text=True)
        
        print("✅ Experiment completed successfully!")
        print(f"📊 Output preview:\n{result.stdout[-500:]}")  # Last 500 chars
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed!")
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Run example experiments sequentially."""
    print("🔬 PCB DEFECT DETECTION EXPERIMENT RUNNER")
    print("=" * 60)
    print("This script demonstrates running multiple experiments")
    print("Each experiment includes: Training → Validation → Testing")
    print("=" * 60)
    
    # List of experiments to run (modify as needed)
    experiments = [
        "experiments/configs/01_yolov8n_baseline_standard.yaml",
        # Add more experiments as needed:
        # "experiments/configs/02_yolov8s_baseline_standard.yaml",
        # "experiments/configs/04_yolov8n_eca_standard.yaml",
    ]
    
    results = {}
    total_start_time = time.time()
    
    print(f"📋 Planned experiments: {len(experiments)}")
    for i, config in enumerate(experiments, 1):
        print(f"   {i}. {Path(config).stem}")
    print()
    
    # Ask for confirmation
    response = input("🤔 Do you want to proceed? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Experiments cancelled.")
        return
    
    # Run each experiment
    for i, config in enumerate(experiments, 1):
        start_time = time.time()
        
        print(f"\n📊 EXPERIMENT {i}/{len(experiments)}")
        success = run_experiment(config)
        
        experiment_time = time.time() - start_time
        results[config] = {
            'success': success,
            'time': experiment_time
        }
        
        print(f"⏱️  Experiment time: {experiment_time:.2f} seconds")
        
        if not success:
            response = input("❓ Continue with remaining experiments? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                break
    
    # Final summary
    total_time = time.time() - total_start_time
    successful = sum(1 for r in results.values() if r['success'])
    
    print("\n" + "=" * 60)
    print("📊 EXPERIMENT BATCH SUMMARY")
    print("=" * 60)
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {len(results) - successful}/{len(results)}")
    
    print("\n📋 Individual Results:")
    for config, result in results.items():
        status = "✅" if result['success'] else "❌"
        experiment_name = Path(config).stem
        print(f"   {status} {experiment_name}: {result['time']:.2f}s")
    
    print("\n📁 Results are saved in: experiments/results/[experiment_name]/")
    print("📊 Check WandB dashboard for detailed metrics and comparisons")
    
    if successful == len(results):
        print("\n🎉 All experiments completed successfully!")
    else:
        print(f"\n⚠️  {len(results) - successful} experiments failed. Check logs for details.")

if __name__ == "__main__":
    main()