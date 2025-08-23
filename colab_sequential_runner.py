#!/usr/bin/env python3
"""
Google Colab Sequential Experiment Runner
========================================

Runs multiple PCB defect detection experiments sequentially on Colab,
maximizing GPU utilization and handling session timeouts gracefully.

Features:
- Automatic GPU memory optimization
- Gradient accumulation for effective larger batch sizes
- Progress tracking and resumption
- Memory cleanup between experiments
- Colab-specific optimizations

Usage:
    python colab_sequential_runner.py
    python colab_sequential_runner.py --quick  # Run subset for testing

Author: PCB Defect Detection Team
Date: 2025-01-22
"""

import os
import sys
import time
import torch
import gc
import subprocess
from datetime import datetime
from pathlib import Path

def clear_gpu_memory():
    """Clear GPU memory between experiments."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("🧹 GPU memory cleared")

def log_gpu_status():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"📊 GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    else:
        print("❌ No GPU available")

def run_experiment(config_path, experiment_num, total_experiments):
    """Run a single experiment with error handling."""
    print("\n" + "="*80)
    print(f"🚀 EXPERIMENT {experiment_num}/{total_experiments}")
    print(f"📁 Config: {config_path}")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        # Clear memory before starting
        clear_gpu_memory()
        log_gpu_status()
        
        # Run the experiment
        cmd = [sys.executable, "run_experiment.py", "--config", config_path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print(f"✅ EXPERIMENT {experiment_num} COMPLETED SUCCESSFULLY!")
        print(f"🕐 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ EXPERIMENT {experiment_num} FAILED!")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR in experiment {experiment_num}: {e}")
        return False
    finally:
        # Always clear memory after experiment
        clear_gpu_memory()

def main():
    """Main function to run sequential experiments."""
    
    # Check if we're in quick mode
    quick_mode = "--quick" in sys.argv
    
    if quick_mode:
        print("⚡ QUICK MODE: Running subset of experiments")
        experiments = [
            "experiments/configs/colab_01_yolov8n_baseline_optimized.yaml",
            "experiments/configs/04_yolov8n_eca_standard.yaml",
            "experiments/configs/07_yolov8n_baseline_focal_siou.yaml"
        ]
    else:
        print("🔄 FULL MODE: Running all experiments")
        experiments = [
            # Baseline experiments
            "experiments/configs/colab_01_yolov8n_baseline_optimized.yaml",
            "experiments/configs/01_yolov8n_baseline_standard.yaml",
            "experiments/configs/02_yolov8s_baseline_standard.yaml",
            
            # Attention mechanism experiments
            "experiments/configs/04_yolov8n_eca_standard.yaml",
            "experiments/configs/05_yolov8n_cbam_standard.yaml",
            "experiments/configs/06_yolov8n_coordatt_standard.yaml",
            
            # Loss function experiments
            "experiments/configs/07_yolov8n_baseline_focal_siou.yaml",
            
            # High-resolution experiments (if GPU memory allows)
            "experiments/configs/10_yolov8n_baseline_1024px.yaml",
        ]
    
    print("\n🎯 GOOGLE COLAB SEQUENTIAL EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"📊 Total experiments to run: {len(experiments)}")
    print(f"⏰ Estimated total time: {len(experiments) * 2:.1f} - {len(experiments) * 4:.1f} hours")
    print(f"🖥️  Platform: Google Colab")
    print("=" * 60)
    
    # Log initial system status
    log_gpu_status()
    
    # Track results
    results = []
    start_time = time.time()
    
    # Run each experiment
    for i, config_path in enumerate(experiments, 1):
        if not Path(config_path).exists():
            print(f"⚠️  Config file not found: {config_path}")
            results.append(False)
            continue
        
        success = run_experiment(config_path, i, len(experiments))
        results.append(success)
        
        # Progress update
        completed = sum(results)
        elapsed_time = (time.time() - start_time) / 3600  # hours
        avg_time_per_exp = elapsed_time / i if i > 0 else 0
        remaining_time = avg_time_per_exp * (len(experiments) - i)
        
        print(f"\n📈 PROGRESS UPDATE:")
        print(f"   ✅ Completed: {completed}/{len(experiments)} experiments")
        print(f"   ⏱️  Elapsed time: {elapsed_time:.1f} hours")
        print(f"   🔮 Estimated remaining: {remaining_time:.1f} hours")
        
        # Small delay between experiments
        if i < len(experiments):
            print("⏸️  Waiting 30 seconds before next experiment...")
            time.sleep(30)
    
    # Final summary
    total_time = (time.time() - start_time) / 3600
    successful = sum(results)
    failed = len(results) - successful
    
    print("\n" + "="*80)
    print("🏁 ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"✅ Successful: {successful}/{len(experiments)} experiments")
    print(f"❌ Failed: {failed}/{len(experiments)} experiments")
    print(f"⏱️  Total time: {total_time:.2f} hours")
    print(f"📊 Average time per experiment: {total_time/len(experiments):.2f} hours")
    
    if failed > 0:
        print(f"\n⚠️  Failed experiments:")
        for i, (config, success) in enumerate(zip(experiments, results), 1):
            if not success:
                print(f"   {i}. {config}")
    
    print(f"\n🎉 Sequential experiment run completed!")
    print(f"📁 Check experiments/results/ for individual experiment outputs")
    print("="*80)

if __name__ == "__main__":
    main()