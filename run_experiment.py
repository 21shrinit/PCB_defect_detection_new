#!/usr/bin/env python3
"""
PCB Defect Detection - Experiment Launcher
==========================================

Convenience script to run experiments from the project root.
This is a simple wrapper around the organized experiment runner.

Usage:
    python run_experiment.py --config experiments/configs/01_yolov8n_baseline_standard.yaml
    python run_experiment.py --config experiments/configs/custom_config.yaml --test_only

Author: PCB Defect Detection Team
Date: 2025-01-21
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the organized experiment runner."""
    
    # Path to the organized experiment runner
    runner_script = Path(__file__).parent / 'scripts' / 'experiments' / 'run_single_experiment.py'
    
    if not runner_script.exists():
        print(f"❌ Error: Experiment runner not found at {runner_script}")
        print("Make sure the project structure is correct.")
        sys.exit(1)
    
    # Forward all arguments to the organized runner
    cmd = [sys.executable, str(runner_script)] + sys.argv[1:]
    
    print("PCB Defect Detection - Launching Experiment")
    print(f"Using runner: {runner_script}")
    print("=" * 60)
    
    try:
        # Run the experiment with the same environment
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"\nExperiment failed with return code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()