#!/usr/bin/env python3
"""
Systematic Study Execution Script
=================================

This script orchestrates the complete systematic study of YOLOv8 variants,
attention mechanisms, and loss function combinations for PCB defect detection.

It automatically runs all experiment configurations in the optimal order and
provides comprehensive tracking and analysis.

Usage:
    # Run all experiments
    python experiments/run_systematic_study.py --run_all
    
    # Run specific phase
    python experiments/run_systematic_study.py --phase 1
    
    # Run single experiment
    python experiments/run_systematic_study.py --experiment 01_yolov8n_baseline_standard

Author: PCB Defect Detection Team
Date: 2025-01-20
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'systematic_study_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class SystematicStudyRunner:
    """
    Orchestrates the complete systematic study execution.
    """
    
    # Define experiment phases and configurations
    EXPERIMENT_PHASES = {
        'phase_1_baselines': [
            '01_yolov8n_baseline_standard.yaml',
            '02_yolov8s_baseline_standard.yaml', 
            '03_yolov10s_baseline_standard.yaml'
        ],
        'phase_2_attention': [
            '04_yolov8n_eca_standard.yaml',
            '05_yolov8n_cbam_standard.yaml',
            '06_yolov8n_coordatt_standard.yaml'
        ],
        'phase_3_loss_functions': [
            '07_yolov8n_baseline_focal_siou.yaml',
            '08_yolov8n_verifocal_eiou.yaml',
            '09_yolov8n_verifocal_siou.yaml'
        ],
        'phase_3_resolution': [
            '10_yolov8n_baseline_1024px.yaml',
            '11_yolov8s_baseline_1024px.yaml'
        ]
    }
    
    def __init__(self):
        """Initialize the systematic study runner."""
        self.project_root = project_root
        self.configs_dir = self.project_root / 'experiments' / 'configs'
        self.run_script = self.project_root / 'run_experiment.py'
        self.results_summary = []
        
        # Validate paths
        if not self.run_script.exists():
            raise FileNotFoundError(f"Run script not found: {self.run_script}")
        if not self.configs_dir.exists():
            raise FileNotFoundError(f"Configs directory not found: {self.configs_dir}")
            
        logger.info(f"SystematicStudyRunner initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Configs directory: {self.configs_dir}")
        
    def get_all_experiments(self) -> List[str]:
        """
        Get all experiment configurations in execution order.
        
        Returns:
            List[str]: Ordered list of experiment config files
        """
        all_experiments = []
        for phase_name, experiments in self.EXPERIMENT_PHASES.items():
            all_experiments.extend(experiments)
        return all_experiments
        
    def check_experiment_completed(self, config_file: str) -> bool:
        """
        Check if an experiment has already been completed successfully.
        
        Args:
            config_file (str): Name of the configuration file
            
        Returns:
            bool: True if experiment is completed, False otherwise
        """
        try:
            # Extract experiment name from config file
            config_path = self.configs_dir / config_file
            if not config_path.exists():
                return False
                
            # Load config to get experiment name
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            experiment_name = config.get('experiment', {}).get('name', config_file.replace('.yaml', ''))
            project_dir = config.get('training', {}).get('project', 'experiments')
            
            # Check if best.pt exists in the experiment directory
            best_checkpoint = Path(project_dir) / experiment_name / "weights" / "best.pt"
            
            if best_checkpoint.exists():
                logger.info(f"✅ Experiment already completed: {config_file}")
                logger.info(f"   Found checkpoint: {best_checkpoint}")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️  Error checking experiment status for {config_file}: {e}")
            
        return False
        
    def run_single_experiment(self, config_file: str, skip_if_completed: bool = True) -> Dict[str, Any]:
        """
        Run a single experiment configuration.
        
        Args:
            config_file (str): Name of the configuration file
            skip_if_completed (bool): Skip if experiment already completed
            
        Returns:
            Dict[str, Any]: Experiment results summary
        """
        # Check if experiment is already completed
        if skip_if_completed and self.check_experiment_completed(config_file):
            experiment_result = {
                'config_file': config_file,
                'status': 'skipped_completed',
                'execution_time': 0.0,
                'start_time': datetime.now().isoformat(),
                'message': 'Experiment already completed - skipped'
            }
            self.results_summary.append(experiment_result)
            return experiment_result
        config_path = self.configs_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        logger.info(f"Starting experiment: {config_file}")
        logger.info(f"Config path: {config_path}")
        
        # Debug: Verify paths exist
        if not config_path.exists():
            logger.error(f"Config file does not exist: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if not self.run_script.exists():
            logger.error(f"Run script does not exist: {self.run_script}")
            raise FileNotFoundError(f"Run script not found: {self.run_script}")
        
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Run script: {self.run_script}")
        
        # Prepare command with unbuffered output
        cmd = [
            sys.executable,
            '-u',  # Force unbuffered stdout and stderr
            str(self.run_script),
            '--config', str(config_path)
        ]
        
        # Execute experiment with real-time output
        start_time = time.time()
        try:
            # Set environment for better Unicode handling and memory optimization
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            # PyTorch memory optimization
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            env['CUDA_LAUNCH_BLOCKING'] = '0'  # Allow async CUDA operations
            
            # Use Popen for real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                cwd=str(self.project_root),
                bufsize=1,
                universal_newlines=True,
                env=env,
                encoding='utf-8',
                errors='replace'  # Replace problematic characters instead of crashing
            )
            
            # Stream output in real-time
            stdout_lines = []
            logger.info("=" * 60)
            logger.info(f"EXPERIMENT OUTPUT: {config_file}")
            logger.info("=" * 60)
            
            for line in process.stdout:
                try:
                    # Remove emoji characters and print
                    clean_line = line.encode('ascii', 'ignore').decode('ascii')
                    print(clean_line.rstrip(), flush=True)
                    stdout_lines.append(clean_line)
                except UnicodeError:
                    # Fallback for any remaining Unicode issues
                    safe_line = repr(line)
                    print(safe_line, flush=True)
                    stdout_lines.append(safe_line)
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Create result object
            result = type('Result', (), {
                'stdout': ''.join(stdout_lines),
                'stderr': '',
                'returncode': return_code
            })()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd, ''.join(stdout_lines))
            
            execution_time = time.time() - start_time
            
            experiment_result = {
                'config_file': config_file,
                'status': 'success',
                'execution_time': execution_time,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            logger.info("=" * 60)
            logger.info(f"SUCCESS Experiment completed successfully: {config_file}")
            logger.info(f"Execution time: {execution_time:.2f} seconds")
            logger.info("=" * 60)
            
        except subprocess.CalledProcessError as e:
            execution_time = time.time() - start_time
            
            experiment_result = {
                'config_file': config_file,
                'status': 'failed',
                'execution_time': execution_time,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'stdout': ''.join(stdout_lines),
                'stderr': '',
                'return_code': e.returncode,
                'error': str(e)
            }
            
            logger.error("=" * 60)
            logger.error(f"FAILED Experiment failed: {config_file}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"Working directory: {self.project_root}")
            logger.error("Last few lines of output:")
            for line in stdout_lines[-10:]:  # Show last 10 lines
                logger.error(f"  {line.rstrip()}")
            logger.error("=" * 60)
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            experiment_result = {
                'config_file': config_file,
                'status': 'error',
                'execution_time': execution_time,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'error': str(e)
            }
            
            logger.error(f"ERROR Unexpected error in experiment: {config_file}")
            logger.error(f"Error: {e}")
            
        self.results_summary.append(experiment_result)
        return experiment_result
        
    def run_phase(self, phase_number: int, resume: bool = False) -> List[Dict[str, Any]]:
        """
        Run all experiments in a specific phase.
        
        Args:
            phase_number (int): Phase number (1, 2, or 3)
            resume (bool): Skip experiments that are already completed
            
        Returns:
            List[Dict[str, Any]]: Results for all experiments in the phase
        """
        phase_mapping = {
            1: ['phase_1_baselines'],
            2: ['phase_2_attention'],
            3: ['phase_3_loss_functions', 'phase_3_resolution']
        }
        
        if phase_number not in phase_mapping:
            raise ValueError(f"Invalid phase number: {phase_number}. Use 1, 2, or 3.")
            
        phase_results = []
        
        for phase_name in phase_mapping[phase_number]:
            experiments = self.EXPERIMENT_PHASES[phase_name]
            
            logger.info(f"Starting {phase_name}")
            logger.info(f"Experiments: {experiments}")
            
            for config_file in experiments:
                result = self.run_single_experiment(config_file, skip_if_completed=resume)
                phase_results.append(result)
                
                # Add delay between experiments for system recovery and GPU memory cleanup
                if len(experiments) > 1:
                    logger.info("Cleaning up GPU memory and waiting 30 seconds before next experiment...")
                    # Force Python garbage collection
                    import gc
                    gc.collect()
                    # Try to clear CUDA cache if possible
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            logger.info("GPU memory cache cleared")
                    except Exception as e:
                        logger.info(f"Could not clear GPU cache: {e}")
                    time.sleep(30)
                    
        return phase_results
        
    def run_all_experiments(self, resume: bool = False) -> List[Dict[str, Any]]:
        """
        Run all experiments in the systematic study.
        
        Args:
            resume (bool): Skip experiments that are already completed
            
        Returns:
            List[Dict[str, Any]]: Results for all experiments
        """
        logger.info("Starting complete systematic study")
        logger.info(f"Total experiments: {len(self.get_all_experiments())}")
        
        all_results = []
        
        # Run Phase 1
        logger.info("=" * 80)
        logger.info("PHASE 1: BASELINE MODELS")
        logger.info("=" * 80)
        phase_1_results = self.run_phase(1, resume=resume)
        all_results.extend(phase_1_results)
        
        # Break between phases
        logger.info("Phase 1 completed. Waiting 5 minutes before Phase 2...")
        time.sleep(300)  # 5 minute break
        
        # Run Phase 2
        logger.info("=" * 80)
        logger.info("PHASE 2: ATTENTION MECHANISMS")
        logger.info("=" * 80)
        phase_2_results = self.run_phase(2, resume=resume)
        all_results.extend(phase_2_results)
        
        # Break between phases
        logger.info("Phase 2 completed. Waiting 5 minutes before Phase 3...")
        time.sleep(300)  # 5 minute break
        
        # Run Phase 3
        logger.info("=" * 80)
        logger.info("PHASE 3: LOSS FUNCTIONS AND RESOLUTION STUDY")
        logger.info("=" * 80)
        phase_3_results = self.run_phase(3, resume=resume)
        all_results.extend(phase_3_results)
        
        return all_results
        
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of all experiments.
        
        Returns:
            str: Summary report text
        """
        if not self.results_summary:
            return "No experiments have been run yet."
            
        successful = [r for r in self.results_summary if r['status'] == 'success']
        failed = [r for r in self.results_summary if r['status'] == 'failed']
        errors = [r for r in self.results_summary if r['status'] == 'error']
        skipped = [r for r in self.results_summary if r['status'] == 'skipped_completed']
        
        total_time = sum(r['execution_time'] for r in self.results_summary)
        
        report = f"""
SYSTEMATIC STUDY EXECUTION SUMMARY
=====================================

Overall Statistics:
  Total experiments: {len(self.results_summary)}
  Successful: {len(successful)}
  Failed: {len(failed)}
  Errors: {len(errors)}
  Skipped (already completed): {len(skipped)}
  Total execution time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)

SUCCESS Successful Experiments:
"""
        
        for result in successful:
            report += f"  SUCCESS {result['config_file']} ({result['execution_time']:.1f}s)\n"
            
        if skipped:
            report += "\nSKIPPED Already Completed Experiments:\n"
            for result in skipped:
                report += f"  SKIPPED {result['config_file']} - {result.get('message', 'Already completed')}\n"
            
        if failed:
            report += "\nFAILED Failed Experiments:\n"
            for result in failed:
                report += f"  FAILED {result['config_file']} - {result.get('error', 'Unknown error')}\n"
                
        if errors:
            report += "\nERROR Error Experiments:\n"
            for result in errors:
                report += f"  ERROR {result['config_file']} - {result.get('error', 'Unknown error')}\n"
                
        report += f"\nReport generated: {datetime.now().isoformat()}\n"
        
        return report
        
    def save_summary_report(self, filename: str = None):
        """
        Save the summary report to a file.
        
        Args:
            filename (str, optional): Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"systematic_study_summary_{timestamp}.txt"
            
        report = self.generate_summary_report()
        
        with open(filename, 'w') as f:
            f.write(report)
            
        logger.info(f"Summary report saved: {filename}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Systematic Study Execution for PCB Defect Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python experiments/run_systematic_study.py --run_all
  
  # Run specific phase
  python experiments/run_systematic_study.py --phase 1
  python experiments/run_systematic_study.py --phase 2
  python experiments/run_systematic_study.py --phase 3
  
  # Run single experiment
  python experiments/run_systematic_study.py --experiment 01_yolov8n_baseline_standard.yaml
        """
    )
    
    parser.add_argument('--run_all', action='store_true',
                        help='Run all experiments in the systematic study')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Run experiments in specific phase (1, 2, or 3)')
    parser.add_argument('--experiment', type=str,
                        help='Run single experiment by config filename')
    parser.add_argument('--list_experiments', action='store_true',
                        help='List all available experiments')
    parser.add_argument('--resume', action='store_true',
                        help='Skip experiments that are already completed')
    
    args = parser.parse_args()
    
    try:
        runner = SystematicStudyRunner()
        
        if args.list_experiments:
            # List all experiments
            all_experiments = runner.get_all_experiments()
            print("Available Experiments:")
            for i, exp in enumerate(all_experiments, 1):
                print(f"  {i:2d}. {exp}")
            return
            
        if args.run_all:
            # Run all experiments
            logger.info("Starting complete systematic study")
            if args.resume:
                logger.info("Resume mode: skipping already completed experiments")
            runner.run_all_experiments(resume=args.resume)
            
        elif args.phase:
            # Run specific phase
            logger.info(f"Running Phase {args.phase}")
            if args.resume:
                logger.info("Resume mode: skipping already completed experiments")
            runner.run_phase(args.phase, resume=args.resume)
            
        elif args.experiment:
            # Run single experiment
            logger.info(f"Running single experiment: {args.experiment}")
            runner.run_single_experiment(args.experiment, skip_if_completed=False)  # Never skip single experiments
            
        else:
            parser.print_help()
            return
            
        # Generate and save summary report
        runner.save_summary_report()
        print(runner.generate_summary_report())
        
    except KeyboardInterrupt:
        logger.info("Study interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Study execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()