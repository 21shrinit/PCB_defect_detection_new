#!/usr/bin/env python3
"""
Systematic Attention Mechanism Benchmarking Script
=================================================

This script runs comprehensive benchmarking of all attention mechanisms:
- Baseline YOLOv8n (no attention)
- ECA (Efficient Channel Attention)
- CBAM (Convolutional Block Attention Module)
- CoordAtt (Coordinate Attention)

Each mechanism is trained using the two-stage approach to prevent
destructive learning dynamics and ensure fair comparison.

Author: MLOps Engineering Team
Version: 2.0.0
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from train_attention_benchmark import AttentionTrainingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AttentionBenchmarkSuite:
    """
    Comprehensive benchmarking suite for attention mechanisms.
    
    This class manages systematic training and evaluation of multiple
    attention mechanisms with proper statistical comparison.
    """
    
    def __init__(self, benchmark_dir: str = "benchmark_results"):
        """
        Initialize the benchmark suite.
        
        Args:
            benchmark_dir (str): Directory to store benchmark results
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Define attention mechanisms to benchmark
        self.attention_configs = {
            'baseline': {
                'config_path': 'configs/config_baseline.yaml',
                'description': 'Standard YOLOv8n without attention',
                'expected_params': '3.2M',
                'expected_improvement': '0% (reference)'
            },
            'eca': {
                'config_path': 'configs/config_eca.yaml',
                'description': 'Efficient Channel Attention',
                'expected_params': '3.2M (+minimal)',
                'expected_improvement': '+1-3% mAP'
            },
            'cbam': {
                'config_path': 'configs/config_cbam.yaml',
                'description': 'Convolutional Block Attention Module',
                'expected_params': '3.3M (+moderate)',
                'expected_improvement': '+2-4% mAP'
            },
            'coordatt': {
                'config_path': 'configs/config_coordatt.yaml',
                'description': 'Coordinate Attention',
                'expected_params': '3.2M (+minimal)',
                'expected_improvement': '+1-2% mAP'
            }
        }
        
        self.results = {}
        self.benchmark_start_time = None
        
    def run_single_benchmark(self, attention_name: str, config_info: Dict) -> Dict[str, Any]:
        """
        Run benchmark for a single attention mechanism.
        
        Args:
            attention_name (str): Name of the attention mechanism
            config_info (Dict): Configuration information
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        logger.info("=" * 80)
        logger.info(f"🚀 BENCHMARKING: {attention_name.upper()}")
        logger.info("=" * 80)
        logger.info(f"📋 Description: {config_info['description']}")
        logger.info(f"📊 Expected Parameters: {config_info['expected_params']}")
        logger.info(f"📈 Expected Improvement: {config_info['expected_improvement']}")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Initialize training pipeline
            config_path = config_info['config_path']
            if not Path(config_path).exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            pipeline = AttentionTrainingPipeline(config_path)
            
            # Run complete training
            warmup_results, finetune_results = pipeline.run_complete_pipeline()
            
            # Export model
            pipeline.export_final_model(['onnx', 'torchscript'])
            
            # Calculate training time
            end_time = datetime.now()
            training_time = end_time - start_time
            
            # Extract key metrics
            warmup_map50 = warmup_results.results_dict.get('metrics/mAP50(B)', 0.0)
            finetune_map50 = finetune_results.results_dict.get('metrics/mAP50(B)', 0.0)
            warmup_map50_95 = warmup_results.results_dict.get('metrics/mAP50-95(B)', 0.0)
            finetune_map50_95 = finetune_results.results_dict.get('metrics/mAP50-95(B)', 0.0)
            
            # Compile results
            results = {
                'attention_mechanism': attention_name,
                'description': config_info['description'],
                'training_time': str(training_time),
                'training_time_seconds': training_time.total_seconds(),
                
                # Warmup stage results
                'warmup_epochs': warmup_results.epochs,
                'warmup_map50': float(warmup_map50),
                'warmup_map50_95': float(warmup_map50_95),
                
                # Fine-tuning stage results
                'finetune_epochs': finetune_results.epochs,
                'final_map50': float(finetune_map50),
                'final_map50_95': float(finetune_map50_95),
                
                # Improvement metrics
                'warmup_to_finetune_improvement': float(finetune_map50 - warmup_map50),
                
                # Paths
                'experiment_dir': str(pipeline.experiment_dir),
                'warmup_weights': str(warmup_results.save_dir / 'weights' / 'best.pt'),
                'final_weights': str(finetune_results.save_dir / 'weights' / 'best.pt'),
                
                # Status
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            
            logger.info("✅ BENCHMARK COMPLETED SUCCESSFULLY!")
            logger.info(f"📊 Final mAP@0.5: {finetune_map50:.4f}")
            logger.info(f"📊 Final mAP@0.5-0.95: {finetune_map50_95:.4f}")
            logger.info(f"⏱️  Training Time: {training_time}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed for {attention_name}: {str(e)}")
            
            return {
                'attention_mechanism': attention_name,
                'description': config_info['description'],
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }
    
    def run_complete_benchmark(self, mechanisms: List[str] = None) -> Dict[str, Any]:
        """
        Run complete benchmark suite for all or specified attention mechanisms.
        
        Args:
            mechanisms (List[str]): List of mechanisms to benchmark. If None, benchmarks all.
            
        Returns:
            Dict[str, Any]: Complete benchmark results
        """
        self.benchmark_start_time = datetime.now()
        
        if mechanisms is None:
            mechanisms = list(self.attention_configs.keys())
        
        logger.info("🏁 STARTING COMPLETE ATTENTION MECHANISM BENCHMARK SUITE")
        logger.info("=" * 80)
        logger.info(f"📅 Started: {self.benchmark_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🎯 Mechanisms: {', '.join(mechanisms)}")
        logger.info(f"📁 Results Directory: {self.benchmark_dir}")
        logger.info("=" * 80)
        
        # Run benchmarks for each mechanism
        for i, mechanism in enumerate(mechanisms, 1):
            if mechanism not in self.attention_configs:
                logger.warning(f"⚠️  Unknown mechanism: {mechanism}. Skipping.")
                continue
            
            logger.info(f"\n🔄 BENCHMARK {i}/{len(mechanisms)}: {mechanism.upper()}")
            
            config_info = self.attention_configs[mechanism]
            result = self.run_single_benchmark(mechanism, config_info)
            self.results[mechanism] = result
            
            # Save intermediate results
            self._save_intermediate_results()
        
        # Generate final report
        benchmark_end_time = datetime.now()
        total_time = benchmark_end_time - self.benchmark_start_time
        
        final_results = {
            'benchmark_info': {
                'started_at': self.benchmark_start_time.isoformat(),
                'completed_at': benchmark_end_time.isoformat(),
                'total_time': str(total_time),
                'total_time_seconds': total_time.total_seconds(),
                'mechanisms_tested': len([r for r in self.results.values() if r.get('status') == 'completed']),
                'mechanisms_failed': len([r for r in self.results.values() if r.get('status') == 'failed'])
            },
            'individual_results': self.results
        }
        
        # Save final results
        self._save_final_results(final_results)
        
        # Generate comparison report
        self._generate_comparison_report(final_results)
        
        logger.info("\n" + "=" * 80)
        logger.info("🏆 COMPLETE BENCHMARK SUITE FINISHED!")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total Time: {total_time}")
        logger.info(f"✅ Successful: {final_results['benchmark_info']['mechanisms_tested']}")
        logger.info(f"❌ Failed: {final_results['benchmark_info']['mechanisms_failed']}")
        logger.info(f"📋 Report: {self.benchmark_dir / 'benchmark_comparison.json'}")
        logger.info("=" * 80)
        
        return final_results
    
    def _save_intermediate_results(self):
        """Save intermediate results during benchmarking."""
        results_file = self.benchmark_dir / 'intermediate_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def _save_final_results(self, final_results: Dict[str, Any]):
        """Save final benchmark results."""
        results_file = self.benchmark_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"💾 Final results saved: {results_file}")
    
    def _generate_comparison_report(self, final_results: Dict[str, Any]):
        """Generate a comprehensive comparison report."""
        completed_results = {
            name: result for name, result in final_results['individual_results'].items()
            if result.get('status') == 'completed'
        }
        
        if not completed_results:
            logger.warning("⚠️  No completed results to compare")
            return
        
        # Create comparison table
        comparison_data = []
        baseline_map50 = None
        
        for name, result in completed_results.items():
            row = {
                'Mechanism': name.upper(),
                'Description': result['description'],
                'Final mAP@0.5': f"{result['final_map50']:.4f}",
                'Final mAP@0.5-0.95': f"{result['final_map50_95']:.4f}",
                'Training Time': result['training_time'],
                'Warmup→Finetune Δ': f"{result['warmup_to_finetune_improvement']:.4f}"
            }
            
            # Calculate improvement over baseline
            if name == 'baseline':
                baseline_map50 = result['final_map50']
                row['vs Baseline'] = '0.0% (reference)'
            elif baseline_map50 is not None:
                improvement = ((result['final_map50'] - baseline_map50) / baseline_map50) * 100
                row['vs Baseline'] = f"{improvement:+.1f}%"
            else:
                row['vs Baseline'] = 'N/A'
            
            comparison_data.append(row)
        
        # Save comparison as JSON
        comparison_file = self.benchmark_dir / 'benchmark_comparison.json'
        comparison_summary = {
            'benchmark_summary': final_results['benchmark_info'],
            'mechanism_comparison': comparison_data,
            'ranking_by_map50': sorted(
                comparison_data, 
                key=lambda x: float(x['Final mAP@0.5']), 
                reverse=True
            )
        }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        # Create CSV for easy analysis
        df = pd.DataFrame(comparison_data)
        csv_file = self.benchmark_dir / 'benchmark_comparison.csv'
        df.to_csv(csv_file, index=False)
        
        # Print summary table
        logger.info("\n📊 BENCHMARK COMPARISON SUMMARY")
        logger.info("=" * 120)
        for row in comparison_data:
            logger.info(f"{row['Mechanism']:<12} | "
                       f"mAP@0.5: {row['Final mAP@0.5']:<8} | "
                       f"vs Baseline: {row['vs Baseline']:<12} | "
                       f"Time: {row['Training Time']}")
        logger.info("=" * 120)
        
        logger.info(f"📋 Detailed comparison saved: {comparison_file}")
        logger.info(f"📊 CSV export saved: {csv_file}")


def main():
    """Main function for running attention mechanism benchmarks."""
    parser = argparse.ArgumentParser(
        description="Systematic Attention Mechanism Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete benchmark suite
  python benchmark_all_attention.py
  
  # Run specific mechanisms
  python benchmark_all_attention.py --mechanisms baseline cbam
  
  # Custom results directory
  python benchmark_all_attention.py --output benchmark_2025
        """
    )
    
    parser.add_argument(
        '--mechanisms',
        nargs='*',
        choices=['baseline', 'eca', 'cbam', 'coordatt'],
        default=None,
        help='Attention mechanisms to benchmark (default: all)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results',
        help='Output directory for benchmark results (default: benchmark_results)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be benchmarked without running training'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 ATTENTION MECHANISM BENCHMARKING SUITE v2.0")
    print("   Systematic Comparison of YOLOv8 Attention Mechanisms")
    print("=" * 80)
    
    # Initialize benchmark suite
    benchmark_suite = AttentionBenchmarkSuite(args.output)
    
    mechanisms_to_test = args.mechanisms or list(benchmark_suite.attention_configs.keys())
    
    print(f"🎯 Mechanisms to benchmark: {', '.join(mechanisms_to_test)}")
    print(f"📁 Results directory: {args.output}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Configuration Check:")
        for mechanism in mechanisms_to_test:
            config_info = benchmark_suite.attention_configs[mechanism]
            config_path = config_info['config_path']
            exists = "✅" if Path(config_path).exists() else "❌"
            print(f"  {mechanism:<12} | {config_path:<40} | {exists}")
        print("\n✅ Dry run completed. Use without --dry-run to start benchmarking.")
        return
    
    try:
        # Run benchmarks
        final_results = benchmark_suite.run_complete_benchmark(mechanisms_to_test)
        
        print("\n🎉 BENCHMARKING COMPLETED SUCCESSFULLY!")
        print(f"📋 Results: {benchmark_suite.benchmark_dir / 'benchmark_comparison.json'}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmarking interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmarking failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()