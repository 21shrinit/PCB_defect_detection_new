#!/usr/bin/env python3
"""
Master Comprehensive Experiment Runner
====================================

Runs ALL experiments with comprehensive data collection for research reporting.
This script will:

✅ Run all experiment configurations using the FIXED training script
✅ Automatically test after each training completion  
✅ Collect all metrics required by research marking schema
✅ Generate publication-ready tables and analysis
✅ Save organized results for easy report writing

Usage: python run_all_experiments_comprehensive.py
"""

import os
import sys
import yaml
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.comprehensive_experiment_runner import ComprehensiveExperimentRunner
from scripts.experiments.generate_research_tables import ResearchTablesGenerator

class MasterExperimentRunner:
    """Master runner for all comprehensive experiments."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results_dir = "COMPREHENSIVE_EXPERIMENT_RESULTS_" + datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup master logging."""
        log_file = Path(self.results_dir) / "master_experiment_log.log"
        log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def discover_experiment_configs(self) -> List[Path]:
        """Discover all available experiment configurations."""
        configs_dir = self.project_root / "experiments" / "configs"
        
        if not configs_dir.exists():
            self.logger.error(f"Configs directory not found: {configs_dir}")
            return []
        
        # Find all YAML config files
        config_files = []
        for config_file in configs_dir.rglob("*.yaml"):
            if config_file.is_file():
                config_files.append(config_file)
        
        # Sort by name for consistent ordering
        config_files.sort(key=lambda x: x.name)
        
        self.logger.info(f"📁 Found {len(config_files)} experiment configurations")
        for config in config_files:
            self.logger.info(f"   - {config.relative_to(self.project_root)}")
        
        return config_files
    
    def validate_config(self, config_path: Path) -> bool:
        """Validate that a config file is properly formatted."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['model', 'training']
            for section in required_sections:
                if section not in config:
                    self.logger.warning(f"Config {config_path.name} missing section: {section}")
                    return False
            
            # Check model type
            model_type = config['model'].get('type')
            if not model_type:
                self.logger.warning(f"Config {config_path.name} missing model type")
                return False
            
            # Check dataset path
            dataset_path = config['training'].get('dataset', {}).get('path')
            if not dataset_path:
                self.logger.warning(f"Config {config_path.name} missing dataset path")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Config validation failed for {config_path.name}: {e}")
            return False
    
    def run_all_experiments(self):
        """Run all experiments with comprehensive data collection."""
        self.logger.info("🚀 Starting Master Comprehensive Experiment Runner")
        self.logger.info("=" * 80)
        
        # Discover configs
        config_files = self.discover_experiment_configs()
        
        if not config_files:
            self.logger.error("❌ No experiment configurations found!")
            return False
        
        # Validate configs
        valid_configs = []
        for config_path in config_files:
            if self.validate_config(config_path):
                valid_configs.append(config_path)
            else:
                self.logger.warning(f"⚠️  Skipping invalid config: {config_path.name}")
        
        self.logger.info(f"✅ Validated {len(valid_configs)} out of {len(config_files)} configurations")
        
        if not valid_configs:
            self.logger.error("❌ No valid configurations found!")
            return False
        
        # Initialize comprehensive runner
        runner = ComprehensiveExperimentRunner(self.results_dir)
        
        # Run all experiments
        successful_experiments = []
        failed_experiments = []
        
        for i, config_path in enumerate(valid_configs, 1):
            self.logger.info(f"\n{'='*20} EXPERIMENT {i}/{len(valid_configs)} {'='*20}")
            self.logger.info(f"📋 Config: {config_path.relative_to(self.project_root)}")
            
            start_time = time.time()
            
            try:
                result = runner.run_single_experiment(str(config_path))
                
                if result['status'] == 'completed':
                    elapsed_time = time.time() - start_time
                    self.logger.info(f"✅ Experiment {i} completed in {elapsed_time/3600:.2f} hours")
                    successful_experiments.append({
                        'config': config_path.name,
                        'result': result,
                        'duration_hours': elapsed_time / 3600
                    })
                else:
                    self.logger.error(f"❌ Experiment {i} failed: {result.get('error', 'Unknown error')}")
                    failed_experiments.append({
                        'config': config_path.name,
                        'error': result.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                self.logger.error(f"❌ Experiment {i} crashed: {e}")
                failed_experiments.append({
                    'config': config_path.name,
                    'error': f"Crashed: {str(e)}"
                })
        
        # Generate summary
        self.generate_master_summary(successful_experiments, failed_experiments)
        
        # Generate research tables if we have successful results
        if successful_experiments:
            self.logger.info("\n📊 Generating comprehensive research analysis...")
            try:
                generator = ResearchTablesGenerator(self.results_dir)
                generator.generate_all_research_tables()
                self.logger.info("✅ Research analysis completed")
            except Exception as e:
                self.logger.error(f"❌ Research analysis failed: {e}")
        
        return len(successful_experiments) > 0
    
    def generate_master_summary(self, successful: List[Dict], failed: List[Dict]):
        """Generate master summary of all experiments."""
        summary_file = Path(self.results_dir) / "MASTER_EXPERIMENT_SUMMARY.md"
        
        total_experiments = len(successful) + len(failed)
        success_rate = len(successful) / total_experiments * 100 if total_experiments > 0 else 0
        
        total_duration = sum(exp['duration_hours'] for exp in successful)
        avg_duration = total_duration / len(successful) if successful else 0
        
        with open(summary_file, 'w') as f:
            f.write(f"""# Master Experiment Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Results Directory**: `{self.results_dir}`

## Overview

- **Total Experiments**: {total_experiments}
- **Successful**: {len(successful)}
- **Failed**: {len(failed)}
- **Success Rate**: {success_rate:.1f}%
- **Total Training Time**: {total_duration:.2f} hours
- **Average Time per Experiment**: {avg_duration:.2f} hours

## Successful Experiments

""")
            
            if successful:
                f.write("| # | Configuration | Duration (hours) | Status |\n")
                f.write("|---|---------------|------------------|--------|\n")
                
                for i, exp in enumerate(successful, 1):
                    f.write(f"| {i} | {exp['config']} | {exp['duration_hours']:.2f} | ✅ Complete |\n")
                
                f.write("\n### Performance Summary\n\n")
                
                # Quick performance overview
                best_performers = []
                for exp in successful:
                    testing = exp['result'].get('comprehensive_testing', {}).get('validation_metrics', {})
                    if testing:
                        best_performers.append({
                            'config': exp['config'],
                            'mAP': testing.get('mAP_0.5_0.95', 0),
                            'precision': testing.get('precision', 0),
                            'recall': testing.get('recall', 0)
                        })
                
                if best_performers:
                    best_performers.sort(key=lambda x: x['mAP'], reverse=True)
                    
                    f.write("**Top 5 Performers by mAP@0.5:0.95:**\n\n")
                    f.write("| Rank | Configuration | mAP@0.5:0.95 | Precision | Recall |\n")
                    f.write("|------|---------------|--------------|-----------|--------|\n")
                    
                    for i, perf in enumerate(best_performers[:5], 1):
                        f.write(f"| {i} | {perf['config']} | {perf['mAP']:.4f} | {perf['precision']:.4f} | {perf['recall']:.4f} |\n")
                    
                    f.write("\n")
            else:
                f.write("No successful experiments.\n\n")
            
            if failed:
                f.write("## Failed Experiments\n\n")
                f.write("| # | Configuration | Error |\n")
                f.write("|---|---------------|-------|\n")
                
                for i, exp in enumerate(failed, 1):
                    f.write(f"| {i} | {exp['config']} | {exp['error']} |\n")
                
                f.write("\n")
            
            f.write(f"""## Next Steps

### For Academic Reporting

All results have been processed and organized for easy academic reporting:

1. **Complete Results**: `{self.results_dir}/summary_reports/`
   - Individual experiment summaries and JSON data
   - Training logs and model checkpoints

2. **Research Tables**: `{self.results_dir}/research_tables/`
   - Model architecture comparison table
   - Loss function ablation matrix  
   - Computational efficiency analysis
   - Attention mechanism impact analysis
   - LaTeX formatted tables for papers

3. **Main Report**: `{self.results_dir}/research_tables/RESEARCH_SUMMARY_REPORT.md`
   - Comprehensive analysis meeting marking schema requirements
   - Publication-ready findings and recommendations

### For Further Development

- Use the best performing configurations for production deployment
- Investigate ensemble methods combining top performers
- Test generalization on additional PCB datasets
- Explore model optimization techniques (quantization, pruning)

---

**Note**: This summary provides the high-level overview. Detailed analysis including statistical significance testing, cross-dataset evaluation, and failure case analysis is available in the research tables directory.
""")
        
        self.logger.info(f"📋 Master summary saved to: {summary_file}")

def main():
    """Main entry point."""
    print("🚀 PCB Defect Detection - Master Comprehensive Experiment Runner")
    print("=" * 80)
    print("This will run ALL experiments with comprehensive data collection.")
    print("Results will be organized for easy academic reporting.")
    print("")
    
    # Confirmation
    response = input("Do you want to proceed? This may take several hours. [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Experiment run cancelled.")
        return
    
    # Run all experiments
    runner = MasterExperimentRunner()
    success = runner.run_all_experiments()
    
    if success:
        print(f"\n🎉 Master experiment run completed successfully!")
        print(f"📁 All results saved to: {runner.results_dir}")
        print(f"📊 Research tables and analysis ready for academic reporting")
    else:
        print(f"\n❌ Master experiment run completed with issues.")
        print(f"📁 Check logs in: {runner.results_dir}")

if __name__ == "__main__":
    main()