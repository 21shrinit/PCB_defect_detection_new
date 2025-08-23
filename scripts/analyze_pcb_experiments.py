#!/usr/bin/env python3
"""
PCB Defect Detection Experiment Analysis Script

This script analyzes all experiments in the pcb-defect-150epochs-v1 directory
and generates comprehensive outputs and tables for reporting.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class PCBExperimentAnalyzer:
    def __init__(self, experiments_dir: str):
        self.experiments_dir = Path(experiments_dir)
        self.output_dir = Path("analysis_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "summary").mkdir(exist_ok=True)
        
        self.experiments_data = {}
        
    def load_experiment_data(self) -> None:
        """Load data from all experiments"""
        print("Loading experiment data...")
        
        for exp_folder in self.experiments_dir.iterdir():
            if exp_folder.is_dir():
                exp_name = exp_folder.name
                print(f"Processing: {exp_name}")
                
                exp_data = {}
                
                # Load results CSV
                results_file = exp_folder / "results.csv"
                if results_file.exists():
                    exp_data['results'] = pd.read_csv(results_file)
                
                # Load args YAML
                args_file = exp_folder / "args.yaml"
                if args_file.exists():
                    with open(args_file, 'r') as f:
                        exp_data['config'] = yaml.safe_load(f)
                
                # Check for weights
                weights_dir = exp_folder / "weights"
                if weights_dir.exists():
                    exp_data['weights'] = list(weights_dir.glob("*.pt"))
                
                # Store experiment data
                self.experiments_data[exp_name] = exp_data
                
        print(f"Loaded {len(self.experiments_data)} experiments")
    
    def extract_final_metrics(self) -> pd.DataFrame:
        """Extract final epoch metrics from all experiments"""
        metrics_data = []
        
        for exp_name, exp_data in self.experiments_data.items():
            if 'results' not in exp_data:
                continue
                
            results_df = exp_data['results']
            config = exp_data.get('config', {})
            
            # Get final epoch metrics
            final_metrics = results_df.iloc[-1]
            
            # Extract experiment details from name and config
            exp_info = {
                'experiment': exp_name,
                'model': self._extract_model_type(exp_name, config),
                'loss_function': self._extract_loss_function(exp_name),
                'attention_mechanism': self._extract_attention(exp_name),
                'resolution': self._extract_resolution(exp_name, config),
                'training_type': self._extract_training_type(exp_name),
                'epochs': config.get('epochs', 150),
                'batch_size': config.get('batch', 16),
                'optimizer': config.get('optimizer', 'SGD')
            }
            
            # Add final metrics
            metrics = {
                'final_epoch': int(final_metrics['epoch']),
                'final_precision': final_metrics['metrics/precision(B)'],
                'final_recall': final_metrics['metrics/recall(B)'],
                'final_mAP50': final_metrics['metrics/mAP50(B)'],
                'final_mAP50_95': final_metrics['metrics/mAP50-95(B)'],
                'final_box_loss': final_metrics['val/box_loss'],
                'final_cls_loss': final_metrics['val/cls_loss'],
                'training_time': final_metrics['time']
            }
            
            # Find best mAP50 epoch
            best_map50_idx = results_df['metrics/mAP50(B)'].idxmax()
            best_metrics = {
                'best_epoch': int(results_df.loc[best_map50_idx, 'epoch']),
                'best_precision': results_df.loc[best_map50_idx, 'metrics/precision(B)'],
                'best_recall': results_df.loc[best_map50_idx, 'metrics/recall(B)'],
                'best_mAP50': results_df.loc[best_map50_idx, 'metrics/mAP50(B)'],
                'best_mAP50_95': results_df.loc[best_map50_idx, 'metrics/mAP50-95(B)']
            }
            
            # Combine all data
            row_data = {**exp_info, **metrics, **best_metrics}
            metrics_data.append(row_data)
        
        return pd.DataFrame(metrics_data)
    
    def _extract_model_type(self, exp_name: str, config: dict) -> str:
        """Extract model type from experiment name or config"""
        if 'yolov8s' in exp_name.lower():
            return 'YOLOv8s'
        elif 'yolov8n' in exp_name.lower():
            return 'YOLOv8n'
        else:
            model = config.get('model', '')
            if 'yolov8s' in model.lower():
                return 'YOLOv8s'
            else:
                return 'YOLOv8n'
    
    def _extract_loss_function(self, exp_name: str) -> str:
        """Extract loss function from experiment name"""
        if 'siou' in exp_name.lower():
            return 'SIoU'
        elif 'eiou' in exp_name.lower():
            return 'EIoU'
        else:
            return 'IoU'
    
    def _extract_attention(self, exp_name: str) -> str:
        """Extract attention mechanism from experiment name"""
        if 'cbam' in exp_name.lower():
            return 'CBAM'
        elif 'eca' in exp_name.lower():
            return 'ECA'
        elif 'coordatt' in exp_name.lower():
            return 'CoordAtt'
        else:
            return 'None'
    
    def _extract_resolution(self, exp_name: str, config: dict) -> int:
        """Extract image resolution"""
        if '1024px' in exp_name:
            return 1024
        elif '640px' in exp_name:
            return 640
        else:
            return config.get('imgsz', 640)
    
    def _extract_training_type(self, exp_name: str) -> str:
        """Extract training type (stable vs standard)"""
        if 'stable' in exp_name.lower():
            return 'Stable'
        elif 'standard' in exp_name.lower():
            return 'Standard'
        else:
            return 'Standard'
    
    def create_performance_comparison_table(self) -> None:
        """Create comprehensive performance comparison table"""
        df_metrics = self.extract_final_metrics()
        
        # Sort by best mAP50
        df_sorted = df_metrics.sort_values('best_mAP50', ascending=False)
        
        # Create formatted table
        table_cols = [
            'experiment', 'model', 'loss_function', 'attention_mechanism',
            'resolution', 'training_type', 'best_mAP50', 'best_mAP50_95',
            'best_precision', 'best_recall', 'best_epoch', 'training_time'
        ]
        
        df_table = df_sorted[table_cols].copy()
        
        # Format numeric columns
        numeric_cols = ['best_mAP50', 'best_mAP50_95', 'best_precision', 'best_recall']
        for col in numeric_cols:
            df_table[col] = df_table[col].round(4)
        
        df_table['training_time'] = df_table['training_time'].round(1)
        
        # Save table
        df_table.to_csv(self.output_dir / "tables" / "performance_comparison.csv", index=False)
        df_table.to_excel(self.output_dir / "tables" / "performance_comparison.xlsx", index=False)
        
        print("✓ Performance comparison table saved")
        return df_table
    
    def create_training_curves_analysis(self) -> None:
        """Generate training curves analysis"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Curves Comparison - Top 5 Performers', fontsize=16)
        
        # Get top 5 experiments by mAP50
        df_metrics = self.extract_final_metrics()
        top_5 = df_metrics.nlargest(5, 'best_mAP50')['experiment'].tolist()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(top_5)))
        
        for i, exp_name in enumerate(top_5):
            if exp_name not in self.experiments_data:
                continue
                
            results_df = self.experiments_data[exp_name]['results']
            
            # mAP50
            axes[0, 0].plot(results_df['epoch'], results_df['metrics/mAP50(B)'], 
                           label=exp_name[:20], color=colors[i], linewidth=2)
            
            # mAP50-95
            axes[0, 1].plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], 
                           label=exp_name[:20], color=colors[i], linewidth=2)
            
            # Box Loss
            axes[1, 0].plot(results_df['epoch'], results_df['val/box_loss'], 
                           label=exp_name[:20], color=colors[i], linewidth=2)
            
            # Class Loss
            axes[1, 1].plot(results_df['epoch'], results_df['val/cls_loss'], 
                           label=exp_name[:20], color=colors[i], linewidth=2)
        
        # Customize plots
        axes[0, 0].set_title('mAP@0.5')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('mAP@0.5')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('mAP@0.5:0.95')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP@0.5:0.95')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Validation Box Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Box Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Validation Classification Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Classification Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "training_curves_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Training curves analysis saved")
    
    def create_ablation_study_tables(self) -> None:
        """Create ablation study tables"""
        df_metrics = self.extract_final_metrics()
        
        # 1. Architecture Comparison (YOLOv8n vs YOLOv8s)
        arch_comparison = df_metrics.groupby('model').agg({
            'best_mAP50': ['mean', 'std', 'max'],
            'best_mAP50_95': ['mean', 'std', 'max'],
            'training_time': 'mean'
        }).round(4)
        
        arch_comparison.to_csv(self.output_dir / "tables" / "architecture_comparison.csv")
        
        # 2. Loss Function Comparison
        loss_comparison = df_metrics.groupby('loss_function').agg({
            'best_mAP50': ['mean', 'std', 'max'],
            'best_mAP50_95': ['mean', 'std', 'max'],
            'best_precision': 'mean',
            'best_recall': 'mean'
        }).round(4)
        
        loss_comparison.to_csv(self.output_dir / "tables" / "loss_function_comparison.csv")
        
        # 3. Attention Mechanism Comparison
        attention_comparison = df_metrics.groupby('attention_mechanism').agg({
            'best_mAP50': ['mean', 'std', 'max'],
            'best_mAP50_95': ['mean', 'std', 'max'],
            'training_time': 'mean'
        }).round(4)
        
        attention_comparison.to_csv(self.output_dir / "tables" / "attention_mechanism_comparison.csv")
        
        # 4. Resolution Study
        resolution_study = df_metrics.groupby('resolution').agg({
            'best_mAP50': ['mean', 'std'],
            'best_mAP50_95': ['mean', 'std'],
            'training_time': 'mean'
        }).round(4)
        
        resolution_study.to_csv(self.output_dir / "tables" / "resolution_study.csv")
        
        # 5. Training Stability Comparison
        stability_comparison = df_metrics.groupby('training_type').agg({
            'best_mAP50': ['mean', 'std'],
            'best_mAP50_95': ['mean', 'std'],
            'final_mAP50': ['mean', 'std']
        }).round(4)
        
        stability_comparison.to_csv(self.output_dir / "tables" / "training_stability_comparison.csv")
        
        print("✓ Ablation study tables saved")
    
    def create_performance_heatmaps(self) -> None:
        """Create performance heatmaps"""
        df_metrics = self.extract_final_metrics()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Create pivot tables for heatmaps
        pivot_map50 = df_metrics.pivot_table(
            values='best_mAP50', 
            index='attention_mechanism', 
            columns='loss_function', 
            aggfunc='mean'
        )
        
        pivot_map50_95 = df_metrics.pivot_table(
            values='best_mAP50_95', 
            index='attention_mechanism', 
            columns='loss_function', 
            aggfunc='mean'
        )
        
        # Create heatmaps
        sns.heatmap(pivot_map50, annot=True, fmt='.4f', cmap='YlOrRd', 
                   ax=axes[0], cbar_kws={'label': 'mAP@0.5'})
        axes[0].set_title('mAP@0.5 by Attention Mechanism and Loss Function')
        
        sns.heatmap(pivot_map50_95, annot=True, fmt='.4f', cmap='YlOrRd', 
                   ax=axes[1], cbar_kws={'label': 'mAP@0.5:0.95'})
        axes[1].set_title('mAP@0.5:0.95 by Attention Mechanism and Loss Function')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "performance_heatmaps.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Performance heatmaps saved")
    
    def create_summary_report(self) -> None:
        """Create comprehensive summary report"""
        df_metrics = self.extract_final_metrics()
        
        # Best performing model
        best_model = df_metrics.loc[df_metrics['best_mAP50'].idxmax()]
        
        # Calculate statistics
        summary_stats = {
            'Total Experiments': len(df_metrics),
            'Best mAP@0.5': f"{best_model['best_mAP50']:.4f}",
            'Best Model': best_model['experiment'],
            'Average mAP@0.5': f"{df_metrics['best_mAP50'].mean():.4f}",
            'mAP@0.5 Std': f"{df_metrics['best_mAP50'].std():.4f}",
            'Average Training Time': f"{df_metrics['training_time'].mean():.1f} seconds"
        }
        
        # Top 5 performers
        top_5 = df_metrics.nlargest(5, 'best_mAP50')[['experiment', 'best_mAP50', 'best_mAP50_95']]
        
        # Architecture performance
        arch_stats = df_metrics.groupby('model')['best_mAP50'].agg(['mean', 'std']).round(4)
        
        # Create summary report
        report = {
            'summary_statistics': summary_stats,
            'top_5_performers': top_5.to_dict('records'),
            'architecture_performance': arch_stats.to_dict(),
            'loss_function_performance': df_metrics.groupby('loss_function')['best_mAP50'].mean().to_dict(),
            'attention_mechanism_performance': df_metrics.groupby('attention_mechanism')['best_mAP50'].mean().to_dict()
        }
        
        # Save as JSON
        with open(self.output_dir / "summary" / "experiment_summary.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create text summary
        with open(self.output_dir / "summary" / "experiment_summary.txt", 'w') as f:
            f.write("PCB DEFECT DETECTION EXPERIMENT ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            f.write("OVERVIEW:\n")
            f.write("---------\n")
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("TOP 5 PERFORMERS:\n")
            f.write("-----------------\n")
            for i, row in top_5.iterrows():
                f.write(f"{row['experiment'][:40]:40} | mAP@0.5: {row['best_mAP50']:.4f} | mAP@0.5:0.95: {row['best_mAP50_95']:.4f}\n")
            f.write("\n")
            
            f.write("ARCHITECTURE COMPARISON:\n")
            f.write("------------------------\n")
            for arch, stats in arch_stats.iterrows():
                f.write(f"{arch}: Mean mAP@0.5 = {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        
        print("✓ Summary report saved")
    
    def run_complete_analysis(self) -> None:
        """Run complete analysis pipeline"""
        print("Starting PCB Defect Detection Experiment Analysis...")
        print("="*50)
        
        # Load all data
        self.load_experiment_data()
        
        # Generate all outputs
        print("\nGenerating analysis outputs...")
        self.create_performance_comparison_table()
        self.create_training_curves_analysis()
        self.create_ablation_study_tables()
        self.create_performance_heatmaps()
        self.create_summary_report()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print(f"All outputs saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        print("- tables/: CSV and Excel files with comparative analysis")
        print("- plots/: Training curves and performance visualizations") 
        print("- summary/: Comprehensive experiment summary")
        
def main():
    # Configuration
    experiments_dir = "F:/PCB_defect/experiments/pcb-defect-150epochs-v1"
    
    # Initialize analyzer
    analyzer = PCBExperimentAnalyzer(experiments_dir)
    
    # Run complete analysis
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()