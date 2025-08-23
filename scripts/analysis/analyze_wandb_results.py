#!/usr/bin/env python3
"""
WandB Results Analysis Script for PCB Defect Detection Experiments
================================================================

This script fetches experimental results from Weights & Biases API and generates
comprehensive analysis and insights for the systematic study of YOLOv8 variants,
attention mechanisms, and loss function combinations.

Features:
- Fetch all runs from WandB project
- Performance comparison across models and attention mechanisms
- Statistical analysis and significance testing
- Visualization of key metrics
- Automated insights generation
- Export results to CSV/Excel for further analysis

Usage:
    python analyze_wandb_results.py --project pcb-defect-150epochs-v1
    python analyze_wandb_results.py --project pcb-defect-150epochs-v1 --export-csv
    python analyze_wandb_results.py --project pcb-defect-150epochs-v1 --detailed-analysis

Author: PCB Defect Detection Team
Date: 2025-01-21
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# WandB and statistics imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("❌ WandB not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False
    sys.exit(1)

try:
    from scipy import stats
    from scipy.stats import ttest_ind, mannwhitneyu, kruskal
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️  SciPy not available. Statistical tests will be limited.")
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure plotting
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class WandBResultsAnalyzer:
    """
    Comprehensive analyzer for WandB experimental results.
    """
    
    def __init__(self, project_name: str, entity: str = None):
        """
        Initialize the WandB results analyzer.
        
        Args:
            project_name (str): WandB project name
            entity (str, optional): WandB entity/username
        """
        self.project_name = project_name
        self.entity = entity
        self.api = wandb.Api()
        self.runs_data = []
        self.df = None
        
        # Create output directory
        self.output_dir = Path("wandb_analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔍 WandB Results Analyzer initialized")
        logger.info(f"📊 Project: {project_name}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
    def fetch_runs(self) -> pd.DataFrame:
        """
        Fetch all runs from the WandB project.
        
        Returns:
            pd.DataFrame: DataFrame containing all run data
        """
        try:
            logger.info(f"🔄 Fetching runs from WandB project: {self.project_name}")
            
            # Get runs from WandB
            if self.entity:
                project_path = f"{self.entity}/{self.project_name}"
            else:
                project_path = self.project_name
                
            runs = self.api.runs(project_path)
            
            logger.info(f"📥 Found {len(runs)} runs")
            
            # Extract run data
            runs_data = []
            for run in runs:
                try:
                    # Basic run information
                    run_data = {
                        'run_id': run.id,
                        'run_name': run.name,
                        'state': run.state,
                        'created_at': run.created_at,
                        'duration': run._attrs.get('runtime', 0),
                        'tags': run.tags,
                        'notes': run.notes,
                    }
                    
                    # Extract config
                    config = run.config
                    run_data.update({
                        'model_type': config.get('model_config', {}).get('type', 'unknown'),
                        'attention_mechanism': config.get('model_config', {}).get('attention_mechanism', 'none'),
                        'batch_size': config.get('training_config', {}).get('batch', 'unknown'),
                        'image_size': config.get('training_config', {}).get('imgsz', 640),
                        'epochs': config.get('training_config', {}).get('epochs', 'unknown'),
                        'optimizer': config.get('training_config', {}).get('optimizer', 'unknown'),
                        'learning_rate': config.get('training_config', {}).get('lr0', 'unknown'),
                    })
                    
                    # Extract summary metrics (final values)
                    summary = run.summary
                    metrics_mapping = {
                        'final_map50': ['metrics/mAP50', 'val/mAP50', 'mAP50'],
                        'final_map50_95': ['metrics/mAP50-95', 'val/mAP50-95', 'mAP50-95'],
                        'final_precision': ['metrics/precision(B)', 'val/precision', 'precision'],
                        'final_recall': ['metrics/recall(B)', 'val/recall', 'recall'],
                        'final_f1': ['metrics/F1(B)', 'val/F1', 'f1'],
                        'final_box_loss': ['train/box_loss', 'box_loss'],
                        'final_cls_loss': ['train/cls_loss', 'cls_loss'],
                        'final_dfl_loss': ['train/dfl_loss', 'dfl_loss'],
                        'val_box_loss': ['val/box_loss'],
                        'val_cls_loss': ['val/cls_loss'],
                        'val_dfl_loss': ['val/dfl_loss'],
                        'total_parameters': ['model/total_parameters'],
                        'trainable_parameters': ['model/trainable_parameters'],
                        'training_time': ['final/training_time_seconds', 'training_time'],
                    }
                    
                    # Try to get metrics with fallback names
                    for metric_name, possible_keys in metrics_mapping.items():
                        value = None
                        for key in possible_keys:
                            if key in summary:
                                value = summary[key]
                                break
                        run_data[metric_name] = value
                    
                    # Get per-class metrics if available
                    for i in range(6):  # HRIPCB has 6 classes
                        class_map_key = f'metrics/mAP50(B)_{i}'
                        if class_map_key in summary:
                            run_data[f'class_{i}_map50'] = summary[class_map_key]
                    
                    # Extract experiment metadata from tags
                    phase = 'unknown'
                    experiment_type = 'unknown'
                    for tag in run.tags:
                        if 'phase_1' in tag:
                            phase = 'Phase 1: Baselines'
                        elif 'phase_2' in tag:
                            phase = 'Phase 2: Attention'
                        elif 'phase_3' in tag:
                            phase = 'Phase 3: Loss/Resolution'
                        
                        if 'model_scaling' in tag:
                            experiment_type = 'Model Scaling'
                        elif 'attention_study' in tag:
                            experiment_type = 'Attention Study'
                        elif 'architecture_study' in tag:
                            experiment_type = 'Architecture Study'
                        elif 'resolution_study' in tag:
                            experiment_type = 'Resolution Study'
                    
                    run_data['phase'] = phase
                    run_data['experiment_type'] = experiment_type
                    
                    runs_data.append(run_data)
                    
                except Exception as e:
                    logger.warning(f"⚠️  Error processing run {run.name}: {e}")
                    continue
            
            # Create DataFrame
            self.df = pd.DataFrame(runs_data)
            self.runs_data = runs_data
            
            # Clean and process data
            self.df = self._clean_dataframe(self.df)
            
            logger.info(f"✅ Successfully processed {len(self.df)} runs")
            logger.info(f"📊 Completed runs: {len(self.df[self.df['state'] == 'finished'])}")
            logger.info(f"🔄 Running runs: {len(self.df[self.df['state'] == 'running'])}")
            logger.info(f"❌ Failed runs: {len(self.df[self.df['state'] == 'failed'])}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch runs: {e}")
            raise
            
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the dataframe.
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Convert numeric columns
        numeric_columns = [
            'final_map50', 'final_map50_95', 'final_precision', 'final_recall', 'final_f1',
            'final_box_loss', 'final_cls_loss', 'final_dfl_loss',
            'val_box_loss', 'val_cls_loss', 'val_dfl_loss',
            'total_parameters', 'trainable_parameters', 'training_time',
            'batch_size', 'image_size', 'epochs', 'learning_rate'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert duration to hours
        if 'duration' in df.columns:
            df['duration_hours'] = df['duration'] / 3600
        
        # Create model variant column
        df['model_variant'] = df.apply(self._create_model_variant, axis=1)
        
        # Filter only completed runs for analysis
        df_completed = df[df['state'] == 'finished'].copy()
        
        return df_completed
        
    def _create_model_variant(self, row) -> str:
        """Create a descriptive model variant name."""
        model_type = row.get('model_type', 'unknown')
        attention = row.get('attention_mechanism', 'none')
        image_size = row.get('image_size', 640)
        
        if attention == 'none' or attention == 'unknown':
            attention_str = ""
        else:
            attention_str = f" + {attention.upper()}"
        
        if image_size != 640:
            size_str = f" ({image_size}px)"
        else:
            size_str = ""
        
        return f"{model_type.upper()}{attention_str}{size_str}"
        
    def generate_performance_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary.
        
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        if self.df is None or len(self.df) == 0:
            logger.warning("No data available for analysis")
            return {}
        
        logger.info("📊 Generating performance summary...")
        
        summary = {}
        
        # Overall statistics
        summary['total_experiments'] = len(self.df)
        summary['successful_experiments'] = len(self.df[self.df['state'] == 'finished'])
        summary['total_training_time_hours'] = self.df['training_time'].sum() / 3600 if 'training_time' in self.df.columns else 0
        
        # Performance metrics summary
        metrics = ['final_map50', 'final_map50_95', 'final_precision', 'final_recall', 'final_f1']
        for metric in metrics:
            if metric in self.df.columns and not self.df[metric].isna().all():
                summary[f'{metric}_mean'] = self.df[metric].mean()
                summary[f'{metric}_std'] = self.df[metric].std()
                summary[f'{metric}_max'] = self.df[metric].max()
                summary[f'{metric}_min'] = self.df[metric].min()
                summary[f'{metric}_best_model'] = self.df.loc[self.df[metric].idxmax(), 'model_variant']
        
        # Model comparison
        if 'model_variant' in self.df.columns:
            model_performance = self.df.groupby('model_variant')['final_map50'].agg(['mean', 'std', 'count'])
            summary['model_performance'] = model_performance.to_dict()
        
        # Phase comparison
        if 'phase' in self.df.columns:
            phase_performance = self.df.groupby('phase')['final_map50'].agg(['mean', 'std', 'count'])
            summary['phase_performance'] = phase_performance.to_dict()
        
        return summary
        
    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """
        Perform statistical analysis and significance testing.
        
        Returns:
            Dict[str, Any]: Statistical analysis results
        """
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available - skipping statistical analysis")
            return {}
        
        if self.df is None or len(self.df) == 0:
            return {}
        
        logger.info("📈 Performing statistical analysis...")
        
        results = {}
        
        # Compare model types
        if 'model_type' in self.df.columns and 'final_map50' in self.df.columns:
            model_groups = []
            model_names = []
            for model_type in self.df['model_type'].unique():
                if pd.notna(model_type):
                    group_data = self.df[self.df['model_type'] == model_type]['final_map50'].dropna()
                    if len(group_data) > 0:
                        model_groups.append(group_data)
                        model_names.append(model_type)
            
            if len(model_groups) >= 2:
                # Kruskal-Wallis test for multiple groups
                if len(model_groups) > 2:
                    stat, p_value = kruskal(*model_groups)
                    results['model_comparison'] = {
                        'test': 'Kruskal-Wallis',
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'groups': model_names
                    }
                # Mann-Whitney U test for two groups
                elif len(model_groups) == 2:
                    stat, p_value = mannwhitneyu(model_groups[0], model_groups[1], alternative='two-sided')
                    results['model_comparison'] = {
                        'test': 'Mann-Whitney U',
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'groups': model_names
                    }
        
        # Compare attention mechanisms
        if 'attention_mechanism' in self.df.columns and 'final_map50' in self.df.columns:
            attention_groups = []
            attention_names = []
            for attention in self.df['attention_mechanism'].unique():
                if pd.notna(attention):
                    group_data = self.df[self.df['attention_mechanism'] == attention]['final_map50'].dropna()
                    if len(group_data) > 0:
                        attention_groups.append(group_data)
                        attention_names.append(attention)
            
            if len(attention_groups) >= 2:
                if len(attention_groups) > 2:
                    stat, p_value = kruskal(*attention_groups)
                    results['attention_comparison'] = {
                        'test': 'Kruskal-Wallis',
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'groups': attention_names
                    }
                elif len(attention_groups) == 2:
                    stat, p_value = mannwhitneyu(attention_groups[0], attention_groups[1], alternative='two-sided')
                    results['attention_comparison'] = {
                        'test': 'Mann-Whitney U',
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'groups': attention_names
                    }
        
        return results
        
    def generate_insights(self) -> List[str]:
        """
        Generate automated insights from the experimental results.
        
        Returns:
            List[str]: List of insights and recommendations
        """
        if self.df is None or len(self.df) == 0:
            return ["No data available for analysis"]
        
        logger.info("🔍 Generating insights...")
        
        insights = []
        
        # Performance insights
        if 'final_map50' in self.df.columns and not self.df['final_map50'].isna().all():
            valid_map50_data = self.df.dropna(subset=['final_map50'])
            if len(valid_map50_data) > 0:
                best_map50 = valid_map50_data['final_map50'].max()
                best_model_idx = valid_map50_data['final_map50'].idxmax()
                best_model = valid_map50_data.loc[best_model_idx, 'model_variant']
                mean_map50 = valid_map50_data['final_map50'].mean()
            
                insights.append(f"🏆 Best performing model: {best_model} with mAP@0.5 = {best_map50:.4f}")
                insights.append(f"📊 Average mAP@0.5 across all experiments: {mean_map50:.4f}")
                
                if best_map50 > 0.8:
                    insights.append("✅ Excellent detection performance achieved (mAP@0.5 > 0.8)")
                elif best_map50 > 0.6:
                    insights.append("👍 Good detection performance achieved (mAP@0.5 > 0.6)")
                else:
                    insights.append("⚠️  Detection performance needs improvement (mAP@0.5 < 0.6)")
            else:
                insights.append("⚠️  No valid mAP@0.5 data available for analysis")
        
        # Model comparison insights
        if 'model_type' in self.df.columns and 'final_map50' in self.df.columns:
            valid_model_data = self.df.dropna(subset=['model_type', 'final_map50'])
            if len(valid_model_data) > 0:
                model_performance = valid_model_data.groupby('model_type')['final_map50'].mean().sort_values(ascending=False)
                if len(model_performance) > 1:
                    best_model_type = model_performance.index[0]
                    worst_model_type = model_performance.index[-1]
                    performance_gap = model_performance.iloc[0] - model_performance.iloc[-1]
                    
                    insights.append(f"🔄 {best_model_type.upper()} outperforms {worst_model_type.upper()} by {performance_gap:.4f} mAP@0.5")
                    
                    if performance_gap > 0.05:
                        insights.append("📈 Significant performance difference between model architectures")
                    else:
                        insights.append("📊 Model architectures show similar performance levels")
        
        # Attention mechanism insights
        if 'attention_mechanism' in self.df.columns and 'final_map50' in self.df.columns:
            valid_attention_data = self.df.dropna(subset=['attention_mechanism', 'final_map50'])
            if len(valid_attention_data) > 0:
                attention_data = valid_attention_data[valid_attention_data['attention_mechanism'] != 'none']
                baseline_data = valid_attention_data[valid_attention_data['attention_mechanism'] == 'none']
                
                if len(attention_data) > 0 and len(baseline_data) > 0:
                    attention_mean = attention_data['final_map50'].mean()
                    baseline_mean = baseline_data['final_map50'].mean()
                    improvement = attention_mean - baseline_mean
                    
                    if improvement > 0.01:
                        insights.append(f"✨ Attention mechanisms improve performance by {improvement:.4f} mAP@0.5 on average")
                    elif improvement > 0:
                        insights.append(f"🔍 Attention mechanisms show modest improvement: +{improvement:.4f} mAP@0.5")
                    else:
                        insights.append("⚠️  Attention mechanisms do not show clear performance benefits")
                    
                    # Best attention mechanism
                    if len(attention_data) > 0:
                        attention_performance = attention_data.groupby('attention_mechanism')['final_map50'].mean().sort_values(ascending=False)
                        if len(attention_performance) > 0:
                            best_attention = attention_performance.index[0]
                            best_attention_score = attention_performance.iloc[0]
                            insights.append(f"🎯 Best attention mechanism: {best_attention.upper()} with {best_attention_score:.4f} mAP@0.5")
        
        # Resolution insights
        if 'image_size' in self.df.columns and 'final_map50' in self.df.columns:
            valid_resolution_data = self.df.dropna(subset=['image_size', 'final_map50'])
            if len(valid_resolution_data) > 0:
                resolution_performance = valid_resolution_data.groupby('image_size')['final_map50'].mean().sort_values(ascending=False)
                if len(resolution_performance) > 1:
                    high_res = resolution_performance.index[0]
                    high_res_score = resolution_performance.iloc[0]
                    standard_res_score = resolution_performance.get(640, 0)
                    
                    if high_res > 640 and high_res_score > standard_res_score:
                        improvement = high_res_score - standard_res_score
                        insights.append(f"📏 High resolution ({high_res}px) improves performance by {improvement:.4f} mAP@0.5")
                    else:
                        insights.append("📏 Higher resolution does not show significant benefits")
        
        # Training efficiency insights
        if 'training_time' in self.df.columns and 'final_map50' in self.df.columns:
            # Efficiency score: performance per hour (only for valid data)
            valid_time_data = self.df.dropna(subset=['training_time', 'final_map50'])
            if len(valid_time_data) > 0:
                valid_time_data = valid_time_data[valid_time_data['training_time'] > 0]  # Avoid division by zero
                if len(valid_time_data) > 0:
                    efficiency = valid_time_data['final_map50'] / (valid_time_data['training_time'] / 3600)
                    max_efficiency_idx = efficiency.idxmax()
                    most_efficient = valid_time_data.loc[max_efficiency_idx, 'model_variant']
                    efficiency_score = efficiency.max()
                    
                    insights.append(f"⚡ Most efficient model: {most_efficient} ({efficiency_score:.4f} mAP@0.5 per hour)")
        
        # Parameter efficiency insights
        if 'total_parameters' in self.df.columns and 'final_map50' in self.df.columns:
            valid_param_data = self.df.dropna(subset=['total_parameters', 'final_map50'])
            if len(valid_param_data) > 0:
                valid_param_data = valid_param_data[valid_param_data['total_parameters'] > 0]  # Avoid division by zero
                if len(valid_param_data) > 0:
                    param_efficiency = valid_param_data['final_map50'] / (valid_param_data['total_parameters'] / 1e6)  # per million params
                    max_param_efficiency_idx = param_efficiency.idxmax()
                    most_param_efficient = valid_param_data.loc[max_param_efficiency_idx, 'model_variant']
                    param_efficiency_score = param_efficiency.max()
                    
                    insights.append(f"🎯 Most parameter-efficient model: {most_param_efficient} ({param_efficiency_score:.4f} mAP@0.5 per million parameters)")
        
        return insights
        
    def create_visualizations(self, save_plots: bool = True) -> None:
        """
        Create comprehensive visualizations of the results.
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        if self.df is None or len(self.df) == 0:
            logger.warning("No data available for visualization")
            return
        
        logger.info("📊 Creating visualizations...")
        
        # Set up the plotting environment
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # 1. Model Performance Comparison
        if 'model_variant' in self.df.columns and 'final_map50' in self.df.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Box plot of performance by model variant
            model_data = []
            model_labels = []
            for variant in self.df['model_variant'].unique():
                if pd.notna(variant):
                    data = self.df[self.df['model_variant'] == variant]['final_map50'].dropna()
                    if len(data) > 0:
                        model_data.append(data)
                        model_labels.append(variant)
            
            if model_data:
                ax.boxplot(model_data, labels=model_labels)
                ax.set_ylabel('mAP@0.5')
                ax.set_title('Model Performance Comparison')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(self.output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # 2. Metrics Correlation Heatmap
        metrics_cols = ['final_map50', 'final_map50_95', 'final_precision', 'final_recall', 'final_f1']
        available_metrics = [col for col in metrics_cols if col in self.df.columns and not self.df[col].isna().all()]
        
        if len(available_metrics) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = self.df[available_metrics].corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                       square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
            ax.set_title('Metrics Correlation Matrix')
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(self.output_dir / 'metrics_correlation.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Training Time vs Performance
        if 'training_time' in self.df.columns and 'final_map50' in self.df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            scatter = ax.scatter(self.df['training_time'] / 3600, self.df['final_map50'], 
                               c=self.df['total_parameters'] if 'total_parameters' in self.df.columns else 'blue',
                               alpha=0.7, s=100, cmap='viridis')
            
            ax.set_xlabel('Training Time (hours)')
            ax.set_ylabel('mAP@0.5')
            ax.set_title('Training Time vs Performance')
            
            if 'total_parameters' in self.df.columns:
                cbar = plt.colorbar(scatter)
                cbar.set_label('Total Parameters')
            
            # Add model variant labels
            if 'model_variant' in self.df.columns:
                for i, row in self.df.iterrows():
                    if pd.notna(row['training_time']) and pd.notna(row['final_map50']):
                        ax.annotate(row['model_variant'], 
                                  (row['training_time'] / 3600, row['final_map50']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(self.output_dir / 'training_time_vs_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 4. Phase Comparison
        if 'phase' in self.df.columns and 'final_map50' in self.df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            phase_means = self.df.groupby('phase')['final_map50'].mean().sort_values(ascending=True)
            phase_stds = self.df.groupby('phase')['final_map50'].std()
            
            bars = ax.barh(range(len(phase_means)), phase_means.values, 
                          xerr=phase_stds.values, capsize=5, alpha=0.7)
            ax.set_yticks(range(len(phase_means)))
            ax.set_yticklabels(phase_means.index)
            ax.set_xlabel('Average mAP@0.5')
            ax.set_title('Performance by Experimental Phase')
            
            # Add value labels on bars
            for i, (bar, mean) in enumerate(zip(bars, phase_means.values)):
                ax.text(mean + 0.005, i, f'{mean:.3f}', va='center')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(self.output_dir / 'phase_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        logger.info(f"✅ Visualizations saved to {self.output_dir}")
        
    def export_results(self, format: str = 'csv') -> Path:
        """
        Export results to CSV or Excel file.
        
        Args:
            format (str): Export format ('csv' or 'excel')
            
        Returns:
            Path: Path to exported file
        """
        if self.df is None or len(self.df) == 0:
            logger.warning("No data available for export")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == 'csv':
            filename = self.output_dir / f"wandb_results_{timestamp}.csv"
            self.df.to_csv(filename, index=False)
        elif format.lower() == 'excel':
            filename = self.output_dir / f"wandb_results_{timestamp}.xlsx"
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                self.df.to_excel(writer, sheet_name='All_Results', index=False)
                
                # Summary statistics
                summary_stats = self.df.describe()
                summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
                
                # Model comparison
                if 'model_variant' in self.df.columns:
                    model_summary = self.df.groupby('model_variant').agg({
                        'final_map50': ['mean', 'std', 'count'],
                        'final_map50_95': ['mean', 'std'],
                        'training_time': ['mean', 'std']
                    }).round(4)
                    model_summary.to_excel(writer, sheet_name='Model_Comparison')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"✅ Results exported to {filename}")
        return filename
        
    def generate_report(self) -> str:
        """
        Generate a comprehensive text report.
        
        Returns:
            str: Formatted report text
        """
        if self.df is None or len(self.df) == 0:
            return "No data available for report generation"
        
        logger.info("📝 Generating comprehensive report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PCB DEFECT DETECTION EXPERIMENT ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"WandB Project: {self.project_name}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 50)
        summary = self.generate_performance_summary()
        
        report_lines.append(f"Total Experiments: {summary.get('total_experiments', 0)}")
        report_lines.append(f"Successful Experiments: {summary.get('successful_experiments', 0)}")
        report_lines.append(f"Total Training Time: {summary.get('total_training_time_hours', 0):.1f} hours")
        
        if 'final_map50_mean' in summary:
            report_lines.append(f"Average mAP@0.5: {summary['final_map50_mean']:.4f} ± {summary['final_map50_std']:.4f}")
            report_lines.append(f"Best mAP@0.5: {summary['final_map50_max']:.4f} ({summary['final_map50_best_model']})")
            report_lines.append(f"Worst mAP@0.5: {summary['final_map50_min']:.4f}")
        
        report_lines.append("")
        
        # Key Insights
        report_lines.append("KEY INSIGHTS")
        report_lines.append("-" * 50)
        insights = self.generate_insights()
        for insight in insights:
            report_lines.append(f"• {insight}")
        report_lines.append("")
        
        # Statistical Analysis
        if SCIPY_AVAILABLE:
            report_lines.append("STATISTICAL ANALYSIS")
            report_lines.append("-" * 50)
            stats_results = self.perform_statistical_analysis()
            
            if 'model_comparison' in stats_results:
                model_stats = stats_results['model_comparison']
                report_lines.append(f"Model Comparison ({model_stats['test']}):")
                report_lines.append(f"  Statistic: {model_stats['statistic']:.4f}")
                report_lines.append(f"  P-value: {model_stats['p_value']:.4f}")
                report_lines.append(f"  Significant: {'Yes' if model_stats['significant'] else 'No'}")
                report_lines.append("")
            
            if 'attention_comparison' in stats_results:
                attention_stats = stats_results['attention_comparison']
                report_lines.append(f"Attention Mechanism Comparison ({attention_stats['test']}):")
                report_lines.append(f"  Statistic: {attention_stats['statistic']:.4f}")
                report_lines.append(f"  P-value: {attention_stats['p_value']:.4f}")
                report_lines.append(f"  Significant: {'Yes' if attention_stats['significant'] else 'No'}")
                report_lines.append("")
        
        # Detailed Results
        if 'model_variant' in self.df.columns:
            report_lines.append("DETAILED RESULTS BY MODEL")
            report_lines.append("-" * 50)
            
            model_stats = self.df.groupby('model_variant').agg({
                'final_map50': ['count', 'mean', 'std', 'min', 'max'],
                'final_map50_95': ['mean', 'std'],
                'training_time': ['mean', 'std']
            }).round(4)
            
            for model in model_stats.index:
                report_lines.append(f"\n{model}:")
                if ('final_map50', 'count') in model_stats.columns:
                    report_lines.append(f"  Experiments: {model_stats.loc[model, ('final_map50', 'count')]}")
                if ('final_map50', 'mean') in model_stats.columns:
                    report_lines.append(f"  mAP@0.5: {model_stats.loc[model, ('final_map50', 'mean')]:.4f} ± {model_stats.loc[model, ('final_map50', 'std')]:.4f}")
                if ('final_map50_95', 'mean') in model_stats.columns:
                    report_lines.append(f"  mAP@0.5-0.95: {model_stats.loc[model, ('final_map50_95', 'mean')]:.4f} ± {model_stats.loc[model, ('final_map50_95', 'std')]:.4f}")
                if ('training_time', 'mean') in model_stats.columns:
                    report_lines.append(f"  Training Time: {model_stats.loc[model, ('training_time', 'mean')]/3600:.1f} ± {model_stats.loc[model, ('training_time', 'std')]/3600:.1f} hours")
        
        # Recommendations
        report_lines.append("\n\nRECOMMENDATIONS")
        report_lines.append("-" * 50)
        
        if 'final_map50' in self.df.columns and not self.df['final_map50'].isna().all():
            best_model = self.df.loc[self.df['final_map50'].idxmax(), 'model_variant']
            best_score = self.df['final_map50'].max()
            
            report_lines.append(f"• Deploy {best_model} for production (mAP@0.5: {best_score:.4f})")
            
            if best_score < 0.6:
                report_lines.append("• Consider data augmentation strategies to improve performance")
                report_lines.append("• Investigate class imbalance issues")
                report_lines.append("• Consider ensemble methods")
            elif best_score < 0.8:
                report_lines.append("• Performance is good but could be optimized further")
                report_lines.append("• Consider hyperparameter tuning")
            else:
                report_lines.append("• Excellent performance achieved - ready for deployment")
        
        # Technical details
        report_lines.append("\n• Continue monitoring model performance on new data")
        report_lines.append("• Implement model versioning and monitoring in production")
        report_lines.append("• Consider automated retraining pipelines")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)
        
    def run_complete_analysis(self, export_csv: bool = True, export_excel: bool = True, 
                            save_plots: bool = True, save_report: bool = True) -> None:
        """
        Run the complete analysis pipeline.
        
        Args:
            export_csv (bool): Export results to CSV
            export_excel (bool): Export results to Excel
            save_plots (bool): Save visualization plots
            save_report (bool): Save text report
        """
        logger.info("🚀 Starting complete WandB analysis pipeline...")
        
        try:
            # Fetch data
            self.fetch_runs()
            
            if self.df is None or len(self.df) == 0:
                logger.error("❌ No data available for analysis")
                return
            
            # Generate analysis
            logger.info("📊 Generating performance summary...")
            summary = self.generate_performance_summary()
            
            logger.info("🔍 Generating insights...")
            insights = self.generate_insights()
            
            logger.info("📈 Performing statistical analysis...")
            stats_results = self.perform_statistical_analysis()
            
            # Create visualizations
            if save_plots:
                self.create_visualizations(save_plots=True)
            
            # Export data
            if export_csv:
                self.export_results('csv')
            
            if export_excel:
                self.export_results('excel')
            
            # Generate and save report
            if save_report:
                report = self.generate_report()
                report_file = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(report_file, 'w') as f:
                    f.write(report)
                logger.info(f"📝 Report saved to {report_file}")
            
            # Print summary to console
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE - SUMMARY")
            print("="*80)
            
            print(f"📊 Analyzed {len(self.df)} completed experiments")
            if 'final_map50_mean' in summary:
                print(f"🎯 Average Performance: {summary['final_map50_mean']:.4f} mAP@0.5")
                print(f"🏆 Best Performance: {summary['final_map50_max']:.4f} mAP@0.5 ({summary['final_map50_best_model']})")
            
            print(f"\n📁 All results saved to: {self.output_dir}")
            print(f"📊 Visualizations: {'✅ Generated' if save_plots else '❌ Skipped'}")
            print(f"📄 CSV Export: {'✅ Generated' if export_csv else '❌ Skipped'}")
            print(f"📊 Excel Export: {'✅ Generated' if export_excel else '❌ Skipped'}")
            print(f"📝 Report: {'✅ Generated' if save_report else '❌ Skipped'}")
            
            print("\n🔍 KEY INSIGHTS:")
            for insight in insights[:5]:  # Show top 5 insights
                print(f"• {insight}")
            
            if len(insights) > 5:
                print(f"... and {len(insights) - 5} more insights in the full report")
            
            logger.info("✅ Complete analysis pipeline finished successfully!")
            
        except Exception as e:
            logger.error(f"❌ Analysis pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="WandB Results Analysis for PCB Defect Detection Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_wandb_results.py --project pcb-defect-systematic-study
  
  # Export to CSV only
  python analyze_wandb_results.py --project pcb-defect-systematic-study --export-csv
  
  # Full analysis with all exports
  python analyze_wandb_results.py --project pcb-defect-systematic-study --detailed-analysis
  
  # Specify entity/username
  python analyze_wandb_results.py --project pcb-defect-systematic-study --entity your-username
        """
    )
    
    parser.add_argument('--project', type=str, required=True,
                        help='WandB project name')
    parser.add_argument('--entity', type=str, 
                        help='WandB entity/username (optional)')
    parser.add_argument('--export-csv', action='store_true',
                        help='Export results to CSV file')
    parser.add_argument('--export-excel', action='store_true',
                        help='Export results to Excel file')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--no-report', action='store_true',
                        help='Skip generating text report')
    parser.add_argument('--detailed-analysis', action='store_true',
                        help='Run complete analysis with all exports')
    
    args = parser.parse_args()
    
    # Set defaults for detailed analysis
    if args.detailed_analysis:
        export_csv = True
        export_excel = True
        save_plots = not args.no_plots
        save_report = not args.no_report
    else:
        export_csv = args.export_csv
        export_excel = args.export_excel
        save_plots = not args.no_plots
        save_report = not args.no_report
    
    try:
        # Create analyzer
        analyzer = WandBResultsAnalyzer(
            project_name=args.project,
            entity=args.entity
        )
        
        # Run analysis
        analyzer.run_complete_analysis(
            export_csv=export_csv,
            export_excel=export_excel,
            save_plots=save_plots,
            save_report=save_report
        )
        
    except KeyboardInterrupt:
        logger.info("⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()