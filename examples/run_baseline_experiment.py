#!/usr/bin/env python3
"""
Example: Baseline YOLOv8n Training
=================================

This example demonstrates how to train baseline YOLOv8n (without attention)
using the production-ready two-stage training framework.

This serves as the reference for comparing attention mechanism performance.

Baseline Features:
- Standard YOLOv8n architecture
- No attention mechanisms
- Reference performance for comparison
- Fastest training and inference
"""

import sys
from pathlib import Path
from train_attention_benchmark import AttentionTrainingPipeline

def main():
    """Run baseline YOLOv8n training experiment."""
    
    print("🚀 YOLOv8n Baseline Training (No Attention)")
    print("=" * 60)
    print("📋 Mechanism: None (Standard YOLOv8n)")
    print("🎯 Purpose: Reference baseline for comparison")
    print("📊 Expected: Reference performance (0% improvement)")
    print("🚀 Use Case: Speed-critical applications")
    print("=" * 60)
    
    # Configuration for baseline
    config_path = "configs/config_baseline.yaml"
    
    try:
        # Verify config exists
        if not Path(config_path).exists():
            print(f"❌ Configuration not found: {config_path}")
            print("💡 Make sure you have created the baseline configuration file")
            sys.exit(1)
        
        # Initialize training pipeline
        print("🔧 Initializing baseline training pipeline...")
        pipeline = AttentionTrainingPipeline(config_path)
        
        # Run two-stage training
        print("🎯 Starting two-stage baseline training...")
        print("   Stage 1: Warmup with frozen backbone (25 epochs)")
        print("   Stage 2: Fine-tuning with reduced LR (125 epochs)")
        print("   ⚡ Fastest training - no attention overhead!")
        
        warmup_results, finetune_results = pipeline.run_complete_pipeline()
        
        # Export the final model
        print("📦 Exporting baseline model...")
        pipeline.export_final_model(['onnx', 'torchscript'])
        
        # Display results
        warmup_map = warmup_results.results_dict.get('metrics/mAP50(B)', 0.0)
        final_map = finetune_results.results_dict.get('metrics/mAP50(B)', 0.0)
        improvement = final_map - warmup_map
        
        print("\n" + "=" * 60)
        print("📊 BASELINE TRAINING RESULTS")
        print("=" * 60)
        print(f"🎯 Stage 1 (Warmup) mAP@0.5: {warmup_map:.4f}")
        print(f"🎯 Stage 2 (Final) mAP@0.5: {final_map:.4f}")
        print(f"📈 Warmup→Final Improvement: {improvement:.4f}")
        print(f"📁 Experiment Directory: {pipeline.experiment_dir}")
        print("=" * 60)
        print("📋 This baseline will be used to measure attention mechanism improvements")
        print("=" * 60)
        
        print("\n✅ Baseline training completed successfully!")
        print("🎉 Your reference baseline model is ready!")
        print("📊 Use this as comparison for attention mechanisms!")
        
    except Exception as e:
        print(f"\n❌ Baseline training failed: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Check dataset paths in hripcb_data.yaml")
        print("   2. Ensure CUDA GPU is available")
        print("   3. Verify all dependencies are installed")
        print("   4. Check available memory (reduce batch_size if needed)")
        sys.exit(1)

if __name__ == "__main__":
    main()