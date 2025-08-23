#!/usr/bin/env python3
"""
Example: ECA (Efficient Channel Attention) Training
==================================================

This example demonstrates how to train YOLOv8 with ECA attention
using the production-ready two-stage training framework.

ECA Features:
- Lightweight channel attention with minimal overhead
- Adaptive kernel size based on channel dimensions
- Perfect for resource-constrained environments
- 1-3% mAP improvement with minimal computational cost
"""

import sys
from pathlib import Path
from train_attention_benchmark import AttentionTrainingPipeline

def main():
    """Run ECA attention training experiment."""
    
    print("🚀 YOLOv8-ECA (Efficient Channel Attention) Training")
    print("=" * 60)
    print("📋 Mechanism: ECA (Efficient Channel Attention)")
    print("🎯 Focus: Lightweight channel attention")
    print("📊 Expected: +1-3% mAP with minimal overhead")
    print("🚀 Use Case: Resource-constrained applications")
    print("=" * 60)
    
    # Configuration for ECA
    config_path = "configs/config_eca.yaml"
    
    try:
        # Verify config exists
        if not Path(config_path).exists():
            print(f"❌ Configuration not found: {config_path}")
            print("💡 Make sure you have created the ECA configuration file")
            sys.exit(1)
        
        # Initialize training pipeline
        print("🔧 Initializing ECA training pipeline...")
        pipeline = AttentionTrainingPipeline(config_path)
        
        # Run two-stage training
        print("🎯 Starting two-stage ECA training...")
        print("   Stage 1: Warmup with frozen backbone (25 epochs)")
        print("   Stage 2: Fine-tuning with reduced LR (125 epochs)")
        
        warmup_results, finetune_results = pipeline.run_complete_pipeline()
        
        # Export the final model
        print("📦 Exporting ECA-enhanced model...")
        pipeline.export_final_model(['onnx', 'torchscript'])
        
        # Display results
        warmup_map = warmup_results.results_dict.get('metrics/mAP50(B)', 0.0)
        final_map = finetune_results.results_dict.get('metrics/mAP50(B)', 0.0)
        improvement = final_map - warmup_map
        
        print("\n" + "=" * 60)
        print("📊 ECA TRAINING RESULTS")
        print("=" * 60)
        print(f"🎯 Stage 1 (Warmup) mAP@0.5: {warmup_map:.4f}")
        print(f"🎯 Stage 2 (Final) mAP@0.5: {final_map:.4f}")
        print(f"📈 Warmup→Final Improvement: {improvement:.4f}")
        print(f"📁 Experiment Directory: {pipeline.experiment_dir}")
        print("=" * 60)
        
        print("\n✅ ECA training completed successfully!")
        print("🎉 Your ECA-enhanced YOLOv8 model is ready!")
        
    except Exception as e:
        print(f"\n❌ ECA training failed: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Check dataset paths in hripcb_data.yaml")
        print("   2. Ensure CUDA GPU is available")
        print("   3. Verify all dependencies are installed")
        print("   4. Check available memory (reduce batch_size if needed)")
        sys.exit(1)

if __name__ == "__main__":
    main()