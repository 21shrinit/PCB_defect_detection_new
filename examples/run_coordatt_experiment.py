#!/usr/bin/env python3
"""
Example: CoordAtt (Coordinate Attention) Training
================================================

This example demonstrates how to train YOLOv8 with Coordinate Attention
using the production-ready two-stage training framework.

CoordAtt Features:
- Mobile-friendly positional attention
- Factorizes attention into height and width dimensions
- Captures long-range dependencies with spatial information
- 1-2% mAP improvement with mobile optimization
"""

import sys
from pathlib import Path
from train_attention_benchmark import AttentionTrainingPipeline

def main():
    """Run Coordinate Attention training experiment."""
    
    print("🚀 YOLOv8-CoordAtt (Coordinate Attention) Training")
    print("=" * 60)
    print("📋 Mechanism: CoordAtt (Coordinate Attention)")
    print("🎯 Focus: Mobile-friendly positional attention")
    print("📊 Expected: +1-2% mAP with mobile optimization")
    print("🚀 Use Case: Mobile and edge deployment")
    print("=" * 60)
    
    # Configuration for CoordAtt
    config_path = "configs/config_coordatt.yaml"
    
    try:
        # Verify config exists
        if not Path(config_path).exists():
            print(f"❌ Configuration not found: {config_path}")
            print("💡 Make sure you have created the CoordAtt configuration file")
            sys.exit(1)
        
        # Initialize training pipeline
        print("🔧 Initializing CoordAtt training pipeline...")
        pipeline = AttentionTrainingPipeline(config_path)
        
        # Run two-stage training
        print("🎯 Starting two-stage CoordAtt training...")
        print("   Stage 1: Warmup with frozen backbone (25 epochs)")
        print("   Stage 2: Fine-tuning with reduced LR (125 epochs)")
        print("   📱 Optimized for mobile deployment!")
        
        warmup_results, finetune_results = pipeline.run_complete_pipeline()
        
        # Export the final model
        print("📦 Exporting CoordAtt-enhanced model...")
        pipeline.export_final_model(['onnx', 'torchscript'])
        
        # Display results
        warmup_map = warmup_results.results_dict.get('metrics/mAP50(B)', 0.0)
        final_map = finetune_results.results_dict.get('metrics/mAP50(B)', 0.0)
        improvement = final_map - warmup_map
        
        print("\n" + "=" * 60)
        print("📊 COORDINATE ATTENTION TRAINING RESULTS")
        print("=" * 60)
        print(f"🎯 Stage 1 (Warmup) mAP@0.5: {warmup_map:.4f}")
        print(f"🎯 Stage 2 (Final) mAP@0.5: {final_map:.4f}")
        print(f"📈 Warmup→Final Improvement: {improvement:.4f}")
        print(f"📁 Experiment Directory: {pipeline.experiment_dir}")
        print("=" * 60)
        
        print("\n✅ CoordAtt training completed successfully!")
        print("🎉 Your mobile-optimized attention model is ready!")
        print("📱 Perfect for edge deployment and mobile applications!")
        
    except Exception as e:
        print(f"\n❌ CoordAtt training failed: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Check dataset paths in hripcb_data.yaml")
        print("   2. Ensure CUDA GPU is available")
        print("   3. Verify all dependencies are installed")
        print("   4. Check available memory (reduce batch_size if needed)")
        print("   5. Consider adjusting reduction ratio for your input size")
        sys.exit(1)

if __name__ == "__main__":
    main()