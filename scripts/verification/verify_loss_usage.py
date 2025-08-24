#!/usr/bin/env python3
"""
Real-time Loss Function Usage Verification
==========================================

This script verifies that the correct loss functions are actually being used
during training by inspecting the model's loss criterion.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from scripts.experiments.run_single_experiment_FIXED import FixedExperimentRunner

def verify_loss_usage(config_path: str):
    """Verify which loss functions are actually being used."""
    print(f"🔍 VERIFYING LOSS USAGE FOR: {config_path}")
    print("=" * 60)
    
    try:
        # Initialize experiment runner
        runner = FixedExperimentRunner(config_path)
        
        # Create model (this applies our loss configuration)
        model = runner.create_model()
        
        # Apply loss configuration 
        dummy_train_args = {}
        runner.apply_loss_configuration(model, dummy_train_args, runner.config['training'])
        
        print("\n🔍 INSPECTING MODEL LOSS CRITERION:")
        print("-" * 40)
        
        # Get the loss criterion (this triggers init_criterion)
        criterion = model.model.init_criterion()
        
        print(f"✅ Loss Criterion Type: {type(criterion).__name__}")
        
        # Check IoU loss type
        if hasattr(criterion, 'bbox_loss'):
            bbox_loss = criterion.bbox_loss
            print(f"✅ BBox Loss Type: {type(bbox_loss).__name__}")
            if hasattr(bbox_loss, 'iou_type'):
                print(f"✅ IoU Loss Function: {bbox_loss.iou_type.upper()}")
            else:
                print("❌ IoU type not found on bbox_loss")
        else:
            print("❌ bbox_loss not found on criterion")
        
        # Check classification loss type
        if hasattr(criterion, 'cls_type'):
            print(f"✅ Classification Loss: {criterion.cls_type.upper()}")
        else:
            print("❌ cls_type not found on criterion")
        
        # Check if focal/varifocal loss objects exist
        if hasattr(criterion, 'focal_loss'):
            print(f"✅ Focal Loss Object: {type(criterion.focal_loss).__name__}")
        if hasattr(criterion, 'varifocal_loss'):
            print(f"✅ VariFocal Loss Object: {type(criterion.varifocal_loss).__name__}")
        
        # Check model args
        print(f"\n🔍 MODEL ARGS VERIFICATION:")
        print("-" * 40)
        if hasattr(model.model, 'args'):
            args = model.model.args
            print(f"✅ Model args type: {type(args).__name__}")
            if hasattr(args, 'iou_type'):
                print(f"✅ model.model.args.iou_type: {args.iou_type}")
            if hasattr(args, 'cls_type'):
                print(f"✅ model.model.args.cls_type: {args.cls_type}")
        else:
            print("❌ model.model.args not found")
            
        # Expected vs Actual comparison
        print(f"\n🎯 EXPECTED VS ACTUAL:")
        print("-" * 40)
        
        loss_config = runner.config['training'].get('loss', {})
        expected_loss_type = loss_config.get('type', 'standard')
        
        print(f"Expected Loss Type: {expected_loss_type}")
        
        if expected_loss_type == 'siou':
            print("Expected: SIoU IoU loss + BCE classification")
        elif expected_loss_type == 'eiou':
            print("Expected: EIoU IoU loss + BCE classification")
        elif expected_loss_type == 'focal':
            print("Expected: CIoU IoU loss + Focal classification")
        elif expected_loss_type == 'varifocal':
            print("Expected: CIoU IoU loss + VariFocal classification")
        elif expected_loss_type.startswith('focal_'):
            iou_part = expected_loss_type.split('_')[1]
            print(f"Expected: {iou_part.upper()} IoU loss + Focal classification")
        elif expected_loss_type.startswith('verifocal_'):
            iou_part = expected_loss_type.split('_')[1]
            print(f"Expected: {iou_part.upper()} IoU loss + VariFocal classification")
        else:
            print("Expected: CIoU IoU loss + BCE classification (default)")
            
        # Verification result
        if hasattr(criterion, 'bbox_loss') and hasattr(criterion.bbox_loss, 'iou_type'):
            actual_iou = criterion.bbox_loss.iou_type
            actual_cls = getattr(criterion, 'cls_type', 'bce')
            
            print(f"Actual: {actual_iou.upper()} IoU loss + {actual_cls.upper()} classification")
            
            # Check if they match expectations
            if expected_loss_type in ['siou', 'eiou', 'ciou', 'giou']:
                expected_match = (actual_iou == expected_loss_type and actual_cls == 'bce')
            elif expected_loss_type == 'focal':
                expected_match = (actual_iou == 'ciou' and actual_cls == 'focal')
            elif expected_loss_type == 'varifocal':
                expected_match = (actual_iou == 'ciou' and actual_cls == 'varifocal')
            elif expected_loss_type.startswith('focal_'):
                expected_iou = expected_loss_type.split('_')[1]
                expected_match = (actual_iou == expected_iou and actual_cls == 'focal')
            elif expected_loss_type.startswith('verifocal_'):
                expected_iou = expected_loss_type.split('_')[1]
                expected_match = (actual_iou == expected_iou and actual_cls == 'varifocal')
            else:
                expected_match = (actual_iou == 'ciou' and actual_cls == 'bce')
            
            if expected_match:
                print("🎉 ✅ VERIFICATION PASSED: Expected and actual loss functions match!")
            else:
                print("❌ ⚠️  VERIFICATION FAILED: Expected and actual loss functions don't match!")
        else:
            print("❌ Cannot verify - loss function information not accessible")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Test multiple configurations."""
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        verify_loss_usage(config_path)
    else:
        print("Usage: python verify_loss_usage.py <config_path>")
        print("\nExample configs to test:")
        test_configs = [
            "experiments/configs/roboflow_pcb/RB00_YOLOv8n_Baseline.yaml",
            "experiments/configs/roboflow_pcb/RB01_YOLOv8n_SIoU_ECA.yaml", 
            "experiments/configs/roboflow_pcb/RB04_YOLOv8n_EIoU_ECA.yaml",
            "experiments/configs/roboflow_pcb/RB07_YOLOv8n_Focal.yaml",
            "experiments/configs/roboflow_pcb/RB08_YOLOv8n_VeriFocal.yaml"
        ]
        
        for config in test_configs:
            if Path(config).exists():
                print(f"python verify_loss_usage.py {config}")

if __name__ == "__main__":
    main()