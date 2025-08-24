#!/usr/bin/env python3
"""
Loss Function Comparison Tool
============================

Compare training behavior with different loss functions to verify they're working.
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils.loss import v8DetectionLoss, BboxLoss

def compare_loss_functions():
    """Compare different loss functions with identical inputs."""
    print("🔍 COMPARING LOSS FUNCTION OUTPUTS")
    print("=" * 50)
    
    # Create dummy data (same for all loss functions)
    batch_size = 2
    num_anchors = 100
    num_classes = 6
    
    # Dummy predictions
    pred_scores = torch.randn(batch_size, num_anchors, num_classes).sigmoid()
    pred_boxes = torch.randn(batch_size, num_anchors, 4)
    pred_distri = torch.randn(batch_size, num_anchors, 64)  # reg_max * 4
    
    # Dummy targets
    target_scores = torch.zeros_like(pred_scores)
    target_scores[:, :10, 0] = 1.0  # First 10 anchors have class 0
    target_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3]] * num_anchors).unsqueeze(0).repeat(batch_size, 1, 1)
    fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool)
    fg_mask[:, :10] = True  # First 10 are foreground
    
    anchor_points = torch.randn(num_anchors, 2)
    target_scores_sum = target_scores.sum()
    
    print("🧪 Testing different IoU loss functions:")
    print("-" * 40)
    
    # Test different IoU loss functions
    iou_types = ['ciou', 'siou', 'eiou', 'giou']
    results = {}
    
    for iou_type in iou_types:
        try:
            bbox_loss = BboxLoss(reg_max=16, iou_type=iou_type)
            
            # Calculate loss
            loss_iou, loss_dfl = bbox_loss(
                pred_distri, pred_boxes, anchor_points, target_boxes, target_scores, target_scores_sum, fg_mask
            )
            
            results[iou_type] = {
                'iou_loss': loss_iou.item(),
                'dfl_loss': loss_dfl.item()
            }
            
            print(f"✅ {iou_type.upper():4} | IoU Loss: {loss_iou.item():.4f} | DFL Loss: {loss_dfl.item():.4f}")
            
        except Exception as e:
            print(f"❌ {iou_type.upper():4} | Error: {e}")
    
    print("\n🧪 Testing different classification loss functions:")
    print("-" * 40)
    
    # Test classification losses (dummy model for testing)
    from ultralytics import YOLO
    try:
        model = YOLO('yolov8n.pt')
        
        cls_types = ['bce', 'focal', 'varifocal']
        
        for cls_type in cls_types:
            try:
                detection_loss = v8DetectionLoss(model.model, iou_type='ciou', cls_type=cls_type)
                
                # Create dummy batch for classification loss testing
                dummy_batch = {
                    'batch_idx': torch.zeros(10, 1),
                    'cls': torch.zeros(10, 1),
                    'bboxes': torch.tensor([[0.1, 0.1, 0.2, 0.2]] * 10)
                }
                
                # This is complex to test in isolation, so just verify initialization
                print(f"✅ {cls_type.upper():10} | Classification loss initialized successfully")
                print(f"    └─ Type: {detection_loss.cls_type}")
                
                if hasattr(detection_loss, 'focal_loss'):
                    print(f"    └─ Has focal_loss: {type(detection_loss.focal_loss).__name__}")
                if hasattr(detection_loss, 'varifocal_loss'):
                    print(f"    └─ Has varifocal_loss: {type(detection_loss.varifocal_loss).__name__}")
                
            except Exception as e:
                print(f"❌ {cls_type.upper():10} | Error: {e}")
                
    except Exception as e:
        print(f"❌ Could not load model for classification testing: {e}")
    
    print("\n🎯 INTERPRETATION:")
    print("-" * 40)
    print("✅ If different IoU types show DIFFERENT loss values → IoU functions are working")
    print("✅ If classification types initialize successfully → Classification losses are available") 
    print("❌ If all IoU types show SAME loss values → Something is wrong")
    print("❌ If any errors occur → Integration issues exist")
    
    # Compare results
    if len(results) > 1:
        loss_values = [results[iou]['iou_loss'] for iou in results]
        if len(set(f"{v:.6f}" for v in loss_values)) > 1:
            print("\n🎉 SUCCESS: Different IoU loss functions produce different values!")
            print("    This confirms that IoU loss selection is working correctly.")
        else:
            print("\n⚠️  WARNING: All IoU loss functions produce identical values.")
            print("    This might indicate the loss selection isn't working properly.")

if __name__ == "__main__":
    compare_loss_functions()