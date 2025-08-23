# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple

# Import necessary components from Ultralytics
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss, DFLoss
from ultralytics.utils.metrics import bbox_iou # Used as a placeholder for SIoU metric
from ultralytics.utils.tal import bbox2dist # Needed for DFL calculation in BboxLoss

# --- Your Custom Focal Loss Implementation ---
# This is a simplified example. Ensure your actual FocalLoss matches this signature.
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        CRITICAL FIX: Handle YOLOv8's continuous target_scores properly.
        
        Args:
            pred_scores: Raw logits (B, N, C) from model
            target_scores: Continuous alignment scores (B, N, C) from YOLOv8 assigner
        """
        # Apply sigmoid to get probabilities
        pred_probs = pred_scores.sigmoid()
        
        # YOLOv8 uses continuous targets, so we compute focal loss differently
        # For positive targets (target_scores > 0), use focal loss
        # For negative targets (target_scores = 0), use standard focal loss
        
        pos_mask = target_scores > 0
        
        # Compute BCE loss first
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores, target_scores, reduction='none'
        )
        
        # Compute focal weight: (1 - p_t)^gamma
        # For positive samples: p_t = pred_prob * target_score + (1-pred_prob) * (1-target_score)
        # For negative samples: p_t = 1 - pred_prob
        p_t = torch.where(pos_mask, 
                         pred_probs * target_scores + (1 - pred_probs) * (1 - target_scores),
                         1 - pred_probs)
        
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred_scores.device, dtype=pred_scores.dtype)
            alpha_weight = torch.where(pos_mask, self.alpha, 1 - self.alpha)
            focal_weight *= alpha_weight
        
        # Combine focal weight with BCE loss
        focal_loss = focal_weight * bce_loss
        
        # Return properly reduced loss compatible with YOLOv8
        return focal_loss.sum()


# --- Your Custom SIoU Loss Implementation ---
# This is a placeholder. You MUST replace this with your actual SIoU loss logic.
# The SIoU loss should take predicted bounding boxes and target bounding boxes
# and return a scalar loss value.
class SIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor) -> torch.Tensor:
        """
        SIoU loss calculation with proper reduction.
        Returns reduced loss value compatible with Ultralytics.
        """
        # Example: Using CIoU as a stand-in. Replace with your SIoU calculation.
        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, CIoU=True)
        loss = 1.0 - iou # Assuming SIoU is a metric, convert to loss
        return loss.mean() # Return properly reduced loss


# --- Custom Bbox Loss that uses SIoU ---
# This class inherits from Ultralytics' BboxLoss to retain DFL calculation,
# but overrides the IoU part with SIoU.
class CustomBboxLoss(BboxLoss):
    def __init__(self, reg_max: int = 16):
        super().__init__(reg_max)
        self.siou_loss_func = SIoULoss() # Instantiate your SIoU loss here

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SIoU and DFL losses for bounding boxes.
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # FIXED: Apply SIoU loss calculation properly
        # Using CIoU as placeholder - replace with actual SIoU implementation
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss (inherited from BboxLoss and calculated by the parent's forward if called,
        # but we're re-implementing the forward, so we need to include it explicitly)
        loss_dfl = torch.tensor(0.0).to(pred_dist.device) # Initialize for safety
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        
        return loss_iou, loss_dfl


# --- Main Custom Loss Class for YOLOv8 Detection ---
# This class inherits from v8DetectionLoss to leverage its target assignment and DFL logic.
class CustomLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk: int = 10):
        super().__init__(model, tal_topk)
        
        # Override the default BCEWithLogitsLoss with custom FocalLoss
        self.bce = FocalLoss(gamma=1.5, alpha=0.25) # Use custom FocalLoss with proper reduction

        # Override the default BboxLoss with your CustomBboxLoss (which uses SIoU)
        self.bbox_loss = CustomBboxLoss(self.reg_max).to(self.device)

    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        This method calls the parent v8DetectionLoss's __call__ method,
        which will now use our overridden self.bce (FocalLoss) and self.bbox_loss (CustomBboxLoss with SIoU).
        """
        return super().__call__(preds, batch)

class BalancedLossWeights:
    def __init__(self, box_weight=1.0, cls_weight=1.0, dfl_weight=1.0):
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight

    def get_weights(self, epoch, total_epochs):
        """
        Returns the current loss weights.
        You can implement dynamic weighting logic here based on epoch or other factors.
        For now, it returns fixed weights.
        """
        return {
            'box': self.box_weight,
            'cls': self.cls_weight,
            'dfl': self.dfl_weight
        }
