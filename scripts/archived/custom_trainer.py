"""
Custom trainer for PCB defect detection with Focal-SIoU loss.
This trainer overrides the default loss calculation to use enhanced loss functions.
Based on research findings for optimal integration with Ultralytics.
"""

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss as DetectionLoss
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tal import make_anchors
from custom_modules.loss import FocalLoss, SIoULoss, BalancedLossWeights, CustomBboxLoss
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomDetectionLoss(DetectionLoss):
    """
    FIXED: Custom loss function that properly integrates with YOLOv8's target assignment.
    Handles continuous target_scores from TaskAlignedAssigner correctly.
    """
    def __init__(self, model):
        super().__init__(model)
        
        # CRITICAL FIX: Use custom FocalLoss that handles YOLOv8 target format
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
        
        # Override bbox loss with SIoU
        self.bbox_loss = CustomBboxLoss(self.reg_max).to(self.device)

    def __call__(self, preds, batch):
        """
        FIXED: Loss calculation that properly handles YOLOv8 target assignment.
        This is the complete reimplementation to ensure correct integration.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Standard YOLOv8 target preprocessing
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Bbox decoding
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # YOLOv8 target assignment
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # CRITICAL FIX: Apply Focal Loss correctly to YOLOv8's continuous targets
        loss[1] = self.focal_loss(pred_scores, target_scores.to(dtype)) / target_scores_sum

        # Bbox loss with SIoU
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        # Apply hyperparameter weights
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain  
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()


class MyCustomModel(DetectionModel):
    """
    A custom model class that uses our custom loss function.
    """
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
        """Initialize with proper args attribute."""
        super().__init__(cfg, ch, nc, verbose)
        # Ensure args attribute exists for compatibility
        if not hasattr(self, 'args'):
            from ultralytics.utils import DEFAULT_CFG
            self.args = DEFAULT_CFG
    
    def init_criterion(self):
        """Overrides the default criterion to use our CustomDetectionLoss."""
        return CustomDetectionLoss(self)


class MyCustomTrainer(DetectionTrainer):
    """
    A custom trainer class that ensures our MyCustomModel is used.
    Based on research findings for optimal integration approach.
    """
    def get_model(self, cfg, weights):
        """Returns an instance of MyCustomModel."""
        model = MyCustomModel(cfg, ch=3, nc=self.data['nc'])
        if weights:
            model.load(weights)
        return model
    
    def _do_train(self, world_size=1):
        """
        Override training loop to update loss weights dynamically.
        """
        # Update total epochs in loss function
        if hasattr(self.model, 'loss') and hasattr(self.model.loss, 'update_epoch'):
            self.model.loss.update_epoch(0, self.epochs)
        
        # Call parent training method
        return super()._do_train(world_size)
    
    def _do_epoch(self, epoch):
        """
        Override epoch method to update loss weights.
        """
        # Update current epoch in loss function
        if hasattr(self.model, 'loss') and hasattr(self.model.loss, 'update_epoch'):
            self.model.loss.update_epoch(epoch, self.epochs)
        
        # Call parent epoch method
        return super()._do_epoch(epoch)
