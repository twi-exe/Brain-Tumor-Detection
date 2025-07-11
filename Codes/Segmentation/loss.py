import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, TverskyLoss
from monai.networks.utils import one_hot
from monai.utils import LossReduction
from monai.metrics.utils import get_surface_distance
from monai.metrics import compute_average_surface_distance
from typing import Tuple, Union, List, Dict, Any

class VolumeAwareLoss(nn.Module):
    def __init__(
        self,
        include_background: bool = False,
        to_onehot_y: bool = True,
        softmax: bool = True,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        class_weights=[1.0, 2.0, 1.5, 2.5, 1.5],
        baseline_volumes=[0.0, 2000.0, 8000.0, 1000.0, 3000.0],
        epsilon: float = 1e-5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.epsilon = epsilon
        self.reduction = reduction

        self.dice_ce = DiceCELoss(
            include_background=include_background,
            to_onehot_y=False,
            softmax=softmax,
            lambda_dice=0.5,
            lambda_ce=0.5,
            reduction=LossReduction.MEAN
        )

        self.tversky = TverskyLoss(
            include_background=include_background,
            to_onehot_y=False,
            softmax=softmax,
            alpha=tversky_alpha,
            beta=tversky_beta,
            reduction=LossReduction.MEAN
        )

        self.register_buffer("static_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.register_buffer("baseline_volumes", torch.tensor(baseline_volumes, dtype=torch.float32))
        self.n_classes = len(class_weights)

    def _compute_dynamic_weights(self, onehot):
        volumes = torch.sum(onehot, dim=(2, 3, 4))
        total_volumes = torch.clamp(torch.sum(volumes, dim=1, keepdim=True), min=self.epsilon)
        relative_volumes = volumes / total_volumes
        volume_multipliers = torch.sqrt(self.baseline_volumes / torch.clamp(volumes, min=self.epsilon))
        volume_multipliers = torch.clamp(volume_multipliers, max=3.0)
        effective_weights = self.static_weights * volume_multipliers
        norm_factors = torch.sum(effective_weights, dim=1, keepdim=True) / self.n_classes
        normalized_weights = effective_weights / torch.clamp(norm_factors, min=self.epsilon)
        return normalized_weights

    def _compute_surface_loss(self, pred, target, weights):
        pred_bin = (pred > 0.5).float()
        B, C = pred.shape[:2]
        losses = torch.zeros(B, device=pred.device)
        for b in range(B):
            loss = 0.0
            for c in range(C):
                if not self.include_background and c == 0:
                    continue
                pred_c = pred_bin[b, c].cpu().numpy().astype(bool)
                target_c = target[b, c].cpu().numpy().astype(bool)
                if not target_c.any() and not pred_c.any():
                    continue
                if not target_c.any() or not pred_c.any():
                    loss += weights[b, c] * 10.0
                    continue
                dist = get_surface_distance(target_c, pred_c, spacing=(1, 1, 1), distance_metric="euclidean")
                avg = compute_average_surface_distance(dist)
                mean_surface = (avg[0] + avg[1]) / 2.0
                loss += weights[b, c] * mean_surface
            losses[b] = loss / (C if self.include_background else C - 1)
        return losses

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the composite loss.

        Args:
            pred: Model predictions (B, C, H, W, D)
            target: Ground truth labels (B, H, W, D) or one-hot (B, C, H, W, D)

        Returns:
            Dictionary of all individual and final loss components.
        """
        # Determine target format
        if target.dim() == pred.dim():  # one-hot
            y_onehot = target
            target_indices = torch.argmax(target, dim=1)  # Convert for MONAI loss
        elif target.dim() == pred.dim() - 1:  # class indices
            target_indices = target.long()
            y_onehot = one_hot(target_indices, num_classes=self.n_classes)
        else:
            raise ValueError(f"[ERROR] Target shape must be [B,C,...] or [B,...], got {target.shape}")

        # Ensure prediction is float and apply softmax for surface loss
        pred_softmax = F.softmax(pred, dim=1) if self.softmax else pred
        pred = pred.float()

        # Compute dynamic class weights based on volume
        normalized_weights = self._compute_dynamic_weights(y_onehot)

        # === MONAI LOSSES ===
        # These expect: pred: [B,C,...], target: [B,...] or [B,1,...]
        dice_ce_loss = self.dice_ce(pred, target_indices)
        tversky_loss = self.tversky(pred, target_indices)

        # Apply class weights
        weighted_dice = (dice_ce_loss * normalized_weights).sum(dim=1)  # [B]
        weighted_tversky = (tversky_loss * normalized_weights).sum(dim=1)  # [B]

        # === CUSTOM SURFACE LOSS ===
        surface_loss = self._compute_surface_loss(pred_softmax, y_onehot, normalized_weights)

        # Combine losses
        total_loss = (
            0.4 * weighted_dice +
            0.4 * weighted_tversky +
            0.2 * surface_loss
        )

        # Final reduction
        if self.reduction == "mean":
            return {
                "loss": total_loss.mean(),
                "dice_ce_loss": weighted_dice.mean(),
                "tversky_loss": weighted_tversky.mean(),
                "surface_loss": surface_loss.mean(),
                "per_sample_loss": total_loss,
                "normalized_weights": normalized_weights
            }
        elif self.reduction == "sum":
            return {
                "loss": total_loss.sum(),
                "dice_ce_loss": weighted_dice.sum(),
                "tversky_loss": weighted_tversky.sum(),
                "surface_loss": surface_loss.sum(),
                "per_sample_loss": total_loss,
                "normalized_weights": normalized_weights
            }
        else:
            return {
                "loss": total_loss,
                "dice_ce_loss": weighted_dice,
                "tversky_loss": weighted_tversky,
                "surface_loss": surface_loss,
                "per_sample_loss": total_loss,
                "normalized_weights": normalized_weights
            }

def get_brats_loss():
    return VolumeAwareLoss()



