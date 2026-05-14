"""
Loss functions and scheduled-sampling utilities for precipitation nowcasting.

The main training loss is computed in log1p precipitation space and combines:

    1. Mean absolute error
    2. Mean squared error
    3. Extra L1 penalty on heavy-rain pixels

This follows the original ConvLSTM training script.
"""

from __future__ import annotations

import torch


def nowcast_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    heavy_threshold: float,
    heavy_weight: float,
) -> torch.Tensor:
    """
    Compute weighted nowcasting loss.

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted precipitation sequence in log1p space.
        Shape: `(B, T, 1, H, W)`.
    target : torch.Tensor
        Target precipitation sequence in log1p space.
        Shape: `(B, T, 1, H, W)`.
    heavy_threshold : float
        Heavy-rain threshold in log1p space.
        Example: `math.log1p(5.0)` for 5 mm/hr.
    heavy_weight : float
        Extra weight applied to heavy-rain pixels.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """

    absolute_error = torch.abs(prediction - target)
    squared_error = (prediction - target) ** 2

    base_loss = 0.5 * absolute_error.mean() + 0.5 * squared_error.mean()

    heavy_mask = (target > heavy_threshold).float()
    heavy_count = heavy_mask.sum().clamp(min=1.0)

    heavy_loss = (
        heavy_weight * absolute_error * heavy_mask
    ).sum() / heavy_count

    return base_loss + heavy_loss


def teacher_forcing_ratio(
    epoch: int,
    start_ratio: float = 1.0,
    end_ratio: float = 0.0,
    decay_epochs: int = 20,
) -> float:
    """
    Linearly decay teacher-forcing ratio across training epochs.

    Parameters
    ----------
    epoch : int
        Current epoch number.
    start_ratio : float
        Initial teacher-forcing ratio.
    end_ratio : float
        Final teacher-forcing ratio.
    decay_epochs : int
        Number of epochs over which to decay.

    Returns
    -------
    float
        Teacher-forcing ratio for the current epoch.
    """

    if decay_epochs <= 0:
        return end_ratio

    if epoch >= decay_epochs:
        return end_ratio

    progress = epoch / decay_epochs
    ratio = start_ratio + progress * (end_ratio - start_ratio)

    return float(ratio)