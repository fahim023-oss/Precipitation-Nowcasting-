"""
Training utilities for ConvLSTM precipitation nowcasting.

This module contains reusable training and validation loops. The command-line
entry point should live separately in:

    scripts/train_convlstm.py
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nowcasting.losses import nowcast_loss, teacher_forcing_ratio


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epoch: int,
    out_len: int,
    heavy_threshold: float,
    heavy_weight: float,
    grad_clip: float = 1.0,
    teacher_forcing_start: float = 1.0,
    teacher_forcing_end: float = 0.0,
    teacher_forcing_epochs: int = 20,
) -> float:
    """
    Train model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        ConvLSTM nowcasting model.
    loader : DataLoader
        Training dataloader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : torch.device or str
        Training device.
    epoch : int
        Current epoch.
    out_len : int
        Number of forecast frames.
    heavy_threshold : float
        Heavy-rain threshold in log1p space.
    heavy_weight : float
        Extra weight for heavy-rain pixels.
    grad_clip : float
        Maximum gradient norm.
    teacher_forcing_start : float
        Initial teacher-forcing ratio.
    teacher_forcing_end : float
        Final teacher-forcing ratio.
    teacher_forcing_epochs : int
        Number of epochs over which teacher forcing decays.

    Returns
    -------
    float
        Mean training loss.
    """

    model.train()

    tf_ratio = teacher_forcing_ratio(
        epoch=epoch,
        start_ratio=teacher_forcing_start,
        end_ratio=teacher_forcing_end,
        decay_epochs=teacher_forcing_epochs,
    )

    total_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} [train] tf={tf_ratio:.2f}",
    )

    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        prediction = model(
            x,
            y=y,
            teacher_forcing_ratio=tf_ratio,
            future_steps=out_len,
        )

        loss = nowcast_loss(
            prediction=prediction,
            target=y,
            heavy_threshold=heavy_threshold,
            heavy_weight=heavy_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    out_len: int,
    heavy_threshold: float,
    heavy_weight: float,
) -> float:
    """
    Evaluate validation loss.

    Parameters
    ----------
    model : torch.nn.Module
        ConvLSTM nowcasting model.
    loader : DataLoader
        Validation dataloader.
    device : torch.device or str
        Evaluation device.
    out_len : int
        Number of forecast frames.
    heavy_threshold : float
        Heavy-rain threshold in log1p space.
    heavy_weight : float
        Extra weight for heavy-rain pixels.

    Returns
    -------
    float
        Mean validation loss.
    """

    model.eval()

    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        prediction = model(
            x,
            future_steps=out_len,
        )

        loss = nowcast_loss(
            prediction=prediction,
            target=y,
            heavy_threshold=heavy_threshold,
            heavy_weight=heavy_weight,
        )

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch: int | None = None,
    best_val_loss: float | None = None,
    config: Dict | None = None,
) -> None:
    """
    Save model checkpoint.

    The saved checkpoint contains enough information to resume or inspect
    training, while still remaining simple to load for inference.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": config,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Dict:
    """
    Load model checkpoint.

    Supports both:
        1. New checkpoint dictionary format.
        2. Old raw `model.state_dict()` format from the original script.
    """

    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    # Backward compatibility with old:
    # torch.save(model.state_dict(), "best_convlstm.pt")
    model.load_state_dict(checkpoint)

    return {
        "model_state_dict": checkpoint,
        "epoch": None,
        "best_val_loss": None,
        "config": None,
    }


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device | str,
    epochs: int,
    patience: int,
    checkpoint_path: str,
    out_len: int,
    heavy_threshold: float,
    heavy_weight: float,
    grad_clip: float = 1.0,
    teacher_forcing_start: float = 1.0,
    teacher_forcing_end: float = 0.0,
    teacher_forcing_epochs: int = 20,
    config: Dict | None = None,
) -> Tuple[List[float], List[float], float]:
    """
    Full training loop with validation and early stopping.

    Parameters
    ----------
    model : torch.nn.Module
        ConvLSTM nowcasting model.
    train_loader : DataLoader
        Training dataloader.
    val_loader : DataLoader
        Validation dataloader.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : torch.optim.lr_scheduler.LRScheduler or None
        Learning-rate scheduler.
    device : torch.device or str
        Training device.
    epochs : int
        Maximum number of epochs.
    patience : int
        Early stopping patience.
    checkpoint_path : str
        Where to save the best checkpoint.
    out_len : int
        Number of forecast frames.
    heavy_threshold : float
        Heavy-rain threshold in log1p space.
    heavy_weight : float
        Extra heavy-rain loss weight.
    grad_clip : float
        Gradient clipping norm.
    teacher_forcing_start : float
        Starting teacher-forcing ratio.
    teacher_forcing_end : float
        Ending teacher-forcing ratio.
    teacher_forcing_epochs : int
        Teacher-forcing decay length.
    config : dict or None
        Optional config dictionary saved inside the checkpoint.

    Returns
    -------
    train_history : list[float]
        Epoch-wise training loss.
    val_history : list[float]
        Epoch-wise validation loss.
    best_val_loss : float
        Best validation loss.
    """

    best_val_loss = float("inf")
    wait = 0

    train_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            out_len=out_len,
            heavy_threshold=heavy_threshold,
            heavy_weight=heavy_weight,
            grad_clip=grad_clip,
            teacher_forcing_start=teacher_forcing_start,
            teacher_forcing_end=teacher_forcing_end,
            teacher_forcing_epochs=teacher_forcing_epochs,
        )

        val_loss = validate(
            model=model,
            loader=val_loader,
            device=device,
            out_len=out_len,
            heavy_threshold=heavy_threshold,
            heavy_weight=heavy_weight,
        )

        if scheduler is not None:
            scheduler.step()

        train_history.append(train_loss)
        val_history.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:03d} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
            )

            print(f"  ✓ Best checkpoint saved: {checkpoint_path}")
        else:
            wait += 1
            print(f"  No improvement: {wait}/{patience}")

            if wait >= patience:
                print("\nEarly stopping triggered.")
                break

    return train_history, val_history, best_val_loss