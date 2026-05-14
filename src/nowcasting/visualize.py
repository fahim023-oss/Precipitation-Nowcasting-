"""
Visualization utilities for ConvLSTM precipitation nowcasting.

This module contains:

    - NWS-like precipitation colormap
    - training/validation loss curve plotting
    - qualitative prediction plotting

The precipitation arrays are converted from log1p space back to mm/hr
before plotting.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import matplotlib

# Safe default for HPC / non-interactive environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap


def make_precip_colormap() -> Tuple[ListedColormap, BoundaryNorm, List[float]]:
    """
    Create discrete precipitation colormap.

    Returns
    -------
    cmap : ListedColormap
        Discrete precipitation colormap.
    norm : BoundaryNorm
        Boundary normalization.
    bounds : list[float]
        Precipitation bin boundaries in mm/hr.
    """

    colors = [
        "white",
        "#c0e8c0",
        "#00a600",
        "#f0f000",
        "#e07000",
        "#e00000",
        "#c000c0",
        "#7030c0",
    ]

    bounds = [0, 0.1, 1, 5, 10, 20, 40, 70, 150]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm, bounds


def plot_training_curve(
    train_history: Iterable[float],
    val_history: Iterable[float],
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    Plot training and validation loss curves.

    Parameters
    ----------
    train_history : iterable of float
        Training loss values.
    val_history : iterable of float
        Validation loss values.
    output_path : str
        Figure save path.
    dpi : int
        Output figure resolution.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(list(train_history), label="Train")
    plt.plot(list(val_history), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ConvLSTM Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    print(f"Training curve saved -> {output_path}")


@torch.no_grad()
def plot_prediction_example(
    model: torch.nn.Module,
    dataset,
    sample_index: int,
    device: torch.device | str,
    output_path: str,
    out_len: int = 30,
    lead_times: Iterable[int] = (0, 9, 19, 29),
    dpi: int = 150,
) -> None:
    """
    Plot qualitative prediction example.

    The figure layout follows the original training script:

        column 1 : last observed frame
        column 2 : ground truth
        column 3 : ConvLSTM prediction

    Parameters
    ----------
    model : torch.nn.Module
        Trained nowcasting model.
    dataset : Dataset or Subset
        Dataset from which one sample will be selected.
    sample_index : int
        Index of the sample to plot.
    device : torch.device or str
        Inference device.
    output_path : str
        Figure save path.
    out_len : int
        Number of forecast frames.
    lead_times : iterable of int
        Forecast frame indices to plot. Example: 0 means t+1.
    dpi : int
        Output figure resolution.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model.eval()

    cmap, norm, bounds = make_precip_colormap()

    x, y = dataset[sample_index]

    x_batch = x.unsqueeze(0).to(device)

    prediction = model(
        x_batch,
        future_steps=out_len,
    )

    x_mm = torch.expm1(x).clamp(min=0.0).squeeze().cpu().numpy()
    y_mm = torch.expm1(y).clamp(min=0.0).squeeze().cpu().numpy()
    pred_mm = (
        torch.expm1(prediction)
        .clamp(min=0.0)
        .squeeze()
        .cpu()
        .numpy()
    )

    lead_times = list(lead_times)

    fig, axes = plt.subplots(
        len(lead_times),
        3,
        figsize=(12, 3.7 * len(lead_times)),
    )

    if len(lead_times) == 1:
        axes = axes.reshape(1, 3)

    image_handle = None

    for row, lead_idx in enumerate(lead_times):
        lead_label = f"t+{lead_idx + 1} ({(lead_idx + 1) * 2} min)"

        if lead_idx >= y_mm.shape[0]:
            raise IndexError(
                f"Requested lead index {lead_idx}, "
                f"but target sequence has only {y_mm.shape[0]} frames."
            )

        if row == 0:
            image_handle = axes[row, 0].imshow(
                x_mm[-1],
                origin="lower",
                cmap=cmap,
                norm=norm,
            )
            axes[row, 0].set_title(
                "Last Observed Frame",
                fontsize=11,
                fontweight="bold",
            )
        else:
            axes[row, 0].axis("off")

        axes[row, 1].imshow(
            y_mm[lead_idx],
            origin="lower",
            cmap=cmap,
            norm=norm,
        )
        axes[row, 1].set_title(
            f"Ground Truth {lead_label}",
            fontsize=10,
        )

        image_handle = axes[row, 2].imshow(
            pred_mm[lead_idx],
            origin="lower",
            cmap=cmap,
            norm=norm,
        )
        axes[row, 2].set_title(
            f"Prediction {lead_label}",
            fontsize=10,
        )

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    colorbar = fig.colorbar(
        image_handle,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        pad=0.03,
        fraction=0.03,
        aspect=60,
        ticks=bounds,
        extend="max",
    )

    colorbar.set_label("Precipitation (mm/hr)", fontsize=12)

    plt.suptitle(
        "ConvLSTM Precipitation Nowcasting",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Prediction figure saved -> {output_path}")


@torch.no_grad()
def plot_selected_prediction_event(
    model: torch.nn.Module,
    dataset,
    sample_index: int,
    device: torch.device | str,
    output_path: str,
    out_len: int = 30,
    lead_times: Iterable[int] = (0, 9),
    row_labels: Iterable[str] | None = None,
    dpi: int = 300,
) -> None:
    """
    Plot compact selected-event figure.

    This is the cleaned version of your separate figure script. It creates
    a 2-row by 3-column plot by default:

        row 1 : last observed frame | ground truth t+1  | prediction t+1
        row 2 : blank               | ground truth t+10 | prediction t+10

    Parameters
    ----------
    model : torch.nn.Module
        Trained nowcasting model.
    dataset : Dataset
        Dataset containing the selected event.
    sample_index : int
        Dataset index of the selected event.
    device : torch.device or str
        Inference device.
    output_path : str
        Figure save path.
    out_len : int
        Number of forecast frames.
    lead_times : iterable of int
        Forecast frame indices to plot.
    row_labels : iterable of str or None
        Optional row labels. If None, labels are generated automatically.
    dpi : int
        Output figure resolution.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model.eval()

    lead_times = list(lead_times)

    if row_labels is None:
        row_labels = [
            f"t+{lead + 1} ({(lead + 1) * 2} min)"
            for lead in lead_times
        ]
    else:
        row_labels = list(row_labels)

    if len(row_labels) != len(lead_times):
        raise ValueError("row_labels and lead_times must have the same length.")

    cmap, norm, bounds = make_precip_colormap()

    x, y = dataset[sample_index]
    x_batch = x.unsqueeze(0).to(device)

    prediction = model(
        x_batch,
        future_steps=out_len,
    )

    x_mm = torch.expm1(x).clamp(min=0.0).squeeze().cpu().numpy()
    y_mm = torch.expm1(y).clamp(min=0.0).squeeze().cpu().numpy()
    pred_mm = (
        torch.expm1(prediction)
        .clamp(min=0.0)
        .squeeze()
        .cpu()
        .numpy()
    )

    fig, axes = plt.subplots(
        len(lead_times),
        3,
        figsize=(12, 4 * len(lead_times)),
        gridspec_kw={"hspace": 0.38, "wspace": 0.06},
    )

    if len(lead_times) == 1:
        axes = axes.reshape(1, 3)

    image_handle = None

    for row, (lead_idx, row_label) in enumerate(zip(lead_times, row_labels)):
        if lead_idx >= y_mm.shape[0]:
            raise IndexError(
                f"Requested lead index {lead_idx}, "
                f"but target sequence has only {y_mm.shape[0]} frames."
            )

        if row == 0:
            image_handle = axes[row, 0].imshow(
                x_mm[-1],
                origin="lower",
                cmap=cmap,
                norm=norm,
            )
            axes[row, 0].set_title(
                "Last Observed\nFrame",
                fontsize=15,
                fontweight="bold",
            )
        else:
            axes[row, 0].axis("off")

        axes[row, 1].imshow(
            y_mm[lead_idx],
            origin="lower",
            cmap=cmap,
            norm=norm,
        )
        axes[row, 1].set_title(
            f"Ground Truth\n{row_label}",
            fontsize=15,
            fontweight="bold",
        )

        image_handle = axes[row, 2].imshow(
            pred_mm[lead_idx],
            origin="lower",
            cmap=cmap,
            norm=norm,
        )
        axes[row, 2].set_title(
            f"ConvLSTM Prediction\n{row_label}",
            fontsize=15,
            fontweight="bold",
        )

    for axis in axes.ravel():
        if axis.get_visible() and axis.axison:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.9)
                spine.set_edgecolor("black")

    fig.canvas.draw()

    # Place colorbar below the ground-truth and prediction panels.
    pos_left = axes[-1, 1].get_position()
    pos_right = axes[-1, 2].get_position()

    colorbar_axis = fig.add_axes(
        [
            pos_left.x0,
            pos_left.y0 - 0.10,
            pos_right.x1 - pos_left.x0,
            0.022,
        ]
    )

    colorbar = fig.colorbar(
        image_handle,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=bounds,
        extend="max",
    )

    colorbar.ax.set_xticklabels([str(bound) for bound in bounds])
    colorbar.set_label(
        "Precipitation (mm hr$^{-1}$)",
        fontsize=14,
        labelpad=7,
    )
    colorbar.outline.set_linewidth(0.8)

    fig.suptitle(
        "ConvLSTM Precipitation Nowcasting",
        fontsize=19,
        fontweight="bold",
        y=1.02,
    )

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Selected-event figure saved -> {output_path}")