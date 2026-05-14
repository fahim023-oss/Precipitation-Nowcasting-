"""
Dataset and dataloader utilities for MRMS precipitation nowcasting.

The dataset expects each sample to be stored as a `.npz` file containing
a `precip` array with shape:

    (time, height, width)

For the default MRMS 3-hour cubes:
    time = 90
    height = 128
    width = 128

The model uses:
    60 input frames  ->  30 forecast frames
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class MRMSDataset(Dataset):
    """
    PyTorch Dataset for MRMS precipitation cubes.

    Parameters
    ----------
    folder : str
        Directory containing `.npz` files.
    in_len : int
        Number of input frames.
    out_len : int
        Number of target forecast frames.
    precip_key : str
        Name of the precipitation array inside each `.npz` file.

    Returns
    -------
    x : torch.Tensor
        Input sequence with shape `(in_len, 1, H, W)`.
    y : torch.Tensor
        Target sequence with shape `(out_len, 1, H, W)`.
    """

    def __init__(
        self,
        folder: str,
        in_len: int = 60,
        out_len: int = 30,
        precip_key: str = "precip",
    ) -> None:
        self.folder = folder
        self.in_len = in_len
        self.out_len = out_len
        self.precip_key = precip_key

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Data folder not found: {folder}")

        self.files = sorted(
            f for f in os.listdir(folder)
            if f.endswith(".npz")
        )

        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in: {folder}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = os.path.join(self.folder, self.files[idx])

        with np.load(file_path) as npz:
            if self.precip_key not in npz:
                raise KeyError(
                    f"Key '{self.precip_key}' not found in {file_path}. "
                    f"Available keys: {list(npz.keys())}"
                )
            cube = npz[self.precip_key].astype(np.float32)

        expected_len = self.in_len + self.out_len
        if cube.shape[0] < expected_len:
            raise ValueError(
                f"Cube {file_path} has only {cube.shape[0]} frames, "
                f"but {expected_len} are required."
            )

        # Preprocessing follows the original script:
        # 1. Replace NaNs with zero.
        # 2. Remove negative precipitation.
        # 3. Apply log1p transform.
        cube = np.nan_to_num(cube, nan=0.0)
        cube = np.maximum(cube, 0.0)
        cube = np.log1p(cube)

        x = torch.from_numpy(cube[: self.in_len]).unsqueeze(1)
        y = torch.from_numpy(
            cube[self.in_len : self.in_len + self.out_len]
        ).unsqueeze(1)

        return x, y

    def get_filename(self, idx: int) -> str:
        """Return the filename associated with a dataset index."""
        return self.files[idx]


def split_dataset(
    dataset: Dataset,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[Subset, Subset, Subset]:
    """
    Deterministically split dataset into train, validation, and test subsets.

    This follows the original chronological split strategy:
    first train, then validation, then test.
    """

    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")

    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")

    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.")

    n_total = len(dataset)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n_total))

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)

    return train_ds, val_ds, test_ds


def build_dataloaders(
    data_folder: str,
    in_len: int = 60,
    out_len: int = 30,
    batch_size: int = 4,
    num_workers: int = 4,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Subset]:
    """
    Build train, validation, and test dataloaders.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    test_dataset : Subset
        Returned separately because plotting scripts often need direct indexing.
    """

    full_dataset = MRMSDataset(
        folder=data_folder,
        in_len=in_len,
        out_len=out_len,
    )

    train_ds, val_ds, test_ds = split_dataset(
        full_dataset,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_kwargs,
    )

    print(
        "Dataset split -> "
        f"train: {len(train_ds)}  "
        f"val: {len(val_ds)}  "
        f"test: {len(test_ds)}"
    )

    return train_loader, val_loader, test_loader, test_ds
    