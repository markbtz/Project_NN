"""
Shared runtime and reproducibility utilities.

This module centralizes:
- automatic device selection;
- global seed initialization.
"""

import random

import torch


def get_device() -> torch.device:
    """Seleziona automaticamente cuda (Colab) > mps (M1 Max) > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed: int):
    """
    Global seed for reproducibility (see configs/data.yaml -> seed).

    IMPORTANT: SaliconDataset usa random.random() (modulo Python standard,
    rather than torch) to decide the horizontal flip; setting only
    torch.manual_seed() is NOT enough; augmentation would remain non-
    deterministic. Call this function once at script startup,
    before building the dataset/dataloader/model.
    """

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)