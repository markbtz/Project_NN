"""
Utility condivise per runtime e riproducibilita'.

Questo modulo centralizza:
- selezione automatica del device;
- inizializzazione dei seed globali.
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
    Seed globale per riproducibilita' (vedi configs/data.yaml -> seed).

    IMPORTANTE: SaliconDataset usa random.random() (modulo Python standard,
    non torch) per decidere l'horizontal flip — settare solo
    torch.manual_seed() NON basta, l'augmentation resterebbe non
    deterministica. Chiamare questa funzione una sola volta, a inizio script,
    prima di costruire dataset/dataloader/modello.
    """

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)