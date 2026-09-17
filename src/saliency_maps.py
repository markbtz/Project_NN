"""
Utility condivise per la normalizzazione delle saliency map.
"""

import torch


def normalize_probability_map(
    saliency_map: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Normalizza gli ultimi due assi spaziali affinché
    ogni mappa abbia somma uguale a 1.
    """

    saliency_map = torch.clamp(
        saliency_map,
        min=0.0,
    )

    numerator = saliency_map + eps

    denominator = numerator.sum(
        dim=(-2, -1),
        keepdim=True,
    )

    return numerator / denominator