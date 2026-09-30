"""
Shared utilities for saliency-map normalization.
"""

import torch


def normalize_probability_map(
    saliency_map: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Normalize the last two spatial dimensions so that
    each map sums to 1.
    """
    if eps <= 0:
        raise ValueError(
            "eps must be > 0."
        )
    
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