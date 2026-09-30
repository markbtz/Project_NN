"""
Differentiable saliency losses used by the NNDL SALICON project.

Conventions
-----------
B1 / M1:
    prediction raw in [0, 1]
    target = density_map_raw
    loss = MSE

M1-L / G / M2:
    target = density_map_prob
    loss = CC-loss + KLD

KLD convention used project-wide:
    KLD(target || prediction)

The model is not required to output a probability distribution directly.
For probability-based losses, predictions are normalized internally.
"""

import torch
import torch.nn.functional as F
from src.saliency_maps import normalize_probability_map


def _check_same_shape(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )

def mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Mean Squared Error.

    Used by B1 and M1 with:
        prediction: raw model output in [0, 1]
        target: density_map_raw
    """
    _check_same_shape(
        prediction,
        target,
    )

    return F.mse_loss(
        prediction,
        target,
    )


def cc_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Correlation Coefficient loss:

        CC-loss = 1 - CC

    CC is computed independently for each batch item
    e poi mediato.

    Valore ideale:
        0  (CC = 1)

    For constant maps, correlation is mathematically undefined;
    eps keeps the computation numerically stable and the loss finite.
    """
    _check_same_shape(
        prediction,
        target,
    )

    if eps <= 0:
        raise ValueError("eps must be > 0.")

    pred = prediction.flatten(
        start_dim=1
    )

    tgt = target.flatten(
        start_dim=1
    )

    pred_centered = (
        pred
        - pred.mean(
            dim=1,
            keepdim=True,
        )
    )

    tgt_centered = (
        tgt
        - tgt.mean(
            dim=1,
            keepdim=True,
        )
    )

    numerator = (
        pred_centered
        * tgt_centered
    ).sum(
        dim=1
    )

    pred_norm = torch.sqrt(
        (
            pred_centered ** 2
        ).sum(
            dim=1
        )
    )

    tgt_norm = torch.sqrt(
        (
            tgt_centered ** 2
        ).sum(
            dim=1
        )
    )

    denominator = (
        pred_norm
        * tgt_norm
    ).clamp_min(
        eps
    )

    cc = (
        numerator
        / denominator
    ).clamp(
        min=-1.0,
        max=1.0,
    )

    return (
        1.0
        - cc
    ).mean()


def kld_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Kullback-Leibler divergence using the project convention:

        KLD(target || prediction)

    Prediction and target are internally normalized as
    probability distributions.

    Valore ideale:
        0
    """
    _check_same_shape(
        prediction,
        target,
    )

    pred_prob = normalize_probability_map(
        prediction,
        eps=eps,
    )

    target_prob = normalize_probability_map(
        target,
        eps=eps,
    )

    per_pixel = (
        target_prob
        * (
            torch.log(
                target_prob
            )
            - torch.log(
                pred_prob
            )
        )
    )

    per_sample = per_pixel.sum(
        dim=(-2, -1)
    )

    # Also average over the channel dimension, if present.
    return per_sample.mean()


def cc_kld_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cc_weight: float,
    kld_weight: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Combined loss for M1-L / G / M2:

        total = cc_weight * CC-loss
              + kld_weight * KLD(target || prediction)

    The weights are intentionally required.
    Values in configs/experiments.yaml remain null until selected in the tuning set
    """
    if cc_weight is None:
        raise ValueError(
            "cc_weight cannot be None."
        )

    if kld_weight is None:
        raise ValueError(
            "kld_weight cannot be None."
        )

    if cc_weight < 0:
        raise ValueError(
            "cc_weight must be >= 0."
        )

    if kld_weight < 0:
        raise ValueError(
            "kld_weight must be >= 0."
        )

    loss_cc = cc_loss(
        prediction,
        target,
        eps=eps,
    )

    loss_kld = kld_loss(
        prediction,
        target,
        eps=eps,
    )

    return (
        cc_weight
        * loss_cc
        + kld_weight
        * loss_kld
    )