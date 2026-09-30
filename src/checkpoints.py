"""
Shared utilities for saving and loading checkpoints.
"""

import os

import torch


def initial_best_score(selection_mode):
    """
    Initial value used for best-checkpoint selection.
    """

    if selection_mode == "max":
        return float("-inf")

    if selection_mode == "min":
        return float("inf")

    raise ValueError(
        "selection_mode must be 'max' or 'min'."
    )


def is_better(
    current_score,
    best_score,
    selection_mode,
):
    """
    Compare the current score with the best score according to
    the direction defined in experiments.yaml.
    """

    if selection_mode == "max":
        return current_score > best_score

    if selection_mode == "min":
        return current_score < best_score

    raise ValueError(
        "selection_mode must be 'max' or 'min'."
    )


def save_training_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_score,
    selection_metric,
    selection_mode,
    seed,
    experiment,
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),
            "best_score": best_score,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "seed": seed,
            "experiment": experiment,
        },
        path,
    )


def load_training_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
    selection_metric="cc",
    selection_mode="max",
):
    """
    Return (start_epoch, best_score).

    If no checkpoint exists, start from scratch using the
    initial value consistent with selection_mode.

    Legacy checkpoints based on best_loss are not reused:
    best-model selection now uses the tuning set and therefore
    follows a different protocol.

    Important on Colab: a session may
    disconnect at any time; this prevents
    having to restart from scratch.
    """

    initial_score = initial_best_score(
        selection_mode
    )

    if not os.path.exists(path):
        return 0, initial_score

    ckpt = torch.load(
        path,
        map_location=device,
    )

    if "best_score" not in ckpt:
        raise ValueError(
            "Incompatible legacy checkpoint: contains best_loss "
            "but not best_score. Remove or rename B1_last.pt "
            "and restart with the new validation protocol."
        )

    if (
        ckpt.get("selection_metric")
        != selection_metric
    ):
        raise ValueError(
            "Checkpoint uses a different selection_metric: "
            f"{ckpt.get('selection_metric')} != {selection_metric}."
        )

    if (
        ckpt.get("selection_mode")
        != selection_mode
    ):
        raise ValueError(
            "Checkpoint uses a different selection_mode: "
            f"{ckpt.get('selection_mode')} != {selection_mode}."
        )

    model.load_state_dict(
        ckpt["model_state"]
    )

    if (
        optimizer is not None
        and ckpt.get("optimizer_state") is not None
    ):
        optimizer.load_state_dict(
            ckpt["optimizer_state"]
        )

    print(
        f"Resumed checkpoint from {path} "
        f"(epoch {ckpt['epoch']}, "
        f"best {selection_metric} "
        f"{ckpt['best_score']:.6f})"
    )

    return (
        ckpt["epoch"],
        ckpt["best_score"],
    )