"""
Utility condivise per il salvataggio e il caricamento dei checkpoint.
"""

import os

import torch


def initial_best_score(selection_mode):
    """
    Valore iniziale per la selezione del best checkpoint.
    """

    if selection_mode == "max":
        return float("-inf")

    if selection_mode == "min":
        return float("inf")

    raise ValueError(
        "selection_mode deve essere 'max' oppure 'min'."
    )


def is_better(
    current_score,
    best_score,
    selection_mode,
):
    """
    Confronta lo score corrente con il best score secondo
    la direzione definita in experiments.yaml.
    """

    if selection_mode == "max":
        return current_score > best_score

    if selection_mode == "min":
        return current_score < best_score

    raise ValueError(
        "selection_mode deve essere 'max' oppure 'min'."
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
    Ritorna (epoca_di_partenza, best_score).

    Se non c'e' checkpoint, riparte da zero usando il valore
    iniziale coerente con selection_mode.

    I vecchi checkpoint basati su best_loss non vengono riutilizzati:
    la selezione del best model ora avviene sul tuning set e quindi
    rappresenta un protocollo diverso.

    Fondamentale su Colab: una sessione puo'
    disconnettersi in qualunque momento, questo evita
    di ripartire da capo ogni volta.
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
            "Checkpoint legacy non compatibile: contiene best_loss "
            "ma non best_score. Rimuovere o rinominare B1_last.pt "
            "e ripartire con il nuovo protocollo di validazione."
        )

    if (
        ckpt.get("selection_metric")
        != selection_metric
    ):
        raise ValueError(
            "Il checkpoint usa una selection_metric diversa: "
            f"{ckpt.get('selection_metric')} != {selection_metric}."
        )

    if (
        ckpt.get("selection_mode")
        != selection_mode
    ):
        raise ValueError(
            "Il checkpoint usa una selection_mode diversa: "
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
        f"Checkpoint ripreso da {path} "
        f"(epoca {ckpt['epoch']}, "
        f"best {selection_metric} "
        f"{ckpt['best_score']:.6f})"
    )

    return (
        ckpt["epoch"],
        ckpt["best_score"],
    )