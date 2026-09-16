"""
Training loop per B0/B1 (vedi configs/experiments.yaml).

Il training utilizza il dataset reale SALICON tramite SaliconDataset.

B1 usa:
    density_map_raw
    valori in [0,1]
    loss MSE

B0 usa:
    density_map_prob
    normalizzata a somma 1

Le configurazioni condivise vengono lette da:
    configs/data.yaml
    configs/experiments.yaml

Uso:
    python scripts/train.py --experiment B1
    python scripts/train.py --experiment B0

Gli argomenti CLI --epochs, --batch_size, --height, --width,
--data_dir e --manifest_path restano disponibili come override opzionali.

Su Colab, passare --checkpoint_dir sul path di Drive, es.:
    python scripts/train.py --experiment B1 \
        --checkpoint_dir /content/drive/MyDrive/nndl-saliency/checkpoints
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    REPO_ROOT,
)


from src.models.baseline import (
    B1Baseline,
    CenterPriorB0,
    get_device,
    set_seed,
)

from src.data.dataset import SaliconDataset


DEFAULT_DATA_CONFIG = os.path.join(
    REPO_ROOT,
    "configs",
    "data.yaml",
)

DEFAULT_EXPERIMENTS_CONFIG = os.path.join(
    REPO_ROOT,
    "configs",
    "experiments.yaml",
)


def load_yaml_config(path):
    """Carica un file YAML e verifica che contenga un mapping."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configurazione YAML non valida: {path}"
        )

    return config


def resolve_repo_path(path):
    """Rende assoluti i path relativi alla root del repository."""
    if os.path.isabs(path):
        return path

    return os.path.join(
        REPO_ROOT,
        path,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_loss,
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
            "best_loss": best_loss,
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
):
    """
    Ritorna (epoca_di_partenza, best_loss).

    Se non c'e' checkpoint, riparte da zero.

    Fondamentale su Colab: una sessione puo'
    disconnettersi in qualunque momento, questo evita
    di ripartire da capo ogni volta.
    """

    if not os.path.exists(path):
        return 0, float("inf")

    ckpt = torch.load(
        path,
        map_location=device,
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
        f"best_loss {ckpt['best_loss']:.4f})"
    )

    return (
        ckpt["epoch"],
        ckpt["best_loss"],
    )


def train_b1(args, device):
    model = B1Baseline(
        pretrained=args.pretrained,
        decoder_width=args.decoder_width,
    ).to(device)

    if args.optimizer_name != "AdamW":
        raise ValueError(
            "B1 supporta attualmente solo optimizer AdamW, "
            f"ma experiments.yaml contiene: {args.optimizer_name}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.loss_name != "mse":
        raise ValueError(
            "B1 supporta attualmente solo loss MSE, "
            f"ma experiments.yaml contiene: {args.loss_name}"
        )

    criterion = nn.MSELoss()

    # -----------------------------------------------------
    # Dataset reale SALICON
    #
    # Il target e' definito in configs/experiments.yaml.
    # Per B1 deve essere density_map_raw.
    # -----------------------------------------------------

    train_dataset = SaliconDataset(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        split=args.train_split,
        input_size=(
            args.width,
            args.height,
        ),
        density_map_epsilon=args.density_map_epsilon,
        augmentation=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        "B1_last.pt",
    )

    start_epoch, best_loss = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        device=device,
    )

    for epoch in range(
        start_epoch,
        args.epochs,
    ):
        model.train()

        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:

            images = batch[
                "image"
            ].to(device)

            targets = batch[
                args.target_key
            ].to(device)

            optimizer.zero_grad()

            preds = model(
                images
            )

            loss = criterion(
                preds,
                targets,
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.item()
                * images.size(0)
            )

        epoch_loss /= len(
            train_dataset
        )

        print(
            f"[B1] Epoca "
            f"{epoch + 1}/{args.epochs} "
            f"- loss MSE: "
            f"{epoch_loss:.6e} "
            f"- {time.time() - t0:.1f}s"
        )

        is_best = (
            epoch_loss < best_loss
        )

        best_loss = min(
            epoch_loss,
            best_loss,
        )

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch + 1,
            best_loss,
        )

        if is_best:
            save_checkpoint(
                os.path.join(
                    args.checkpoint_dir,
                    "B1_best.pt",
                ),
                model,
                optimizer,
                epoch + 1,
                best_loss,
            )

    print(
        "Training B1 completato. "
        f"Checkpoint in: {args.checkpoint_dir}"
    )


def fit_b0(args, device):
    """
    B0 non si allena via backprop.

    Calcola il center prior come media delle density_map_prob del training
    set, in streaming (fit_from_loader): con 10.000 immagini, tenere tutte
    le density map insieme in memoria costa circa 1.9 GB solo per quel
    tensore. fit_from_loader accumula una somma incrementale, un batch alla
    volta, senza mai avere tutto il dataset in RAM contemporaneamente.
    """

    train_dataset = SaliconDataset(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        split=args.train_split,
        input_size=(
            args.width,
            args.height,
        ),
        density_map_epsilon=args.density_map_epsilon,
        augmentation=False,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
    )

    def density_batches():
        for batch in loader:
            yield batch[
                args.target_key
            ].to(device)

    model = CenterPriorB0(
        height=args.height,
        width=args.width,
    ).to(device)

    model.fit_from_loader(
        density_batches(),
        eps=args.density_map_epsilon,
    )

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        "B0_center_map.pt",
    )

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    print(
        f"B0 (center prior) calcolato su "
        f"{len(train_dataset)} mappe "
        f"e salvato in {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["B0", "B1"],
        required=True,
    )

    parser.add_argument(
        "--data_config",
        type=str,
        default=DEFAULT_DATA_CONFIG,
    )

    parser.add_argument(
        "--experiments_config",
        type=str,
        default=DEFAULT_EXPERIMENTS_CONFIG,
    )

    # Override opzionali da CLI.
    # Se omessi, i valori vengono letti dai file YAML.
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help=(
            "Su Colab passare il path su Drive, es. "
            "/content/drive/MyDrive/"
            "nndl-saliency/checkpoints"
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Caricamento configurazioni YAML
    # -----------------------------------------------------

    data_config = load_yaml_config(
        args.data_config
    )

    experiments_config = load_yaml_config(
        args.experiments_config
    )

    training_config = experiments_config[
        "training"
    ]

    experiment_config = experiments_config[
        "experiments"
    ][args.experiment]

    # -----------------------------------------------------
    # data.yaml
    # -----------------------------------------------------

    args.seed = int(
        data_config["seed"]
    )

    config_width, config_height = data_config[
        "input_size"
    ]

    if args.width is None:
        args.width = int(config_width)

    if args.height is None:
        args.height = int(config_height)

    args.density_map_epsilon = float(
        data_config["density_map_epsilon"]
    )

    if args.data_dir is None:
        colab_cache = data_config.get(
            "colab_local_cache_dir"
        )

        if (
            colab_cache
            and os.path.isdir(colab_cache)
        ):
            args.data_dir = colab_cache
        else:
            args.data_dir = resolve_repo_path(
                data_config["dataset_root"]
            )

    if args.manifest_path is None:
        args.manifest_path = resolve_repo_path(
            data_config["manifest_path"]
        )

    # -----------------------------------------------------
    # experiments.yaml -> training
    # -----------------------------------------------------

    args.train_split = training_config[
        "train_split"
    ]

    args.num_workers = int(
        training_config.get(
            "num_workers",
            0,
        )
    )

    if args.batch_size is None:
        args.batch_size = int(
            training_config["batch_size"]
        )

    if args.epochs is None:
        args.epochs = int(
            training_config["epochs"]
        )

    optimizer_config = training_config[
        "optimizer"
    ]

    args.optimizer_name = optimizer_config[
        "name"
    ]

    args.learning_rate = float(
        optimizer_config[
            "learning_rate"
        ]
    )

    args.weight_decay = float(
        optimizer_config[
            "weight_decay"
        ]
    )

    # -----------------------------------------------------
    # experiments.yaml -> esperimento selezionato
    # -----------------------------------------------------

    args.target_key = experiment_config[
        "target"
    ]

    loss_config = experiment_config[
        "loss"
    ]

    args.loss_name = loss_config[
        "name"
    ]

    if args.experiment == "B1":
        args.pretrained = bool(
            experiment_config[
                "encoder"
            ][
                "pretrained"
            ]
        )

        args.decoder_width = int(
            experiment_config[
                "decoder"
            ][
                "width"
            ]
        )

    # -----------------------------------------------------
    # Riproducibilita' e riepilogo
    # -----------------------------------------------------

    set_seed(
        args.seed
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Data config: {args.data_config}"
    )

    print(
        f"Experiments config: "
        f"{args.experiments_config}"
    )

    print(
        f"Dataset: {args.data_dir}"
    )

    print(
        f"Manifest: {args.manifest_path}"
    )

    print(
        f"Split: {args.train_split}"
    )

    print(
        f"Input size: "
        f"{args.width}x{args.height}"
    )

    print(
        f"Epsilon: "
        f"{args.density_map_epsilon}"
    )

    print(
        f"Target: {args.target_key}"
    )

    if args.experiment == "B1":
        print(
            f"Optimizer: "
            f"{args.optimizer_name}"
        )

        print(
            f"Learning rate: "
            f"{args.learning_rate}"
        )

        print(
            f"Weight decay: "
            f"{args.weight_decay}"
        )

        print(
            f"Batch size: "
            f"{args.batch_size}"
        )

        print(
            f"Epochs: "
            f"{args.epochs}"
        )

        print(
            f"Loss: "
            f"{args.loss_name}"
        )

    if args.experiment == "B1":
        train_b1(
            args,
            device,
        )

    else:
        fit_b0(
            args,
            device,
        )


if __name__ == "__main__":
    main()