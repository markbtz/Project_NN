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

Uso:
    python scripts/train.py --experiment B1 --epochs 3 --batch_size 8
    python scripts/train.py --experiment B0

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
from torch.utils.data import DataLoader


sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)


from src.models.baseline import (
    B1Baseline,
    CenterPriorB0,
    get_device,
    set_seed,
)

from src.data.dataset import SaliconDataset


# Seed unico per tutto il progetto (vedi configs/data.yaml -> seed).
# Tenuto qui come costante esplicita finche' non e' collegato il caricamento
# reale dello YAML in questo script (vedi roadmap README, punto ancora aperto).
GLOBAL_SEED = 42


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
        pretrained=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()

    # -----------------------------------------------------
    # Dataset reale SALICON
    #
    # B1 usa density_map_raw perché la loss è MSE.
    # -----------------------------------------------------

    train_dataset = SaliconDataset(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        split="train",
        input_size=(
            args.width,
            args.height,
        ),
        density_map_epsilon=1e-6,
        augmentation=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
                "density_map_raw"
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
    le density map insieme in memoria (come faceva la versione precedente
    con torch.cat) costa circa 1.9 GB solo per quel tensore — un rischio
    concreto di OOM su Colab free. fit_from_loader accumula una somma
    incrementale, un batch alla volta, senza mai avere tutto il dataset in
    RAM contemporaneamente.
    """

    train_dataset = SaliconDataset(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        split="train",
        input_size=(
            args.width,
            args.height,
        ),
        density_map_epsilon=1e-6,
        augmentation=False,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
    )

    def density_batches():
        for batch in loader:
            yield batch["density_map_prob"].to(device)

    model = CenterPriorB0(
        height=args.height,
        width=args.width,
    ).to(device)

    model.fit_from_loader(density_batches())

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
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
    )

    # -----------------------------------------------------
    # Percorsi dataset reale
    # -----------------------------------------------------

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/content/data_local",
        help=(
            "Root del dataset SALICON. "
            "Su Colab: /content/data_local"
        ),
    )

    parser.add_argument(
        "--manifest_path",
        type=str,
        default="results/split_manifest.csv",
        help="Path al manifest dello split.",
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

    set_seed(GLOBAL_SEED)

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Seed: {GLOBAL_SEED}"
    )

    print(
        f"Dataset: {args.data_dir}"
    )

    print(
        f"Manifest: {args.manifest_path}"
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
