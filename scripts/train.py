"""
Training loop per B0/B1 (vedi configs/experiments.yaml).

ATTENZIONE: usa ancora un dataset FITTIZIO (tensori random) perche' il vero
DataLoader (src/data/dataset.py, di competenza di A) non e' ancora pronto.
Appena disponibile, sostituire DummyDensityDataset con l'import reale (vedi
il TODO piu' sotto) — l'interfaccia (Dataset che restituisce (image, density))
e' gia' pensata per essere compatibile, quindi il resto dello script non
dovrebbe cambiare.

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
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.baseline import B1Baseline, CenterPriorB0, get_device


class DummyDensityDataset(Dataset):
    """
    Placeholder finche' src/data/dataset.py non e' pronto (proprieta' di A).
    Genera immagini RGB casuali e density map casuali "grezze" (0-1 per pixel,
    NON a somma 1 — vedi baseline.py per il perche') della stessa shape che
    avra' il dataset vero.

    ATTENZIONE: qui ogni __getitem__ genera rumore nuovo, quindi non c'e' un
    segnale imparabile — questo dataset serve solo a validare che il loop
    (forward/backward/checkpoint) non crashi, NON a verificare che la loss
    scenda in modo sensato. Quella verifica arriva solo con il dataset vero.

    TODO (rimuovere appena pronto):
        from src.data.dataset import SaliconDataset
        dataset = SaliconDataset(split="train", config_path="configs/data.yaml")
    """
    def __init__(self, n_samples: int = 64, height: int = 192, width: int = 256):
        self.n_samples = n_samples
        self.height = height
        self.width = width

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image = torch.rand(3, self.height, self.width)
        density = torch.rand(1, self.height, self.width)  # grezza, 0-1, non a somma 1
        return image, density


def save_checkpoint(path, model, optimizer, epoch, best_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "best_loss": best_loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Ritorna (epoca_di_partenza, best_loss). Se non c'e' checkpoint, riparte da zero.
    Fondamentale su Colab: una sessione puo' disconnettersi in qualunque momento,
    questo evita di ripartire da capo ogni volta."""
    if not os.path.exists(path):
        return 0, float("inf")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Checkpoint ripreso da {path} (epoca {ckpt['epoch']}, best_loss {ckpt['best_loss']:.4f})")
    return ckpt["epoch"], ckpt["best_loss"]


def train_b1(args, device):
    model = B1Baseline(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()  # loss di B1 secondo configs/experiments.yaml

    train_dataset = DummyDensityDataset(n_samples=64, height=args.height, width=args.width)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    checkpoint_path = os.path.join(args.checkpoint_dir, "B1_last.pt")
    start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer, device=device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        epoch_loss /= len(train_dataset)
        print(f"[B1] Epoca {epoch + 1}/{args.epochs} - loss MSE: {epoch_loss:.6e} "
              f"- {time.time() - t0:.1f}s")

        is_best = epoch_loss < best_loss
        best_loss = min(epoch_loss, best_loss)
        save_checkpoint(checkpoint_path, model, optimizer, epoch + 1, best_loss)
        if is_best:
            save_checkpoint(os.path.join(args.checkpoint_dir, "B1_best.pt"),
                             model, optimizer, epoch + 1, best_loss)

    print(f"Training B1 completato. Checkpoint in: {args.checkpoint_dir}")


def fit_b0(args, device):
    """B0 non si allena via backprop: si calcola una volta sola la media delle
    density map di training. Con dati fittizi serve solo a validare il meccanismo
    (fit + salvataggio); va rilanciato con il dataset vero appena disponibile."""
    train_dataset = DummyDensityDataset(n_samples=256, height=args.height, width=args.width)
    loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    all_maps = []
    for _, targets in loader:
        all_maps.append(targets)
    all_maps = torch.cat(all_maps, dim=0).to(device)

    model = CenterPriorB0(height=args.height, width=args.width).to(device)
    model.fit(all_maps)

    checkpoint_path = os.path.join(args.checkpoint_dir, "B0_center_map.pt")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"B0 (center prior) calcolato su {len(train_dataset)} mappe e salvato in {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["B0", "B1"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                         help="Su Colab passare il path su Drive, es. "
                              "/content/drive/MyDrive/nndl-saliency/checkpoints")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print("ATTENZIONE: dataset ancora fittizio (DummyDensityDataset) — "
          "sostituire con il DataLoader reale appena pronto (vedi TODO nel file).")

    if args.experiment == "B1":
        train_b1(args, device)
    else:
        fit_b0(args, device)


if __name__ == "__main__":
    main()
