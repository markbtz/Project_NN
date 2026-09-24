"""
Diagnostica il comportamento del gate di G: distribuzione di alpha sulle
immagini di tuning. Se alpha e' quasi costante, il gate non sta imparando
un peso adattivo per-immagine (gate collapse) — sta solo aggiungendo una
piccola quantita' fissa di center prior a ogni predizione, il che spiega
un peggioramento uniforme rispetto a M1-L da solo.

Uso:
    python scripts/diagnose_gate.py \
        --checkpoint /content/drive/MyDrive/nndl-saliency/checkpoints/G_best.pt \
        --data_dir /content/data_local \
        --split tuning
"""
import argparse
import sys
import os

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.adaptive_center_prior import AdaptiveCenterPriorG
from src.data.dataset import SaliconDataset
from src.runtime import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--manifest_path", default="results/split_manifest.csv")
    parser.add_argument("--split", default="tuning")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_batches", type=int, default=None,
                         help="Limita il numero di batch per un check veloce")
    args = parser.parse_args()

    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    if ckpt.get("experiment") != "G":
        raise ValueError(f"Checkpoint non e' di G: experiment={ckpt.get('experiment')}")

    model = AdaptiveCenterPriorG(height=args.height, width=args.width)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    dataset = SaliconDataset(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        split=args.split,
        input_size=(args.width, args.height),
        augmentation=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_alphas = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.n_batches is not None and i >= args.n_batches:
                break
            images = batch["image"].to(device)
            _, alpha = model(images, return_alpha=True)
            all_alphas.append(alpha.flatten().cpu())

    alphas = torch.cat(all_alphas)

    print(f"Campioni analizzati: {alphas.numel()}")
    print(f"alpha - media: {alphas.mean().item():.6f}")
    print(f"alpha - deviazione standard: {alphas.std().item():.6f}")
    print(f"alpha - min: {alphas.min().item():.6f}")
    print(f"alpha - max: {alphas.max().item():.6f}")
    print(f"alpha - percentili [5,25,50,75,95]: "
          f"{torch.quantile(alphas, torch.tensor([0.05,0.25,0.5,0.75,0.95])).tolist()}")

    print()
    if alphas.std().item() < 0.01:
        print("DIAGNOSI: alpha e' praticamente costante su tutte le immagini.")
        print("Il gate NON sta imparando un peso adattivo per-immagine — e' collassato")
        print("a un valore fisso. Questo spiega un peggioramento uniforme rispetto a M1-L.")
    else:
        print("alpha varia in modo non trascurabile tra le immagini — nessun collasso")
        print("evidente. Il peggioramento (se confermato) ha probabilmente un'altra causa")
        print("(es. il gate si affida al prior proprio nei casi in cui M1-L era gia' forte).")


if __name__ == "__main__":
    main()
