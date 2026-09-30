"""
Diagnose G gate behavior: alpha distribution over
tuning images. If alpha is nearly constant, the gate is not learning
an adaptive per-image weight (gate collapse); it is only adding a
small fixed amount of center prior to every prediction, which explains
a consistent degradation relative to M1-L alone.

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
                         help="Limit the number of batches for a quick check")
    args = parser.parse_args()

    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    if ckpt.get("experiment") != "G":
        raise ValueError(f"Checkpoint is not for G: experiment={ckpt.get('experiment')}")

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

    print(f"Samples analyzed: {alphas.numel()}")
    print(f"alpha - mean: {alphas.mean().item():.6f}")
    print(f"alpha - standard deviation: {alphas.std().item():.6f}")
    print(f"alpha - min: {alphas.min().item():.6f}")
    print(f"alpha - max: {alphas.max().item():.6f}")
    print(f"alpha - percentiles [5,25,50,75,95]: "
          f"{torch.quantile(alphas, torch.tensor([0.05,0.25,0.5,0.75,0.95])).tolist()}")

    print()
    if alphas.std().item() < 0.01:
        print("DIAGNOSIS: alpha is nearly constant across all images.")
        print("The gate is NOT learning an image-adaptive weight; it has collapsed")
        print("to a fixed value. This explains a consistent degradation relative to M1-L.")
    else:
        print("alpha varies meaningfully across images; no obvious collapse")
        print("is visible. If the degradation is confirmed, it likely has another cause")
        print("(e.g. the gate relies on the prior when M1-L is already strong).")


if __name__ == "__main__":
    main()
