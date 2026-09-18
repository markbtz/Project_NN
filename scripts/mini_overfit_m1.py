from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import SaliconDataset
from src.models.multiscale import M1MultiScale
from src.models.baseline import get_device, set_seed


DATA_DIR = "/content/data_local"
MANIFEST_PATH = REPO_ROOT / "results" / "split_manifest.csv"

SEED = 42
N_IMAGES = 16
BATCH_SIZE = 8
N_UPDATES = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


def main():
    print("=" * 60)
    print("M1 - MINI OVERFIT TEST")
    print("=" * 60)

    set_seed(SEED)
    device = get_device()

    print("Device:", device)

    dataset = SaliconDataset(
        data_dir=DATA_DIR,
        manifest_path=MANIFEST_PATH,
        split="train",
        input_size=(256, 192),
        density_map_epsilon=1e-6,
        augmentation=False,
    )

    # Usiamo sempre le stesse 16 immagini.
    subset = Subset(
        dataset,
        list(range(N_IMAGES)),
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = M1MultiScale(
        pretrained=True,
        decoder_width=96,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    model.train()

    iterator = iter(loader)

    first_loss = None
    last_loss = None

    for update in range(1, N_UPDATES + 1):

        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        images = batch["image"].to(device)
        targets = batch["density_map_raw"].to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(
            predictions,
            targets,
        )

        loss.backward()
        optimizer.step()

        last_loss = loss.item()

        if first_loss is None:
            first_loss = last_loss

        if (
            update == 1
            or update % 10 == 0
            or update == N_UPDATES
        ):
            print(
                f"Update {update:02d}/{N_UPDATES} "
                f"- MSE: {last_loss:.6f}"
            )

    print("-" * 60)
    print(f"MSE iniziale: {first_loss:.6f}")
    print(f"MSE finale:   {last_loss:.6f}")

    if last_loss < first_loss:
        print("MINI OVERFIT M1: OK - la loss e' diminuita")
    else:
        raise RuntimeError(
            "MINI OVERFIT M1 FALLITO: "
            "la loss non e' diminuita."
        )


if __name__ == "__main__":
    main()