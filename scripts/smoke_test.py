from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------
# Permette di importare src anche eseguendo:
# python scripts/smoke_test.py
# ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import SaliconDataset


# ---------------------------------------------------------
# Configurazione
# ---------------------------------------------------------

DATA_DIR = "/content/data_local"
MANIFEST_PATH = REPO_ROOT / "results" / "split_manifest.csv"

EXPECTED_SPLITS = {
    "train": 10000,
    "tuning": 2500,
    "internal_test": 2500,
}

EXPECTED_IMAGE_SHAPE = (3, 192, 256)
EXPECTED_MAP_SHAPE = (1, 192, 256)


# ---------------------------------------------------------
# Controllo singolo campione
# ---------------------------------------------------------

def check_sample(dataset, index, split_name):
    sample = dataset[index]

    image = sample["image"]
    density_map = sample["density_map"]
    fixation_path = Path(sample["fixation_path"])

    assert image.shape == EXPECTED_IMAGE_SHAPE, (
        f"{split_name}: image shape errata: {image.shape}"
    )

    assert density_map.shape == EXPECTED_MAP_SHAPE, (
        f"{split_name}: density map shape errata: {density_map.shape}"
    )

    assert torch.isfinite(image).all(), (
        f"{split_name}: trovati NaN/Inf nell'immagine"
    )

    assert torch.isfinite(density_map).all(), (
        f"{split_name}: trovati NaN/Inf nella density map"
    )

    density_sum = density_map.sum().item()

    assert abs(density_sum - 1.0) < 1e-4, (
        f"{split_name}: density map non normalizzata "
        f"(somma={density_sum})"
    )

    assert fixation_path.exists(), (
        f"{split_name}: fixation file non trovato: "
        f"{fixation_path}"
    )

    print(
        f"  [OK] {split_name} | "
        f"{sample['image_id']} | "
        f"map sum={density_sum:.6f}"
    )


# ---------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("SALICON DATA PIPELINE - SMOKE TEST")
    print("=" * 60)

    data_dir = Path(DATA_DIR)

    assert data_dir.exists(), (
        f"Dataset non trovato: {DATA_DIR}\n"
        "Esegui prima la sezione 6bis del notebook Colab."
    )

    assert MANIFEST_PATH.exists(), (
        f"Manifest non trovato: {MANIFEST_PATH}"
    )

    datasets = {}

    # -----------------------------------------------------
    # 1. Controllo dimensione split
    # -----------------------------------------------------

    print("\n[1/3] Controllo split")

    for split_name, expected_size in EXPECTED_SPLITS.items():

        dataset = SaliconDataset(
            data_dir=DATA_DIR,
            manifest_path=MANIFEST_PATH,
            split=split_name,
            input_size=(256, 192),
            density_map_epsilon=1e-6,
            augmentation=False,
        )

        actual_size = len(dataset)

        assert actual_size == expected_size, (
            f"{split_name}: attesi {expected_size} campioni, "
            f"trovati {actual_size}"
        )

        datasets[split_name] = dataset

        print(
            f"  [OK] {split_name}: "
            f"{actual_size} campioni"
        )

    # -----------------------------------------------------
    # 2. Controllo campioni
    # -----------------------------------------------------

    print("\n[2/3] Controllo campioni")

    for split_name, dataset in datasets.items():

        # Controlliamo inizio, centro e fine dello split
        indices = [
            0,
            len(dataset) // 2,
            len(dataset) - 1,
        ]

        for index in indices:
            check_sample(
                dataset,
                index,
                split_name,
            )

    # -----------------------------------------------------
    # 3. Controllo DataLoader
    # -----------------------------------------------------

    print("\n[3/3] Controllo DataLoader")

    train_loader = DataLoader(
        datasets["train"],
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(train_loader))

    images = batch["image"]
    density_maps = batch["density_map"]

    assert images.shape == (4, 3, 192, 256), (
        f"Batch immagini con shape errata: {images.shape}"
    )

    assert density_maps.shape == (4, 1, 192, 256), (
        f"Batch density maps con shape errata: "
        f"{density_maps.shape}"
    )

    assert torch.isfinite(images).all(), (
        "NaN/Inf nel batch immagini"
    )

    assert torch.isfinite(density_maps).all(), (
        "NaN/Inf nel batch density maps"
    )

    map_sums = density_maps.sum(dim=(1, 2, 3))

    assert torch.allclose(
        map_sums,
        torch.ones_like(map_sums),
        atol=1e-4,
    ), (
        f"Density maps del batch non normalizzate: "
        f"{map_sums}"
    )

    print(
        "  [OK] Batch immagini:",
        tuple(images.shape),
    )

    print(
        "  [OK] Batch density maps:",
        tuple(density_maps.shape),
    )

    print(
        "  [OK] Somme density maps:",
        map_sums.tolist(),
    )

    # -----------------------------------------------------
    # Fine
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()