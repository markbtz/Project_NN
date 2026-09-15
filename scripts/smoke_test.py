from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import SaliconDataset


DATA_DIR = "/content/data_local"
MANIFEST_PATH = REPO_ROOT / "results" / "split_manifest.csv"

EXPECTED_SPLITS = {
    "train": 10000,
    "tuning": 2500,
    "internal_test": 2500,
}

EXPECTED_IMAGE_SHAPE = (3, 192, 256)
EXPECTED_MAP_SHAPE = (1, 192, 256)


def check_sample(dataset, index, split_name):
    sample = dataset[index]

    image = sample["image"]
    density_map_raw = sample["density_map_raw"]
    density_map_prob = sample["density_map_prob"]
    fixation_path = Path(sample["fixation_path"])

    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    assert image.shape == EXPECTED_IMAGE_SHAPE, (
        f"{split_name}: image shape errata: {image.shape}"
    )

    assert torch.isfinite(image).all(), (
        f"{split_name}: trovati NaN/Inf nell'immagine"
    )

    # -----------------------------------------------------
    # RAW DENSITY MAP
    # -----------------------------------------------------

    assert density_map_raw.shape == EXPECTED_MAP_SHAPE, (
        f"{split_name}: raw map shape errata: "
        f"{density_map_raw.shape}"
    )

    assert density_map_raw.dtype == torch.float32, (
        f"{split_name}: raw map dtype errato: "
        f"{density_map_raw.dtype}"
    )

    assert torch.isfinite(density_map_raw).all(), (
        f"{split_name}: trovati NaN/Inf nella raw density map"
    )

    assert torch.all(density_map_raw >= 0.0), (
        f"{split_name}: raw density map con valori < 0"
    )

    assert torch.all(density_map_raw <= 1.0), (
        f"{split_name}: raw density map con valori > 1"
    )

    # -----------------------------------------------------
    # PROBABILITY DENSITY MAP
    # -----------------------------------------------------

    assert density_map_prob.shape == EXPECTED_MAP_SHAPE, (
        f"{split_name}: probability map shape errata: "
        f"{density_map_prob.shape}"
    )

    assert density_map_prob.dtype == torch.float32, (
        f"{split_name}: probability map dtype errato: "
        f"{density_map_prob.dtype}"
    )

    assert torch.isfinite(density_map_prob).all(), (
        f"{split_name}: trovati NaN/Inf "
        f"nella probability density map"
    )

    assert torch.all(density_map_prob >= 0.0), (
        f"{split_name}: probability map con valori < 0"
    )

    probability_sum = density_map_prob.sum().item()

    assert abs(probability_sum - 1.0) < 1e-4, (
        f"{split_name}: probability map non normalizzata "
        f"(somma={probability_sum})"
    )

    # -----------------------------------------------------
    # FIXATION
    # -----------------------------------------------------

    assert fixation_path.exists(), (
        f"{split_name}: fixation file non trovato: "
        f"{fixation_path}"
    )

    print(
        f"  [OK] {split_name} | "
        f"{sample['image_id']} | "
        f"raw_min={density_map_raw.min().item():.4f} | "
        f"raw_max={density_map_raw.max().item():.4f} | "
        f"prob_sum={probability_sum:.6f}"
    )


def main():
    print("=" * 60)
    print("SALICON DATA PIPELINE - SMOKE TEST")
    print("=" * 60)

    data_dir = Path(DATA_DIR)

    assert data_dir.exists(), (
        f"Dataset non trovato: {DATA_DIR}\n"
        "Esegui prima la preparazione del dataset in Colab."
    )

    assert MANIFEST_PATH.exists(), (
        f"Manifest non trovato: {MANIFEST_PATH}"
    )

    datasets = {}

    # -----------------------------------------------------
    # 1. CONTROLLO SPLIT
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
    # 2. CONTROLLO CAMPIONI
    # -----------------------------------------------------

    print("\n[2/3] Controllo campioni")

    for split_name, dataset in datasets.items():

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
    # 3. CONTROLLO DATALOADER
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
    raw_maps = batch["density_map_raw"]
    prob_maps = batch["density_map_prob"]

    assert images.shape == (
        4,
        3,
        192,
        256,
    ), (
        f"Batch immagini con shape errata: "
        f"{images.shape}"
    )

    assert raw_maps.shape == (
        4,
        1,
        192,
        256,
    ), (
        f"Batch raw maps con shape errata: "
        f"{raw_maps.shape}"
    )

    assert prob_maps.shape == (
        4,
        1,
        192,
        256,
    ), (
        f"Batch probability maps con shape errata: "
        f"{prob_maps.shape}"
    )

    assert torch.isfinite(images).all(), (
        "NaN/Inf nel batch immagini"
    )

    assert torch.isfinite(raw_maps).all(), (
        "NaN/Inf nel batch raw maps"
    )

    assert torch.isfinite(prob_maps).all(), (
        "NaN/Inf nel batch probability maps"
    )

    assert torch.all(raw_maps >= 0.0), (
        "Il batch raw contiene valori < 0"
    )

    assert torch.all(raw_maps <= 1.0), (
        "Il batch raw contiene valori > 1"
    )

    prob_sums = prob_maps.sum(
        dim=(1, 2, 3)
    )

    assert torch.allclose(
        prob_sums,
        torch.ones_like(prob_sums),
        atol=1e-4,
    ), (
        f"Probability maps non normalizzate: "
        f"{prob_sums}"
    )

    print(
        "  [OK] Batch immagini:",
        tuple(images.shape),
    )

    print(
        "  [OK] Batch raw maps:",
        tuple(raw_maps.shape),
    )

    print(
        "  [OK] Batch probability maps:",
        tuple(prob_maps.shape),
    )

    print(
        "  [OK] Somme probability maps:",
        prob_sums.tolist(),
    )

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()