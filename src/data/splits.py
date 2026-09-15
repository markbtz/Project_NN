from pathlib import Path
import argparse
import csv
import random


def build_split_manifest(data_dir, output_csv, seed=42):
    data_dir = Path(data_dir)

    train_dir = data_dir / "images" / "train"
    val_dir = data_dir / "images" / "val"

    train_images = sorted(train_dir.glob("*"))
    val_images = sorted(val_dir.glob("*"))

    train_images = [
        p for p in train_images
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    val_images = [
        p for p in val_images
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    print("Train ufficiale:", len(train_images))
    print("Validation ufficiale:", len(val_images))

    if len(train_images) != 10000:
        raise ValueError(
            f"Attese 10000 immagini train, trovate {len(train_images)}"
        )

    if len(val_images) != 5000:
        raise ValueError(
            f"Attese 5000 immagini validation, trovate {len(val_images)}"
        )

    rng = random.Random(seed)

    shuffled_val = val_images.copy()
    rng.shuffle(shuffled_val)

    tuning_images = shuffled_val[:2500]
    internal_test_images = shuffled_val[2500:]

    rows = []

    for image_path in train_images:
        rows.append({
            "image_id": image_path.stem,
            "official_split": "train",
            "split": "train",
        })

    for image_path in tuning_images:
        rows.append({
            "image_id": image_path.stem,
            "official_split": "val",
            "split": "tuning",
        })

    for image_path in internal_test_images:
        rows.append({
            "image_id": image_path.stem,
            "official_split": "val",
            "split": "internal_test",
        })

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "official_split", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Manifest creato:", output_csv)
    print("Train:", sum(r["split"] == "train" for r in rows))
    print("Tuning:", sum(r["split"] == "tuning" for r in rows))
    print(
        "Internal test:",
        sum(r["split"] == "internal_test" for r in rows)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/split_manifest.csv",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    build_split_manifest(
        data_dir=args.data_dir,
        output_csv=args.output,
        seed=args.seed,
    )