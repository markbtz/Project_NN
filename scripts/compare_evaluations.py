import csv
import argparse
from pathlib import Path

METRICS = (
    "cc",
    "sim",
    "kld",
)


def load_per_image_csv(path):
    """
    Carica un CSV per-image prodotto da evaluate.py.

    Le righe vengono indicizzate per image_id in modo che
    i confronti tra modelli non dipendano dall'ordine del CSV.
    """

    samples = {}

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "image_id",
            *METRICS,
        }

        if not required_columns.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                f"CSV non valido: {path}. "
                "Sono richieste le colonne "
                "image_id, cc, sim, kld."
            )

        for row in reader:
            image_id = row["image_id"]

            if image_id in samples:
                raise ValueError(
                    "image_id duplicato nel CSV "
                    f"{path}: {image_id}"
                )

            samples[image_id] = {
                metric: float(row[metric])
                for metric in METRICS
            }

    return samples

def align_by_image_id(
    left_samples,
    right_samples,
    *,
    left_name,
    right_name,
):
    """
    Allinea due evaluation tramite image_id.

    Fallisce esplicitamente se uno dei due CSV contiene
    immagini che mancano nell'altro.
    """

    left_ids = set(left_samples)
    right_ids = set(right_samples)

    missing_in_right = left_ids - right_ids
    missing_in_left = right_ids - left_ids

    if missing_in_right or missing_in_left:
        raise ValueError(
            "I CSV non contengono gli stessi image_id. "
            f"Mancanti in {right_name}: "
            f"{sorted(missing_in_right)}. "
            f"Mancanti in {left_name}: "
            f"{sorted(missing_in_left)}."
        )

    return [
        (
            image_id,
            left_samples[image_id],
            right_samples[image_id],
        )
        for image_id in sorted(left_ids)
    ]

def compute_differences(
    aligned_samples,
    *,
    left_name,
    right_name,
):
    """
    Calcola le differenze per-image come:

        right - left

    Esempi:
        B1 - B0
        M1 - B1
    """

    rows = []

    for (
        image_id,
        left_metrics,
        right_metrics,
    ) in aligned_samples:
        row = {
            "image_id": image_id,
        }

        for metric in METRICS:
            left_value = left_metrics[metric]
            right_value = right_metrics[metric]

            row[f"{left_name}_{metric}"] = (
                left_value
            )
            row[f"{right_name}_{metric}"] = (
                right_value
            )
            row[f"delta_{metric}"] = (
                right_value - left_value
            )

        rows.append(row)

    return rows

def save_differences_csv(
    rows,
    output_path,
    *,
    left_name,
    right_name,
):
    fieldnames = ["image_id"]

    for metric in METRICS:
        fieldnames.extend(
            [
                f"{left_name}_{metric}",
                f"{right_name}_{metric}",
                f"delta_{metric}",
            ]
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Confronta due CSV per-image "
            "allineandoli tramite image_id."
        )
    )

    parser.add_argument("left_csv")
    parser.add_argument("right_csv")

    parser.add_argument(
        "--left-name",
        required=True,
    )

    parser.add_argument(
        "--right-name",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    left_samples = load_per_image_csv(
        args.left_csv
    )
    right_samples = load_per_image_csv(
        args.right_csv
    )

    aligned = align_by_image_id(
        left_samples,
        right_samples,
        left_name=args.left_name,
        right_name=args.right_name,
    )

    rows = compute_differences(
        aligned,
        left_name=args.left_name,
        right_name=args.right_name,
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_differences_csv(
        rows,
        output_path,
        left_name=args.left_name,
        right_name=args.right_name,
    )

    print(
        f"Confrontati {len(rows)} campioni. "
        f"Risultato salvato in: {output_path}"
    )


if __name__ == "__main__":
    main()