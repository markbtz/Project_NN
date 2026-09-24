"""
Genera la figura qualitativa sul tuning set.

Logica derivata dalle celle 7.6.1–7.6.4 del notebook VER_FINALE.
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from scripts.compare_evaluations import align_by_image_id, load_per_image_csv
from scripts.evaluate import load_model_for_evaluation
from src.data.dataset import SaliconDataset
from src.saliency_maps import normalize_probability_map


MODELS = ("B0", "B1", "M1", "M1-L", "G", "M2")

CHECKPOINT_FILES = {
    "B0": "B0_center_map.pt",
    "B1": "B1_best.pt",
    "M1": "M1_best.pt",
    "M1-L": "M1-L_best.pt",
    "G": "G_best.pt",
    "M2": "M2_best.pt",
}


def select_examples(results_dir):
    results_dir = Path(results_dir)

    scores = {
        model: load_per_image_csv(
            results_dir / model / "tuning_per_image.csv"
        )
        for model in ("B1", "M1", "M1-L", "G")
    }

    b1_m1 = align_by_image_id(
        scores["B1"], scores["M1"], left_name="B1", right_name="M1"
    )
    m1_m1l = align_by_image_id(
        scores["M1"], scores["M1-L"], left_name="M1", right_name="M1-L"
    )
    m1l_g = align_by_image_id(
        scores["M1-L"], scores["G"], left_name="M1-L", right_name="G"
    )

    if any(
        not math.isfinite(values["cc"])
        for row in b1_m1 + m1_m1l + m1l_g
        for values in row[1:]
    ):
        raise ValueError("Sono presenti valori CC non finiti.")

    rankings = [
        ("G migliore per CC", sorted(m1l_g, key=lambda r: (-r[2]["cc"], r[0]))),
        ("G peggiore per CC", sorted(m1l_g, key=lambda r: (r[2]["cc"], r[0]))),
        (
            "Maggiore ΔCC M1 − B1",
            sorted(b1_m1, key=lambda r: (-(r[2]["cc"] - r[1]["cc"]), r[0])),
        ),
        (
            "Maggiore ΔCC M1-L − M1",
            sorted(m1_m1l, key=lambda r: (-(r[2]["cc"] - r[1]["cc"]), r[0])),
        ),
        (
            "Maggiore ΔCC G − M1-L",
            sorted(m1l_g, key=lambda r: (-(r[2]["cc"] - r[1]["cc"]), r[0])),
        ),
    ]

    selected, used = [], set()

    for label, candidates in rankings:
        choice = next((row for row in candidates if row[0] not in used), None)

        if choice is None:
            raise ValueError("Non ci sono abbastanza image_id distinti.")

        selected.append({"label": label, "image_id": choice[0]})
        used.add(choice[0])

    return selected


def prepare_dataset(data_dir, selected):
    with open(REPO_DIR / "configs" / "data.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    width, height = config["input_size"]
    eps = config["density_map_epsilon"]
    manifest = REPO_DIR / config["manifest_path"]

    dataset = SaliconDataset(
        data_dir=data_dir,
        manifest_path=str(manifest),
        split="tuning",
        input_size=(width, height),
        density_map_epsilon=eps,
        augmentation=False,
    )

    wanted = {item["image_id"] for item in selected}
    indices = {}

    # Stesso metodo usato nella versione testata del notebook.
    for index in range(len(dataset)):
        image_id = dataset[index]["image_id"]
        if image_id in wanted:
            indices[image_id] = index
        if len(indices) == len(wanted):
            break

    missing = wanted - set(indices)
    if missing:
        raise ValueError(
            f"Image ID selezionati non trovati nel tuning set: {sorted(missing)}"
        )

    return dataset, indices, width, height, eps


def load_models(checkpoint_dir, experiments_config, device, width, height):
    checkpoint_dir = Path(checkpoint_dir)
    models, prediction_fns = {}, {}

    for experiment in MODELS:
        checkpoint = checkpoint_dir / CHECKPOINT_FILES[experiment]

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint {experiment} non trovato: {checkpoint}"
            )

        model, prediction_fn, _, _, _ = load_model_for_evaluation(
            experiment=experiment,
            experiment_config=experiments_config["experiments"][experiment],
            checkpoint_path=str(checkpoint),
            device=device,
            height=int(height),
            width=int(width),
        )

        model.eval()
        models[experiment] = model
        prediction_fns[experiment] = prediction_fn

    return models, prediction_fns


def run_inference(selected, dataset, indices, checkpoint_dir, width, height, eps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(
        REPO_DIR / "configs" / "experiments.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        experiments_config = yaml.safe_load(f)

    models, prediction_fns = load_models(
        checkpoint_dir,
        experiments_config,
        device,
        width,
        height,
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    results = []

    for example in selected:
        image_id = example["image_id"]
        sample = dataset[indices[image_id]]
        image_cpu = sample["image"]
        image_batch = image_cpu.unsqueeze(0).to(device)

        display_image = (
            (image_cpu * std + mean)
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .numpy()
        )

        predictions = {}
        g_alpha = None

        with torch.inference_mode():
            for experiment in MODELS:
                model = models[experiment]
                prediction_fn = prediction_fns[experiment]

                if experiment == "G":
                    prediction, alpha = model(image_batch, return_alpha=True)
                    g_alpha = float(alpha.reshape(-1)[0].detach().cpu())
                elif prediction_fn is None:
                    prediction = model(image_batch)
                else:
                    prediction = prediction_fn(
                        model,
                        {"image": image_batch},
                        device,
                    )

                prediction = normalize_probability_map(prediction, eps=eps)
                predictions[experiment] = (
                    prediction[0, 0].detach().cpu().numpy()
                )

        results.append({
            "label": example["label"],
            "image_id": image_id,
            "image": display_image,
            "ground_truth": sample["density_map_prob"].squeeze(0).numpy(),
            "predictions": predictions,
            "g_alpha": g_alpha,
        })

    for result in results:
        print(
            f"- {result['label']} | {result['image_id']} | "
            f"alpha G = {result['g_alpha']:.4f}"
        )

    return results


def save_figure(results, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    titles = ["Immagine", "Ground truth", "B0", "B1", "M1", "M1-L", "G", "M2"]

    fig, axes = plt.subplots(
        len(results),
        len(titles),
        figsize=(21, 3.2 * len(results)),
        squeeze=False,
    )

    for row_index, result in enumerate(results):
        maps = [
            result["ground_truth"],
            *(result["predictions"][model] for model in MODELS),
        ]
        vmax = max(float(saliency_map.max()) for saliency_map in maps)

        axes[row_index, 0].imshow(result["image"])
        axes[row_index, 0].set_ylabel(
            f"{result['label']}\n{result['image_id']}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
            labelpad=10,
        )

        axes[row_index, 1].imshow(
            result["ground_truth"],
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )

        for column_index, experiment in enumerate(MODELS, start=2):
            axes[row_index, column_index].imshow(
                result["predictions"][experiment],
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )

        g_column = 2 + MODELS.index("G")
        axes[row_index, g_column].set_xlabel(
            f"α = {result['g_alpha']:.3f}",
            fontsize=9,
        )

        for column_index, title in enumerate(titles):
            axis = axes[row_index, column_index]
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(title, fontsize=11)

    fig.suptitle("Confronto qualitativo sul tuning set", fontsize=14)
    plt.tight_layout(rect=[0.08, 0.0, 1.0, 0.97])
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura salvata in: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selected = select_examples(args.results_dir)

    print("Esempi selezionati:")
    for item in selected:
        print(f"- {item['label']}: {item['image_id']}")

    dataset, indices, width, height, eps = prepare_dataset(
        args.data_dir,
        selected,
    )

    results = run_inference(
        selected,
        dataset,
        indices,
        args.checkpoint_dir,
        width,
        height,
        eps,
    )

    save_figure(results, args.output)


if __name__ == "__main__":
    main()
