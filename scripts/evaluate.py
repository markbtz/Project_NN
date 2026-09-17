"""
Evaluation CLI condivisa per B0 e B1.

Prima versione:
- valuta sul tuning set per default;
- calcola CC / SIM / KLD per immagine;
- calcola opzionalmente la validation loss del modello;
- salva CSV per-image e JSON summary;
- supporta B0 tramite prediction_fn dedicata;
- protegge internal_test: richiede --final_evaluation.

Esempi:

    python scripts/evaluate.py \
        --experiment B1 \
        --checkpoint_path /content/checkpoints/B1_best.pt

    python scripts/evaluate.py \
        --experiment B0 \
        --checkpoint_path /content/checkpoints/B0_center_map.pt

L'internal test resta congelato durante lo sviluppo. Per usarlo serve
esplicitamente:

    --split internal_test --final_evaluation
"""

import argparse
import csv
import json
import os
import sys

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    REPO_ROOT,
)

from src.models.factory import build_model

from src.config_utils import (
    load_yaml_config,
    resolve_project_path,
)


from src.data.dataset import SaliconDataset
from src.evaluation import evaluate_model

from src.runtime import (
    get_device,
    set_seed,
)


DEFAULT_DATA_CONFIG = os.path.join(
    REPO_ROOT,
    "configs",
    "data.yaml",
)

DEFAULT_EXPERIMENTS_CONFIG = os.path.join(
    REPO_ROOT,
    "configs",
    "experiments.yaml",
)


def load_model_for_evaluation(
    experiment,
    experiment_config,
    checkpoint_path,
    device,
    height,
    width,
):
    """
    Costruisce il modello e carica il checkpoint.

    Ritorna:
        model,
        prediction_fn,
        loss_fn,
        loss_target_key,
        checkpoint_metadata
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint non trovato: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if experiment == "B0":
        model = build_model(
            "B0",
            experiment_config,
            height=height,
            width=width,
            pretrained=False,
        ).to(device)

        # train.py salva B0 direttamente come state_dict.
        # Supportiamo anche un eventuale formato wrapped futuro.
        if (
            isinstance(checkpoint, dict)
            and "model_state" in checkpoint
        ):
            model_state = checkpoint[
                "model_state"
            ]
        else:
            model_state = checkpoint

        model.load_state_dict(
            model_state
        )

        def b0_prediction_fn(
            model,
            batch,
            device,
        ):
            batch_size = batch[
                "image"
            ].shape[0]

            return model(
                batch_size
            )

        return (
            model,
            b0_prediction_fn,
            None,
            None,
            {},
        )

    if experiment == "B1":
        
        # Non serve scaricare nuovamente i pesi ImageNet:
        # il checkpoint contiene gia' tutto il model_state.
        model = build_model(
            "B1",
            experiment_config,
            height=height,
            width=width,
            pretrained=False,
        ).to(device)

        if (
            not isinstance(checkpoint, dict)
            or "model_state" not in checkpoint
        ):
            raise ValueError(
                "Checkpoint B1 non valido: "
                "manca 'model_state'."
            )

        checkpoint_experiment = checkpoint.get(
            "experiment"
        )

        if (
            checkpoint_experiment is not None
            and checkpoint_experiment != "B1"
        ):
            raise ValueError(
                "Checkpoint incompatibile: "
                f"experiment={checkpoint_experiment}"
            )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        loss_config = experiment_config[
            "loss"
        ]

        if loss_config["name"] != "mse":
            raise ValueError(
                "B1 supporta attualmente solo loss MSE, "
                f"ma experiments.yaml contiene: "
                f"{loss_config['name']}"
            )

        checkpoint_metadata = {
            "epoch": checkpoint.get(
                "epoch"
            ),
            "best_score": checkpoint.get(
                "best_score"
            ),
            "selection_metric": checkpoint.get(
                "selection_metric"
            ),
            "selection_mode": checkpoint.get(
                "selection_mode"
            ),
        }

        return (
            model,
            None,
            nn.MSELoss(),
            experiment_config[
                "target"
            ],
            checkpoint_metadata,
        )

    raise ValueError(
        f"Esperimento non supportato: {experiment}"
    )


def save_evaluation_results(
    result,
    *,
    experiment,
    split,
    checkpoint_path,
    checkpoint_metadata,
    results_dir,
):
    """
    Salva:
        results/<experiment>/<split>_per_image.csv
        results/<experiment>/<split>_summary.json
    """

    experiment_dir = os.path.join(
        results_dir,
        experiment,
    )

    os.makedirs(
        experiment_dir,
        exist_ok=True,
    )

    per_image_path = os.path.join(
        experiment_dir,
        f"{split}_per_image.csv",
    )

    summary_path = os.path.join(
        experiment_dir,
        f"{split}_summary.json",
    )

    rows = result[
        "per_sample"
    ]

    with open(
        per_image_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "cc",
                "sim",
                "kld",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    summary_payload = {
        "experiment": experiment,
        "split": split,
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint_metadata,
        **result["summary"],
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary_payload,
            f,
            indent=2,
        )

    return (
        per_image_path,
        summary_path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=[
            "B0",
            "B1",
        ],
        required=True,
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help=(
            "Checkpoint da valutare. "
            "Per B0: B0_center_map.pt. "
            "Per B1: preferibilmente B1_best.pt."
        ),
    )

    parser.add_argument(
        "--split",
        choices=[
            "tuning",
            "internal_test",
        ],
        default="tuning",
    )

    parser.add_argument(
        "--final_evaluation",
        action="store_true",
        help=(
            "Autorizza esplicitamente l'uso di internal_test. "
            "Non usare durante sviluppo/tuning."
        ),
    )

    parser.add_argument(
        "--data_config",
        type=str,
        default=DEFAULT_DATA_CONFIG,
    )

    parser.add_argument(
        "--experiments_config",
        type=str,
        default=DEFAULT_EXPERIMENTS_CONFIG,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Protezione internal test
    # -----------------------------------------------------

    if (
        args.split == "internal_test"
        and not args.final_evaluation
    ):
        parser.error(
            "internal_test e' congelato durante lo sviluppo. "
            "Per una valutazione finale esplicita usare "
            "--final_evaluation."
        )

    # -----------------------------------------------------
    # Configurazioni
    # -----------------------------------------------------

    data_config = load_yaml_config(
        args.data_config
    )

    experiments_config = load_yaml_config(
        args.experiments_config
    )

    training_config = experiments_config[
        "training"
    ]

    experiment_config = experiments_config[
        "experiments"
    ][
        args.experiment
    ]

    seed = int(
        data_config["seed"]
    )

    width, height = data_config[
        "input_size"
    ]

    width = int(width)
    height = int(height)

    density_map_epsilon = float(
        data_config[
            "density_map_epsilon"
        ]
    )

    if args.data_dir is None:
        colab_cache = data_config.get(
            "colab_local_cache_dir"
        )

        if (
            colab_cache
            and os.path.isdir(
                colab_cache
            )
        ):
            args.data_dir = colab_cache

        else:
            args.data_dir = resolve_project_path(
                data_config[
                    "dataset_root"
                ]
            )

    if args.manifest_path is None:
        args.manifest_path = resolve_project_path(
            data_config[
                "manifest_path"
            ]
        )

    if args.batch_size is None:
        args.batch_size = int(
            training_config[
                "batch_size"
            ]
        )

    if args.num_workers is None:
        args.num_workers = int(
            training_config.get(
                "num_workers",
                0,
            )
        )

    checkpoint_path = (
        args.checkpoint_path
        if os.path.isabs(
            args.checkpoint_path
        )
        else resolve_project_path(
            args.checkpoint_path
        )
    )

    results_dir = resolve_project_path(
        args.results_dir
    )

    # -----------------------------------------------------
    # Riproducibilita' / device
    # -----------------------------------------------------

    set_seed(
        seed
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Experiment: {args.experiment}"
    )

    print(
        f"Split: {args.split}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        f"Dataset: {args.data_dir}"
    )

    print(
        f"Manifest: {args.manifest_path}"
    )

    print(
        f"Input size: {width}x{height}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = SaliconDataset(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        split=args.split,
        input_size=(
            width,
            height,
        ),
        density_map_epsilon=density_map_epsilon,
        augmentation=False,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # -----------------------------------------------------
    # Modello / checkpoint
    # -----------------------------------------------------

    (
        model,
        prediction_fn,
        loss_fn,
        loss_target_key,
        checkpoint_metadata,
    ) = load_model_for_evaluation(
        experiment=args.experiment,
        experiment_config=experiment_config,
        checkpoint_path=checkpoint_path,
        device=device,
        height=height,
        width=width,
    )

    # -----------------------------------------------------
    # Evaluation
    #
    # Prima versione: CC / SIM / KLD.
    # NSS e sAUC verranno aggiunti quando verra' fissato
    # il protocollo delle fixation negative per sAUC.
    # -----------------------------------------------------

    result = evaluate_model(
        model,
        data_loader,
        device,
        prediction_fn=prediction_fn,
        loss_fn=loss_fn,
        loss_target_key=loss_target_key,
        metric_target_key="density_map_prob",
        eps=density_map_epsilon,
        collect_per_sample=True,
    )

    summary = result[
        "summary"
    ]

    print()
    print(
        f"Samples: {summary['n_samples']}"
    )

    if summary[
        "loss"
    ] is not None:
        print(
            f"Loss: {summary['loss']:.6e}"
        )

    print(
        f"CC: {summary['cc']:.6f}"
    )

    print(
        f"SIM: {summary['sim']:.6f}"
    )

    print(
        f"KLD: {summary['kld']:.6f}"
    )

    (
        per_image_path,
        summary_path,
    ) = save_evaluation_results(
        result,
        experiment=args.experiment,
        split=args.split,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        results_dir=results_dir,
    )

    print()
    print(
        f"Per-image CSV: {per_image_path}"
    )

    print(
        f"Summary JSON: {summary_path}"
    )


if __name__ == "__main__":
    main()
