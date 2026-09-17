"""
Training loop per B0/B1 (vedi configs/experiments.yaml).

Il training utilizza il dataset reale SALICON tramite SaliconDataset.

B1 usa:
    density_map_raw
    valori in [0,1]
    loss MSE

B0 usa:
    density_map_prob
    normalizzata a somma 1

Le configurazioni condivise vengono lette da:
    configs/data.yaml
    configs/experiments.yaml

Uso:
    python scripts/train.py --experiment B1
    python scripts/train.py --experiment B0

Gli argomenti CLI --epochs, --batch_size, --height, --width,
--data_dir, --manifest_path e --dev_subset restano disponibili
come override opzionali.

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
import yaml
from torch.utils.data import DataLoader, Subset


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

from src.runtime import (
    get_device,
    set_seed,
)

from src.data.dataset import SaliconDataset
from src.evaluation import evaluate_model


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

def _initial_best_score(selection_mode):
    """
    Valore iniziale per la selezione del best checkpoint.
    """

    if selection_mode == "max":
        return float("-inf")

    if selection_mode == "min":
        return float("inf")

    raise ValueError(
        "selection_mode deve essere 'max' oppure 'min'."
    )


def _is_better(
    current_score,
    best_score,
    selection_mode,
):
    """
    Confronta lo score corrente con il best score secondo
    la direzione definita in experiments.yaml.
    """

    if selection_mode == "max":
        return current_score > best_score

    if selection_mode == "min":
        return current_score < best_score

    raise ValueError(
        "selection_mode deve essere 'max' oppure 'min'."
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_score,
    selection_metric,
    selection_mode,
    seed,
    experiment,
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),
            "best_score": best_score,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "seed": seed,
            "experiment": experiment,
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
    selection_metric="cc",
    selection_mode="max",
):
    """
    Ritorna (epoca_di_partenza, best_score).

    Se non c'e' checkpoint, riparte da zero usando il valore
    iniziale coerente con selection_mode.

    I vecchi checkpoint basati su best_loss non vengono riutilizzati:
    la selezione del best model ora avviene sul tuning set e quindi
    rappresenta un protocollo diverso.

    Fondamentale su Colab: una sessione puo'
    disconnettersi in qualunque momento, questo evita
    di ripartire da capo ogni volta.
    """

    initial_best_score = _initial_best_score(
        selection_mode
    )

    if not os.path.exists(path):
        return 0, initial_best_score

    ckpt = torch.load(
        path,
        map_location=device,
    )

    if "best_score" not in ckpt:
        raise ValueError(
            "Checkpoint legacy non compatibile: contiene best_loss "
            "ma non best_score. Rimuovere o rinominare B1_last.pt "
            "e ripartire con il nuovo protocollo di validazione."
        )

    if (
        ckpt.get("selection_metric")
        != selection_metric
    ):
        raise ValueError(
            "Il checkpoint usa una selection_metric diversa: "
            f"{ckpt.get('selection_metric')} != {selection_metric}."
        )

    if (
        ckpt.get("selection_mode")
        != selection_mode
    ):
        raise ValueError(
            "Il checkpoint usa una selection_mode diversa: "
            f"{ckpt.get('selection_mode')} != {selection_mode}."
        )

    model.load_state_dict(
        ckpt["model_state"]
    )

    if (
        optimizer is not None
        and ckpt.get("optimizer_state") is not None
    ):
        optimizer.load_state_dict(
            ckpt["optimizer_state"]
        )

    print(
        f"Checkpoint ripreso da {path} "
        f"(epoca {ckpt['epoch']}, "
        f"best {selection_metric} "
        f"{ckpt['best_score']:.6f})"
    )

    return (
        ckpt["epoch"],
        ckpt["best_score"],
    )


def train_b1(args, device, experiment_config):
    model = build_model(
        "B1",
        experiment_config,
        height=args.height,
        width=args.width,
        pretrained=args.pretrained,
    ).to(device)

    if args.optimizer_name != "AdamW":
        raise ValueError(
            "B1 supporta attualmente solo optimizer AdamW, "
            f"ma experiments.yaml contiene: {args.optimizer_name}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.loss_name != "mse":
        raise ValueError(
            "B1 supporta attualmente solo loss MSE, "
            f"ma experiments.yaml contiene: {args.loss_name}"
        )

    criterion = nn.MSELoss()

    # -----------------------------------------------------
    # Dataset reale SALICON
    #
    # Il target e' definito in configs/experiments.yaml.
    # Per B1 deve essere density_map_raw.
    # -----------------------------------------------------

    train_dataset = SaliconDataset(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        split=args.train_split,
        input_size=(
            args.width,
            args.height,
        ),
        density_map_epsilon=args.density_map_epsilon,
        augmentation=True,
    )

    # -----------------------------------------------------
    # Development subset opzionale
    #
    # Quando --dev_subset e' attivo, B1 usa un sottoinsieme
    # deterministico del training set. La dimensione viene
    # letta da data.yaml (dev_subset_n_train).
    #
    # Usiamo un generatore locale con seed fisso per non
    # alterare lo stato RNG globale usato da training e
    # augmentation.
    # -----------------------------------------------------

    if args.dev_subset:
        if args.dev_subset_n_train > len(
            train_dataset
        ):
            raise ValueError(
                "dev_subset_n_train e' maggiore del numero "
                "di campioni disponibili nel training set."
            )

        dev_generator = torch.Generator()
        dev_generator.manual_seed(
            args.seed
        )

        dev_indices = torch.randperm(
            len(train_dataset),
            generator=dev_generator,
        )[
            :args.dev_subset_n_train
        ].tolist()

        train_dataset = Subset(
            train_dataset,
            dev_indices,
        )

        print(
            f"[B1] Development subset attivo: "
            f"{len(train_dataset)} campioni"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # -----------------------------------------------------
    # Tuning set
    #
    # Nessuna augmentation: il tuning deve essere stabile
    # e riproducibile tra epoche ed esperimenti.
    #
    # La validation loss di B1 usa density_map_raw (MSE),
    # mentre CC/SIM/KLD usano density_map_prob.
    # -----------------------------------------------------

    tuning_dataset = SaliconDataset(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        split=args.validation_split,
        input_size=(
            args.width,
            args.height,
        ),
        density_map_epsilon=args.density_map_epsilon,
        augmentation=False,
    )

    tuning_loader = DataLoader(
        tuning_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        "B1_last.pt",
    )

    start_epoch, best_score = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        device=device,
        selection_metric=args.selection_metric,
        selection_mode=args.selection_mode,
    )

    for epoch in range(
        start_epoch,
        args.epochs,
    ):
        model.train()

        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:

            images = batch[
                "image"
            ].to(device)

            targets = batch[
                args.target_key
            ].to(device)

            optimizer.zero_grad()

            preds = model(
                images
            )

            loss = criterion(
                preds,
                targets,
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.item()
                * images.size(0)
            )

        epoch_loss /= len(
            train_dataset
        )

        print(
            f"[B1] Epoca "
            f"{epoch + 1}/{args.epochs} "
            f"- train MSE: "
            f"{epoch_loss:.6e} "
            f"- {time.time() - t0:.1f}s"
        )

        # -------------------------------------------------
        # Validazione sul tuning set
        #
        # Loss:
        #   prediction raw vs density_map_raw -> MSE
        #
        # Metriche:
        #   prediction raw vs density_map_prob
        #   -> CC / SIM / KLD
        #
        # evaluate_model usa reduction="none" e aggrega
        # correttamente per immagine.
        # -------------------------------------------------

        validation = evaluate_model(
            model,
            tuning_loader,
            device,
            loss_fn=criterion,
            loss_target_key=args.target_key,
            metric_target_key="density_map_prob",
            eps=args.density_map_epsilon,
            collect_per_sample=False,
        )

        validation_summary = validation[
            "summary"
        ]

        print(
            f"[B1] Tuning "
            f"- MSE: "
            f"{validation_summary['loss']:.6e} "
            f"- CC: "
            f"{validation_summary['cc']:.6f} "
            f"- SIM: "
            f"{validation_summary['sim']:.6f} "
            f"- KLD: "
            f"{validation_summary['kld']:.6f}"
        )

        if (
            args.selection_metric
            not in validation_summary
        ):
            raise ValueError(
                "selection_metric non disponibile nel summary: "
                f"{args.selection_metric}"
            )

        current_score = validation_summary[
            args.selection_metric
        ]

        if current_score is None:
            raise ValueError(
                "La selection_metric scelta non ha un valore: "
                f"{args.selection_metric}"
            )

        is_best = _is_better(
            current_score,
            best_score,
            args.selection_mode,
        )

        if is_best:
            best_score = current_score

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch + 1,
            best_score,
            args.selection_metric,
            args.selection_mode,
            args.seed,
            args.experiment,
        )

        if is_best:
            best_checkpoint_path = os.path.join(
                args.checkpoint_dir,
                "B1_best.pt",
            )

            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch + 1,
                best_score,
                args.selection_metric,
                args.selection_mode,
                args.seed,
                args.experiment,
            )

            print(
                f"[B1] Nuovo best checkpoint "
                f"- {args.selection_metric}: "
                f"{best_score:.6f}"
            )

    print(
        "Training B1 completato. "
        f"Checkpoint in: {args.checkpoint_dir}"
    )


def fit_b0(args, device, experiment_config):
    """
    B0 non si allena via backprop.

    Calcola il center prior come media delle density_map_prob del training
    set, in streaming (fit_from_loader): con 10.000 immagini, tenere tutte
    le density map insieme in memoria costa circa 1.9 GB solo per quel
    tensore. fit_from_loader accumula una somma incrementale, un batch alla
    volta, senza mai avere tutto il dataset in RAM contemporaneamente.
    """

    train_dataset = SaliconDataset(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        split=args.train_split,
        input_size=(
            args.width,
            args.height,
        ),
        density_map_epsilon=args.density_map_epsilon,
        augmentation=False,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
    )

    def density_batches():
        for batch in loader:
            yield batch[
                args.target_key
            ].to(device)

    model = build_model(
        "B0",
        experiment_config,
        height=args.height,
        width=args.width,
        pretrained=False,
    ).to(device)

    model.fit_from_loader(
        density_batches(),
        eps=args.density_map_epsilon,
    )

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        "B0_center_map.pt",
    )

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    print(
        f"B0 (center prior) calcolato su "
        f"{len(train_dataset)} mappe "
        f"e salvato in {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["B0", "B1"],
        required=True,
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

    # Override opzionali da CLI.
    # Se omessi, i valori vengono letti dai file YAML.
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
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
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help=(
            "Su Colab passare il path su Drive, es. "
            "/content/drive/MyDrive/"
            "nndl-saliency/checkpoints"
        ),
    )

    parser.add_argument(
        "--dev_subset",
        action="store_true",
        help=(
            "Usa il development subset del training set. "
            "La dimensione e' letta da "
            "data.yaml -> dev_subset_n_train."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Caricamento configurazioni YAML
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
    ][args.experiment]

    # -----------------------------------------------------
    # data.yaml
    # -----------------------------------------------------

    args.seed = int(
        data_config["seed"]
    )

    config_width, config_height = data_config[
        "input_size"
    ]

    if args.width is None:
        args.width = int(config_width)

    if args.height is None:
        args.height = int(config_height)

    args.density_map_epsilon = float(
        data_config["density_map_epsilon"]
    )

    args.dev_subset_n_train = int(
        data_config["dev_subset_n_train"]
    )

    if args.data_dir is None:
        colab_cache = data_config.get(
            "colab_local_cache_dir"
        )

        if (
            colab_cache
            and os.path.isdir(colab_cache)
        ):
            args.data_dir = colab_cache
        else:
            args.data_dir = resolve_project_path(
                data_config["dataset_root"]
            )

    if args.manifest_path is None:
        args.manifest_path = resolve_project_path(
            data_config["manifest_path"]
        )

    # -----------------------------------------------------
    # experiments.yaml -> training
    # -----------------------------------------------------

    args.train_split = training_config[
        "train_split"
    ]

    args.validation_split = training_config[
        "validation_split"
    ]

    args.selection_metric = training_config[
        "selection_metric"
    ]

    args.selection_mode = training_config[
        "selection_mode"
    ]

    if args.selection_mode not in (
        "max",
        "min",
    ):
        raise ValueError(
            "selection_mode deve essere 'max' oppure 'min'."
        )

    if args.selection_metric not in (
        "loss",
        "cc",
        "sim",
        "kld",
    ):
        raise ValueError(
            "selection_metric deve essere una tra: "
            "loss, cc, sim, kld."
        )

    args.num_workers = int(
        training_config.get(
            "num_workers",
            0,
        )
    )

    if args.batch_size is None:
        args.batch_size = int(
            training_config["batch_size"]
        )

    if args.epochs is None:
        args.epochs = int(
            training_config["epochs"]
        )

    optimizer_config = training_config[
        "optimizer"
    ]

    args.optimizer_name = optimizer_config[
        "name"
    ]

    args.learning_rate = float(
        optimizer_config[
            "learning_rate"
        ]
    )

    args.weight_decay = float(
        optimizer_config[
            "weight_decay"
        ]
    )

    # -----------------------------------------------------
    # experiments.yaml -> esperimento selezionato
    # -----------------------------------------------------

    args.target_key = experiment_config[
        "target"
    ]

    loss_config = experiment_config[
        "loss"
    ]

    args.loss_name = loss_config[
        "name"
    ]

    if args.experiment == "B1":
        args.pretrained = bool(
            experiment_config[
                "encoder"
            ][
                "pretrained"
            ]
        )


    # -----------------------------------------------------
    # Protezione del protocollo B0
    # -----------------------------------------------------

    if (
        args.experiment == "B0"
        and args.dev_subset
    ):
        raise ValueError(
            "--dev_subset non e' previsto per B0: "
            "il center prior va calcolato sul training set completo."
        )

    # -----------------------------------------------------
    # Riproducibilita' e riepilogo
    # -----------------------------------------------------

    set_seed(
        args.seed
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Seed: {args.seed}"
    )

    print(
        f"Data config: {args.data_config}"
    )

    print(
        f"Experiments config: "
        f"{args.experiments_config}"
    )

    print(
        f"Dataset: {args.data_dir}"
    )

    print(
        f"Manifest: {args.manifest_path}"
    )

    print(
        f"Split: {args.train_split}"
    )

    print(
        f"Input size: "
        f"{args.width}x{args.height}"
    )

    print(
        f"Epsilon: "
        f"{args.density_map_epsilon}"
    )

    print(
        f"Target: {args.target_key}"
    )

    if args.experiment == "B1":
        print(
            f"Optimizer: "
            f"{args.optimizer_name}"
        )

        print(
            f"Learning rate: "
            f"{args.learning_rate}"
        )

        print(
            f"Weight decay: "
            f"{args.weight_decay}"
        )

        print(
            f"Batch size: "
            f"{args.batch_size}"
        )

        print(
            f"Development subset: "
            f"{'yes' if args.dev_subset else 'no'}"
        )

        if args.dev_subset:
            print(
                f"Development subset size: "
                f"{args.dev_subset_n_train}"
            )

        print(
            f"Epochs: "
            f"{args.epochs}"
        )

        print(
            f"Loss: "
            f"{args.loss_name}"
        )

        print(
            f"Validation split: "
            f"{args.validation_split}"
        )

        print(
            f"Checkpoint selection: "
            f"{args.selection_metric} "
            f"({args.selection_mode})"
        )

    if args.experiment == "B1":
        train_b1(
            args,
            device,
            experiment_config,
        )

    else:
        fit_b0(
            args,
            device,
            experiment_config,
        )


if __name__ == "__main__":
    main()