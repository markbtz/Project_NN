"""
Training loop for B0/B1/M1 (see configs/experiments.yaml).

Training uses the real SALICON dataset through SaliconDataset.

B1 uses:
    density_map_raw
    values in [0, 1]
    MSE loss

B0 uses:
    density_map_prob
    normalized to sum to 1

Shared configuration is loaded from:
    configs/data.yaml
    configs/experiments.yaml

Usage:
    python scripts/train.py --experiment B1
    python scripts/train.py --experiment B0

CLI arguments --epochs, --batch_size, --height, --width,
--data_dir, --manifest_path, and --dev_subset remain available
as optional overrides.

On Colab, set --checkpoint_dir to a Drive path, e.g.:
    python scripts/train.py --experiment B1 \
        --checkpoint_dir /content/drive/MyDrive/nndl-saliency/checkpoints
"""

import argparse
import os
import sys
import time

from functools import partial

import torch
import torch.nn as nn
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

from src.losses.saliency_losses import cc_kld_loss

from src.training_monitoring import (
    EarlyStopping,
    TrainingHistory,
)

from src.checkpoints import (
    is_better,
    load_training_checkpoint,
    save_training_checkpoint,
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


def load_g_base_state_dict(checkpoint_path, device):
    """Load a base checkpoint only if it is explicitly marked as M1-L."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("experiment") != "M1-L"
    ):
        found = (
            checkpoint.get("experiment")
            if isinstance(checkpoint, dict)
            else None
        )
        raise ValueError(
            "Incompatible base checkpoint for G: "
            f"expected experiment=M1-L, found experiment={found}."
        )

    if not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError(
            "The M1-L checkpoint does not contain a valid 'model_state'."
        )

    return checkpoint["model_state"]


def train_mse_model(args, device, experiment_config):
    if args.experiment not in ("B1", "M1", "M1-L", "G", "M2"):
        raise ValueError(
            f"Unsupported experiment: {args.experiment}"
        )

    model = build_model(
        args.experiment,
        experiment_config,
        height=args.height,
        width=args.width,
        pretrained=args.pretrained,
    ).to(device)

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"{args.experiment}_last.pt",
    )

    # -----------------------------------------------------
    # Special initialization for G.
    #
    # If G_last.pt does not exist yet:
    # - load the best M1-L checkpoint
    # - load the B0 center prior
    #
    # If G_last.pt exists, normal resume restores
    # the complete G state.
    # -----------------------------------------------------
    if (
        args.experiment == "G"
        and not os.path.isfile(checkpoint_path)
    ):
        if args.base_checkpoint is None:
            raise ValueError(
                "For the first G training run, specify "
                "--base_checkpoint con M1-L_best.pt."
            )

        if args.center_prior_checkpoint is None:
            raise ValueError(
                "For the first G training run, specify "
                "--center_prior_checkpoint con B0_center_map.pt."
            )

        if not os.path.isfile(args.base_checkpoint):
            raise FileNotFoundError(
                f"M1-L checkpoint not found: "
                f"{args.base_checkpoint}"
            )

        if not os.path.isfile(
            args.center_prior_checkpoint
        ):
            raise FileNotFoundError(
                f"B0 checkpoint not found: "
                f"{args.center_prior_checkpoint}"
            )

        model.load_base_state_dict(
            load_g_base_state_dict(
                args.base_checkpoint,
                device,
            )
        )

        center_prior_state = torch.load(
            args.center_prior_checkpoint,
            map_location=device,
        )

        model.load_center_prior_state_dict(
            center_prior_state
        )

        print(
            f"[G] M1-L caricato da: "
            f"{args.base_checkpoint}"
        )

        print(
            f"[G] B0 caricato da: "
            f"{args.center_prior_checkpoint}"
        )

    if args.experiment == "G":
        model.freeze_base_and_prior()

    if args.optimizer_name != "AdamW":
        raise ValueError(
            f"{args.experiment} currently supports only the AdamW optimizer, "
            f"ma experiments.yaml contiene: {args.optimizer_name}"
        )

    if args.experiment == "G":
        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        trainable_count = sum(
            parameter.numel()
            for parameter in trainable_parameters
        )

        print(
            f"[G] Parametri allenabili: "
            f"{trainable_count}"
        )

    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    if args.loss_name == "mse":
        criterion = nn.MSELoss()

    elif args.loss_name == "cc_kld":
        criterion = partial(
        cc_kld_loss,
        cc_weight=args.cc_weight,
        kld_weight=args.kld_weight,
        eps=args.density_map_epsilon,
    )

    else:
        raise ValueError(
            f"Unsupported loss for {args.experiment}:"
            f"{args.loss_name}"
        )

    # -----------------------------------------------------
    # Dataset reale SALICON
    #
    # The target is defined in configs/experiments.yaml.
    # B1 must use density_map_raw.
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
    # Optional development subset.
    #
    # With --dev_subset, B1 uses a deterministic
    # training subset. Its size is read from
    # read from data.yaml (dev_subset_n_train).
    #
    # Use a local generator with a fixed seed to avoid
    # changing the global RNG state used by training and
    # augmentation.
    # -----------------------------------------------------

    if args.dev_subset:
        if args.dev_subset_n_train > len(
            train_dataset
        ):
            raise ValueError(
                "dev_subset_n_train is greater than the number "
                "of samples available in the training set."
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
            f"[{args.experiment}] Development subset enabled: "
            f"{len(train_dataset)} samples"
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
    # No augmentation: tuning must be deterministic.
    # and reproducible across epochs and experiments.
    #
    # B1 validation loss uses density_map_raw (MSE),
    # while CC/SIM/KLD use density_map_prob.
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

    start_epoch, best_score = load_training_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        device=device,
        selection_metric=args.selection_metric,
        selection_mode=args.selection_mode,
    )

    early_stopping = EarlyStopping(
        enabled=args.early_stopping_enabled,
        patience=args.early_stopping_patience,
    )

    history = TrainingHistory()

    history_path = os.path.join(
        args.checkpoint_dir,
        f"{args.experiment}_training_history.csv",
    )

    loss_plot_path = os.path.join(
        args.checkpoint_dir,
        f"{args.experiment}_training_loss.png",
    )

    history.load_csv(
        history_path
    )

    early_stopping.epochs_without_improvement = (
        history.consecutive_non_improving_epochs(
            args.selection_mode
        )
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
            f"[{args.experiment}] Epoch "
            f"{epoch + 1}/{args.epochs} "
            f"- train loss ({args.loss_name}): "
            f"{epoch_loss:.6e} "
            f"- {time.time() - t0:.1f}s"
        )

        # -------------------------------------------------
        # Validation on tuning set
        #
        # Loss:
        #   prediction raw vs density_map_raw -> MSE
        #
        # Metriche:
        #   prediction raw vs density_map_prob
        #   -> CC / SIM / KLD
        #
        # evaluate_model uses reduction="none" and aggregates
        # correctly on a per-image basis.
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
            f"[{args.experiment}] Tuning "
            f"- Loss ({args.loss_name}): "
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
                "selection_metric is not available in the summary: "
                f"{args.selection_metric}"
            )

        current_score = validation_summary[
            args.selection_metric
        ]

        if current_score is None:
            raise ValueError(
                "The selected selection_metric has no value: "
                f"{args.selection_metric}"
            )

        history.add_epoch(
            epoch=epoch + 1,
            train_loss=epoch_loss,
            validation_loss=validation_summary["loss"],
            selection_metric=args.selection_metric,
            selection_value=current_score,
        )

        history.save_csv(
            history_path
        )

        history.save_loss_plot(
            loss_plot_path
        )

        is_best = is_better(
            current_score,
            best_score,
            args.selection_mode,
        )

        if is_best:
            best_score = current_score

        save_training_checkpoint(
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
                f"{args.experiment}_best.pt",
            )

            save_training_checkpoint(
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
                f"[{args.experiment}] New best checkpoint "
                f"- {args.selection_metric}: "
                f"{best_score:.6f}"
            )

        should_stop = early_stopping.step(
        improved=is_best,
        )

        if should_stop:
            print(
                f"[{args.experiment}] Early stopping after "
                f"{early_stopping.epochs_without_improvement} "
                f"epochs without improvement."
            )
            break

    print(
        f"Training {args.experiment} completed. "
        f"Checkpoint in: {args.checkpoint_dir}"
    )


def fit_b0(args, device, experiment_config):
    """
    B0 is not trained via backpropagation.

    Compute the center prior as the mean training-set density_map_prob using
    streaming accumulation (fit_from_loader). With 10,000 images, keeping all
    density maps in memory would require about 1.9 GB for that tensor alone.
    fit_from_loader accumulates the sum batch by batch without loading the
    entire dataset into RAM at once.
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
        f"B0 (center prior) computed from "
        f"{len(train_dataset)} maps "
        f"e salvato in {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["B0", "B1", "M1", "M1-L", "G", "M2"],
        required=True,
    )

    parser.add_argument(
        "--cc_weight",
        type=float,
        default=None,
        help="Optional CC-weight override for the CC+KLD loss.",
    )

    parser.add_argument(
        "--kld_weight",
        type=float,
        default=None,
        help="Optional KLD-weight override for the CC+KLD loss.",
    )

    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default=None,
        help=(
            "Best checkpoint of the M1-L base model "
            "used to initialize G."
        ),
    )

    parser.add_argument(
        "--center_prior_checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint B0_center_map.pt "
            "used to initialize G."
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

    # Optional CLI overrides.
    # If omitted, values are read from the YAML files.
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
            "On Colab, provide a Drive path, e.g. "
            "/content/drive/MyDrive/"
            "nndl-saliency/checkpoints"
        ),
    )

    parser.add_argument(
        "--dev_subset",
        action="store_true",
        help=(
            "Use the development subset of the training set. "
            "Its size is read from "
            "data.yaml -> dev_subset_n_train."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load YAML configuration.
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
            "selection_mode must be either 'max' or 'min'."
        )

    if args.selection_metric not in (
        "loss",
        "cc",
        "sim",
        "kld",
    ):
        raise ValueError(
            "selection_metric must be one of: "
            "loss, cc, sim, kld."
        )

    args.num_workers = int(
        training_config.get(
            "num_workers",
            0,
        )
    )

    early_stopping_config = training_config.get(
    "early_stopping",
    {},
    )

    args.early_stopping_enabled = bool(
        early_stopping_config.get(
            "enabled",
            False,
        )
    )

    args.early_stopping_patience = int(
        early_stopping_config.get(
            "patience",
            5,
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
    # experiments.yaml -> selected experiment
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

    if args.loss_name == "cc_kld":

        if args.cc_weight is None:
            args.cc_weight = loss_config.get(
                "cc_weight"
            )

        if args.kld_weight is None:
            args.kld_weight = loss_config.get(
                "kld_weight"
            )

        if (
            args.cc_weight is None
            or args.kld_weight is None
        ):
            raise ValueError(
                "For the cc_kld loss, specify "
                "cc_weight and kld_weight in the YAML "
                "oppure tramite CLI."
            )

        args.cc_weight = float(
            args.cc_weight
        )

        args.kld_weight = float(
            args.kld_weight
        )

    if args.experiment in ("B1", "M1", "M1-L", "M2"):
        args.pretrained = bool(
            experiment_config[
                "encoder"
            ][
                "pretrained"
            ]
        )

    elif args.experiment == "G":
        # G receives weights from the best M1-L checkpoint.
        # No need to download another pretrained ResNet.
        args.pretrained = False


    # -----------------------------------------------------
    # Protect the B0 protocol.
    # -----------------------------------------------------

    if (
        args.experiment == "B0"
        and args.dev_subset
    ):
        raise ValueError(
            "--dev_subset is not supported for B0: "
            "the center prior must be computed on the full training set."
        )

    # -----------------------------------------------------
    # Reproducibility and summary
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

    if args.experiment in ("B1", "M1", "M1-L", "G", "M2"):
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
            f"Early stopping: "
            f"{'yes' if args.early_stopping_enabled else 'no'}"
        )

        print(
            f"Early stopping patience: "
            f"{args.early_stopping_patience}"
        )

        print(
            f"Loss: "
            f"{args.loss_name}"
        )

        if args.loss_name == "cc_kld":
            print(
                f"CC weight: {args.cc_weight}"
            )

            print(
                f"KLD weight: {args.kld_weight}"
            )

        if args.experiment == "G":
            print(
                f"Base checkpoint: "
                f"{args.base_checkpoint}"
            )

            print(
                f"Center prior checkpoint: "
                f"{args.center_prior_checkpoint}"
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

    if args.experiment in ("B1", "M1", "M1-L", "G", "M2"):
        train_mse_model(
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
