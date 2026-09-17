from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Permette di importare src quando eseguiamo pytest
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import evaluate_model
from src.metrics import cc, sim, kld


class DictSaliencyDataset(Dataset):
    """
    Dataset minimale per testare l'evaluator senza SALICON reale.
    """

    def __init__(
        self,
        predictions,
        metric_targets,
        raw_targets=None,
    ):
        self.predictions = predictions
        self.metric_targets = metric_targets

        if raw_targets is None:
            raw_targets = metric_targets

        self.raw_targets = raw_targets

    def __len__(self):
        return self.predictions.shape[0]

    def __getitem__(self, index):
        return {
            "image": self.predictions[index],
            "density_map_prob": self.metric_targets[index],
            "density_map_raw": self.raw_targets[index],
            "image_id": f"img_{index}",
        }


class IdentityModel(nn.Module):
    """
    Usa direttamente batch['image'] come prediction.
    """

    def forward(self, x):
        return x


class StateTrackingModel(nn.Module):
    """
    Registra se il forward e' stato eseguito in eval
    e dentro torch.inference_mode().
    """

    def __init__(self):
        super().__init__()
        self.forward_training_state = None
        self.inference_mode_enabled = None

    def forward(self, x):
        self.forward_training_state = self.training
        self.inference_mode_enabled = (
            torch.is_inference_mode_enabled()
        )
        return x


class DummyB0(nn.Module):
    """
    Modello con la stessa idea di interfaccia di CenterPriorB0:
    forward riceve la batch size, non le immagini.
    """

    def __init__(self, center_map):
        super().__init__()

        self.register_buffer(
            "center_map",
            center_map.clone(),
        )

    def forward(self, batch_size):
        return self.center_map.expand(
            batch_size,
            -1,
            -1,
            -1,
        )


def test_identical_maps_have_ideal_metrics():
    target = torch.tensor(
        [
            [[[0.1, 0.2], [0.3, 0.4]]],
            [[[0.4, 0.1], [0.2, 0.3]]],
        ],
        dtype=torch.float32,
    )

    dataset = DictSaliencyDataset(
        predictions=target.clone(),
        metric_targets=target,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    model = IdentityModel()

    result = evaluate_model(
        model,
        loader,
        torch.device("cpu"),
    )

    summary = result["summary"]

    assert summary["n_samples"] == 2

    assert abs(
        summary["cc"] - 1.0
    ) < 1e-4

    assert abs(
        summary["sim"] - 1.0
    ) < 1e-5

    assert abs(
        summary["kld"] - 0.0
    ) < 1e-5


def test_collect_per_sample_returns_one_row_per_image_id():
    target = torch.tensor(
        [
            [[[0.1, 0.2], [0.3, 0.4]]],
            [[[0.4, 0.1], [0.2, 0.3]]],
            [[[0.2, 0.4], [0.1, 0.3]]],
        ],
        dtype=torch.float32,
    )

    dataset = DictSaliencyDataset(
        predictions=target.clone(),
        metric_targets=target,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    result = evaluate_model(
        IdentityModel(),
        loader,
        torch.device("cpu"),
        collect_per_sample=True,
    )

    rows = result["per_sample"]

    assert len(rows) == 3

    assert [
        row["image_id"]
        for row in rows
    ] == [
        "img_0",
        "img_1",
        "img_2",
    ]

    for row in rows:
        assert set(row.keys()) == {
            "image_id",
            "cc",
            "sim",
            "kld",
        }


def test_aggregation_is_correct_with_uneven_batch_sizes():
    target = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[0.0, 1.0], [2.0, 3.0]]],
        ],
        dtype=torch.float32,
    )

    prediction = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[3.0, 2.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [1.0, 2.0]]],
            [[[0.0, 0.0], [1.0, 3.0]]],
            [[[0.0, 1.0], [2.0, 3.0]]],
        ],
        dtype=torch.float32,
    )

    dataset = DictSaliencyDataset(
        predictions=prediction,
        metric_targets=target,
    )

    # 5 campioni -> batch 2 + 2 + 1
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    result = evaluate_model(
        IdentityModel(),
        loader,
        torch.device("cpu"),
    )

    summary = result["summary"]

    expected_cc = cc(
        prediction,
        target,
        eps=1e-6,
    ).item()

    expected_sim = sim(
        prediction,
        target,
        eps=1e-6,
    ).item()

    expected_kld = kld(
        prediction,
        target,
        eps=1e-6,
    ).item()

    assert summary["n_samples"] == 5

    assert abs(
        summary["cc"] - expected_cc
    ) < 1e-6

    assert abs(
        summary["sim"] - expected_sim
    ) < 1e-6

    assert abs(
        summary["kld"] - expected_kld
    ) < 1e-6


def test_collect_per_sample_false_does_not_store_rows():
    target = torch.tensor(
        [
            [[[0.1, 0.2], [0.3, 0.4]]],
            [[[0.4, 0.1], [0.2, 0.3]]],
        ],
        dtype=torch.float32,
    )

    dataset = DictSaliencyDataset(
        predictions=target.clone(),
        metric_targets=target,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    result = evaluate_model(
        IdentityModel(),
        loader,
        torch.device("cpu"),
        collect_per_sample=False,
    )

    assert result["per_sample"] == []


def test_custom_prediction_fn_supports_b0_style_model():
    center_map = torch.tensor(
        [[[[0.1, 0.2], [0.3, 0.4]]]],
        dtype=torch.float32,
    )

    metric_targets = center_map.repeat(
        3,
        1,
        1,
        1,
    )

    dummy_images = torch.zeros(
        3,
        1,
        2,
        2,
        dtype=torch.float32,
    )

    dataset = DictSaliencyDataset(
        predictions=dummy_images,
        metric_targets=metric_targets,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    model = DummyB0(
        center_map=center_map,
    )

    def b0_prediction_fn(
        model,
        batch,
        device,
    ):
        batch_size = batch["image"].shape[0]

        return model(
            batch_size
        )

    result = evaluate_model(
        model,
        loader,
        torch.device("cpu"),
        prediction_fn=b0_prediction_fn,
    )

    summary = result["summary"]

    assert summary["n_samples"] == 3
    assert abs(summary["cc"] - 1.0) < 1e-4
    assert abs(summary["sim"] - 1.0) < 1e-5
    assert abs(summary["kld"] - 0.0) < 1e-5


def test_model_state_is_restored_after_evaluation():
    target = torch.tensor(
        [
            [[[0.1, 0.2], [0.3, 0.4]]],
        ],
        dtype=torch.float32,
    )

    dataset = DictSaliencyDataset(
        predictions=target.clone(),
        metric_targets=target,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    model = StateTrackingModel()
    model.train()

    assert model.training is True

    evaluate_model(
        model,
        loader,
        torch.device("cpu"),
    )

    # Durante evaluation il modello deve essere in eval().
    assert model.forward_training_state is False

    # Il forward deve avvenire dentro inference_mode().
    assert model.inference_mode_enabled is True

    # Al termine deve tornare allo stato originale.
    assert model.training is True


def test_validation_loss_is_weighted_by_number_of_samples():
    prediction = torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[2.0, 2.0], [2.0, 2.0]]],
            [[[3.0, 3.0], [3.0, 3.0]]],
            [[[4.0, 4.0], [4.0, 4.0]]],
        ],
        dtype=torch.float32,
    )

    # Target metriche non costante, per evitare casi degeneri di CC.
    metric_target = torch.tensor(
        [[[[0.1, 0.2], [0.3, 0.4]]]],
        dtype=torch.float32,
    ).repeat(
        5,
        1,
        1,
        1,
    )

    raw_target = torch.zeros_like(
        prediction
    )

    dataset = DictSaliencyDataset(
        predictions=prediction,
        metric_targets=metric_target,
        raw_targets=raw_target,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    def mse_loss(pred, target):
        return torch.mean(
            (pred - target) ** 2
        )

    result = evaluate_model(
        IdentityModel(),
        loader,
        torch.device("cpu"),
        loss_fn=mse_loss,
        loss_target_key="density_map_raw",
    )

    expected_loss = torch.mean(
        (prediction - raw_target) ** 2
    ).item()

    assert abs(
        result["summary"]["loss"]
        - expected_loss
    ) < 1e-6
