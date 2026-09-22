import csv
import json
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

import pytest

from scripts import evaluate as evaluate_script
from scripts.evaluate import save_evaluation_results
from src.evaluation import evaluate_model


@pytest.mark.parametrize("experiment", ["B1", "M1", "M1-L"])
def test_save_evaluation_results_writes_csv_and_json(
    tmp_path,
    experiment,
):
    result = {
        "summary": {
            "loss": None if experiment == "M1-L" else 0.01,
            "cc": 0.80,
            "sim": 0.70,
            "kld": 0.30,
            "n_samples": 2,
        },
        "per_sample": [
            {
                "image_id": "img_1",
                "cc": 0.81,
                "sim": 0.71,
                "kld": 0.29,
            },
            {
                "image_id": "img_2",
                "cc": 0.79,
                "sim": 0.69,
                "kld": 0.31,
            },
        ],
    }

    per_image_path, summary_path = (
        save_evaluation_results(
            result,
            experiment=experiment,
            split="tuning",
            checkpoint_path=f"{experiment}_best.pt",
            checkpoint_metadata={
                "epoch": 3,
                "best_score": 0.80,
            },
            results_dir=str(tmp_path),
        )
    )

    with open(
        per_image_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 2

    assert [
        row["image_id"]
        for row in rows
    ] == [
        "img_1",
        "img_2",
    ]

    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    assert summary["experiment"] == experiment
    assert summary["split"] == "tuning"
    assert summary["n_samples"] == 2
    assert summary["checkpoint"]["epoch"] == 3

@pytest.mark.parametrize("experiment", ["B1", "M1", "M1-L"])
def test_internal_test_requires_final_evaluation(
    monkeypatch,
    capsys,
    experiment,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--experiment",
            experiment,
            "--checkpoint_path",
            "dummy.pt",
            "--split",
            "internal_test",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        evaluate_script.main()

    assert exc_info.value.code == 2

    captured = capsys.readouterr()

    assert "--final_evaluation" in captured.err


class TinySaliencyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, images):
        return torch.sigmoid(self.conv(images))


@pytest.mark.parametrize("experiment", ["M1", "M1-L"])
def test_multiscale_checkpoint_evaluation_writes_one_row_per_image(
    tmp_path,
    monkeypatch,
    experiment,
):
    checkpoint_path = tmp_path / f"{experiment}_best.pt"
    torch.save(
        {
            "model_state": TinySaliencyModel().state_dict(),
            "experiment": experiment,
            "epoch": 3,
            "best_score": 0.8,
            "selection_metric": "cc",
            "selection_mode": "max",
        },
        checkpoint_path,
    )

    factory_calls = []

    def fake_build_model(
        experiment,
        experiment_config,
        *,
        height,
        width,
        pretrained,
    ):
        factory_calls.append(
            (experiment, height, width, pretrained)
        )
        return TinySaliencyModel()

    monkeypatch.setattr(
        evaluate_script,
        "build_model",
        fake_build_model,
    )

    experiment_config = {
        "target": (
            "density_map_prob"
            if experiment == "M1-L"
            else "density_map_raw"
        ),
        "loss": {
            "name": "cc_kld" if experiment == "M1-L" else "mse"
        },
    }
    (
        model,
        prediction_fn,
        loss_fn,
        loss_target_key,
        checkpoint_metadata,
    ) = evaluate_script.load_model_for_evaluation(
        experiment=experiment,
        experiment_config=experiment_config,
        checkpoint_path=str(checkpoint_path),
        device=torch.device("cpu"),
        height=2,
        width=2,
    )

    assert factory_calls == [(experiment, 2, 2, False)]
    assert prediction_fn is None
    if experiment == "M1-L":
        assert loss_fn is None
        assert loss_target_key is None
    else:
        assert isinstance(loss_fn, nn.MSELoss)
        assert loss_target_key == "density_map_raw"

    targets = torch.tensor(
        [
            [[[0.1, 0.2], [0.3, 0.4]]],
            [[[0.4, 0.1], [0.2, 0.3]]],
            [[[0.2, 0.4], [0.1, 0.3]]],
        ],
        dtype=torch.float32,
    )
    samples = [
        {
            "image": torch.rand(3, 2, 2),
            "density_map_raw": targets[index],
            "density_map_prob": targets[index],
            "image_id": f"img_{index}",
        }
        for index in range(3)
    ]

    result = evaluate_model(
        model,
        DataLoader(samples, batch_size=2, shuffle=False),
        torch.device("cpu"),
        prediction_fn=prediction_fn,
        loss_fn=loss_fn,
        loss_target_key=loss_target_key,
        collect_per_sample=True,
    )
    per_image_path, summary_path = (
        save_evaluation_results(
            result,
            experiment=experiment,
            split="tuning",
            checkpoint_path=str(checkpoint_path),
            checkpoint_metadata=checkpoint_metadata,
            results_dir=str(tmp_path / "results"),
        )
    )

    with open(
        per_image_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    assert [row["image_id"] for row in rows] == [
        "img_0",
        "img_1",
        "img_2",
    ]
    assert summary["experiment"] == experiment
    assert summary["n_samples"] == len(rows) == 3
    if experiment == "M1-L":
        assert summary["loss"] is None
    else:
        assert summary["loss"] is not None


@pytest.mark.parametrize(
    ("experiment", "checkpoint_experiment"),
    [
        ("M1", "B1"),
        ("M1", "M1-L"),
        ("M1", None),
        ("M1-L", "M1"),
        ("M1-L", "B1"),
        ("M1-L", None),
    ],
)
def test_multiscale_rejects_wrong_or_missing_experiment_metadata(
    tmp_path,
    experiment,
    checkpoint_experiment,
):
    checkpoint_path = tmp_path / "wrong_checkpoint.pt"
    checkpoint = {"model_state": {}}
    if checkpoint_experiment is not None:
        checkpoint["experiment"] = checkpoint_experiment
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="Checkpoint incompatibile"):
        evaluate_script.load_model_for_evaluation(
            experiment=experiment,
            experiment_config={
                "target": (
                    "density_map_prob"
                    if experiment == "M1-L"
                    else "density_map_raw"
                ),
                "loss": {
                    "name": (
                        "cc_kld" if experiment == "M1-L" else "mse"
                    )
                },
            },
            checkpoint_path=str(checkpoint_path),
            device=torch.device("cpu"),
            height=192,
            width=256,
        )


@pytest.mark.parametrize(
    ("target", "loss_name"),
    [
        ("density_map_prob", "mse"),
        ("density_map_raw", "cc_kld"),
    ],
)
def test_m1_l_rejects_wrong_loss_or_target(
    tmp_path,
    target,
    loss_name,
):
    checkpoint_path = tmp_path / "M1-L_best.pt"
    torch.save(
        {"model_state": {}, "experiment": "M1-L"},
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="M1-L richiede"):
        evaluate_script.load_model_for_evaluation(
            experiment="M1-L",
            experiment_config={
                "target": target,
                "loss": {"name": loss_name},
            },
            checkpoint_path=str(checkpoint_path),
            device=torch.device("cpu"),
            height=192,
            width=256,
        )


def test_b1_rejects_checkpoint_from_other_experiment(
    tmp_path,
):
    checkpoint_path = tmp_path / "wrong_checkpoint.pt"

    torch.save(
        {
            "model_state": {},
            "experiment": "M1",
        },
        checkpoint_path,
    )

    experiment_config = {
        "decoder": {
            "width": 96,
        },
        "target": "density_map_raw",
        "loss": {
            "name": "mse",
        },
    }

    with pytest.raises(
        ValueError,
        match="Checkpoint incompatibile",
    ):
        evaluate_script.load_model_for_evaluation(
            experiment="B1",
            experiment_config=experiment_config,
            checkpoint_path=str(checkpoint_path),
            device=torch.device("cpu"),
            height=192,
            width=256,
        )
