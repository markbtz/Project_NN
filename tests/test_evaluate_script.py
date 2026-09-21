import csv
import json
import sys
import torch

import pytest

from scripts import evaluate as evaluate_script
from scripts.evaluate import save_evaluation_results

def test_save_evaluation_results_writes_csv_and_json(
    tmp_path,
):
    result = {
        "summary": {
            "loss": 0.01,
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
            experiment="B1",
            split="tuning",
            checkpoint_path="B1_best.pt",
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

    assert summary["experiment"] == "B1"
    assert summary["split"] == "tuning"
    assert summary["n_samples"] == 2
    assert summary["checkpoint"]["epoch"] == 3

def test_internal_test_requires_final_evaluation(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--experiment",
            "B1",
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