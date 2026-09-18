import pytest
import torch

from src.checkpoints import (
    initial_best_score,
    is_better,
    load_training_checkpoint,
    save_training_checkpoint,
)


def test_initial_best_score():
    assert initial_best_score("max") == float("-inf")
    assert initial_best_score("min") == float("inf")


def test_is_better_max():
    assert is_better(0.8, 0.7, "max")
    assert not is_better(0.6, 0.7, "max")


def test_is_better_min():
    assert is_better(0.2, 0.3, "min")
    assert not is_better(0.4, 0.3, "min")


def test_checkpoint_selection_rejects_invalid_mode():
    with pytest.raises(ValueError):
        initial_best_score("invalid")

    with pytest.raises(ValueError):
        is_better(1.0, 0.0, "invalid")

def test_save_and_load_training_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    # Esegue uno step per creare realmente lo stato dell'optimizer.
    inputs = torch.tensor(
        [[1.0, 2.0]]
    )
    loss = model(inputs).sum()
    loss.backward()
    optimizer.step()

    expected_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    save_training_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_score=0.75,
        selection_metric="cc",
        selection_mode="max",
        seed=42,
        experiment="B1",
    )

    resumed_model = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=1e-4,
    )

    start_epoch, best_score = load_training_checkpoint(
        path=str(checkpoint_path),
        model=resumed_model,
        optimizer=resumed_optimizer,
        device="cpu",
        selection_metric="cc",
        selection_mode="max",
    )

    assert start_epoch == 3
    assert best_score == 0.75

    for expected, resumed in zip(
        expected_parameters,
        resumed_model.parameters(),
    ):
        assert torch.equal(
            expected,
            resumed.detach(),
        )

    assert resumed_optimizer.state_dict()["state"]