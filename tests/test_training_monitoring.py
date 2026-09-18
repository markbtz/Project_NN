import pytest

from src.training_monitoring import EarlyStopping

from src.training_monitoring import (
    EarlyStopping,
    TrainingHistory,
)

def test_early_stopping_resets_on_improvement():
    early_stopping = EarlyStopping(
        enabled=True,
        patience=3,
    )

    assert not early_stopping.step(False)
    assert not early_stopping.step(False)

    assert early_stopping.epochs_without_improvement == 2

    assert not early_stopping.step(True)

    assert early_stopping.epochs_without_improvement == 0


def test_early_stopping_triggers_after_patience():
    early_stopping = EarlyStopping(
        enabled=True,
        patience=3,
    )

    assert not early_stopping.step(False)
    assert not early_stopping.step(False)
    assert early_stopping.step(False)


def test_disabled_early_stopping_never_triggers():
    early_stopping = EarlyStopping(
        enabled=False,
        patience=2,
    )

    assert not early_stopping.step(False)
    assert not early_stopping.step(False)
    assert not early_stopping.step(False)


def test_early_stopping_rejects_invalid_patience():
    with pytest.raises(ValueError):
        EarlyStopping(
            enabled=True,
            patience=0,
        )

def test_training_history_adds_epoch():
    history = TrainingHistory()

    history.add_epoch(
        epoch=1,
        train_loss=0.015,
        validation_loss=0.010,
        selection_metric="cc",
        selection_value=0.84,
    )

    assert len(history.records) == 1

    record = history.records[0]

    assert record["epoch"] == 1
    assert record["train_loss"] == 0.015
    assert record["validation_loss"] == 0.010
    assert record["selection_metric"] == "cc"
    assert record["selection_value"] == 0.84


def test_training_history_preserves_multiple_epochs():
    history = TrainingHistory()

    history.add_epoch(
        epoch=1,
        train_loss=0.015,
        validation_loss=0.010,
        selection_metric="cc",
        selection_value=0.84,
    )

    history.add_epoch(
        epoch=2,
        train_loss=0.012,
        validation_loss=0.009,
        selection_metric="cc",
        selection_value=0.86,
    )

    assert len(history.records) == 2
    assert history.records[0]["epoch"] == 1
    assert history.records[1]["epoch"] == 2

def test_training_history_saves_csv(tmp_path):
    history = TrainingHistory()

    history.add_epoch(
        epoch=1,
        train_loss=0.015,
        validation_loss=0.010,
        selection_metric="cc",
        selection_value=0.84,
    )

    csv_path = tmp_path / "history.csv"

    history.save_csv(
        str(csv_path)
    )

    content = csv_path.read_text(
        encoding="utf-8"
    )

    assert (
        "epoch,train_loss,validation_loss,"
        "selection_metric,selection_value"
        in content
    )

    assert (
        "1,0.015,0.01,cc,0.84"
        in content
    )

def test_training_history_loads_csv(tmp_path):
    csv_path = tmp_path / "history.csv"

    csv_path.write_text(
        (
            "epoch,train_loss,validation_loss,"
            "selection_metric,selection_value\n"
            "1,0.015,0.01,cc,0.84\n"
            "2,0.012,0.009,cc,0.86\n"
        ),
        encoding="utf-8",
    )

    history = TrainingHistory()

    history.load_csv(
        str(csv_path)
    )

    assert len(history.records) == 2

    assert history.records[0] == {
        "epoch": 1,
        "train_loss": 0.015,
        "validation_loss": 0.01,
        "selection_metric": "cc",
        "selection_value": 0.84,
    }

    assert history.records[1] == {
        "epoch": 2,
        "train_loss": 0.012,
        "validation_loss": 0.009,
        "selection_metric": "cc",
        "selection_value": 0.86,
    }

def test_training_history_saves_loss_plot(tmp_path):
    history = TrainingHistory()

    history.add_epoch(
        epoch=1,
        train_loss=0.015,
        validation_loss=0.010,
        selection_metric="cc",
        selection_value=0.84,
    )

    history.add_epoch(
        epoch=2,
        train_loss=0.012,
        validation_loss=0.009,
        selection_metric="cc",
        selection_value=0.86,
    )

    plot_path = tmp_path / "training_loss.png"

    history.save_loss_plot(
        str(plot_path)
    )

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0

def test_training_history_counts_non_improving_epochs_max():
    history = TrainingHistory()

    values = [
        0.84,
        0.86,
        0.85,
        0.84,
        0.83,
    ]

    for epoch, value in enumerate(
        values,
        start=1,
    ):
        history.add_epoch(
            epoch=epoch,
            train_loss=0.0,
            validation_loss=0.0,
            selection_metric="cc",
            selection_value=value,
        )

    assert (
        history.consecutive_non_improving_epochs(
            "max"
        )
        == 3
    )


def test_training_history_resets_non_improving_count():
    history = TrainingHistory()

    values = [
        0.84,
        0.83,
        0.82,
        0.87,
    ]

    for epoch, value in enumerate(
        values,
        start=1,
    ):
        history.add_epoch(
            epoch=epoch,
            train_loss=0.0,
            validation_loss=0.0,
            selection_metric="cc",
            selection_value=value,
        )

    assert (
        history.consecutive_non_improving_epochs(
            "max"
        )
        == 0
    )


def test_training_history_counts_non_improving_epochs_min():
    history = TrainingHistory()

    values = [
        0.40,
        0.30,
        0.31,
        0.32,
    ]

    for epoch, value in enumerate(
        values,
        start=1,
    ):
        history.add_epoch(
            epoch=epoch,
            train_loss=0.0,
            validation_loss=0.0,
            selection_metric="kld",
            selection_value=value,
        )

    assert (
        history.consecutive_non_improving_epochs(
            "min"
        )
        == 2
    )

def test_early_stopping_resumes_from_history():
    history = TrainingHistory()

    values = [
        0.84,
        0.83,
        0.82,
    ]

    for epoch, value in enumerate(
        values,
        start=1,
    ):
        history.add_epoch(
            epoch=epoch,
            train_loss=0.0,
            validation_loss=0.0,
            selection_metric="cc",
            selection_value=value,
        )

    early_stopping = EarlyStopping(
        enabled=True,
        patience=3,
    )

    early_stopping.epochs_without_improvement = (
        history.consecutive_non_improving_epochs(
            "max"
        )
    )

    assert (
        early_stopping.epochs_without_improvement
        == 2
    )

    assert early_stopping.step(
        improved=False
    )