import pytest

from src.models.baseline import (
    B1Baseline,
    CenterPriorB0,
)
from src.models.factory import build_model


def test_build_model_b0():
    model = build_model(
        "B0",
        {},
        height=192,
        width=256,
        pretrained=False,
    )

    assert isinstance(
        model,
        CenterPriorB0,
    )

    assert model.height == 192
    assert model.width == 256


def test_build_model_b1():
    experiment_config = {
        "decoder": {
            "width": 96,
        }
    }

    model = build_model(
        "B1",
        experiment_config,
        height=192,
        width=256,
        pretrained=False,
    )

    assert isinstance(
        model,
        B1Baseline,
    )

    assert (
        model.decoder.blocks[0][1].out_channels
        == 96
    )


def test_build_model_rejects_unknown_experiment():
    with pytest.raises(ValueError):
        build_model(
            "UNKNOWN",
            {},
            height=192,
            width=256,
            pretrained=False,
        )