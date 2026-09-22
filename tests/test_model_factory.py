import pytest

from src.models.baseline import (
    B1Baseline,
    CenterPriorB0,
)
from src.models.multiscale import M1MultiScale
from src.models.factory import build_model
from src.models.adaptive_center_prior import AdaptiveCenterPriorG


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

def test_build_model_m1_l():
    experiment_config = {
        "decoder": {
            "width": 96,
        }
    }

    model = build_model(
        "M1-L",
        experiment_config,
        height=192,
        width=256,
        pretrained=False,
    )

    assert isinstance(
        model,
        M1MultiScale,
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
def test_build_model_g():
    config = {
        "model": "adaptive_center_prior",
        "base_model": "M1-L",
        "center_prior": "B0",
        "gate": {
            "hidden_dim": 64,
        },
        "target": "density_map_prob",
        "loss": {
            "name": "cc_kld",
            "cc_weight": 0.5,
            "kld_weight": 0.5,
        },
    }

    model = build_model(
        "G",
        config,
        height=64,
        width=64,
        pretrained=False,
    )

    assert isinstance(
        model,
        AdaptiveCenterPriorG,
    )