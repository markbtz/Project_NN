from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.factory import build_model
from src.models.multiscale import M1MultiScale


def build_m1():
    """
    Crea M1 senza pesi pretrained, cosi' i test
    non richiedono download da Internet.
    """
    return M1MultiScale(
        pretrained=False,
        decoder_width=96,
    )


def test_m1_output_shape_and_range():
    model = build_m1()
    model.eval()

    x = torch.rand(
        1,
        3,
        192,
        256,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (
        1,
        1,
        192,
        256,
    )

    assert torch.isfinite(output).all()

    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0


def test_m1_encoder_feature_shapes():
    model = build_m1()
    model.eval()

    x = torch.rand(
        1,
        3,
        192,
        256,
    )

    with torch.no_grad():
        features = model.encoder(x)

    assert features["C3"].shape == (
        1,
        128,
        24,
        32,
    )

    assert features["C4"].shape == (
        1,
        256,
        12,
        16,
    )

    assert features["C5"].shape == (
        1,
        512,
        6,
        8,
    )


def test_m1_backward_produces_finite_gradients():
    model = build_m1()
    model.train()

    x = torch.rand(
        2,
        3,
        64,
        64,
    )

    output = model(x)

    loss = output.mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    assert len(gradients) > 0

    for gradient in gradients:
        assert torch.isfinite(
            gradient
        ).all()


def test_m1_probability_map_sums_to_one():
    model = build_m1()
    model.eval()

    x = torch.rand(
        2,
        3,
        64,
        64,
    )

    with torch.no_grad():
        probability = model.predict_probability(x)

    sums = probability.sum(
        dim=(-2, -1)
    )

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
    )


def test_factory_builds_m1():
    experiment_config = {
        "decoder": {
            "name": "multiscale",
            "width": 96,
        }
    }

    model = build_model(
        "M1",
        experiment_config,
        height=192,
        width=256,
        pretrained=False,
    )

    assert isinstance(
        model,
        M1MultiScale,
    )