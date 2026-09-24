import torch

from src.models.factory import build_model
from src.models.hierarchical_transformer import (
    M2HierarchicalTransformer,
)


def test_m2_forward_shape_and_finite():
    model = M2HierarchicalTransformer(
        pretrained=False,
        decoder_width=96,
    )

    x = torch.randn(1, 3, 192, 256)

    y = model(x)

    assert y.shape == (1, 1, 192, 256)
    assert torch.isfinite(y).all()
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


def test_m2_backward():
    model = M2HierarchicalTransformer(
        pretrained=False,
        decoder_width=96,
    )

    x = torch.randn(1, 3, 192, 256)

    y = model(x)
    loss = y.mean()

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    assert len(gradients) > 0

    for gradient in gradients:
        assert torch.isfinite(gradient).all()


def test_m2_probability_map():
    model = M2HierarchicalTransformer(
        pretrained=False,
        decoder_width=96,
    )

    model.eval()

    x = torch.randn(1, 3, 192, 256)

    with torch.no_grad():
        probability = model.predict_probability(x)

    assert probability.shape == (1, 1, 192, 256)
    assert torch.isfinite(probability).all()

    sums = probability.sum(dim=(1, 2, 3))

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
    )


def test_m2_factory():
    config = {
        "decoder": {
            "width": 96,
        }
    }

    model = build_model(
        "M2",
        config,
        height=192,
        width=256,
        pretrained=False,
    )

    assert isinstance(
        model,
        M2HierarchicalTransformer,
    )