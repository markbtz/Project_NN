import torch

from src.models.adaptive_center_prior import AdaptiveCenterPriorG


def test_g_output_shape_alpha_and_probability_sum():
    model = AdaptiveCenterPriorG(
        height=64,
        width=64,
        pretrained=False,
        gate_hidden=16,
    )

    model.eval()

    x = torch.rand(2, 3, 64, 64)

    with torch.no_grad():
        output, alpha = model(
            x,
            return_alpha=True,
        )

    assert output.shape == (2, 1, 64, 64)
    assert alpha.shape == (2, 1, 1, 1)

    assert torch.all(alpha >= 0.0)
    assert torch.all(alpha <= 1.0)

    assert torch.all(output >= 0.0)

    sums = output.sum(dim=(-2, -1))

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
    )


def test_g_predict_probability_matches_forward():
    model = AdaptiveCenterPriorG(
        height=64,
        width=64,
        pretrained=False,
        gate_hidden=16,
    )

    model.eval()

    x = torch.rand(2, 3, 64, 64)

    with torch.no_grad():
        output = model(x)
        probability = model.predict_probability(x)

    assert torch.allclose(
        output,
        probability,
        atol=1e-6,
    )


def test_g_gate_receives_gradients():
    model = AdaptiveCenterPriorG(
        height=64,
        width=64,
        pretrained=False,
        gate_hidden=16,
    )

    x = torch.rand(2, 3, 64, 64)

    output = model(x)

    spatial_weights = torch.linspace(
        0.0,
        1.0,
        64 * 64,
    ).reshape(1, 1, 64, 64)

    loss = (output * spatial_weights).sum()

    loss.backward()

    gate_gradients = [
        parameter.grad
        for parameter in model.gate.parameters()
        if parameter.requires_grad
    ]

    assert gate_gradients
    assert all(
        gradient is not None
        for gradient in gate_gradients
    )

    assert all(
        torch.isfinite(gradient).all()
        for gradient in gate_gradients
    )