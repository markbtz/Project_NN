mport pytest
import torch
import torch.nn.functional as F

from src.losses.saliency_losses import (
    cc_kld_loss,
    cc_loss,
    kld_loss,
    mse_loss,
    normalize_probability_map,
)

from src.metrics import cc, kld


def test_mse_loss_zero_for_identical_maps():
    x = torch.tensor(
        [[[[0.0, 0.2], [0.7, 1.0]]]],
        dtype=torch.float32,
    )

    loss = mse_loss(
        x,
        x,
    )

    assert torch.allclose(
        loss,
        torch.tensor(0.0),
        atol=1e-7,
    )


def test_mse_loss_matches_pytorch():
    prediction = torch.tensor(
        [[[[0.1, 0.2], [0.3, 0.4]]]],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [[[[0.4, 0.3], [0.2, 0.1]]]],
        dtype=torch.float32,
    )

    expected = F.mse_loss(
        prediction,
        target,
    )

    actual = mse_loss(
        prediction,
        target,
    )

    assert torch.allclose(
        actual,
        expected,
        atol=1e-7,
    )


def test_probability_normalization_sums_to_one():
    x = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[4.0, 3.0], [2.0, 1.0]]],
        ],
        dtype=torch.float32,
    )

    prob = normalize_probability_map(
        x
    )

    sums = prob.sum(
        dim=(-2, -1)
    )

    assert torch.allclose(
        sums,
        torch.ones_like(
            sums
        ),
        atol=1e-6,
    )


def test_cc_loss_zero_for_identical_nonconstant_maps():
    x = torch.tensor(
        [[[[0.0, 1.0], [2.0, 4.0]]]],
        dtype=torch.float32,
    )

    loss = cc_loss(
        x,
        x,
    )

    assert torch.allclose(
        loss,
        torch.tensor(0.0),
        atol=1e-6,
    )


def test_cc_loss_two_for_perfect_inverse_maps():
    prediction = torch.tensor(
        [[[[0.0, 1.0], [2.0, 3.0]]]],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [[[[3.0, 2.0], [1.0, 0.0]]]],
        dtype=torch.float32,
    )

    loss = cc_loss(
        prediction,
        target,
    )

    assert torch.allclose(
        loss,
        torch.tensor(2.0),
        atol=1e-6,
    )


def test_cc_loss_is_finite_for_constant_maps():
    prediction = torch.ones(
        2,
        1,
        4,
        4,
    )

    target = torch.ones(
        2,
        1,
        4,
        4,
    )

    loss = cc_loss(
        prediction,
        target,
    )

    assert torch.isfinite(
        loss
    )


def test_kld_zero_for_identical_maps():
    x = torch.tensor(
        [[[[0.1, 0.2], [0.3, 0.4]]]],
        dtype=torch.float32,
    )

    loss = kld_loss(
        x,
        x,
    )

    assert torch.allclose(
        loss,
        torch.tensor(0.0),
        atol=1e-6,
    )


def test_kld_positive_for_different_maps():
    prediction = torch.tensor(
        [[[[0.7, 0.1], [0.1, 0.1]]]],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [[[[0.1, 0.1], [0.1, 0.7]]]],
        dtype=torch.float32,
    )

    loss = kld_loss(
        prediction,
        target,
    )

    assert loss.item() > 0.0


def test_cc_kld_matches_weighted_sum():
    prediction = torch.tensor(
        [[[[0.6, 0.2], [0.1, 0.1]]]],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [[[[0.1, 0.2], [0.2, 0.5]]]],
        dtype=torch.float32,
    )

    cc_weight = 0.7
    kld_weight = 0.3

    expected = (
        cc_weight
        * cc_loss(
            prediction,
            target,
        )
        + kld_weight
        * kld_loss(
            prediction,
            target,
        )
    )

    actual = cc_kld_loss(
        prediction,
        target,
        cc_weight=cc_weight,
        kld_weight=kld_weight,
    )

    assert torch.allclose(
        actual,
        expected,
        atol=1e-7,
    )


def test_cc_kld_backward_produces_finite_gradients():
    prediction = torch.rand(
        2,
        1,
        8,
        8,
        requires_grad=True,
    )

    target = torch.rand(
        2,
        1,
        8,
        8,
    )

    loss = cc_kld_loss(
        prediction,
        target,
        cc_weight=1.0,
        kld_weight=1.0,
    )

    assert torch.isfinite(
        loss
    )

    loss.backward()

    assert prediction.grad is not None

    assert torch.isfinite(
        prediction.grad
    ).all()


def test_combined_loss_rejects_missing_weights():
    prediction = torch.rand(
        1,
        1,
        4,
        4,
    )

    target = torch.rand(
        1,
        1,
        4,
        4,
    )

    with pytest.raises(
        ValueError
    ):
        cc_kld_loss(
            prediction,
            target,
            cc_weight=None,
            kld_weight=1.0,
        )


def test_losses_reject_shape_mismatch():
    prediction = torch.rand(
        1,
        1,
        4,
        4,
    )

    target = torch.rand(
        1,
        1,
        8,
        8,
    )

    with pytest.raises(
        ValueError
    ):
        mse_loss(
            prediction,
            target,
        )

    with pytest.raises(
        ValueError
    ):
        cc_loss(
            prediction,
            target,
        )

    with pytest.raises(
        ValueError
    ):
        kld_loss(
            prediction,
            target,
        )


def test_cc_loss_matches_one_minus_cc_metric():
    """
    CC-loss e metrica CC devono seguire la stessa convenzione:
        CC-loss ~= 1 - CC
    usando lo stesso epsilon.
    """

    eps = 1e-6

    prediction = torch.tensor(
        [[[[0.6, 0.2], [0.1, 0.1]]]],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [[[[0.1, 0.2], [0.2, 0.5]]]],
        dtype=torch.float32,
    )

    actual = cc_loss(
        prediction,
        target,
        eps=eps,
    )

    expected = (
        1.0
        - cc(
            prediction,
            target,
            eps=eps,
        )
    )

    assert torch.allclose(
        actual,
        expected,
        atol=1e-5,
    )


def test_kld_loss_matches_kld_metric():
    """
    KLD loss e metrica KLD devono seguire la stessa convenzione:
        KLD(target || prediction)
    usando lo stesso epsilon.
    """

    eps = 1e-6

    prediction = torch.tensor(
        [[[[0.6, 0.2], [0.1, 0.1]]]],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [[[[0.1, 0.2], [0.2, 0.5]]]],
        dtype=torch.float32,
    )

    actual = kld_loss(
        prediction,
        target,
        eps=eps,
    )

    expected = kld(
        prediction,
        target,
        eps=eps,
    )

    assert torch.allclose(
        actual,
        expected,
        atol=1e-5,
    )
