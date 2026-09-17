import pytest
import torch

from src.saliency_maps import normalize_probability_map


def test_normalization_sums_to_one():
    saliency_map = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )

    normalized = normalize_probability_map(
        saliency_map,
        eps=1e-6,
    )

    assert torch.allclose(
        normalized.sum(
            dim=(-2, -1)
        ),
        torch.ones(
            normalized.shape[:-2],
            dtype=normalized.dtype,
        ),
        atol=1e-6,
    )


def test_normalization_clamps_negative_values():
    saliency_map = torch.tensor(
        [
            [
                [-1.0, 1.0],
                [2.0, -3.0],
            ]
        ],
        dtype=torch.float32,
    )

    normalized = normalize_probability_map(
        saliency_map,
        eps=1e-6,
    )

    assert torch.all(
        normalized >= 0.0
    )


def test_normalization_preserves_shape():
    saliency_map = torch.rand(
        3,
        1,
        8,
        10,
    )

    normalized = normalize_probability_map(
        saliency_map,
        eps=1e-6,
    )

    assert normalized.shape == saliency_map.shape


def test_normalization_is_finite():
    saliency_map = torch.zeros(
        2,
        1,
        4,
        4,
    )

    normalized = normalize_probability_map(
        saliency_map,
        eps=1e-6,
    )

    assert torch.isfinite(
        normalized
    ).all()