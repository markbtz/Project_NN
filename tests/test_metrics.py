from pathlib import Path
import sys

import torch


# Allow importing src when running pytest
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import cc, sim, kld, nss, sauc


def test_identical_maps():
    """
    If prediction and target are identical:
    CC should be approximately 1
    SIM should be approximately 1
    KLD should be approximately 0
    """

    target = torch.tensor(
        [
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    prediction = target.clone()

    cc_score = cc(prediction, target)
    sim_score = sim(prediction, target)
    kld_score = kld(prediction, target)

    assert torch.isclose(
        cc_score,
        torch.tensor(1.0),
        atol=1e-5,
    )

    assert torch.isclose(
        sim_score,
        torch.tensor(1.0),
        atol=1e-5,
    )

    assert torch.isclose(
        kld_score,
        torch.tensor(0.0),
        atol=1e-5,
    )


def test_different_maps():
    """
    Different maps must not achieve
    ideal metric values.
    """

    target = torch.tensor(
        [
            [
                [
                    [0.7, 0.1],
                    [0.1, 0.1],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    prediction = torch.tensor(
        [
            [
                [
                    [0.1, 0.1],
                    [0.1, 0.7],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    cc_score = cc(prediction, target)
    sim_score = sim(prediction, target)
    kld_score = kld(prediction, target)

    assert cc_score < 1.0
    assert sim_score < 1.0
    assert kld_score > 0.0


def test_batch_metrics():
    """
    Check that the metrics also work
    on a batch with multiple elements.
    """

    target = torch.tensor(
        [
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ]
            ],
            [
                [
                    [0.4, 0.3],
                    [0.2, 0.1],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    prediction = target.clone()

    assert torch.isclose(
        cc(prediction, target),
        torch.tensor(1.0),
        atol=1e-5,
    )

    assert torch.isclose(
        sim(prediction, target),
        torch.tensor(1.0),
        atol=1e-5,
    )

    assert torch.isclose(
        kld(prediction, target),
        torch.tensor(0.0),
        atol=1e-5,
    )


def test_metrics_are_finite():
    """
    Metrics must not produce NaN or Inf
    on valid maps.
    """

    target = torch.rand(
        4,
        1,
        192,
        256,
    )

    prediction = torch.rand(
        4,
        1,
        192,
        256,
    )

    cc_score = cc(prediction, target)
    sim_score = sim(prediction, target)
    kld_score = kld(prediction, target)

    assert torch.isfinite(cc_score)
    assert torch.isfinite(sim_score)
    assert torch.isfinite(kld_score)


def test_sim_range():
    """
    SIM must remain between 0 and 1.
    """

    target = torch.rand(
        4,
        1,
        32,
        32,
    )

    prediction = torch.rand(
        4,
        1,
        32,
        32,
    )

    score = sim(prediction, target)

    assert score >= 0.0
    assert score <= 1.0


def test_wrong_shapes_raise_error():
    """
    Prediction and target with different shapes
    must raise an error.
    """

    target = torch.rand(
        1,
        1,
        192,
        256,
    )

    prediction = torch.rand(
        1,
        1,
        100,
        100,
    )

    try:
        cc(prediction, target)

    except ValueError:
        pass

    else:
        raise AssertionError(
            "CC avrebbe dovuto generare ValueError."
        )

def test_nss_high_saliency_at_fixation():
    """
    NSS should be positive when the fixation falls
    in the highest-saliency region.
    """

    prediction = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # x=1, y=1 -> map center
    fixations = [
        [1, 1]
    ]

    score = nss(
        prediction,
        fixations,
    )

    assert torch.isfinite(score)
    assert score > 0.0


def test_nss_low_saliency_at_fixation():
    """
    NSS should be negative when the fixation falls
    in a region less salient than the mean.
    """

    prediction = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # x=0, y=0 -> non-salient region
    fixations = [
        [0, 0]
    ]

    score = nss(
        prediction,
        fixations,
    )

    assert torch.isfinite(score)
    assert score < 0.0


def test_nss_constant_map():
    """
    A completely constant map contains no
    spatial information and should return NSS = 0.
    """

    prediction = torch.ones(
        192,
        256,
        dtype=torch.float32,
    )

    fixations = [
        [10, 10],
        [100, 50],
        [200, 150],
    ]

    score = nss(
        prediction,
        fixations,
    )

    assert torch.isclose(
        score,
        torch.tensor(0.0),
        atol=1e-6,
    )

def test_sauc_perfect_separation():
    """
    Positive fixations fall on high saliency,
    while negative fixations fall on low saliency.

    sAUC should be 1.
    """

    prediction = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    positive_fixations = [
        [1, 1],
    ]

    negative_fixations = [
        [0, 0],
        [2, 0],
        [0, 2],
        [2, 2],
    ]

    score = sauc(
        prediction,
        positive_fixations,
        negative_fixations,
    )

    assert torch.isclose(
        score,
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_sauc_wrong_separation():
    """
    Positive fixations fall on low saliency
    while negative fixations fall on maximum saliency.

    sAUC should be 0.
    """

    prediction = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    positive_fixations = [
        [0, 0],
    ]

    negative_fixations = [
        [1, 1],
    ]

    score = sauc(
        prediction,
        positive_fixations,
        negative_fixations,
    )

    assert torch.isclose(
        score,
        torch.tensor(0.0),
        atol=1e-6,
    )


def test_sauc_ties():
    """
    If positive and negative samples have exactly
    the same saliency, AUC should be 0.5.
    """

    prediction = torch.ones(
        3,
        3,
        dtype=torch.float32,
    )

    positive_fixations = [
        [1, 1],
        [0, 0],
    ]

    negative_fixations = [
        [2, 2],
        [0, 2],
    ]

    score = sauc(
        prediction,
        positive_fixations,
        negative_fixations,
    )

    assert torch.isclose(
        score,
        torch.tensor(0.5),
        atol=1e-6,
    )


def test_reduction_none_returns_one_value_per_sample():
    """
    CC/SIM/KLD with reduction="none" must return
    one value for each batch element.
    """

    target = torch.rand(
        4,
        1,
        16,
        16,
    )

    prediction = torch.rand(
        4,
        1,
        16,
        16,
    )

    cc_scores = cc(
        prediction,
        target,
        reduction="none",
    )

    sim_scores = sim(
        prediction,
        target,
        reduction="none",
    )

    kld_scores = kld(
        prediction,
        target,
        reduction="none",
    )

    assert cc_scores.shape == (4,)
    assert sim_scores.shape == (4,)
    assert kld_scores.shape == (4,)

    assert torch.isfinite(cc_scores).all()
    assert torch.isfinite(sim_scores).all()
    assert torch.isfinite(kld_scores).all()


def test_reduction_mean_matches_none_mean():
    """
    Default behavior must match
    the mean of per-image values.
    """

    target = torch.rand(
        5,
        1,
        16,
        16,
    )

    prediction = torch.rand(
        5,
        1,
        16,
        16,
    )

    for metric in (cc, sim, kld):
        score_mean = metric(
            prediction,
            target,
        )

        scores_none = metric(
            prediction,
            target,
            reduction="none",
        )

        assert torch.allclose(
            score_mean,
            scores_none.mean(),
            atol=1e-7,
        )


def test_invalid_reduction_raises_error():
    """
    An unsupported reduction must fail explicitly.
    """

    target = torch.rand(
        2,
        1,
        8,
        8,
    )

    prediction = torch.rand(
        2,
        1,
        8,
        8,
    )

    for metric in (cc, sim, kld):
        try:
            metric(
                prediction,
                target,
                reduction="invalid",
            )

        except ValueError:
            pass

        else:
            raise AssertionError(
                "The metric should have raised ValueError "
                "for an invalid reduction."
            )


def test_reduction_none_supports_single_2d_map():
    """
    A single saliency map [H, W] must be supported
    also with reduction="none".
    """

    target = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=torch.float32,
    )

    prediction = target.clone()

    for metric in (cc, sim, kld):
        scores = metric(
            prediction,
            target,
            reduction="none",
        )

        assert scores.shape == (1,)


def test_reduction_none_preserves_sample_scores():
    """
    reduction="none" must preserve the value associated
    with each batch element, without averaging samples.
    """

    target = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[0.0, 1.0], [2.0, 3.0]]],
        ],
        dtype=torch.float32,
    )

    prediction = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[3.0, 2.0], [1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    cc_scores = cc(
        prediction,
        target,
        reduction="none",
    )

    assert torch.isclose(
        cc_scores[0],
        torch.tensor(1.0),
        atol=1e-6,
    )

    assert torch.isclose(
        cc_scores[1],
        torch.tensor(-1.0),
        atol=1e-6,
    )
