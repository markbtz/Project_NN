from pathlib import Path
import sys

import torch


# Permette di importare src quando eseguiamo pytest
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import cc, sim, kld


def test_identical_maps():
    """
    Se prediction e target sono identiche:
    CC deve essere circa 1
    SIM deve essere circa 1
    KLD deve essere circa 0
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
    Due mappe diverse non devono ottenere
    i valori ideali delle metriche.
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
    Verifica che le metriche funzionino anche
    su un batch con più elementi.
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
    Le metriche non devono produrre NaN o Inf
    su mappe valide.
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
    SIM deve rimanere tra 0 e 1.
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
    Prediction e target con shape diverse
    devono generare un errore.
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