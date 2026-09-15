from pathlib import Path
import sys

import torch


# Permette di importare src quando eseguiamo pytest
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.metrics import cc, sim, kld, nss, sauc


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

def test_nss_high_saliency_at_fixation():
    """
    NSS deve essere positivo quando la fixation cade
    nella zona con saliency più alta.
    """

    prediction = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # x=1, y=1 -> centro della mappa
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
    NSS deve essere negativo quando la fixation cade
    in una zona meno saliente rispetto alla media.
    """

    prediction = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # x=0, y=0 -> zona non saliente
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
    Una mappa completamente costante non contiene
    informazione spaziale e deve restituire NSS = 0.
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
    Le fixation positive cadono su saliency alta,
    quelle negative su saliency bassa.

    sAUC deve essere 1.
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
    Le fixation positive cadono su saliency bassa
    mentre le negative cadono sulla saliency massima.

    sAUC deve essere 0.
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
    Se positivi e negativi hanno esattamente
    la stessa saliency, l'AUC deve essere 0.5.
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