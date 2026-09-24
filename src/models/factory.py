"""
Factory condivisa per la costruzione dei modelli.
"""

from src.models.adaptive_center_prior import AdaptiveCenterPriorG
from src.models.baseline import (
    B1Baseline,
    CenterPriorB0,
)
from src.models.multiscale import M1MultiScale
from src.models.hierarchical_transformer import M2HierarchicalTransformer


def build_model(
    experiment: str,
    experiment_config: dict,
    *,
    height: int,
    width: int,
    pretrained: bool,
):
    """
    Costruisce il modello associato all'esperimento richiesto.
    """

    if experiment == "B0":
        return CenterPriorB0(
            height=height,
            width=width,
        )

    if experiment == "B1":
        decoder_width = int(
            experiment_config["decoder"]["width"]
        )

        return B1Baseline(
            pretrained=pretrained,
            decoder_width=decoder_width,
        )

    if experiment in {"M1", "M1-L"}:
        decoder_width = int(
            experiment_config["decoder"]["width"]
        )

        return M1MultiScale(
            pretrained=pretrained,
            decoder_width=decoder_width,
        )

    if experiment == "M2":
        decoder_width = int(
            experiment_config["decoder"]["width"]
        )

        return M2HierarchicalTransformer(
            pretrained=pretrained,
            decoder_width=decoder_width,
        )

    if experiment == "G":
        gate_config = experiment_config.get(
            "gate",
            {},
        )

        return AdaptiveCenterPriorG(
            height=height,
            width=width,
            pretrained=pretrained,
            decoder_width=96,
            gate_hidden=int(
                gate_config.get(
                    "hidden_dim",
                    64,
                )
            ),
        )

    raise ValueError(
        f"Esperimento non supportato dalla model factory: {experiment}"
    )