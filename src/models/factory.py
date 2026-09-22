"""
Factory condivisa per la costruzione dei modelli.
"""

from src.models.baseline import (
    B1Baseline,
    CenterPriorB0,
)

from src.models.multiscale import M1MultiScale

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

    raise ValueError(
        f"Esperimento non supportato dalla model factory: {experiment}"
    )