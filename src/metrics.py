import torch


def _check_shapes(prediction: torch.Tensor, target: torch.Tensor) -> None:
    """
    Controlla che prediction e target abbiano la stessa shape.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape diverse: prediction={prediction.shape}, "
            f"target={target.shape}"
        )


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Converte le mappe nella forma [B, N],
    dove B è la batch size e N il numero di pixel.

    Supporta:
    [H, W]
    [1, H, W]
    [B, 1, H, W]
    """

    if x.ndim == 2:
        x = x.unsqueeze(0)

    elif x.ndim == 3:
        x = x.unsqueeze(0)

    if x.ndim != 4:
        raise ValueError(
            "Input non valido. Attese shape "
            "[H,W], [1,H,W] oppure [B,1,H,W]."
        )

    return x.reshape(x.shape[0], -1)


def _normalize_distribution(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalizza ogni mappa affinché la somma dei pixel sia 1.
    """

    x = torch.clamp(x, min=0.0)

    x = _flatten_batch(x)

    sums = x.sum(
        dim=1,
        keepdim=True,
    )

    return (x + eps) / (
        sums + eps * x.shape[1]
    )


def cc(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Correlation Coefficient (CC).

    Misura la correlazione lineare tra prediction e target.

    Valori:
        1  -> correlazione perfetta
        0  -> nessuna correlazione
       -1  -> correlazione inversa

    Più alto è meglio.
    """

    _check_shapes(prediction, target)

    pred = _flatten_batch(
        prediction.float()
    )

    gt = _flatten_batch(
        target.float()
    )

    pred = pred - pred.mean(
        dim=1,
        keepdim=True,
    )

    gt = gt - gt.mean(
        dim=1,
        keepdim=True,
    )

    numerator = (
        pred * gt
    ).sum(dim=1)

    pred_norm = torch.sqrt(
        (pred ** 2).sum(dim=1)
    )

    gt_norm = torch.sqrt(
        (gt ** 2).sum(dim=1)
    )

    denominator = (
        pred_norm * gt_norm
    )

    score = numerator / (
        denominator + eps
    )

    return score.mean()


def sim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Similarity Metric (SIM).

    Le due mappe vengono normalizzate come distribuzioni
    di probabilità.

    SIM = sum(min(prediction, target))

    Valori:
        1 -> mappe identiche
        0 -> nessuna sovrapposizione

    Più alto è meglio.
    """

    _check_shapes(prediction, target)

    pred = _normalize_distribution(
        prediction.float(),
        eps=eps,
    )

    gt = _normalize_distribution(
        target.float(),
        eps=eps,
    )

    score = torch.minimum(
        pred,
        gt,
    ).sum(dim=1)

    return score.mean()


def kld(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Kullback-Leibler Divergence.

    Convenzione utilizzata:

        KLD(target || prediction)

    cioè:

        sum(
            target * log(target / prediction)
        )

    0 significa distribuzioni identiche.
    Più basso è meglio.
    """

    _check_shapes(prediction, target)

    pred = _normalize_distribution(
        prediction.float(),
        eps=eps,
    )

    gt = _normalize_distribution(
        target.float(),
        eps=eps,
    )

    score = (
        gt
        * torch.log(
            (gt + eps)
            / (pred + eps)
        )
    ).sum(dim=1)

    return score.mean()

def nss(
    prediction: torch.Tensor,
    fixation_coordinates,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalized Scanpath Saliency (NSS).

    Parameters
    ----------
    prediction : torch.Tensor
        Saliency map prevista.

        Shape supportate:
            [H, W]
            [1, H, W]
            [1, 1, H, W]

    fixation_coordinates : array-like
        Coordinate delle fixation con shape [N, 2].

        Ogni coordinata è:
            [x, y]

        Le coordinate devono essere già:
            - 0-based
            - ridimensionate alla stessa risoluzione
              della prediction.

    eps : float
        Valore utilizzato per stabilità numerica.

    Returns
    -------
    torch.Tensor
        NSS medio sulle fixation.

    Note
    ----
    Più alto è meglio.

    NSS standardizza la saliency map:

        S_norm = (S - mean(S)) / std(S)

    e calcola la media dei valori standardizzati
    nelle posizioni fissate dagli osservatori.
    """

    # ---------------------------------------------
    # Portiamo prediction alla forma [H, W]
    # ---------------------------------------------

    pred = prediction.float()

    if pred.ndim == 4:

        if pred.shape[0] != 1 or pred.shape[1] != 1:
            raise ValueError(
                "NSS accetta una singola saliency map. "
                "Per input 4D è richiesta shape [1, 1, H, W]."
            )

        pred = pred[0, 0]

    elif pred.ndim == 3:

        if pred.shape[0] != 1:
            raise ValueError(
                "Per input 3D è richiesta shape [1, H, W]."
            )

        pred = pred[0]

    elif pred.ndim != 2:
        raise ValueError(
            "Prediction deve avere shape "
            "[H,W], [1,H,W] oppure [1,1,H,W]."
        )

    height, width = pred.shape

    # ---------------------------------------------
    # Coordinate fixation
    # ---------------------------------------------

    fix = torch.as_tensor(
        fixation_coordinates,
        dtype=torch.long,
        device=pred.device,
    )

    if fix.ndim != 2 or fix.shape[1] != 2:
        raise ValueError(
            "fixation_coordinates deve avere shape [N, 2]."
        )

    if fix.shape[0] == 0:
        raise ValueError(
            "NSS richiede almeno una fixation."
        )

    x = fix[:, 0]
    y = fix[:, 1]

    # ---------------------------------------------
    # Controllo coordinate
    # ---------------------------------------------

    if (
        torch.any(x < 0)
        or torch.any(x >= width)
        or torch.any(y < 0)
        or torch.any(y >= height)
    ):
        raise ValueError(
            "Sono presenti fixation fuori dai limiti "
            "della saliency map."
        )

    # ---------------------------------------------
    # Z-score della saliency map
    # ---------------------------------------------

    mean = pred.mean()

    std = pred.std(
        unbiased=False
    )

    # Mappa costante:
    # non contiene informazione spaziale.
    if std < eps:
        return torch.zeros(
            (),
            dtype=pred.dtype,
            device=pred.device,
        )

    normalized = (
        pred - mean
    ) / (std + eps)

    # ---------------------------------------------
    # Le fixation sono [x, y],
    # mentre PyTorch indicizza [y, x]
    # ---------------------------------------------

    fixation_values = normalized[
        y,
        x,
    ]

    return fixation_values.mean()