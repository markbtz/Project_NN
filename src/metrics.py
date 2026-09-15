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