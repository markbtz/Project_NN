import torch

from src.saliency_maps import normalize_probability_map


def _check_shapes(prediction: torch.Tensor, target: torch.Tensor) -> None:
    """
    Check that prediction and target have the same shape.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            f"Different shapes: prediction={prediction.shape}, "
            f"target={target.shape}"
        )


def _apply_reduction(
    values: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """
    Apply reduction to per-sample metric values.

    reduction="mean":
        return the batch mean.

    reduction="none":
        return one value for each batch element.
    """

    if reduction == "mean":
        return values.mean()

    if reduction == "none":
        return values

    raise ValueError(
        f"Unsupported reduction: {reduction}. "
        "Use 'mean' or 'none'."
    )


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Convert maps to shape [B, N],
    where B is the batch size and N is the number of pixels.

    Supports:
        [H, W]
        [1, H, W]
        [B, 1, H, W]
    """

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)

    elif x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError(
                "3D input must have shape [1, H, W]."
            )

        x = x.unsqueeze(0)

    elif x.ndim == 4:
        if x.shape[1] != 1:
            raise ValueError(
                "4D input must have shape [B, 1, H, W]."
            )

    else:
        raise ValueError(
            "Invalid input. Expected shape "
            "[H,W], [1,H,W] or [B,1,H,W]."
        )

    return x.flatten(start_dim=1)


def _normalize_distribution(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize each map so that its pixel values sum to 1.
    """

    x = normalize_probability_map(
        x,
        eps=eps,
    )

    return _flatten_batch(x)


def cc(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Correlation Coefficient (CC).

    Measure linear correlation between prediction and target.

    Values:
        1  -> perfect correlation
        0  -> no correlation
       -1  -> inverse correlation

    Higher is better.

    reduction:
        "mean" -> batch mean (default)
        "none" -> one value per image
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

    return _apply_reduction(
        score,
        reduction,
    )


def sim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Similarity Metric (SIM).

    The two maps are normalized as probability
    distributions.

    SIM = sum(min(prediction, target))

    Values:
        1 -> identical maps
        0 -> no overlap

    Higher is better.

    reduction:
        "mean" -> batch mean (default)
        "none" -> one value per image
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

    return _apply_reduction(
        score,
        reduction,
    )


def kld(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Kullback-Leibler Divergence.

    Convention used:

        KLD(target || prediction)

    that is:

        sum(
            target * log(target / prediction)
        )

    0 means identical distributions.
    Lower is better.

    Note on eps: the default (1e-6) matches
    configs/data.yaml -> density_map_epsilon, the same value
    already used by kld_loss in src/losses/saliency_losses.py.

    In production (scripts/evaluate.py), eps is explicitly
    passed from there; this default only applies to direct
    kld() calls (e.g. tests), to avoid validating behavior
    different from what is actually used in training/evaluation.

    reduction:
        "mean" -> batch mean (default)
        "none" -> one value per image
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
        * torch.log(gt / pred)
    ).sum(dim=1)

    return _apply_reduction(
        score,
        reduction,
    )


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
        Predicted saliency map.

        Supported shapes:
            [H, W]
            [1, H, W]
            [1, 1, H, W]

    fixation_coordinates : array-like
        Fixation coordinates with shape [N, 2].

        Each coordinate is:
            [x, y]

        Coordinates must already be:
            - 0-based
            - resized to the same resolution
              as the prediction.

    eps : float
        Value used for numerical stability.

    Returns
    -------
    torch.Tensor
        Mean NSS over fixations.

    Note
    ----
    Higher is better.

    NSS standardizes the saliency map:

        S_norm = (S - mean(S)) / std(S)

    and averages the standardized values
    at observer fixation locations.
    """

    # ---------------------------------------------
    # Convert prediction to [H, W]
    # ---------------------------------------------

    pred = prediction.float()

    if pred.ndim == 4:

        if pred.shape[0] != 1 or pred.shape[1] != 1:
            raise ValueError(
                "NSS accepts a single saliency map. "
                "4D input must have shape [1, 1, H, W]."
            )

        pred = pred[0, 0]

    elif pred.ndim == 3:

        if pred.shape[0] != 1:
            raise ValueError(
                "3D input must have shape [1, H, W]."
            )

        pred = pred[0]

    elif pred.ndim != 2:
        raise ValueError(
            "Prediction must have shape "
            "[H,W], [1,H,W] or [1,1,H,W]."
        )

    height, width = pred.shape

    # ---------------------------------------------
    # Fixation coordinates
    # ---------------------------------------------

    fix = torch.as_tensor(
        fixation_coordinates,
        dtype=torch.long,
        device=pred.device,
    )

    if fix.ndim != 2 or fix.shape[1] != 2:
        raise ValueError(
            "fixation_coordinates must have shape [N, 2]."
        )

    if fix.shape[0] == 0:
        raise ValueError(
            "NSS requires at least one fixation."
        )

    x = fix[:, 0]
    y = fix[:, 1]

    # ---------------------------------------------
    # Validate coordinates
    # ---------------------------------------------

    if (
        torch.any(x < 0)
        or torch.any(x >= width)
        or torch.any(y < 0)
        or torch.any(y >= height)
    ):
        raise ValueError(
            "Some fixations are outside the bounds "
            "of the saliency map."
        )

    # ---------------------------------------------
    # Z-score the saliency map.
    # ---------------------------------------------

    mean = pred.mean()

    std = pred.std(
        unbiased=False
    )

    # Constant map:
    # contains no spatial information.
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
    # Fixations use [x, y] coordinates,
    # while PyTorch indexes [y, x]
    # ---------------------------------------------

    fixation_values = normalized[
        y,
        x,
    ]

    return fixation_values.mean()


def sauc(
    prediction: torch.Tensor,
    fixation_coordinates,
    negative_fixation_coordinates,
) -> torch.Tensor:
    """
    Shuffled AUC (sAUC).

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted saliency map.

        Supported shapes:
            [H, W]
            [1, H, W]
            [1, 1, H, W]

    fixation_coordinates : array-like
        Positive fixations from the same image.

        Shape:
            [N, 2]

        Coordinate:
            [x, y]

    negative_fixation_coordinates : array-like
        Fixations from other images,
        used as negative samples.

        Shape:
            [M, 2]

        Coordinate:
            [x, y]

    Returns
    -------
    torch.Tensor
        sAUC value.

    Note
    ----
    Range:
        0.0 -> poor
        0.5 -> random behavior
        1.0 -> perfect separation

    Higher is better.

    Coordinates must already be:
        - 0-based
        - resized to the same resolution
          as the prediction.
    """

    pred = prediction.float()

    # -----------------------------------------------------
    # Convert prediction to [H, W]
    # -----------------------------------------------------

    if pred.ndim == 4:

        if pred.shape[0] != 1 or pred.shape[1] != 1:
            raise ValueError(
                "sAUC accepts a single saliency map. "
                "4D input must have shape [1, 1, H, W]."
            )

        pred = pred[0, 0]

    elif pred.ndim == 3:

        if pred.shape[0] != 1:
            raise ValueError(
                "3D input must have shape [1, H, W]."
            )

        pred = pred[0]

    elif pred.ndim != 2:

        raise ValueError(
            "Prediction must have shape "
            "[H,W], [1,H,W] or [1,1,H,W]."
        )

    height, width = pred.shape

    # -----------------------------------------------------
    # Positive coordinates
    # -----------------------------------------------------

    positives = torch.as_tensor(
        fixation_coordinates,
        dtype=torch.long,
        device=pred.device,
    )

    if positives.ndim != 2 or positives.shape[1] != 2:
        raise ValueError(
            "fixation_coordinates must have shape [N, 2]."
        )

    if positives.shape[0] == 0:
        raise ValueError(
            "sAUC requires at least one positive fixation."
        )

    # -----------------------------------------------------
    # Negative coordinates
    # -----------------------------------------------------

    negatives = torch.as_tensor(
        negative_fixation_coordinates,
        dtype=torch.long,
        device=pred.device,
    )

    if negatives.ndim != 2 or negatives.shape[1] != 2:
        raise ValueError(
            "negative_fixation_coordinates "
            "must have shape [M, 2]."
        )

    if negatives.shape[0] == 0:
        raise ValueError(
            "sAUC requires at least one negative fixation."
        )

    # -----------------------------------------------------
    # Validate coordinates
    # -----------------------------------------------------

    pos_x = positives[:, 0]
    pos_y = positives[:, 1]

    neg_x = negatives[:, 0]
    neg_y = negatives[:, 1]

    positive_out_of_bounds = (
        torch.any(pos_x < 0)
        or torch.any(pos_x >= width)
        or torch.any(pos_y < 0)
        or torch.any(pos_y >= height)
    )

    negative_out_of_bounds = (
        torch.any(neg_x < 0)
        or torch.any(neg_x >= width)
        or torch.any(neg_y < 0)
        or torch.any(neg_y >= height)
    )

    if positive_out_of_bounds:
        raise ValueError(
            "Some positive fixations are outside "
            "the saliency-map bounds."
        )

    if negative_out_of_bounds:
        raise ValueError(
            "Some negative fixations are outside "
            "the saliency-map bounds."
        )

    # -----------------------------------------------------
    # Saliency at positive and negative points
    #
    # Coordinates = [x, y]
    # Tensor      = [y, x]
    # -----------------------------------------------------

    positive_scores = pred[
        pos_y,
        pos_x,
    ]

    negative_scores = pred[
        neg_y,
        neg_x,
    ]

    # -----------------------------------------------------
    # AUC
    #
    # Equivalently:
    #
    # P(positive score > negative score)
    #
    # Assign 0.5 to ties.
    # -----------------------------------------------------

    differences = (
        positive_scores[:, None]
        - negative_scores[None, :]
    )

    wins = (
        differences > 0
    ).float()

    ties = (
        differences == 0
    ).float()

    auc = (
        wins + 0.5 * ties
    ).mean()

    return auc