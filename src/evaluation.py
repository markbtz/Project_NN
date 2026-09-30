import torch

from src.metrics import cc, sim, kld


def _get_batch_size(batch) -> int:
    """
    Infer batch size from the image tensor.
    """

    if "image" not in batch:
        raise KeyError(
            "Batch must contain the 'image' key."
        )

    images = batch["image"]

    if not torch.is_tensor(images):
        raise TypeError(
            "batch['image'] must be a torch.Tensor."
        )

    if images.ndim == 0:
        raise ValueError(
            "batch['image'] must include a batch dimension."
        )

    return int(images.shape[0])


def _get_image_ids(batch, batch_size: int):
    """
    Return image_id values as a list of strings.
    Used only when collect_per_sample=True.
    """

    if "image_id" not in batch:
        raise KeyError(
            "collect_per_sample=True requires the key "
            "'image_id' in the batch."
        )

    image_ids = batch["image_id"]

    if torch.is_tensor(image_ids):
        if image_ids.ndim == 0:
            image_ids = [image_ids.item()]
        else:
            image_ids = image_ids.detach().cpu().tolist()

    elif isinstance(image_ids, (list, tuple)):
        image_ids = list(image_ids)

    else:
        image_ids = [image_ids]

    if len(image_ids) != batch_size:
        raise ValueError(
            "The number of image_id values does not match the batch size: "
            f"{len(image_ids)} != {batch_size}."
        )

    return [str(image_id) for image_id in image_ids]


def evaluate_model(
    model,
    data_loader,
    device,
    *,
    prediction_fn=None,
    metric_prediction_fn=None,
    loss_fn=None,
    loss_target_key=None,
    metric_target_key="density_map_prob",
    eps=1e-6,
    collect_per_sample=False,
):
    """
    Evaluate a model on a DataLoader using CC, SIM, and KLD.

    Parameters
    ----------
    model
        PyTorch model to evaluate.

    data_loader
        DataLoader returning dictionary batches.

    device
        Device used for evaluation.

    prediction_fn
        Optional function with signature:

            prediction_fn(model, batch, device)

        If None, use:

            model(batch["image"].to(device))

        Useful for B0, which produces a prediction
        using only the batch size.

    metric_prediction_fn
        Optional function with signature:

            metric_prediction_fn(model, batch, device)

        If provided, its output is used only
        for CC/SIM/KLD. The original prediction is still
        used for the loss.

        This allows, for example, keeping MSE on the
        raw B1/M1 prediction while evaluating
        CC/SIM/KLD sulla probability map.

    loss_fn
        Optional loss. Must return a scalar batch-mean loss
        for the batch.

    loss_target_key
        Batch key used as the loss target.
        Required when loss_fn is not None.

    metric_target_key
        Batch key used as the CC/SIM/KLD target.

    eps
        Epsilon passato alle metriche.

    collect_per_sample
        If True, keep CC/SIM/KLD for each image_id.

    Returns
    -------
    dict
        {
            "summary": {
                "loss": float oppure None,
                "cc": float,
                "sim": float,
                "kld": float,
                "n_samples": int,
            },
            "per_sample": [
                {
                    "image_id": str,
                    "cc": float,
                    "sim": float,
                    "kld": float,
                },
                ...
            ],
        }
    """

    if eps <= 0:
        raise ValueError(
            "eps must be > 0."
        )

    if loss_fn is not None and loss_target_key is None:
        raise ValueError(
            "loss_target_key is required when loss_fn "
            "is not None."
        )

    was_training = model.training

    metric_sums = {
        "cc": 0.0,
        "sim": 0.0,
        "kld": 0.0,
    }

    loss_sum = 0.0
    n_samples = 0
    per_sample = []

    model.eval()

    try:
        with torch.inference_mode():
            for batch in data_loader:
                batch_size = _get_batch_size(batch)

                if metric_target_key not in batch:
                    raise KeyError(
                        f"Batch does not contain metric_target_key="
                        f"'{metric_target_key}'."
                    )

                if prediction_fn is None:
                    images = batch["image"].to(device)
                    prediction = model(images)

                else:
                    prediction = prediction_fn(
                        model,
                        batch,
                        device,
                    )

                if not torch.is_tensor(prediction):
                    raise TypeError(
                        "Prediction must be a torch.Tensor."
                    )

                if metric_prediction_fn is None:
                    metric_prediction = prediction
                else:
                    metric_prediction = metric_prediction_fn(
                        model,
                        batch,
                        device,
                    )

                if not torch.is_tensor(metric_prediction):
                    raise TypeError(
                        "Metric prediction must be "
                        "a torch.Tensor."
                    )

                metric_target = batch[
                    metric_target_key
                ].to(device)

                cc_values = cc(
                    metric_prediction,
                    metric_target,
                    eps=eps,
                    reduction="none",
                )

                sim_values = sim(
                    metric_prediction,
                    metric_target,
                    eps=eps,
                    reduction="none",
                )

                kld_values = kld(
                    metric_prediction,
                    metric_target,
                    eps=eps,
                    reduction="none",
                )

                if cc_values.shape != (batch_size,):
                    raise ValueError(
                        "CC must return one value per sample."
                    )

                if sim_values.shape != (batch_size,):
                    raise ValueError(
                        "SIM must return one value per sample."
                    )

                if kld_values.shape != (batch_size,):
                    raise ValueError(
                        "KLD must return one value per sample."
                    )

                metric_sums["cc"] += (
                    cc_values.sum().item()
                )

                metric_sums["sim"] += (
                    sim_values.sum().item()
                )

                metric_sums["kld"] += (
                    kld_values.sum().item()
                )

                if loss_fn is not None:
                    if loss_target_key not in batch:
                        raise KeyError(
                            f"Batch does not contain loss_target_key="
                            f"'{loss_target_key}'."
                        )

                    loss_target = batch[
                        loss_target_key
                    ].to(device)

                    batch_loss = loss_fn(
                        prediction,
                        loss_target,
                    )

                    if not torch.is_tensor(batch_loss):
                        batch_loss = torch.as_tensor(
                            batch_loss,
                            device=prediction.device,
                            dtype=prediction.dtype,
                        )

                    if batch_loss.ndim != 0:
                        raise ValueError(
                            "loss_fn must return a "
                            "scalare media for the batch."
                        )

                    loss_sum += (
                        batch_loss.item()
                        * batch_size
                    )

                if collect_per_sample:
                    image_ids = _get_image_ids(
                        batch,
                        batch_size,
                    )

                    cc_cpu = (
                        cc_values.detach().cpu().tolist()
                    )

                    sim_cpu = (
                        sim_values.detach().cpu().tolist()
                    )

                    kld_cpu = (
                        kld_values.detach().cpu().tolist()
                    )

                    for index in range(batch_size):
                        per_sample.append(
                            {
                                "image_id": image_ids[index],
                                "cc": float(cc_cpu[index]),
                                "sim": float(sim_cpu[index]),
                                "kld": float(kld_cpu[index]),
                            }
                        )

                n_samples += batch_size

    finally:
        model.train(was_training)

    if n_samples == 0:
        raise ValueError(
            "Cannot evaluate an empty DataLoader."
        )

    summary = {
        "loss": (
            loss_sum / n_samples
            if loss_fn is not None
            else None
        ),
        "cc": metric_sums["cc"] / n_samples,
        "sim": metric_sums["sim"] / n_samples,
        "kld": metric_sums["kld"] / n_samples,
        "n_samples": n_samples,
    }

    return {
        "summary": summary,
        "per_sample": per_sample,
    }
