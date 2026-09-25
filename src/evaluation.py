import torch

from src.metrics import cc, sim, kld


def _get_batch_size(batch) -> int:
    """
    Ricava la batch size dal tensore delle immagini.
    """

    if "image" not in batch:
        raise KeyError(
            "Il batch deve contenere la chiave 'image'."
        )

    images = batch["image"]

    if not torch.is_tensor(images):
        raise TypeError(
            "batch['image'] deve essere un torch.Tensor."
        )

    if images.ndim == 0:
        raise ValueError(
            "batch['image'] deve avere una dimensione di batch."
        )

    return int(images.shape[0])


def _get_image_ids(batch, batch_size: int):
    """
    Restituisce gli image_id come lista di stringhe.
    Viene usato solo quando collect_per_sample=True.
    """

    if "image_id" not in batch:
        raise KeyError(
            "collect_per_sample=True richiede la chiave "
            "'image_id' nel batch."
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
            "Il numero di image_id non coincide con la batch size: "
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
    Valuta un modello su un DataLoader usando CC, SIM e KLD.

    Parameters
    ----------
    model
        Modello PyTorch da valutare.

    data_loader
        DataLoader che restituisce batch in forma di dizionario.

    device
        Device su cui eseguire la valutazione.

    prediction_fn
        Funzione opzionale con firma:

            prediction_fn(model, batch, device)

        Se None, viene usato:

            model(batch["image"].to(device))

        Serve, per esempio, per B0, che produce una prediction
        a partire dalla sola batch size.

    metric_prediction_fn
        Funzione opzionale con firma:

            metric_prediction_fn(model, batch, device)

        Se specificata, il suo output viene usato esclusivamente
        per CC/SIM/KLD. La prediction originale continua invece
        a essere usata per la loss.

        Questo permette, per esempio, di mantenere la MSE sulla
        prediction raw di B1/M1 e valutare contemporaneamente
        CC/SIM/KLD sulla probability map.

    loss_fn
        Loss opzionale. Deve restituire una loss scalare media
        del batch.

    loss_target_key
        Chiave del batch da usare come target della loss.
        Obbligatoria se loss_fn non e' None.

    metric_target_key
        Chiave del batch da usare come target per CC/SIM/KLD.

    eps
        Epsilon passato alle metriche.

    collect_per_sample
        Se True, conserva CC/SIM/KLD per ogni image_id.

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
            "eps deve essere > 0."
        )

    if loss_fn is not None and loss_target_key is None:
        raise ValueError(
            "loss_target_key e' obbligatoria quando loss_fn "
            "non e' None."
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
                        f"Il batch non contiene metric_target_key="
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
                        "La prediction deve essere un torch.Tensor."
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
                        "La prediction per le metriche deve essere "
                        "un torch.Tensor."
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
                        "CC deve restituire un valore per campione."
                    )

                if sim_values.shape != (batch_size,):
                    raise ValueError(
                        "SIM deve restituire un valore per campione."
                    )

                if kld_values.shape != (batch_size,):
                    raise ValueError(
                        "KLD deve restituire un valore per campione."
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
                            f"Il batch non contiene loss_target_key="
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
                            "loss_fn deve restituire una loss "
                            "scalare media del batch."
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
            "Impossibile valutare un DataLoader vuoto."
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
