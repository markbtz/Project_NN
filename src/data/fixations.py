from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_salicon_fixations(mat_path):
    """
    Legge un file .mat di fixation SALICON.

    Le coordinate SALICON sono memorizzate come:
        (x, y)

    e sono 1-based:
        x = 1 ... width
        y = 1 ... height

    Returns
    -------
    fixations : np.ndarray
        Array di shape [N, 2] contenente tutte le fixation
        di tutti gli osservatori.

    original_size : tuple
        (height, width) dell'immagine originale.
    """

    mat_path = Path(mat_path)

    if not mat_path.exists():
        raise FileNotFoundError(
            f"Fixation file non trovato: {mat_path}"
        )

    data = loadmat(
        mat_path,
        variable_names=[
            "gaze",
            "resolution",
        ],
    )

    if "gaze" not in data:
        raise KeyError(
            f"Campo 'gaze' non trovato in {mat_path}"
        )

    if "resolution" not in data:
        raise KeyError(
            f"Campo 'resolution' non trovato in {mat_path}"
        )

    # SALICON salva:
    # resolution = [height, width]
    resolution = np.asarray(
        data["resolution"]
    ).reshape(-1)

    if resolution.size != 2:
        raise ValueError(
            f"Resolution non valida: {resolution}"
        )

    original_height = int(resolution[0])
    original_width = int(resolution[1])

    gaze = data["gaze"]

    if "fixations" not in gaze.dtype.names:
        raise KeyError(
            f"Campo 'fixations' non trovato in gaze: "
            f"{gaze.dtype.names}"
        )

    all_fixations = []

    for i in range(gaze.shape[0]):

        fix = gaze["fixations"][i, 0]

        if not isinstance(fix, np.ndarray):
            continue

        if fix.size == 0:
            continue

        if fix.ndim != 2 or fix.shape[1] != 2:
            raise ValueError(
                f"Formato fixation inatteso: {fix.shape}"
            )

        all_fixations.append(
            fix.astype(np.float64)
        )

    if not all_fixations:
        fixations = np.empty(
            (0, 2),
            dtype=np.float64,
        )

    else:
        fixations = np.concatenate(
            all_fixations,
            axis=0,
        )

    return fixations, (
        original_height,
        original_width,
    )


def resize_fixations(
    fixations,
    original_size,
    target_size=(192, 256),
):
    """
    Converte e ridimensiona coordinate fixation SALICON.

    SALICON usa coordinate 1-based.
    PyTorch/Python usa coordinate 0-based.

    Parameters
    ----------
    fixations : np.ndarray
        Array [N, 2] con coordinate (x, y) SALICON.

    original_size : tuple
        (height, width) originale.

    target_size : tuple
        (height, width) finale.

        Default:
            (192, 256)

    Returns
    -------
    np.ndarray
        Coordinate (x, y) 0-based e ridimensionate,
        con shape [N, 2].
    """

    fixations = np.asarray(
        fixations,
        dtype=np.float64,
    )

    if fixations.size == 0:
        return np.empty(
            (0, 2),
            dtype=np.int64,
        )

    if fixations.ndim != 2 or fixations.shape[1] != 2:
        raise ValueError(
            "fixations deve avere shape [N, 2]"
        )

    original_height, original_width = original_size
    target_height, target_width = target_size

    # ---------------------------------------------
    # SALICON:
    # x = 1 ... width
    # y = 1 ... height
    #
    # Python:
    # x = 0 ... width-1
    # y = 0 ... height-1
    # ---------------------------------------------

    x = fixations[:, 0] - 1.0
    y = fixations[:, 1] - 1.0

    # Resize delle coordinate
    x = np.floor(
        x * target_width / original_width
    ).astype(np.int64)

    y = np.floor(
        y * target_height / original_height
    ).astype(np.int64)

    # Protezione dai bordi
    x = np.clip(
        x,
        0,
        target_width - 1,
    )

    y = np.clip(
        y,
        0,
        target_height - 1,
    )

    return np.stack(
        [x, y],
        axis=1,
    )


def load_and_resize_fixations(
    mat_path,
    target_size=(192, 256),
):
    """
    Funzione di convenienza:

    .mat SALICON
        ↓
    lettura fixation
        ↓
    conversione 1-based → 0-based
        ↓
    resize alle dimensioni della rete
    """

    fixations, original_size = load_salicon_fixations(
        mat_path
    )

    return resize_fixations(
        fixations=fixations,
        original_size=original_size,
        target_size=target_size,
    )