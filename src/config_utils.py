"""
Utility condivise per il caricamento delle configurazioni.

Questo modulo evita di duplicare la stessa logica in:
- scripts/train.py
- scripts/evaluate.py
"""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(path):
    """
    Carica un file YAML e verifica che il contenuto
    principale sia un mapping/dizionario.
    """

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configurazione YAML non valida: {path}"
        )

    return config


def resolve_project_path(path):
    """
    Converte un path relativo in un path assoluto
    rispetto alla root del repository.

    I path già assoluti vengono restituiti senza modifiche.
    """

    path = Path(path)

    if path.is_absolute():
        return str(path)

    return str(PROJECT_ROOT / path)