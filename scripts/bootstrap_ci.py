"""
Bootstrap paired confidence interval sui delta per-immagine prodotti da
scripts/compare_evaluations.py.

Il confronto tra due modelli (es. M1 - B1) e' significativo solo se il
delta medio non e' compatibile con zero entro l'incertezza — un singolo
numero medio (+0.006 CC) senza intervallo non basta a dire se il
miglioramento e' reale o rumore campionario tra le 2500 immagini del
tuning set.

Metodo: paired bootstrap. Per ogni resample si ricampionano gli image_id
CON reinserimento (stessa dimensione del dataset originale), si calcola
la media dei delta su quel resample, e si ripete N volte. L'intervallo di
confidenza al 95% e' il percentile [2.5, 97.5] della distribuzione delle
medie ricampionate.

Uso:
    python scripts/bootstrap_ci.py results/comparison_B1_M1.csv \
        --metrics delta_cc delta_sim delta_kld \
        --n_resamples 1000 \
        --seed 42 \
        --output results/bootstrap_B1_M1.json
"""
import argparse
import csv
import json
import os

import numpy as np


def load_deltas(csv_path: str, metric_columns: list[str]) -> dict:
    """Legge il CSV di compare_evaluations.py e ritorna un dict
    {metric_column: np.ndarray di valori}, uno per immagine."""
    values = {col: [] for col in metric_columns}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=_sniff_delimiter(f))

        missing = [c for c in metric_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Colonne non trovate nel CSV: {missing}. "
                f"Colonne disponibili: {reader.fieldnames}"
            )

        n_rows = 0
        for row in reader:
            for col in metric_columns:
                values[col].append(float(row[col]))
            n_rows += 1

    if n_rows == 0:
        raise ValueError(f"Il CSV {csv_path} non contiene righe.")

    return {col: np.array(v, dtype=np.float64) for col, v in values.items()}, n_rows


def _sniff_delimiter(f) -> str:
    """compare_evaluations.py potrebbe produrre CSV separato da virgola o
    da tab (l'esempio incollato in chat sembrava tab-separated). Rileviamo
    il delimitatore guardando la prima riga, poi riavvolgiamo il file."""
    pos = f.tell()
    first_line = f.readline()
    f.seek(pos)
    if "\t" in first_line and "," not in first_line:
        return "\t"
    return ","


def paired_bootstrap_ci(
    deltas: np.ndarray,
    n_resamples: int,
    seed: int,
    ci: float = 0.95,
) -> dict:
    """Bootstrap non parametrico sulla media dei delta per-immagine."""
    rng = np.random.default_rng(seed)
    n = len(deltas)

    resampled_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_means[i] = deltas[idx].mean()

    alpha = (1.0 - ci) / 2.0
    lower = np.quantile(resampled_means, alpha)
    upper = np.quantile(resampled_means, 1.0 - alpha)

    observed_mean = deltas.mean()
    # Il CI include lo zero? Se si', il miglioramento non e' distinguibile
    # dal rumore campionario a questo livello di confidenza.
    excludes_zero = (lower > 0.0) or (upper < 0.0)

    return {
        "n_samples": n,
        "n_resamples": n_resamples,
        "observed_mean": float(observed_mean),
        "ci_level": ci,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "significant_at_ci_level": bool(excludes_zero),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_csv", type=str,
                         help="Output di scripts/compare_evaluations.py")
    parser.add_argument("--metrics", type=str, nargs="+",
                         default=["delta_cc", "delta_sim", "delta_kld"],
                         help="Nomi delle colonne delta da analizzare")
    parser.add_argument("--n_resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--output", type=str, default=None,
                         help="Se specificato, salva i risultati come JSON")
    args = parser.parse_args()

    deltas_by_metric, n_rows = load_deltas(args.comparison_csv, args.metrics)
    print(f"Righe lette da {args.comparison_csv}: {n_rows}")
    print(f"Resample: {args.n_resamples} | Seed: {args.seed} | CI: {args.ci:.0%}\n")

    results = {}
    for metric in args.metrics:
        result = paired_bootstrap_ci(
            deltas_by_metric[metric],
            n_resamples=args.n_resamples,
            seed=args.seed,
            ci=args.ci,
        )
        results[metric] = result

        direction = "migliora" if result["observed_mean"] > 0 else "peggiora"
        # Per KLD un valore piu' basso e' meglio: lo segnaliamo ma non invertiamo
        # il segno, il file di comparazione decide gia' la convenzione del segno.
        sig = "SI" if result["significant_at_ci_level"] else "NO (CI include lo zero)"

        print(f"{metric}: media={result['observed_mean']:+.6f}  "
              f"CI[{args.ci:.0%}]=[{result['ci_lower']:+.6f}, {result['ci_upper']:+.6f}]  "
              f"significativo={sig}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nRisultati salvati in {args.output}")


if __name__ == "__main__":
    main()