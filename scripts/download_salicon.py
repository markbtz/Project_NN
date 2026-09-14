"""
Scarica ed estrae il dataset SALICON da Kaggle.

Prerequisiti:
    - pacchetto `kaggle` installato (già in environment-mac.yml / requirements-colab.txt)
    - ~/.kaggle/kaggle.json con le proprie credenziali (vedi README.md)

Uso:
    python scripts/download_salicon.py --output_dir data/

Nota: il nome esatto del dataset Kaggle indicato nella traccia è
"roshan401/salicon" (vedi slide del progetto). Se il link non fosse più
valido, cercare "SALICON" su kaggle.com/datasets e aggiornare KAGGLE_DATASET.
"""
import argparse
import os
import subprocess
import sys


KAGGLE_DATASET = "roshan401/salicon"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data",
                         help="Cartella in cui scaricare ed estrarre il dataset")
    parser.add_argument("--dataset", type=str, default=KAGGLE_DATASET,
                         help="Slug del dataset Kaggle (owner/dataset-name)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print(f"ERRORE: non trovo {kaggle_json}.")
        print("Segui le istruzioni nel README ('Setup Kaggle API') prima di rilanciare.")
        sys.exit(1)

    print(f"Scarico '{args.dataset}' in '{args.output_dir}' ...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", args.dataset,
        "-p", args.output_dir,
        "--unzip",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Download fallito. Controlla lo slug del dataset e le credenziali Kaggle.")
        sys.exit(result.returncode)

    print("\nDownload completato. Contenuto della cartella:")
    for root, dirs, files in os.walk(args.output_dir):
        depth = root.replace(args.output_dir, "").count(os.sep)
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or root}/")
        if depth < 2:
            for f in files[:10]:
                print(f"{indent}  {f}")
            if len(files) > 10:
                print(f"{indent}  ... e altri {len(files) - 10} file")

    print("\nProssimo passo: python scripts/audit_dataset.py --data_dir", args.output_dir)


if __name__ == "__main__":
    main()
