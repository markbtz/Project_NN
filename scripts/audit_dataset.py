"""
Audit del dataset SALICON scaricato — da lanciare SUBITO dopo il download,
prima di scrivere qualunque riga di codice del dataloader.

Risponde a domande bloccanti:
  1) Sono presenti fixation coordinates (oltre alle density map)?
     -> decide se potete usare NSS/sAUC oltre a CC/SIM/KLD.
  2) Le immagini e le rispettive mappe hanno lo stesso ID e la stessa
     geometria (a meno di resize)?
  3) Ci sono NaN/Inf o mappe "vuote" (somma quasi zero)?
  4) La struttura delle cartelle è quella attesa, o va adattata?

Il layout esatto dei pacchetti Kaggle di SALICON varia (alcuni hanno
images/, maps/, fixations/ separate; altri tutto in un'unica cartella
con suffissi diversi). Questo script scansiona ricorsivamente e
classifica i file per estensione/pattern, senza assumere una struttura
fissa: guarda l'output e adatta i path in configs/data.yaml di conseguenza.

Uso:
    python scripts/audit_dataset.py --data_dir data/
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png"}
MAP_HINTS = {"map", "maps", "saliency", "density", "sal"}
FIXATION_EXT = {".mat", ".json", ".csv", ".npy", ".pts"}
FIXATION_HINTS = {"fixation", "fixations", "fix"}


def classify_files(data_dir):
    buckets = defaultdict(list)

    for root, _, files in os.walk(data_dir):
        root_path = Path(root)

        # Usa solo il percorso relativo al dataset.
        # Evita che nomi esterni come "nndl-saliency"
        # facciano classificare tutte le immagini come mappe.
        relative_root = root_path.relative_to(data_dir)
        relative_root_lower = str(relative_root).lower()

        for f in files:
            path = root_path / f
            ext = path.suffix.lower()
            name_lower = f.lower()

            if ext in IMAGE_EXT:
                if any(
                    h in relative_root_lower or h in name_lower
                    for h in MAP_HINTS
                ):
                    buckets["density_maps"].append(path)
                else:
                    buckets["images"].append(path)

            elif ext in FIXATION_EXT and (
                any(h in relative_root_lower for h in FIXATION_HINTS)
                or any(h in name_lower for h in FIXATION_HINTS)
            ):
                buckets["fixation_files"].append(path)

            elif ext in FIXATION_EXT:
                buckets["other_structured_files"].append(path)

            else:
                buckets["other_files"].append(path)

    return buckets

def stem_id(path: Path) -> str:
    """Estrae un ID confrontabile dal nome file, rimuovendo suffissi comuni."""
    stem = path.stem.lower()
    for suffix in ("_fixmap", "_fixpts", "_map", "_sal", "-map", "-sal"):
        stem = stem.replace(suffix, "")
    return stem


def check_image_map_correspondence(images, maps):
    img_ids = {stem_id(p): p for p in images}
    map_ids = {stem_id(p): p for p in maps}
    common = set(img_ids) & set(map_ids)
    only_img = set(img_ids) - set(map_ids)
    only_map = set(map_ids) - set(img_ids)
    return common, only_img, only_map


def check_density_map_values(map_paths, sample_size=30):
    try:
        from PIL import Image
    except ImportError:
        print("  [SKIP] Pillow non installato: impossibile leggere i valori delle mappe.")
        return

    sample = map_paths[:sample_size]
    n_nan, n_empty, sizes = 0, 0, set()
    for p in sample:
        try:
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
        except Exception as e:
            print(f"  [ERRORE] impossibile leggere {p}: {e}")
            continue
        sizes.add(arr.shape)
        if np.isnan(arr).any() or np.isinf(arr).any():
            n_nan += 1
        if arr.sum() < 1e-6:
            n_empty += 1

    print(f"  Campione ispezionato: {len(sample)} mappe")
    print(f"  Dimensioni distinte trovate: {sizes if len(sizes) <= 5 else f'{len(sizes)} valori diversi'}")
    print(f"  Mappe con NaN/Inf: {n_nan}")
    print(f"  Mappe quasi vuote (somma ~0): {n_empty}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERRORE: {data_dir} non esiste. Hai lanciato download_salicon.py?")
        sys.exit(1)

    print(f"Scansione di {data_dir} ...\n")
    buckets = classify_files(data_dir)

    print("=" * 60)
    print("CONTEGGIO FILE PER CATEGORIA (euristico, verificare a occhio)")
    print("=" * 60)
    for key in ("images", "density_maps", "fixation_files", "other_structured_files", "other_files"):
        print(f"  {key:25s}: {len(buckets[key])}")

    print()
    print("=" * 60)
    print("DOMANDA CRITICA: fixation coordinates disponibili?")
    print("=" * 60)
    if buckets["fixation_files"]:
        print(f"  TROVATI {len(buckets['fixation_files'])} file che sembrano fixation data.")
        print("  Esempi:")
        for p in buckets["fixation_files"][:5]:
            print(f"    {p}")
        print("  -> Apri manualmente un file per confermare il formato (es. .mat con array Nx2).")
        print("  -> Se confermato: potete usare NSS e sAUC oltre a CC/SIM/KLD.")
    else:
        print("  NESSUN file di fixation individuato con le euristiche attuali.")
        print("  -> Probabile che il pacchetto Kaggle contenga solo density map.")
        print("  -> Piano B (già previsto): usare solo CC/SIM/KLD, dichiarando il")
        print("     limite nel report. Se servono davvero le fixation, provare")
        print("     l'API SALICON ufficiale (vedi Riferimenti nel piano).")
        print("  -> Se pensi che lo script non le abbia riconosciute, controlla")
        print("     manualmente 'other_structured_files' qui sotto.")
        if buckets["other_structured_files"]:
            print(f"\n  File strutturati non classificati ({len(buckets['other_structured_files'])}), controllare a mano:")
            for p in buckets["other_structured_files"][:10]:
                print(f"    {p}")

    if buckets["images"] and buckets["density_maps"]:
        print()
        print("=" * 60)
        print("CORRISPONDENZA IMMAGINE <-> DENSITY MAP")
        print("=" * 60)
        common, only_img, only_map = check_image_map_correspondence(
            buckets["images"], buckets["density_maps"]
        )
        print(f"  Coppie corrispondenti: {len(common)}")
        print(f"  Immagini senza mappa:  {len(only_img)}")
        print(f"  Mappe senza immagine:  {len(only_map)}")
        if only_img or only_map:
            print("  [ATTENZIONE] disallineamento — verificare naming convention prima di procedere.")

        print()
        print("=" * 60)
        print("VALIDITA' DELLE DENSITY MAP (campione)")
        print("=" * 60)
        check_density_map_values(buckets["density_maps"])
    else:
        print("\n[ATTENZIONE] Non ho trovato sia immagini che density map: la struttura")
        print("del dataset potrebbe non seguire le euristiche di questo script.")
        print("Ispeziona manualmente l'albero delle cartelle e adatta lo script o")
        print("configs/data.yaml di conseguenza.")

    print()
    print("=" * 60)
    print("PROSSIMO PASSO")
    print("=" * 60)
    print("  1. Copia i path corretti (images/, maps/, eventuale fixations/) in configs/data.yaml")
    print("  2. Segna nel gruppo (issue/chat) l'esito della domanda critica sopra")
    print("  3. Procedi al giorno 2: creazione split + unit test delle metriche")


if __name__ == "__main__":
    main()
