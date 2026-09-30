"""Audit a downloaded SALICON dataset before configuring the dataloader.

The audit checks available images, density maps, fixation files, IDs, geometry, and directory layout. Use its output to confirm whether NSS/sAUC can be supported and to adapt paths in configs/data.yaml when necessary.
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

        # Use only the path relative to the dataset root.
        # Evita che nomi esterni come "nndl-saliency"
        # avoid classifying all images as maps.
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
        print("  [SKIP] Pillow is not installed; cannot inspect map values.")
        return

    sample = map_paths[:sample_size]
    n_nan, n_empty, sizes = 0, 0, set()
    for p in sample:
        try:
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
        except Exception as e:
            print(f"  [ERROR] cannot read {p}: {e}")
            continue
        sizes.add(arr.shape)
        if np.isnan(arr).any() or np.isinf(arr).any():
            n_nan += 1
        if arr.sum() < 1e-6:
            n_empty += 1

    print(f"  Sample inspected: {len(sample)} maps")
    print(f"  Distinct sizes found: {sizes if len(sizes) <= 5 else f'{len(sizes)} distinct values'}")
    print(f"  Maps with NaN/Inf: {n_nan}")
    print(f"  Nearly empty maps (sum ~0): {n_empty}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist. Did you run download_salicon.py?")
        sys.exit(1)

    print(f"Scanning {data_dir} ...\n")
    buckets = classify_files(data_dir)

    print("=" * 60)
    print("FILE COUNT BY CATEGORY (heuristic; verify manually)")
    print("=" * 60)
    for key in ("images", "density_maps", "fixation_files", "other_structured_files", "other_files"):
        print(f"  {key:25s}: {len(buckets[key])}")

    print()
    print("=" * 60)
    print("CRITICAL CHECK: are fixation coordinates available?")
    print("=" * 60)
    if buckets["fixation_files"]:
        print(f"  FOUND {len(buckets['fixation_files'])} files that appear to contain fixation data.")
        print("  Examples:")
        for p in buckets["fixation_files"][:5]:
            print(f"    {p}")
        print("  -> Open one file manually to confirm the format (es. .mat con array Nx2).")
        print("  -> If confirmed, NSS and sAUC can be used in addition to CC/SIM/KLD.")
    else:
        print("  NO fixation files detected by the current heuristics.")
        print("  -> The Kaggle package likely contains density maps only.")
        print("  -> Fallback: use CC/SIM/KLD only and document the")
        print("     limitation in the report. If fixations are required, try")
        print("     the official SALICON API (see project references).")
        print("  -> If the script may have missed them, inspect")
        print("     'other_structured_files' below manually.")
        if buckets["other_structured_files"]:
            print(f"\n  Unclassified structured files ({len(buckets['other_structured_files'])}), check manually:")
            for p in buckets["other_structured_files"][:10]:
                print(f"    {p}")

    if buckets["images"] and buckets["density_maps"]:
        print()
        print("=" * 60)
        print("IMAGE <-> DENSITY MAP MATCHING")
        print("=" * 60)
        common, only_img, only_map = check_image_map_correspondence(
            buckets["images"], buckets["density_maps"]
        )
        print(f"  Matched pairs: {len(common)}")
        print(f"  Images without maps:  {len(only_img)}")
        print(f"  Maps without images:  {len(only_map)}")
        if only_img or only_map:
            print("  [WARNING] mismatch detected; verify naming conventions before continuing.")

        print()
        print("=" * 60)
        print("DENSITY MAP VALIDITY (sample)")
        print("=" * 60)
        check_density_map_values(buckets["density_maps"])
    else:
        print("\n[WARNING] Could not find both images and density maps; the dataset")
        print("layout may not match this script's heuristics.")
        print("Inspect the directory tree manually and adjust the script or")
        print("configs/data.yaml accordingly.")

    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("  1. Copy the correct paths (images/, maps/, optional fixations/) into configs/data.yaml")
    print("  2. Record the result of the critical check above in the team issue/chat")
    print("  3. Continue with split creation and metric unit tests")


if __name__ == "__main__":
    main()
