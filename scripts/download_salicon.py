"""
Download and extract the SALICON dataset from Kaggle.

Prerequisites:
    - `kaggle` package installed (already listed in environment-mac.yml / requirements-colab.txt)
    - ~/.kaggle/kaggle.json con le proprie credenziali (vedi README.md)

Use:
    python scripts/download_salicon.py --output_dir data/

Note: the exact Kaggle dataset name specified in the assignment is
"roshan401/salicon" (see the project slides). If the link is no longer
valid, search "SALICON" on kaggle.com/datasets and upload KAGGLE_DATASET.
"""
import argparse
import os
import subprocess
import sys


KAGGLE_DATASET = "roshan401/salicon"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data",
                         help="Directory where the dataset is downloaded and extracted")
    parser.add_argument("--dataset", type=str, default=KAGGLE_DATASET,
                         help="Kaggle dataset slug (owner/dataset-name)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print(f"ERROR: cannot find {kaggle_json}.")
        print("Follow the README instructions ('Setup Kaggle API') before rerunning.")
        sys.exit(1)

    print(f"Downloading '{args.dataset}' in '{args.output_dir}' ...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", args.dataset,
        "-p", args.output_dir,
        "--unzip",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Download failed. Check the dataset slug and Kaggle credentials.")
        sys.exit(result.returncode)

    print("\nDownload complete. Directory contents:")
    for root, dirs, files in os.walk(args.output_dir):
        depth = root.replace(args.output_dir, "").count(os.sep)
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or root}/")
        if depth < 2:
            for f in files[:10]:
                print(f"{indent}  {f}")
            if len(files) > 10:
                print(f"{indent}  ... and {len(files) - 10} files")

    print("\nNext step: python scripts/audit_dataset.py --data_dir", args.output_dir)


if __name__ == "__main__":
    main()
