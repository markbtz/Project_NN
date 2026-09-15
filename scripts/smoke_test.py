from pathlib import Path
import sys

# Aggiunge la root del repository al path di Python
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import SaliconDataset


dataset = SaliconDataset(
    data_dir="/content/data_local",
    manifest_path="results/split_manifest.csv",
    split="train",
    input_size=(256, 192),
    density_map_epsilon=1e-6,
    augmentation=True,
)

sample = dataset[0]

print("Numero campioni:", len(dataset))
print("Image ID:", sample["image_id"])
print("Image shape:", sample["image"].shape)
print("Density map shape:", sample["density_map"].shape)
print("Somma density map:", sample["density_map"].sum().item())
print("Fixation:", sample["fixation_path"])