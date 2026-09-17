from pathlib import Path
import csv
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from src.saliency_maps import normalize_probability_map


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
MAP_EXTENSIONS = [".png", ".jpg", ".jpeg"]
FIXATION_EXTENSIONS = [".mat"]


class SaliconDataset(Dataset):
    def __init__(
        self,
        data_dir,
        manifest_path,
        split,
        input_size=(256, 192),
        density_map_epsilon=1e-6,
        augmentation=False,
    ):
        self.data_dir = Path(data_dir)
        self.manifest_path = Path(manifest_path)
        self.split = split

        # input_size è [width, height]
        self.input_size = tuple(input_size)

        self.eps = float(density_map_epsilon)
        self.augmentation = augmentation

        self.samples = self._load_manifest()

        if len(self.samples) == 0:
            raise ValueError(
                f"Nessun campione trovato per split='{split}' "
                f"nel manifest {manifest_path}"
            )

        print(
            f"SaliconDataset split='{split}': "
            f"{len(self.samples)} campioni"
        )

    def _load_manifest(self):
        samples = []

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as f:
            reader = csv.DictReader(f)

            required_columns = {
                "image_id",
                "official_split",
                "split",
            }

            if not required_columns.issubset(reader.fieldnames):
                raise ValueError(
                    f"Manifest non valido. "
                    f"Colonne trovate: {reader.fieldnames}"
                )

            for row in reader:
                if row["split"] == self.split:
                    samples.append(row)

        return samples

    @staticmethod
    def _find_file(directory, image_id, extensions):
        directory = Path(directory)

        for ext in extensions:
            candidate = directory / f"{image_id}{ext}"

            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"File non trovato per ID '{image_id}' in {directory}"
        )

    def _paths_for_sample(self, sample):
        image_id = sample["image_id"]
        official_split = sample["official_split"]

        image_dir = (
            self.data_dir
            / "images"
            / official_split
        )

        map_dir = (
            self.data_dir
            / "maps"
            / official_split
        )

        fixation_dir = (
            self.data_dir
            / "fixations"
            / official_split
        )

        image_path = self._find_file(
            image_dir,
            image_id,
            IMAGE_EXTENSIONS,
        )

        map_path = self._find_file(
            map_dir,
            image_id,
            MAP_EXTENSIONS,
        )

        fixation_path = self._find_file(
            fixation_dir,
            image_id,
            FIXATION_EXTENSIONS,
        )

        return image_path, map_path, fixation_path

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")

        image = image.resize(
            self.input_size,
            Image.BILINEAR,
        )

        image = TF.to_tensor(image)

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return image

    def _load_density_map(self, path):
        """
        Restituisce due versioni della stessa density map:

        density_map_raw:
            valori in [0, 1]
            usata principalmente con MSE

        density_map_prob:
            valori >= 0
            somma totale = 1
            usata per B0, SIM e KLD
        """

        density_map = Image.open(path).convert("L")

        density_map = density_map.resize(
            self.input_size,
            Image.BILINEAR,
        )

        density_map = np.asarray(
            density_map,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # RAW MAP
        #
        # PIL grayscale produce valori 0...255.
        # Li portiamo nell'intervallo [0,1].
        # -------------------------------------------------

        density_map_raw = density_map / 255.0

        density_map_raw = np.clip(
            density_map_raw,
            a_min=0.0,
            a_max=1.0,
        )

        density_map_raw = torch.from_numpy(
            density_map_raw
        ).unsqueeze(0)

        # -------------------------------------------------
        # PROBABILITY MAP
        #
        # Partiamo dalla raw map e la normalizziamo
        # affinché la somma dei pixel sia 1.
        # -------------------------------------------------
        density_map_prob = normalize_probability_map(
            density_map_raw,
            eps=self.eps,
        )
        

        return (
            density_map_raw,
            density_map_prob,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path, map_path, fixation_path = (
            self._paths_for_sample(sample)
        )

        image = self._load_image(image_path)

        (
            density_map_raw,
            density_map_prob,
        ) = self._load_density_map(
            map_path
        )

        # Flip sincronizzato:
        # immagine + entrambe le density map.
        if self.augmentation and random.random() < 0.5:
            image = torch.flip(
                image,
                dims=[2],
            )

            density_map_raw = torch.flip(
                density_map_raw,
                dims=[2],
            )

            density_map_prob = torch.flip(
                density_map_prob,
                dims=[2],
            )

        return {
            "image": image,
            "density_map_raw": density_map_raw,
            "density_map_prob": density_map_prob,
            "fixation_path": str(fixation_path),
            "image_id": sample["image_id"],
            "split": sample["split"],
        }