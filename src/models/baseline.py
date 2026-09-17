"""
Modelli B0 (Center Prior) e B1 (ResNet18 + decoder singolo, senza skip).
Vedi configs/experiments.yaml per la definizione del disegno sperimentale.

B1 usa SOLO la feature finale dell'encoder (C5, stride 32): niente skip
connections qui — quelle arrivano con M1 (di competenza di C). Tenere questo
file "semplice" e' intenzionale: e' il confronto pulito rispetto a cui M1
deve dimostrare un miglioramento.

Smoke test locale (funziona anche senza dataset reale):
    python src/models/baseline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.runtime import get_device
from src.saliency_maps import normalize_probability_map


def to_probability_map(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Wrapper retrocompatibile per la normalizzazione probabilistica.
    """

    return normalize_probability_map(
        x,
        eps=eps,
    )

class CenterPriorB0(nn.Module):
    """
    B0 — center prior: nessun training via backprop.

    La mappa e' la media delle density_map_prob del training set, cioe'
    density map gia' normalizzate come distribuzioni di probabilita' con
    somma spaziale circa uguale a 1.

    Qui il prior viene inizializzato come distribuzione uniforme.
    Usare fit() o fit_from_loader() per calcolarlo sui dati reali SALICON.
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        # buffer, non parametro: non si allena via backprop
        self.register_buffer("center_map", torch.ones(1, 1, height, width) / (height * width))

    @torch.no_grad()
    def fit(self, density_maps: torch.Tensor, eps: float = 1e-6):
        """
        density_maps: (N, 1, H, W) density_map_prob del training set, GIA'
        interamente in memoria. Va bene per pochi campioni (es. smoke test);
        per il dataset reale (10.000 immagini, ~1.9 GB solo per questo
        tensore) usare fit_from_loader() per non rischiare OOM su Colab
        free o M1 Max.
        """
        mean_map = density_maps.mean(dim=0, keepdim=True)
        self.center_map.copy_(to_probability_map(mean_map, eps=eps))

    @torch.no_grad()
    def fit_from_loader(self, density_map_batches, eps: float = 1e-6):
        """
        Come fit(), ma accumula una somma incrementale batch per batch
        invece di concatenare tutto in un unico tensore in memoria.

        density_map_batches: iterable di tensori (B, 1, H, W) — tipicamente
        un generatore che itera un DataLoader e restituisce
        batch["density_map_prob"]. Esempio d'uso in scripts/train.py:

            def density_batches():
                for batch in loader:
                    yield batch["density_map_prob"].to(device)
            model.fit_from_loader(density_batches())
        """
        total = None
        count = 0
        for batch in density_map_batches:
            batch_sum = batch.sum(dim=0, keepdim=True).to(self.center_map.dtype)
            total = batch_sum if total is None else total + batch_sum
            count += batch.shape[0]
        if count == 0:
            raise ValueError("fit_from_loader: nessun batch ricevuto, il loader e' vuoto.")
        mean_map = total / count
        self.center_map.copy_(to_probability_map(mean_map, eps=eps))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.center_map.expand(batch_size, -1, -1, -1)


class ResNet18Encoder(nn.Module):
    """
    Estrae C2/C3/C4/C5. B1 usa solo C5, ma le altre feature restano
    disponibili come attributi cosi' M1 (skip connections, di competenza di C)
    puo' riusare lo stesso encoder senza riscriverlo.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = torchvision.models.resnet18(weights=weights)
        except AttributeError:
            # API vecchia di torchvision
            backbone = torchvision.models.resnet18(pretrained=pretrained)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # C2, stride 4,  64 canali
        self.layer2 = backbone.layer2  # C3, stride 8,  128 canali
        self.layer3 = backbone.layer3  # C4, stride 16, 256 canali
        self.layer4 = backbone.layer4  # C5, stride 32, 512 canali
        self.out_channels = {"C2": 64, "C3": 128, "C4": 256, "C5": 512}

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"C2": c2, "C3": c3, "C4": c4, "C5": c5}


class SingleBottleneckDecoder(nn.Module):
    """
    Decoder di B1: SOLO C5 in ingresso, 5 blocchi di upsampling bilineare x2
    + conv per tornare da stride 32 alla risoluzione di input (5 blocchi = x32).
    Nessuna skip connection: e' la differenza esplicita rispetto a M1.
    """
    def __init__(self, in_channels: int = 512, width: int = 96):
        super().__init__()
        channels = [in_channels, width, width // 2, width // 4, width // 8, width // 8]
        blocks = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(channels[-1], 1, kernel_size=1)

    def forward(self, c5: torch.Tensor, output_size: tuple) -> torch.Tensor:
        x = c5
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        # resize di sicurezza se lo stride non divide esattamente H/W
        if tuple(x.shape[-2:]) != tuple(output_size):
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class B1Baseline(nn.Module):
    """
    B1 completo: ResNet18Encoder (solo C5) + SingleBottleneckDecoder.
    Loss di riferimento: MSE (vedi configs/experiments.yaml), calcolata in
    train.py direttamente contro density_map_raw del dataset.

    forward() restituisce una mappa "grezza" (0-1 per pixel, via sigmoid),
    NON normalizzata a somma 1 — la MSE di training va calcolata su questa,
    confrontandola con density_map_raw (stessa scala). Usare
    predict_probability() (o to_probability_map() direttamente) solo in
    fase di valutazione, per CC/SIM/KLD, confrontandola con density_map_prob.
    """
    def __init__(self, pretrained: bool = True, decoder_width: int = 96):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained)
        self.decoder = SingleBottleneckDecoder(
            in_channels=self.encoder.out_channels["C5"], width=decoder_width
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        logits = self.decoder(feats["C5"], output_size=tuple(x.shape[-2:]))
        return torch.sigmoid(logits)  # mappa grezza, 0-1 per pixel

    def predict_probability(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Solo per valutazione: converte l'output grezzo in probabilita' (somma 1)."""
        return to_probability_map(self.forward(x), eps=eps)


if __name__ == "__main__":
    # Smoke test locale: verifica solo che shape e normalizzazione siano corrette.
    # Non richiede il dataset reale.
    device = get_device()
    print(f"Device: {device}")

    batch_size, height, width = 4, 192, 256
    dummy_input = torch.rand(batch_size, 3, height, width, device=device)

    model = B1Baseline(pretrained=True).to(device)
    raw_output = model(dummy_input)
    print(f"B1 output grezzo shape: {tuple(raw_output.shape)} (atteso: ({batch_size}, 1, {height}, {width}))")
    print(f"B1 range valori grezzi (min/max, atteso in [0,1]): "
          f"{raw_output.min().item():.4f} / {raw_output.max().item():.4f}")

    prob_output = model.predict_probability(dummy_input)
    sums = prob_output.sum(dim=(-2, -1)).flatten().tolist()
    print(f"B1 probabilita' — somma per immagine (deve essere ~1): {[round(s, 6) for s in sums]}")

    b0 = CenterPriorB0(height=height, width=width).to(device)
    dummy_density_raw = torch.rand(10, 1, height, width, device=device)
    dummy_density_prob = to_probability_map(dummy_density_raw)
    b0.fit(dummy_density_prob)
    center_out = b0(batch_size)
    print(f"B0 output shape: {tuple(center_out.shape)}")
    print(f"B0 somma (deve essere ~1): {center_out[0].sum().item():.6f}")

    # Smoke test aggiuntivo per fit_from_loader (nuova funzionalita')
    b0_loader = CenterPriorB0(height=height, width=width).to(device)

    def dummy_batches():
        for _ in range(3):
            raw = torch.rand(4, 1, height, width, device=device)
            yield to_probability_map(raw)

    b0_loader.fit_from_loader(dummy_batches())
    center_out_loader = b0_loader(batch_size)
    print(f"B0 (fit_from_loader) somma (deve essere ~1): {center_out_loader[0].sum().item():.6f}")
