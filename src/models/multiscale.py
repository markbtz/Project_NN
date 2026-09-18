import torch
import torch.nn as nn

from src.models.baseline import (
    ResNet18Encoder,
    to_probability_map,
)


class MultiScaleDecoder(nn.Module):
    """
    Decoder M1.

    Mantiene il piu' possibile la struttura del decoder B1,
    aggiungendo skip connections da C4 e C3.

    B1:
        C5 -> up -> up -> up -> up -> up -> output

    M1:
        C5 -> up + C4 -> up + C3 -> up -> up -> up -> output
    """

    def __init__(self, channels, width=96):
        super().__init__()

        # -------------------------------------------------
        # C5 -> risoluzione C4
        #
        # Come primo blocco del decoder B1:
        # 512 canali -> 96 canali
        # -------------------------------------------------
        self.up_c5_to_c4 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                channels["C5"],
                width,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # C4 ha 256 canali.
        # La proiettiamo a 96 per poterla sommare a x.
        self.proj_c4 = nn.Conv2d(
            channels["C4"],
            width,
            kernel_size=1,
        )

        # -------------------------------------------------
        # C4 -> risoluzione C3
        #
        # Come secondo blocco del decoder B1:
        # 96 -> 48 canali
        # -------------------------------------------------
        self.up_c4_to_c3 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                width,
                width // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # C3 ha 128 canali.
        # La proiettiamo a 48.
        self.proj_c3 = nn.Conv2d(
            channels["C3"],
            width // 2,
            kernel_size=1,
        )

        # -------------------------------------------------
        # Parte finale uguale concettualmente a B1
        #
        # 48 -> 24 -> 12 -> 12
        # con tre upsampling x2.
        # -------------------------------------------------
        self.tail = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                width // 2,
                width // 4,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                width // 4,
                width // 8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                width // 8,
                width // 8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(
            width // 8,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        c3,
        c4,
        c5,
        output_size,
    ):
        # C5: circa 6x8
        # -> circa 12x16, stessa risoluzione di C4
        x = self.up_c5_to_c4(c5)

        # Skip connection C4
        x = x + self.proj_c4(c4)

        # -> circa 24x32, stessa risoluzione di C3
        x = self.up_c4_to_c3(x)

        # Skip connection C3
        x = x + self.proj_c3(c3)

        # 24x32 -> 48x64 -> 96x128 -> 192x256
        x = self.tail(x)

        x = self.head(x)

        # Sicurezza per eventuali dimensioni non divisibili esattamente
        if tuple(x.shape[-2:]) != tuple(output_size):
            x = nn.functional.interpolate(
                x,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return x


class M1MultiScale(nn.Module):
    """
    M1:
    ResNet18 + feature multi-scala C3/C4/C5.

    Rispetto a B1:
    - stesso encoder ResNet18
    - stessa MSE
    - stesso target density_map_raw
    - decoder simile
    - aggiunge skip connection da C4 e C3
    """

    def __init__(
        self,
        pretrained=True,
        decoder_width=96,
    ):
        super().__init__()

        self.encoder = ResNet18Encoder(
            pretrained=pretrained
        )

        self.decoder = MultiScaleDecoder(
            channels=self.encoder.out_channels,
            width=decoder_width,
        )

    def forward(self, x):
        feats = self.encoder(x)

        logits = self.decoder(
            c3=feats["C3"],
            c4=feats["C4"],
            c5=feats["C5"],
            output_size=tuple(x.shape[-2:]),
        )

        return torch.sigmoid(logits)

    def predict_probability(
        self,
        x,
        eps=1e-6,
    ):
        return to_probability_map(
            self.forward(x),
            eps=eps,
        )

