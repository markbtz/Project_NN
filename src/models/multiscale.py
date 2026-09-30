import torch
import torch.nn as nn

from src.models.baseline import (
    ResNet18Encoder,
    to_probability_map,
)


class MultiScaleDecoder(nn.Module):
    """
    M1 decoder.

    Preserves the B1 decoder structure as much as possible,
    while adding skip connections from C4 and C3.

    B1:
        C5 -> up -> up -> up -> up -> up -> output

    M1:
        C5 -> up + C4 -> up + C3 -> up -> up -> up -> output
    """

    def __init__(self, channels, width=96):
        super().__init__()

        # -------------------------------------------------
        # C5 -> C4 resolution
        #
        # Same structure as the first B1 decoder block:
        # 512 channels -> 96 channels
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

        # C4 has 256 channels.
        # Project to 96 channels before adding to x.
        self.proj_c4 = nn.Conv2d(
            channels["C4"],
            width,
            kernel_size=1,
        )

        # -------------------------------------------------
        # C4 -> C3 resolution
        #
        # Same structure as the second B1 decoder block:
        # 96 -> 48 channels
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

        # C3 has 128 channels.
        # Project to 48 channels.
        self.proj_c3 = nn.Conv2d(
            channels["C3"],
            width // 2,
            kernel_size=1,
        )

        # -------------------------------------------------
        # Final part conceptually identical to B1
        #
        # 48 -> 24 -> 12 -> 12
        # followed by three 2x upsampling blocks.
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
        # C5: approximately 6x8
        # -> about 12x16, matching C4 resolution
        x = self.up_c5_to_c4(c5)

        # C4 skip connection
        x = x + self.proj_c4(c4)

        # -> about 24x32, matching C3 resolution
        x = self.up_c4_to_c3(x)

        # C3 skip connection
        x = x + self.proj_c3(c3)

        # 24x32 -> 48x64 -> 96x128 -> 192x256
        x = self.tail(x)

        x = self.head(x)

        # Handle dimensions that are not exactly divisible.
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
    ResNet18 + C3/C4/C5 multi-scale features.

    Compared to B1:
    - same ResNet18 encoder
    - same MSE
    - same density_map_raw target
    - similar decoder
    - adds skip connections from C4 and C3
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

