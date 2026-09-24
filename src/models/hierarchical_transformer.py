import torch
import torch.nn as nn
import timm

from src.models.baseline import to_probability_map
from src.models.multiscale import MultiScaleDecoder


class PVTv2B1Encoder(nn.Module):
    """
    Encoder Transformer gerarchico per M2.

    Usa PVTv2-B1 tramite timm e restituisce tre feature map
    multi-scala compatibili con l'interfaccia C3/C4/C5 usata da M1.

    Per input 192x256 ci aspettiamo:

        C3 -> (B, 128, 24, 32)
        C4 -> (B, 320, 12, 16)
        C5 -> (B, 512,  6,  8)
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "pvt_v2_b1",
            pretrained=pretrained,
            features_only=True,
        )

        channels = self.backbone.feature_info.channels()
        reductions = self.backbone.feature_info.reduction()

        # PVTv2-B1 deve produrre:
        # channels   = [64, 128, 320, 512]
        # reductions = [4, 8, 16, 32]
        if len(channels) < 4:
            raise RuntimeError(
                f"PVTv2-B1 ha restituito solo {len(channels)} livelli."
            )

        self.out_channels = {
            "C3": channels[1],
            "C4": channels[2],
            "C5": channels[3],
        }

        self.out_reductions = {
            "C3": reductions[1],
            "C4": reductions[2],
            "C5": reductions[3],
        }

    def forward(self, x):
        features = self.backbone(x)

        return {
            "C3": features[1],
            "C4": features[2],
            "C5": features[3],
        }


class M2HierarchicalTransformer(nn.Module):
    """
    M2:
    PVTv2-B1 + decoder multi-scala di M1.

    Rispetto a M1-L:
    - cambia l'encoder: ResNet18 -> PVTv2-B1
    - mantiene lo stesso schema multi-scala C3/C4/C5
    - mantiene MultiScaleDecoder con width=96
    - mantiene target density_map_prob
    - mantiene loss CC+KLD 0.5/0.5

    Obiettivo sperimentale:
    isolare l'effetto del cambio di encoder.
    """

    def __init__(
        self,
        pretrained=True,
        decoder_width=96,
    ):
        super().__init__()

        self.encoder = PVTv2B1Encoder(
            pretrained=pretrained,
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