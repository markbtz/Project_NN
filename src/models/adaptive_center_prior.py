import torch
import torch.nn as nn

from src.models.baseline import (
    CenterPriorB0,
    to_probability_map,
)
from src.models.multiscale import M1MultiScale


class AdaptiveCenterPriorG(nn.Module):
    """
    G - Adaptive Center Prior.

    Base:
        M1-L

    Idea:
        alpha(x) = sigmoid(MLP(GAP(C5)))

        output =
            (1 - alpha) * M1-L
            + alpha * center_prior

    alpha is computed independently for each image.
    """

    def __init__(
        self,
        height,
        width,
        pretrained=True,
        decoder_width=96,
        gate_hidden=64,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        # ---------------------------------------------
        # M1-L base
        # ---------------------------------------------
        self.base_model = M1MultiScale(
            pretrained=pretrained,
            decoder_width=decoder_width,
        )

        # ---------------------------------------------
        # B0 center prior
        #
        # Initially uniform.
        # Before training G, load
        # B0_center_map.pt.
        # ---------------------------------------------
        self.center_prior = CenterPriorB0(
            height=height,
            width=width,
        )

        # ---------------------------------------------
        # Adaptive gate:
        #
        # C5: (B, 512, H/32, W/32)
        # GAP -> (B, 512)
        # MLP -> (B, 1)
        # sigmoid -> alpha in [0, 1]
        # ---------------------------------------------
        c5_channels = (
            self.base_model.encoder.out_channels["C5"]
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.gate = nn.Sequential(
            nn.Linear(
                c5_channels,
                gate_hidden,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                gate_hidden,
                1,
            ),
            nn.Sigmoid(),
        )

        # ---------------------------------------------
        # Zero-initialize the gate output layer.
        #
        # Otherwise, alpha starts from an arbitrary value determined
        # by random weight initialization (possibly near 0
        # or 1 by chance); with weak gradients (few
        # parameters and a frozen base), it may remain there throughout
        # training instead of learning image-specific variation.
        #
        # With a zeroed output layer, the pre-sigmoid logit is
        # always 0, so alpha starts at exactly 0.5 for every image
        # (neutral blend of M1-L and the center prior), and gradients determine
        # how it moves for each image.
        # ---------------------------------------------
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)

        self._base_frozen = False

    def forward(
        self,
        x,
        return_alpha=False,
    ):
        # ---------------------------------------------
        # Extract features once.
        # ---------------------------------------------
        feats = self.base_model.encoder(x)

        logits = self.base_model.decoder(
            c3=feats["C3"],
            c4=feats["C4"],
            c5=feats["C5"],
            output_size=tuple(x.shape[-2:]),
        )

        base_raw = torch.sigmoid(logits)

        # M1-L forward returns values in [0, 1],
        # while B0 is already a probability map.
        #
        # Normalize both before combining them
        # to the same scale: spatial sum = 1.
        base_prob = to_probability_map(
            base_raw,
            eps=self.eps,
        )

        # ---------------------------------------------
        # Compute alpha from global C5 information.
        # ---------------------------------------------
        pooled = self.global_pool(
            feats["C5"]
        )

        pooled = pooled.flatten(1)

        alpha = self.gate(
            pooled
        )

        alpha = alpha.view(
            -1,
            1,
            1,
            1,
        )

        # ---------------------------------------------
        # B0 is identical for all images,
        # but is expanded to the batch size.
        # ---------------------------------------------
        prior = self.center_prior(
            x.shape[0]
        )

        # ---------------------------------------------
        # Adaptive Center Prior
        # ---------------------------------------------
        output = (
            (1.0 - alpha) * base_prob
            + alpha * prior
        )

        if return_alpha:
            return output, alpha

        return output

    def predict_probability(
        self,
        x,
        eps=1e-6,
    ):
        # G.forward() already returns a probability map.
        return self.forward(x)

    def load_base_state_dict(
        self,
        state_dict,
    ):
        """
        Carica i pesi del best checkpoint M1-L.
        """
        self.base_model.load_state_dict(
            state_dict
        )

    def load_center_prior_state_dict(
        self,
        state_dict,
    ):
        """
        Carica B0_center_map.pt.
        """
        self.center_prior.load_state_dict(
            state_dict
        )

    def freeze_base_and_prior(self):
        """
        Congela M1-L e B0.

        Durante il training di G resta allenabile
        soltanto il gate che produce alpha.
        """
        self._base_frozen = True

        self.base_model.requires_grad_(False)
        self.center_prior.requires_grad_(False)

        self.gate.requires_grad_(True)

        self.base_model.eval()
        self.center_prior.eval()

        return self

    def train(self, mode=True):
        """
        Mantiene M1-L e B0 in eval mode anche quando
        il trainer chiama model.train().
        """
        super().train(mode)

        if self._base_frozen:
            self.base_model.eval()
            self.center_prior.eval()
            self.gate.train(mode)

        return self