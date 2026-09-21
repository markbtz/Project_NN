from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.multiscale import M1MultiScale
from src.models.baseline import to_probability_map


device = torch.device("cpu")

batch_size = 2
height = 192
width = 256

print("=== Smoke test M1 ===")

model = M1MultiScale(
    pretrained=False,
    decoder_width=96,
).to(device)

x = torch.rand(
    batch_size,
    3,
    height,
    width,
    device=device,
)

# Controllo feature ResNet18
with torch.no_grad():
    feats = model.encoder(x)

print("C3 shape:", tuple(feats["C3"].shape))
print("C4 shape:", tuple(feats["C4"].shape))
print("C5 shape:", tuple(feats["C5"].shape))

# Forward completo
output = model(x)

print("Output shape:", tuple(output.shape))
print(
    "Output min/max:",
    output.min().item(),
    output.max().item(),
)

assert tuple(output.shape) == (
    batch_size,
    1,
    height,
    width,
)

assert torch.isfinite(output).all()
assert output.min().item() >= 0.0
assert output.max().item() <= 1.0

# Backward
loss = output.mean()
loss.backward()

has_gradients = any(
    p.grad is not None
    for p in model.parameters()
    if p.requires_grad
)

assert has_gradients

print("Backward: OK")

# Probability map
with torch.no_grad():
    prob = to_probability_map(
        output.detach()
    )

sums = prob.sum(
    dim=(-2, -1)
).flatten()

print(
    "Probability sums:",
    sums.tolist(),
)

assert torch.allclose(
    sums,
    torch.ones_like(sums),
    atol=1e-5,
)

print("SMOKE TEST M1: OK")