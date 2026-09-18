# M1 Development Run

## Protocol
- Seed: 42
- Train subset: 6000
- Tuning set: 2500
- Epochs: 3
- Batch size: 8
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Loss: MSE
- Target: density_map_raw
- Selection metric: CC

## Mini-overfit
- Initial MSE: 0.155427
- Final MSE: 0.008665
- Updates: 50
- Result: PASS

## B1
- MSE: 0.009114
- CC: 0.861871
- SIM: 0.760065
- KLD: 0.215867

## M1
- MSE: 0.008979
- CC: 0.866895
- SIM: 0.762698
- KLD: 0.212316

## Notes
M1 uses ResNet18 features C3/C4/C5 with skip connections.
The architecture was compared against B1 using the same training protocol.

## Commit
- Git commit: 0d7ffbb