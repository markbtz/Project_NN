# Evaluation protocol

## Primary metrics

The main evaluation protocol uses:

- CC
- SIM
- KLD

These three metrics must be reported for every model in the main experimental pipeline.

## Optional metrics

NSS and sAUC remain implemented in the project but are not part of the critical path for the main evaluation.

- NSS: optional; include it if time allows.
- sAUC: optional unless explicitly required by the course or later selected by the team.

The existing NSS and sAUC implementations must not be removed.

## Evaluation split

During development, debugging, and model selection, use only the `tuning` split.

The `internal_test` split remains frozen until architectures, losses, and the experimental protocol are finalized.

Access to `internal_test` requires an explicit final evaluation.

## Comparison rule

Per-image model comparisons must be performed using `image_id` alignment.

Do not assume that two CSV files are aligned simply because their rows have the same order.

Missing or duplicate IDs must raise an error.
