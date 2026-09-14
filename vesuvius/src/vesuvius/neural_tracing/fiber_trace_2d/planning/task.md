# Training span eligibility

For training and its test-loss loaders retain only cspline and trace spans;
exclude Lasagna spans and extrapolation outside control points. Do not change
the separate tracing evaluator. Prepare a fresh regular non-embedding five-scroll
JSON-only training config at LR0.01 (no Paris4 NML). Implement and verify before
starting a long training.
