"""Pre-cleanup checkpoints kept usable for inference and tracing only, separate from the current models.

patch_model.py: the frozen patch encoder, context tokens, decoder layer and segment survival scorer both use.
flow_v5.py: the 'flow_matching' follower of the v4-extension and v5 runs (no memory or identity heads).
coordinate_regression.py: the 'coordinate_regression' follower without memory (e.g. mixed_ct_afv_forward35_v1).
    python -m vesuvius.neural_tracing.fiber_follow.legacy.infer --checkpoint CKPT --seed X,Y,Z --family H --out DIR
"""
