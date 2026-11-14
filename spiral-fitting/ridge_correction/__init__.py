"""Post-fit CT-ridge correction of fitted Spiral checkpoints.

Entry point: ``ridge_correct_checkpoint.py``. ``stage`` holds the three steps
(estimate, field, write) with the CT and checkpoint plumbing; ``profiles``,
``estimate``, ``field``, ``gapmap`` and ``writer`` hold the numerical method;
``ctstore`` reads the CT.
"""
