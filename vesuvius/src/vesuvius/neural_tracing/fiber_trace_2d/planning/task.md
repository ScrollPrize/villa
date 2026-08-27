# Task: Support split constraint pieces in fiber BP

Allow `direction-ablation --bp-only` to use ordinary finite `--piece-length`
settings. Constraint extraction already splits each source fiber before spatial
search and creates same-source continuity links between consecutive pieces; BP
must operate on those pieces and consume those links instead of requiring one
piece per source fiber.
