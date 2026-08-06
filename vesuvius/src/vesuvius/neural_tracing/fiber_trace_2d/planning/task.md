# Task: path-aware user completion installation

Add a `las_manager` command that installs shell completion into the standard
per-user completion location. Completion must dispatch to the `las_manager`
resolved from the current shell `PATH`, retain the exact venv executable as a
registered provider, ignore providers whose environments no longer exist, and
allow providers from multiple virtual environments to coexist.

The existing `las_manager completion bash|zsh` output remains compatible.
