# Task log: path-aware completion installation

## Findings

- Standard Python `venv` activation changes `PATH` but has no completion hook
  protocol.
- The host has bash-completion's lazy `_completion_loader`, so the canonical
  user loader path is `${XDG_DATA_HOME:-~/.local/share}/bash-completion/completions/las_manager`.
- A single hard-coded completion file would select the wrong environment after
  switching venvs. The planned loader therefore dispatches by the executable
  currently resolved from `PATH`, backed by additive per-executable providers.

## Deviations

- Installation is intentionally Bash-only. Existing generated Zsh completion
  remains supported; installing into a user's Zsh `fpath` is shell-policy
  dependent and was not requested.

## Independent review

- Use the same portable canonical identity for install and runtime dispatch,
  including symlink coverage and no GNU-only `readlink -f`.
- Use digest-suffixed provider functions and dispatch only to the selected
  provider so multiple sourced files cannot overwrite one another.
- Cover `completion install [bash]`, prefixes, and completion suggestions in
  the shared parser/registry tests.
- All three recommendations were incorporated before implementation.

## Validation

- `35 passed, 6 warnings`: focused manager, open-data, provenance, and packaging
  tests. Warnings are pre-existing Pydantic v2 deprecations from the checked-out
  Atlas models.
- A real installed `/home/hendrik/.venv_las/bin/las_manager completion install`
  smoke test under a temporary `XDG_DATA_HOME` created the canonical loader,
  registry, and provider; the loader and generated Bash output pass `bash -n`.
- Two fake venv providers coexist, reinstall idempotently, and produce distinct
  dynamic results when `PATH` switches between them.
- Canonical provider identity is stable through a symlink.
- `python -m py_compile` and `git diff --check` pass.
