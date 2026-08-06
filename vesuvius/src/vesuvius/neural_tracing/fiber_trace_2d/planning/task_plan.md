# Plan: path-aware completion installation

## Design

1. Preserve `las_manager completion bash|zsh` as the stdout-generation API.
2. Add `las_manager completion install [bash]`. It discovers the running
   console script using `sys.argv[0]`, records its normalized absolute path as
   a provider, and atomically installs files below `${XDG_DATA_HOME:-~/.local/share}`.
3. Store one provider file per executable identity below a manager-owned data
   directory. Install a canonical Bash loader at
   `bash-completion/completions/las_manager`. The loader resolves
   the external `las_manager` executable from `PATH` at completion time and
   dispatches only to the provider whose identity matches. Install and runtime
   identity both use Python's canonical resolved executable path plus SHA-256;
   the selected executable reports that identity through a config-free hidden
   command. This handles symlinks portably without GNU-only `readlink -f`.
   Missing/deleted venv providers are inert. Reinstalling one provider is
   idempotent and never deletes providers belonging to other venvs.
4. Provider functions use their exact executable path for cached dynamic
   snapshot, volume, inference, and run completion. They do not activate a
   venv, execute arbitrary activation hooks, access the network, or select a
   different installed `las_manager`.
5. Keep generated files deterministic, safely shell-quoted, and atomically
   replaced. Print the installed loader path and a concise new-shell/reload
   instruction.
6. Give every provider a digest-suffixed shell function and have the canonical
   dispatcher load and call only the matching one, so sourcing multiple
   providers cannot make the last installed environment win.
7. Update the shared parser/completion grammar for
   `completion install [bash]`, including unique prefixes and Tab suggestions,
   while preserving `completion bash|zsh`.

## Compatibility and scope

- Initial installation support is Bash, matching the current host and the
  standard lazy loader being used. Existing generated Zsh completion remains
  available but is not silently installed into a user-specific `fpath`.
- An editable install and a normal console-script install are treated alike:
  executable identity is the console-script path, not the source checkout.
- If the command is invoked as `python -m lasagna.manager.cli`, installation
  fails clearly because there is no stable `las_manager` console-script path.

## Tests

- Unit-test install paths under a temporary `XDG_DATA_HOME`.
- Install two fake/live executable providers and assert both provider files
  survive and PATH switching produces observably different dynamic completion
  from each selected provider.
- Assert reinstall is idempotent, generated Bash passes `bash -n`, missing
  providers are skipped, symlink invocation has the canonical identity, and
  aliases/functions are not selected in place of an external executable.
- Assert the new install grammar, unique prefixes, and completion suggestions.
- Preserve existing completion-generation and registry coverage tests.
- Run the focused manager suite and `git diff --check`.

## Spec update

Add the per-user completion installer, path-exact provider dispatch, parallel
venv coexistence, no-network behavior, and Bash-only installation scope to
`planning/specs.md`.

## Docs updates

Update `lasagna/docs/manager.md` and `lasagna/README.md` with the canonical
one-time install command, file locations, multi-venv behavior, and reload
instructions. Retain the `eval` form for temporary shells and Zsh.

## Changelog

Add a dated entry for path-aware, multi-venv Bash completion installation.
