# Plan: contextual help and argument completion

## Contextual help

1. Add a preprocessing pass before argparse that recognizes only a final exact
   `help` token and rewrites it to `--help` for the longest valid unique command
   prefix before it. Thus `volume help`, `vol pre help`, and
   `inference run help` select the deepest understood parser, while
   `volume nonsense help` selects `volume` help. An invalid first token has no
   understood prefix and remains an error. Existing `--help` behavior is
   unchanged, and a `help` token after `--` is forwarded untouched.
2. Help exits before global config loading and therefore works on a fresh
   installation.

## Shared contextual completion

3. Replace shell-specific positional case tables with one Python completion
   resolver driven by the shared command registry and the complete command-line
   prefix. Bash and Zsh adapters only transport words/current position and
   render returned candidates. Unique abbreviations are normalized by the same
   resolver used for execution.
4. Complete command/subcommand names, valid flags, static option values, and
   dynamic selectors for snapshots, volumes, durable inferences, and live runs.
   Also complete cached sample IDs and data formats for `volume ls`, and backend
   values for snapshot/inference commands.
   Support both `--option VALUE` and `--option=PREFIX` forms, and stop manager
   completion after `--` so inference-backend arguments are untouched.
5. Complete a selected volume's available scale indices only from exact local
   evidence: OME multiscale dataset paths in the manager volume's `.zattrs`, or
   numeric local groups containing `.zarray` metadata. The open-data catalog
   currently does not declare source pyramid levels. If no local scale metadata
   exists, return no scale candidates rather than inventing levels or accessing
   S3. A prefetch downloads the relevant metadata and makes subsequent scale
   completion available.
6. Preserve tab-separated candidate annotations at the Python boundary while
   shell adapters insert only the candidate value, matching current behavior,
   and preserve all previous safety
   constraints: completion cannot refresh the catalog, open uncached
   checkpoints, reconcile/mutate runs, download chunks, or access the network.
   Missing config/cache produces static candidates where possible and no
   dynamic candidates, not an error printed into the shell.

## Catalog robustness

7. Normalize nullable optional catalog collections before iteration. In
   particular, `properties.shape = null` maps to the existing empty/unknown
   `VolumeRecord.shape`, so one incomplete entry cannot break `volume ls` or
   volume completion. Preserve all raw catalog data and other volume identity.

## Tests

- Test contextual help at top-level groups, leaves, abbreviated prefixes, and
  invalid/ambiguous suffixes without a config, plus forwarding after `--`.
- Test full and abbreviated positional completion for snapshots, volumes,
  scale indices, runs/inferences, samples, formats, backend values, and flags.
- Test `.zattrs` and numeric-group scale discovery plus the no-metadata empty
  result, and assert no network/cache mutation occurs.
- Regress the real nullable-shape catalog form and `volume ls` rendering.
- Test generated Bash/Zsh adapters and the installed multi-venv dispatcher.
- Test separated and equals option values, adapter word-index transport, and
  manager completion termination at `--`.
- Run focused manager suites, installed CLI smoke tests, shell syntax checks,
  Python compilation, and `git diff --check`.

## Spec update

Add final-`help` semantics, shared contextual completion, exact cache-only scale
discovery, abbreviation parity, no-network/no-mutation constraints, and
nullable optional catalog-field handling.

## Docs updates

Update `lasagna/docs/manager.md` and `lasagna/README.md` with contextual help,
argument completion coverage, and the cache-local limitation for first-time
scale proposals.

## Changelog

Add a dated entry for contextual help and argument-aware completion.
