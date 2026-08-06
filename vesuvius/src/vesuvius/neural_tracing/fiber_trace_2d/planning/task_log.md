# Task log: reliable manager tmux attachment

## Findings

- The latest run window exists in the user's attached session as tmux window
  `@32` / index 12 / name `manager`, but its original manager session no longer
  exists. Consequently session-only `run ls` and attach validation lose it.
- The current code links from `<session>:0`, guesses the new numeric index, and
  selects by that mutable index. Tmux exposes stable `@window_id` handles that
  remain valid across links and renumbering.

## Deviations

- None.

## Plan review

- Added atomic `window_id` capture, run-UUID tagging/validation, explicit
  runner PID, stale-ID and no-window behavior, and exact session-qualified
  selection.
- Added existing-config migration and `--device` versus `--devices` precedence
  requirements for global params.

## Validation

- Focused manager/open-data/packaging/bootstrap suite: 51 passed.
- The existing home config loads from outside the checkout with `PYTHONPATH`
  removed and displays the exact requested eight params tokens.
- Python compilation and `git diff --check` completed cleanly.

## Implementation notes

- Further inspection showed the visible `@32` window was the user's ordinary
  Bash `manager` window, not the inference wrapper. The current inference child
  is orphaned (`PPID 1`), so no tmux terminal remains to attach; its log and
  computation remain live. New runs retain and validate wrapper window identity.
- A subsequent healthy run exposed that the wrapper redirected child output
  only to `run.log`, leaving its attached pane blank. The wrapper now tees raw
  child bytes to both destinations, including carriage-return progress.
