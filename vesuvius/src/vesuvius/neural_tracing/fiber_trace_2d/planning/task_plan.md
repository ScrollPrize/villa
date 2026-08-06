# Plan: reliable manager tmux attachment

## Implementation

1. Treat tmux's stable `window_id` value (for example `@32`), not mutable
   session/window indices, as the durable runtime handle. Capture it atomically
   from `new-session -P -F`, tag it with the immutable run UUID, and store it.
   Validate ID+UUID before every attach/live decision so a restarted tmux
   server cannot redirect a stale ID to an unrelated window.
2. When attaching inside tmux, resolve the source window ID, link it after the
   current window using an explicit current-session target, and select it by
   stable ID. If already linked into the current session, only select it.
3. Support records without a stored ID by resolving and tagging their original
   live session. Store `runner_pid` separately from the inference child PID for
   diagnostics; do not claim an orphan inference child is attachable when no
   tmux window remains. Persist recovery only from attach/reconciliation, never
   from completion.
4. Make reconciliation and live `run ls` accept either a live session or stable
   live window. Preserve normal outside-tmux session attachment.
5. Add a validated global string-array `params` configuration field. Render it
   on `config init` with tile size 512, border 32, overlap 96, and all devices;
   insert it before arguments supplied after `--`, so explicit per-run values
   retain final override precedence. Normalize the mutually exclusive
   `--device`/`--devices` defaults when either is explicitly supplied. Record
   the fully resolved argv as before.
6. Tee inference stdout/stderr byte-for-byte from the child pipe into both the
   durable log and wrapper stdout, preserving carriage-return progress, signal
   forwarding, backpressure, and the real child exit code.

## Tests

7. Add command-level tests for atomic window-ID capture and UUID tagging,
   exact link/select targets, already-linked behavior, stale ID rejection,
   legacy records, missing windows, and live listing.
8. Add missing-key backward compatibility, exact TOML token-array rendering,
   validation/config-show, both-backend argv precedence, and singular/plural
   device override coverage.
9. Add byte-exact runner tee coverage including carriage-return progress and a
   nonzero child exit.
   Run focused manager tests, compilation, and diff checks.

## Spec update

Specify stable window-ID ownership and recovery rather than session-name-only
attachment, plus global default inference parameter ordering.

## Docs updates

Document stable tmux window identity and the initialized global inference
parameters, including explicit per-run override order.

## Changelog

Add a short entry for reliable linked-window attachment.
