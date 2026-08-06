# Task: reliable manager tmux attachment

Fix `las_manager tmux attach` so an inference window can be linked and selected
inside an existing tmux session reliably, including when its original detached
session no longer exists after the window was linked. Preserve live-run
discovery for such linked windows.

Add a global manager `params` array whose initialized default is
`--tile-size 512 --border 32 --overlap 96 --devices all`. Apply it to both
managed inference backends before per-run arguments, and update the user's
existing manager config to the same value.

Make attached manager tmux windows display live inference output while keeping
the identical durable byte stream in `run.log`.

Name new tmux tabs with a short scroll-specific label such as
`inf-PHerc0332-84af`, not the generic `inference`.
