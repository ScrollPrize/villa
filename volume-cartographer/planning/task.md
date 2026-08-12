# Task: download queue statistics in the cache status bar

Extend VC3D's existing application cache status bar with queued chunk counts by
pyramid scaledown. During active remote downloads, format the status as
`RAM X/Y GiB disk X/Y GiB net N@XMiB/s qK A/B/C`, omitting leading and trailing
queue levels whose count is zero. For idle remote volumes, show `net idle`.
Do not add a setting or per-slice label.

Only show network/queue information for remote volumes and only show the queue
while remote downloads are in flight. This remains diagnostic groundwork for
later fetch-priority and adaptive parallel-download work.
