# Task: grouped volume listing table

Change human-readable `las_manager volume ls` output from repeated labeled
records to an aligned table. Print labels once in a header, group consecutive
volumes by sample/scroll, print the sample ID once per group, and use tree
branch indicators for each volume. Preserve `volume ls --json` unchanged.
Add a column listing OME scale groups already prefetched into the manager's
local volume store; show `-` when no local scale contains chunk data.
