# Task: grouped volume listing table

Change human-readable `las_manager volume ls` output from repeated labeled
records to an aligned table. Print labels once in a header, group consecutive
volumes by sample/scroll, print the sample ID once per group, and use tree
branch indicators below the first scroll/volume row for additional volumes. Do not print a
separate `ID` column because that ID is already the prefix of the long volume
name. Preserve `volume ls --json` unchanged.
Add a column listing OME scale groups already prefetched into the manager's
local volume store; show `-` when no local scale contains chunk data.
Format 3D volume shapes as depth/height/width with space-padded component
widths of 6/5/5 characters so each dimension aligns vertically.
