# Task: contextual help and argument completion

Extend `las_manager` so a final literal `help` prints help for the longest
valid command prefix (for example `volume help` and `volume prefetch help`).

Extend shell completion beyond command names to cover available positional
arguments and option values, including cached volumes, snapshots, runs,
inferences, samples, formats, backends, and locally known OME-Zarr scales.
Completion must understand unique command abbreviations and remain read-only
and network-free.

Fix `volume ls` for valid catalog entries whose optional `properties.shape` is
explicitly `null`; these entries remain listable with an unknown shape.
