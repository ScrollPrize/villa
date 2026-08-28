# Task: extend and distance-weight signed winding evidence

Increase the H/V diagnostic and winding-BP default exclusive raw
winding-distance cutoff from `1.5` to `4.0`. Preserve the legacy parity
labeler's representable `<1.5` default.

For H/V-aware winding inference, progressively reduce the signed winding
evidence weight as its admitted half-integer target grows:

- `|target| = 0.5`: multiplier `1`
- `|target| = 1.5`: multiplier `0.5`
- `|target| = 2.5`: multiplier `0.25`
- `|target| = 3.5`: multiplier `0.125`

The decay applies to the complete winding-distance factor contribution, not to
the same constraint's independent H/V parallel/perpendicular relation evidence.
Its hard signed-order requirement remains unweighted.
