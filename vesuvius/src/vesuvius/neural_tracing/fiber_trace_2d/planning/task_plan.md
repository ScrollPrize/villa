# Plan: circular winding visibility-mask controls

## Implementation

1. Snapshot every managed winding layer's live `.visible` bit before a shift.
2. Circularly rotate that complete state-by-winding mask by one position over
   the sorted unique winding labels across every managed nonempty H, V, error,
   and tie layer, including Broken/Tie-only windings. `Next` moves source slot
   `i` to `(i+1) mod count`; `Previous` moves it to `(i-1) mod count`. State is
   unchanged. A missing source-state layer contributes `false`; a bit whose
   destination-state layer is absent is discarded. One winding is an exact
   no-op. Apply all bits from the snapshot so hidden space moves as well as
   visible space.
3. Include H, V, error, and tie layers in the roll. Keep the independent
   reference and unmanaged Napari layers untouched.
4. Replace arrow `QToolButton`s with full-size labeled `QPushButton`s. Keep the
   live visibility summary of all states/windings synchronized after controls
   and manual changes.
5. Build one shared palette entry per winding and use that exact entry for both
   H and V layer color arrays. Keep error and tie colors distinct.

## Testing

Test exact Next/Previous direction, complete mask roll in both directions,
wraparound, all-but-one-visible
empty-space movement, per-state bit preservation, sparse state layers, live
layer reads, one-winding no-op, and untouched unmanaged/reference layers. Verify actual fake
Napari H/V layer calls receive equal arrays. Run focused viewer tests and Ruff.

## Spec update

Replace the prior per-visible-H/V navigation contract with circular complete
managed-mask rotation and retain exact shared H/V color semantics.

## Docs updates

Document full-mask wraparound, labeled buttons, and shared H/V colors.

## Changelog

No separate changelog entry; this corrects the current viewer interaction.
