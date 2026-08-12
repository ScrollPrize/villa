# VC3D render and fetch specification

## Invariants

- Rendering values, interpolation, pyramid transforms, and cache contents must
  not change as a side effect of diagnostics or scheduling work.
- Remote fetching must remain asynchronous; UI and render threads must not wait
  on network or persistent-cache I/O.
- Queue diagnostics must come from the existing chunk-cache request state, not
  from a parallel accounting system at viewer call sites.
- A shared cache must report each unresolved chunk once regardless of how many
  viewers requested it.
- Download diagnostics belong in the existing application cache status bar,
  not in per-slice overlays or a separate setting.

## Download label

During active remote downloads, the existing cache status bar appends:

`qK X/Y/Z`

- `K` is the first pyramid scaledown level with unresolved chunk requests.
- `X/Y/Z` are unresolved request counts for consecutive levels from the first
  through the last nonzero level. Interior zero counts are retained; leading
  and trailing zero levels are omitted.
- The queue item is shown only for remote volumes while remote fetches are in
  flight. Remote volumes otherwise show `net idle`; local volumes have no
  network field.
- The full active status is `RAM X/Y GiB disk X/Y GiB net N@XMiB/s qK X/Y/Z`.
