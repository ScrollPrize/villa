# Task: arbitrary staged Fiberlet graph reduction

Replace the fixed aligned-stage-one/half-offset-stage-two experiment with an
ordered sequence of arbitrary reduction stages inside one selected base-space
bbox.

- A stage is defined by a cubic analysis-box side and an XYZ offset relative to
  the selected bbox minimum. Repeated stage specifications define order.
- Each stage tiles every complete analysis box on that offset lattice which is
  contained in the selected bbox. Boxes execute in deterministic XYZ order.
- Every stage owns separate sparse anchor and Fiberlet cache layers with the
  same spatial chunk layout and record format as the initial caches.
- A missing stage-layer chunk means "unchanged from the previous layer".
- A processed analysis box rewrites every intersected storage chunk in its
  current stage layer. It may only remove anchors and physical Fiberlets owned
  by the analysis-box interior; records outside the box remain unchanged.
- Later overlapping boxes in the same stage read prior updates from that stage
  and may remove more records, but may never restore a record removed by an
  earlier box or stage.
- Preserve the existing exact entry-to-first-exit route analysis, regular join
  constraints, and post-reduction simplification semantics.
- Report each stage independently and report the joint original-to-final effect
  over the whole selected bbox, separately for all incident Fiberlets and
  Fiberlets with both endpoints in the bbox.

The existing two-stage experiment is represented by stages `256,0,0,0` and
`256,128,128,128` over a 512-cubed selected bbox. A later whole-bbox pass can be
appended as `512,0,0,0`.
