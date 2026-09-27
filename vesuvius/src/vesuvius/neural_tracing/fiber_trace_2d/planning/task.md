# Local-event BP cost trials

Current user correction: constraints are fully constructed and certified first, then used to place splits. Splits must not influence constraint validity at all. Remove erroneous post-split filtering/clipping, preserve weights independently of segmentation, and re-test against previous grouped-piece results.

Current follow-up: replace the constant continuation-break penalty with a cost increasing with absolute winding displacement, since wrong traces should remain nearby. Test against the matched D10/parallel-off joint-H/V baseline and judge initial winding correctness and ladder structure, not removal score alone. Keep earlier constant and hard options available as controls.

Trial different costs for the newly implemented local/per-side-consistent constraints and clean non-overlapping splits, with and without parallel constraints. Keep extraction fixed; compare full benchmark metrics and preserve results. Use only existing approved run/test commands.

Support independent disabling of pieces and especially cutting continuation connections at orthogonal-event midpoint splits while leaving both pieces and their ordinary constraints active. Audit distinct orthogonal events per piece (coincident events can form a group).
