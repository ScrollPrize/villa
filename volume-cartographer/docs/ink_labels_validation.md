# Ink-label workflow validation

The results below belong to the annotation-only source release
`d3f4816d6dd7e46c90384188ecfdbbe3fcf53c3e`. The subsequent integration with
the five feature groups from PR #2219 has not yet been built or tested.

The annotation-only macOS Apple Silicon build passed 64 Qt results (62 test methods plus
initialization and cleanup), 13 parser controls, and the native annotation
workflow on public PHerc0139 w035. The native run took 70.663 seconds. No paid
compute, model inference or scientific quality evaluation was involved.

| Check | Observed result |
| --- | --- |
| Mode, classes, tools, eraser, background-keeps-ink, visibility, radius, opacity, Undo/Redo | All 17 allowlisted controls exercised; attached and floating selected states readable |
| Viewer | Label visibility and opacity changes visible at one fixed overview camera |
| Guarded history | Agent Undo refused a User-role entry; permitted Agent Undo and ordinary UI Undo restored the exact expected rasters |
| Persistence | All 30,496,800 pixels matched the independent expected raster after save, same-process reopen and production cold restart; all three saved TIFF hashes matched |
| Visual/privacy review | 107 actual captures reviewed against nine criteria; no private checkout path visible |

The build used the source at `b3018581ae99800afd0577ff19ecd8ec4de8a1f6`
plus the 20 feature-file changes packaged by that annotation release. These cover ordinary
viewport/widget events, dock lookup, Undo ownership guards, selected-state
feedback and numeric opacity. Annotation release packaging preserved their exact bytes.
The only omitted local setting caps libbacktrace compilation at two jobs;
the release retains upstream's build-parallelism setting. No application
source was changed between that build/native run and annotation release packaging.

Production executable SHA-256:
`6ccb37aa6fb73530d65e88b865a941f4cd403b169bef2790f94eb808b8ab4dc2`.
Qt test executable SHA-256:
`be2c935d67a81b3b01b2a78b3a67a46548330af757966722e9ca4f12e3326217`.

The workflow uses automated ordinary Qt events. A companion with ordinary app
objects supplies a synthetic User history role; it is not a physical person
drawing. Cold restart uses the production executable. Save runs through the
application bridge. Keyboard shortcuts, Pin open, Assist algorithms, imports
and new-version creation were not exercised by this native run. Polygon and
Fill selection was checked, not a new complete drawing validation for them.

The first supervision toggle also coincided with a CT background update; the
stable return pair and raw inspection show the blue-overlay change. Stored
masks match exactly; screenshots are illustrative and not pixel-identical.
The reviewer was separate from the native-run operator but also contributed
the harness, so this was not a blind third-party review. Other operating
systems and the repository's full test suite were not rerun here.

The accompanying media uses actual public-data captures. PNG stills are
unchanged copies; GIFs hold selected stills for 2.3 seconds, with caption bands
outside the image, proportional downscaling and palette encoding. There is
no invented mouse motion, redacted app frame or claim of verified letters.
Data: Vesuvius Challenge, PHerc0139 segment
`20260317000000-w035_2026031718`, volume `20250728140407`, 9.362 um,
CC BY-NC 4.0. Colored marks are disposable practice annotations.
