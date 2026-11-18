# Live ink preview

VC3D can show an ink detection model's output for the part of a segment that is
on screen, a moment after the view stops moving, without rendering a surface
volume or running a whole-segment prediction first. VC3D samples the layers
under the view from the segment in memory and the open volume, sends them to an
ink preview server running on the same computer, and draws the reply over the
segment view. Because the layers come from the segment in memory, the preview
follows mesh edits.

The preview is model output. It shows what a model computes for these voxels,
not a reading of the scroll, and it is only as good as the model and the
segment's placement.

The controls are the **Live preview** group of the **Ink Detection** dock
(**View → Ink Detection**). The preview is off until you turn it on, and VC3D
only connects to a server you started: it launches nothing.

## Starting a server

VC3D does not include a server. Any server that implements ink preview api 1
(see the protocol below) works: start it on the same computer, listening on
127.0.0.1, with the model you want to see. VC3D treats every such server alike.

The dock starts with the address `http://127.0.0.1:8765`; enter the server's
address there if it listens elsewhere. VC3D does not follow redirects: a `3xx`
answer is reported as an error.

## Dock controls

* **Show the live preview** turns the preview on or off. When on, VC3D asks
  the server for its model (`GET /health`) and then watches the segmentation
  view.
* **Server** is the server's address, `http://HOST:PORT`. It is saved in
  VC3D's settings and applied when you leave the field. An empty field means
  `http://127.0.0.1:8765`.
* **Reverse layers** feeds the model the same layers in reverse order (see
  below).
* **Level** is the volume level the layers are sampled from. **Auto** picks
  the level whose voxel size is nearest the model's training voxel size, as the
  server reports it, or level 0 when either size is unknown. The list shows each
  level's voxel size when the volume's is known.
* The status line says what the preview is doing: connected and the model's
  name, sampling, computing, the size of the last image and how many of its
  patches the server computed or took from its cache, the server's refusal or
  error text, or "Zoom in" when the view is larger than the view budget.

The dock's **Colormap** and **Opacity** rows apply to the preview as well as to
the published detections, and **Space** toggles the preview's opacity with
them. The preview draws byte value `v` as probability `v / 255` with the same
mapping in every view: there is no per-view stretch. It replaces nothing: the
published detections combo works as before, and both can be shown at once.

## What VC3D sends

The preview works on the canvas `vc_render_tifxyz --scale 1 -g LEVEL` would
write for the segment, with no crop, rotation or flip (`vc::RenderCanvas`).
When the view has not changed for about 150 ms, VC3D takes the canvas pixels
the view touches, adds a margin of 64 pixels or an eighth of the view, and asks
the server for that rectangle. The server answers from its cache when it can;
otherwise it names the rectangle whose layers it needs, and VC3D samples that
rectangle on a worker thread and posts it. A small pan therefore uploads only
the new strip, and a view still inside the last image asks nothing.

Layer `i` of the `D` layers the model reads sits `i - D/2` level voxels along
the surface normal (integer division), the layers villa's flat inference reads
from a centered render. VC3D samples them with `vc::sampleRenderLayers`, the
sampling `vc_render_tifxyz` does, from the chosen level alone: no coarser level
stands in for chunks that are not loaded yet. The sampling runs off the GUI
thread from a copy of the segment taken on the GUI thread. It is cancelled when
the segment, its geometry, the level or the layer order changes, or when the
view moves off the requested rectangle. If a chunk cannot be read, the status line shows
the error and nothing new is drawn. A remote volume opened with the remote
offline mode `fill` is refused, because chunks that are missing offline would
read as zeros.

The last image stays on screen, placed by its canvas rectangle, so it moves
with pans and zooms while the next one is computed. One request is in flight at
a time. Views over the view budget (4 Mi pixels, 4,194,304 pixels of the chosen
level, shown as 4.2 Mpx in the status line, or the server's `max_view_px` if that
is smaller) are not requested. The preview is
drawn in the segmentation view only, and not with the volumetric composite.

Every request carries a key that changes whenever the voxels under the canvas
could change: the segment, each edit of it, the volume, the level and the layer
offsets. The server caches by key, so it never serves output computed from an
older geometry.

### Memory and time

The layers of one request take `depth * sh * sw` voxels, and VC3D holds them
twice while it posts them. A server answer may ask for at most `patch - 1`
pixels past the view on each side; a larger request is reported as a server
error and nothing is sampled. For ink_9um (17 layers, patch 128), a square
request at the full view budget on a uint16 volume, with a need that reaches
127 pixels past it on every side, takes about 360 MB at the peak.

The copy of the segment does not depend on the view: the first sample after
each edit copies the whole grid on the GUI thread (12 bytes per vertex), and a
worker then builds the layer surface of the chosen level from the copy (about
46 bytes per vertex). For a segment of 12 million vertices that is about
700 MB, about 0.1 s on the GUI thread and about 5 s on the worker, measured on
one Linux build; the first preview after an edit of a large segment waits for
it.

## The reversed view

**Reverse layers** asks for the same view with the layer order reversed. It is
the control view: model output that depends on the real order of the layers
should fade in it, and output that stays the same does not depend on that
order. It is also the fix when a segment's normals point the other way, so that
its forward layers run from the other side of the sheet.

## Limits

* Model output is not a reading. A preview pixel is what the model computes for
  the sampled layers.
* A preview pixel equals the same pixel of a full prediction from the same
  model only when the volume matches the model's training voxel size and the
  full prediction reads a render of the same level. On scans finer than the
  model was trained on, full predictions usually read a downsampled or pooled
  render, so the preview of a finer level is an approximation of them.
* The server skips patches whose layers are all zero, such as patches off the
  surface; pixels that no other patch covers read 0.
* The preview draws in the segmentation view only.

## Protocol: ink preview api 1

The server listens on 127.0.0.1. `GET /health` returns JSON with at least
`service` (`"ink-preview"`), `api` (1), `backend`, `model`, `patch`, `stride`,
`depth` (the layers the model reads), `voxel_um` (the training voxel size, or
null) and `max_view_px`.

`POST /preview` takes its parameters in the query and, when it uploads layers,
a body of `depth * sh * sw` voxels, layers first, C order, little-endian.

| Parameter | Meaning |
| --- | --- |
| `key` | Opaque text of at most 200 characters naming everything that decides the canvas's voxels. |
| `cw`, `ch` | Canvas size in pixels. |
| `x`, `y`, `w`, `h` | The view rectangle, inside the canvas. |
| `variant` | `f`, or `r` for the reversed control (the server reverses the uploaded layers). |
| `format` | `u8` (one byte per pixel, `floor(255 p)`), `f32` or `png`; VC3D asks for `u8`. |
| `sx`, `sy`, `sw`, `sh` | The canvas rectangle the body covers, with a body only. |
| `dtype` | `uint8` or `uint16`, the body's voxel type, with a body only. |

| Answer | Meaning |
| --- | --- |
| `200` | The view, `w * h` pixels in `format`, with headers `X-Shape: h,w`, `X-Backend` and `X-Tiles: computed,cached`. |
| `409` | JSON `{"need": {"x", "y", "w", "h"}, "depth": D, "missing": N}`: the canvas rectangle whose layers the server needs. Nothing was computed; post again with them. |
| `400`, `413`, `500` | JSON `{"error": TEXT}`: a bad request, a view over `max_view_px`, or a failure. The status line shows the text. |
| `403` | JSON `{"refused": TEXT}`: the server's allow list refused the view. |

The server computes the patches of its sliding window grid on the canvas that
overlap the view, and caches each under the key, the variant and its position.

## Tests

```bash
ctest --test-dir build -R ink_preview --output-on-failure
INK_PREVIEW_TEST_SERVER=http://127.0.0.1:8765 ctest --test-dir build -R ink_preview_server --output-on-failure
```

`test_ink_preview_client` checks the client against an in-test server: health,
the `409` round trip, needs that reach too far past the view, redirects that
are not followed, dropped stale replies, error texts and timeouts.
`test_ink_preview_overlay` checks the canvas rectangle of a known camera, the
image's placement after a pan, the requests that must not start, the key, the
dock's Live preview group, round trips whose uploaded layers are compared with
`vc::sampleRenderLayers` on a synthetic volume (also after switching to a volume
whose levels are placed differently), and a server whose `409` depth disagrees
with its health. The `*_server` tests run one round trip against the server
`INK_PREVIEW_TEST_SERVER` names, and are skipped when it is unset or no server
answers there.
