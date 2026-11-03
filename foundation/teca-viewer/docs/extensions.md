# Extensions

An extension is `src/extensions/<name>.js` and, when it has parts, `src/extensions/<name>/`. `src/main.js` attaches it
with `extensions.extend("<name>", create)`. Nothing else imports it, so a viewer without one deletes that line and its
files and checks.

`create(viewer, entry)` returns `{panel, place, keys, volumeChanged(volume), surfacesChanged(), focusChanged(),
surfaceViewChanged(), kept()}`. All but `panel` and `place` may be left out.

- `place` is `"data"` or `"view"`, a tab of the left panel, or `{title, icon}`, a tab of its own in the right panel.
- `keys` join the key table and the help: `{description, code, shift, command, alt, tool, repeats, run}`. A binding with
  a `tool` works only while that tool is set.
- `volumeChanged` runs after each volume opens, `surfacesChanged` when a surface is loaded, shown, hidden or removed,
  `focusChanged` when a settled frame draws another focus, and `surfaceViewChanged` when the Surface pane shows another
  view.
- `kept()` gives, or resolves to, what the extension keeps in the browser for the view, as JSON, for Save session.
- `entry` is `{saved, save(value), opened}`: the JSON last saved in the address's session, the call that saves it, and
  what `kept()` gave in the session file opened at this start.

## Viewer

`src/ui/extensions.js` builds `viewer`. A seam throws on a key `SEAMS` there does not name. A handle ends with `remove()`
or the next volume.

Volumes and surfaces:

- `openVolume(url, signal)` opens a Zarr volume without showing it; the extension disposes it. Of a volume an extension
  reads only `rootUrl`, `metadata.physicalUnits_ZYX`, `metadata.displayEncoding` and `scales`, finest first, each
  `{shape_ZYX, chunks_ZYX, scale_ZYX, translation_ZYX, downsample_ZYX, offset_ZYX}`, and calls `getChunk(z, y, x,
  channel, scale, signal, priority)`, `clearChunkCache()` and `dispose()`.
- `showVolume(url)` opens a volume as the CT and `addSurface(url)` adds a tifxyz surface; each resolves to whether it
  succeeded.
- `surfaces()` gives the URLs of the surfaces shown, `listedSurfaces()` those listed.
- `surfaceView()` gives `{surface, grid_HW, start_HW, stop_HW, scale_UV, pixel}` for the Surface pane, or null: the
  surface's URL and grid, the box of grid cells drawn, `[row, column]` and unrounded, and the full pixels a device pixel
  spans. A surface's full pixels are its grid cells over its scale.
- `surfacePoints()` gives a function from grid cell `cell_HW` to `{point_ZYX, normal_ZYX}` or null, for the surface and
  offset shown now. It keeps them as the pane moves on.
- `heldCT(level, chunk_ZYX)` gives the voxels of a CT chunk a pane's reader holds, else null. It reads nothing.
- `readSurface(url, signal)` gives `{grid_HW, scale_UV, points_zyx_HW3}`, a cell being off the surface where its values
  are -1 or not finite.
- `editSurface(url, surface)` draws the surface from `surface` in place of its files. It resolves to a handle, or null.
  Its `changed(start_HW, stop_HW)` redraws those grid cells.

Volumes in the page:

- `memoryVolume(name, zeroed)` is a uint8 volume on the CT's levels that the extension writes
  (`src/volume/memory-volume.js`). `zeroed(length)` gives the array a new chunk is written in.
- `madeVolume(name, make)` is a uint8 volume whose chunks `make(level, chunk_ZYX, signal)` makes as panes read them: a
  whole chunk in ZYX order, or null for all zeros (`src/volume/made-volume.js`).

Drawing:

- `addLayer({slices, surface}, style)` draws a volume, given as two readers, over the slices and the Surface; `surface`
  may be null. `style` is `{color_RGB, opacity, range}`, `range` being `[low, high]` display bytes. A null `color_RGB`
  draws labels by `viewer.labelColor(id)`, with no range. The handle has `setStyle(style)` and `changed(changes)`.
- `addSurfaceImage(url, image, style)` draws `image`, `{data_HW, shape_HW}` of grey uint8, stretched across the
  surface's grid in the Surface pane. The handle has `setStyle(style)` and `setDetail(detail)`, where `detail` is
  `{data_HW, shape_HW, start_HW, downsample}` in the surface's full pixels, or null.
- `addMarks({diameter, width, voxels})` draws points and edges on the slice panes. The handle's `set(marks)` takes
  `{points_zyx_N3, colors_N3, edges_E2, selected, depth}` and optionally `neighbours_N2`; `show(shown_N)` hides some, and
  null shows all. Placing happens in the shader, so a move uploads nothing.
- `addSurfaceMarks(options)` draws them on the Surface pane; `set(marks)` takes `{surface, cells_N2, colors_N3,
  edges_E2, selected}`.
- `add3d(object)` draws a three.js object in the 3D pane in world units. The handle's `changed()` redraws it. An object
  with its own shader gives a `raycast` whose distance is the depth it writes.
- `filterCT(passes, reach)` draws the CT through three.js ShaderMaterials, each given `image` and `pixel`. `reach` is how
  many taps from a pixel they read.

Interaction:

- `setTool(tool)` gives the primary button to `{radius, label, surface, down(at, ctrl), move(at), up(at), putAway()}`, or
  back to panning with null. `at` is `{pane, point_ZYX, normal_ZYX, pixel}`, and on the Surface pane also `cell_HW` and
  `edited`. `surface: true` gives the tool the Surface pane too. `label` is `{text, color_RGB}` or null.
- `focus(point_ZYX, {surface})` centres the panes on a world point, `focusPoint()` gives it, and `setStatus(text)` writes
  the status bar.

## Rules

- It imports its own files, `vendor/`, `math.js`, `zip.js` and `storage.js`, nothing else of the core
- Its panel may take the side panel's classes: `card`, `source-dependent`, `button-row`, `control-name`, `readout-line`,
  `range-pair`, `checkbox-row`, `segmented`, `layer-row`, `layer-row-name`, `overlay-form`, `overlay-items`,
  `overlay-swatch`, `icon-btn` and `active`. Its own ids and classes start with its name, and its styles are a `<style>`
  in its panel.
- Its browser checks are `tests/<name>.e2e.py` and `tests/<name>-*.e2e.py`, at least one, and no other check drives its
  elements. A core check draws through `viewer` (`tools/page.py`'s `extension_viewer()`).
- `storage.js`'s `lock(name)` asks for a Web Lock, so one page writes what an extension keeps under it.
