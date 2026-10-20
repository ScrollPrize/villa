# VC3D plugins

A plugin adds an entry to VC3D's **Plugins** menu that runs an external command
on what you have open (the current volume, the active segment, the focus point
or the focus region) and loads what it writes back into VC3D. A plugin is one
JSON file. VC3D never loads plugin code into its own process: it starts the
declared program with an argument list, the same way it already starts
`vc_render_tifxyz` or `vc_grow_seg_from_seed`, and reads the result with its
existing loaders.

This lets people run their own tools (an ink model, a scroll center finder,
a surface check) from inside VC3D without patching it, and lets VC3D call
villa's own `vc_*` commands with fixed settings.

## Where VC3D looks

VC3D reads every `*.json` file (not recursive) in these folders, in order:

1. `~/.vc3d/plugins/`
2. each folder in `VC3D_PLUGIN_PATH` (separated by `:` on Linux and macOS,
   `;` on Windows)

Plugins are read at startup and again with **Plugins > Reload Plugins**.
**Plugins > Plugin Folders...** lists the folders and every manifest that was
skipped and why. Folders inside a project are never searched, so opening a
project someone sent you cannot add plugins.

Run outputs go to `~/.vc3d/plugin-runs/<plugin id>/<UTC time>/` (set
`VC3D_PLUGIN_RUNS` to use another folder). Each run folder holds
`plugin-run.json`: the manifest path and its SHA-256, the exact program and
arguments, the result path and how the run ended.

## Manifest

```json
{
  "schema": "vc3d-plugin/1",
  "id": "villa.tifxyz2obj",
  "name": "Export active segment as OBJ",
  "description": "Runs vc_tifxyz2obj on the active segment.",
  "menu": "Volume Cartographer",
  "program": "vc_tifxyz2obj",
  "args": ["{segment}", "{output}/{segment_id}.obj", "--normalize-uv"],
  "inputs": [],
  "result": {"kind": "file", "path": "{output}/{segment_id}.obj"},
  "homepage": "https://github.com/ScrollPrize/villa",
  "license": "GPL-3.0"
}
```

| Key | Required | Meaning |
| --- | --- | --- |
| `schema` | yes | Always `vc3d-plugin/1`. |
| `id` | yes | 1 to 64 letters, digits, `.`, `_` or `-`; unique across all folders (the first one found wins, later ones are skipped). |
| `name` | yes | Menu text. |
| `program` | yes | A bare name is looked up next to the VC3D executable first (so `vc_*` tools resolve to the ones built with VC3D), then on `PATH`. A relative path with a `/` is relative to the manifest's folder. No placeholders. |
| `args` | no | Array of strings. Each string becomes exactly one argument after placeholders are filled in. |
| `inputs` | no | Values VC3D asks for before each run: `{"name": "radius", "label": "...", "default": "20"}`. Use them as `{input:radius}`. |
| `result` | no | `{"kind": ..., "path": ...}`; see below. Defaults to `none`. |
| `menu` | no | Submenu under Plugins. |
| `description`, `homepage`, `license` | no | Shown as the menu tooltip and kept for reference. |

Unknown keys, unknown placeholders and undeclared inputs make VC3D skip the
manifest, so a typo shows up in Plugin Folders instead of at run time.

### Placeholders

| Placeholder | Value | Needs |
| --- | --- | --- |
| `{volume}` | path of the current volume (its remote location for a remote volume) | an open volume |
| `{volume_id}` | current volume id | an open volume |
| `{segment}` | folder of the active segment (tifxyz) | an active segment |
| `{segment_id}` | active segment id | an active segment |
| `{point}` | focus point as `x,y,z` in full-resolution voxels; also `{point_x}`, `{point_y}`, `{point_z}` | a focus point |
| `{region}` | enabled focus region as `x0,y0,z0,x1,y1,z1` | Focus bounds enabled |
| `{ink_overlay}` | image of the selected ink detection overlay | a selected overlay |
| `{umbilicus}` | the project's attached umbilicus file | an attached umbilicus |
| `{project}` | project folder | a project on disk |
| `{output}` | this run's fresh output folder | always |
| `{plugin_dir}` | folder holding the manifest | always |
| `{input:NAME}` | value typed for input `NAME` | the input |

Write `{{` and `}}` for literal braces. When a plugin uses a placeholder that
has no value (for example `{segment}` with no active segment), VC3D does not
start it and says what is missing.

### Results

`result.path` must start with `{output}/` and stay inside the run folder, so
VC3D only ever loads files the run itself wrote.

| `kind` | What VC3D does with `path` |
| --- | --- |
| `none` | Nothing; reports the run folder. |
| `file` | Shows the path and offers to open it (a report, a mesh, an image). |
| `segment` | Attaches the tifxyz folder to the project as a segments entry and reloads the surface list. |
| `ink_overlay` | Adds the image to the active segment's `ink-detections.json` (keeping the records already there or in `catalog-origin.json`) and selects it in the Ink Detection panel. The image must be on the segment's surface grid, like a published ink prediction. |
| `points` | Reads a VC3D point collection file (`vc_pointcollections_json_version` 1, the format **Save** in the Point Collections panel writes) and adds its collections. Existing collections are never replaced; a name that is taken gets a numbered suffix. |
| `umbilicus` | Checks the file as **File > Attach Umbilicus** does and attaches it, asking first if another umbilicus is attached. |

## Running a plugin

Choosing a plugin asks for its inputs, then shows the exact command line, the
output folder and what will be loaded, and runs only after you confirm. The
program runs through the same runner as VC3D's other command-line tools, so
its output appears in the console window, only one tool runs at a time, and
the log goes where other tool logs go.

## Trust

A plugin runs a program on your computer with your permissions, exactly like
running it from a terminal. Only put manifests you have read in your plugin
folders. What VC3D does and does not do:

- It runs only the `program` named in the manifest. Placeholders cannot change
  it, and arguments never pass through a shell, so a segment or input value
  containing spaces, `;` or `$(...)` stays one literal argument.
- It shows the full command and asks before every run.
- It loads only the declared result, only from the run's own folder, and only
  with the loaders above (no plugin code or libraries are loaded into VC3D).
- It searches only your home folder and `VC3D_PLUGIN_PATH`, never a project.

It cannot check what the program itself does once started.

## Examples

`apps/VC3D/plugins/examples/` has three manifests:

- `vc-tifxyz2obj.json`: villa's `vc_tifxyz2obj` on the active segment (`file`).
- `vc-render-layers.json`: villa's `vc_render_tifxyz` with the Render segment
  defaults, for local volumes (`none`).
- `points-at-focus.json` with `points_at_focus.py`: a format example that
  writes seven points around the focus point (`points`).

Try them with `VC3D_PLUGIN_PATH=/path/to/villa/volume-cartographer/apps/VC3D/plugins/examples VC3D`.
