"""Project attachments beside volumes and segments, and Save As."""

from __future__ import annotations

from typing import Any, Literal, Optional

from mcp.server.fastmcp import Context

from ..core import mcp, _call, _strip_none, _wait_for_job


@mcp.tool()
async def vc3d_attach_to_project(
    kind: Literal["umbilicus", "normal_grid", "lasagna_manifest"],
    location: str,
    fiber_inference: Optional[bool] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Attach a project-level resource, the headless twin of the File menu's
    "Attach Umbilicus...", "Attach Normal Grid..." and "Attach (Remote)
    Lasagna Manifest...". The attachment is saved in the open project and
    comes back when it is reopened. No dialog is shown.

    kind:
      "umbilicus": a local scroll-center JSON file; files with malformed frame
      metadata are refused, files that declare no frame attach with a
      warning. Replaces any other umbilicus (reported as previousLocation).
      Synchronous.
      "normal_grid": a local normal_grids root with xy/xz/yz subfolders and
      metadata.json. Synchronous.
      "lasagna_manifest": a local or remote (s3://, http(s)://, sftp://)
      .lasagna.json manifest, loaded in the background as a source:"project"
      job. fiber_inference=true attaches it as fiber-inference data.
    location: an absolute local path (relative paths are -32602; remote
    locations for kinds other than lasagna_manifest are -32007).
    wait: for the job kinds, block until the attachment finishes (30-minute
    cap) and return the terminal job status inline.

    Attaching an attached location again, in any spelling (trailing slash,
    "..", a relative entry stored by an older project), succeeds with
    alreadyAttached=true and the stored location. Synchronous kinds return
    {"attached", "alreadyAttached", "kind", "location", "previousLocation",
    "warnings"}; job kinds return {"jobId", "kind": "project.attach",
    "source": "project", "attachmentKind", "location"}, and the job's result
    carries the synchronous fields.
    """
    result = await _call(
        "project.attach",
        _strip_none(
            {"kind": kind, "location": location, "fiberInference": fiber_inference}
        ),
    )
    job_id = result.get("jobId") if isinstance(result, dict) else None
    if not job_id:
        return result
    return await _wait_for_job(job_id, wait, result, ctx)


@mcp.tool()
async def vc3d_detach_from_project(
    kind: Literal[
        "umbilicus", "normal_grid", "lasagna_manifest", "volume", "segments"
    ],
    location: Optional[str] = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Remove an attachment from the open project, the headless twin of the File
    menu's "Detach..." and "Detach Umbilicus". Data files are not deleted.
    Requires confirm=true.

    kind: "umbilicus" is single-valued; omit location for it (detaching an umbilicus restores automatic resolution). The other kinds
    need a location, in any spelling of the attached one; an unknown location
    fails with -32007. The project removes that stored location from every
    entry list it appears in, like the menu.
    "volume" and "segments" are refused while segmentation editing is on or
    any job runs (-32004), and the entry backing the current volume is
    refused (-32009); select another volume first.

    Returns {"detached", "kind", "location"}; detached is false when nothing
    of that kind was attached.
    """
    return await _call(
        "project.detach",
        _strip_none({"kind": kind, "location": location, "confirm": confirm}),
    )


@mcp.tool()
async def vc3d_save_project_as(path: str, overwrite: bool = False) -> dict[str, Any]:
    """Write the open project to a new .volpkg.json and continue working on it,
    the headless twin of "Save Project As...". As in the GUI, the session
    switches to the new file: later attachments and edits are saved there, not
    in the original, and relative entry locations resolve against the new
    folder (each one is named in "warnings" when the folder changes). The file
    is listed among recent projects.

    path: absolute file path; ".volpkg.json" is appended when missing.
    overwrite: replace an existing file; without it an existing path fails
    with -32005, as in vc3d_create_project.

    Returns {"saved": true, "path", "warnings"}.
    """
    return await _call("project.save_as", {"path": path, "overwrite": overwrite})
