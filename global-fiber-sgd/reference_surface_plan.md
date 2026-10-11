# Reference tifxyz download and comparison proposal

Download only the requested philodemos Paris4 tifxyz prefix, authenticated using
default AWS credentials through the existing approved benchmark runner. Public
listing returned403; snap AWS CLI unavailable in sandbox. Reuse boto3 installed.
List all objects, reject unsafe relative paths, download concurrently to partial
files then rename, verify sizes and SHA256, preserve source ETag/last-modified
in a provenance manifest. No uploads/deletes or edits to solves.

Inspect metadata and TIFF headers/coordinate samples, compare coordinate bounds
and surface axes to the AFV source metadata. Propose (do not yet implement) robust
fiber-to-surface projection, continuous reference winding phase, one global gauge
alignment, ambiguity/coverage metrics and reference-colored solve maps/3D fibers.
Do not mistake cumulative unwrapped column number for winding number or native
optimizer candidate bucket for physical winding when phase/offset crosses seams.

Spec update: read-only reference acquisition; no new solve or evaluation claimed.
Docs: source manifest and reference_surface.md with findings/proposal. Archive
prior log, update current task/status. Test safe key mapping, local provenance
and complete object counts/sizes, TIFF shape compatibility. Independent review.
