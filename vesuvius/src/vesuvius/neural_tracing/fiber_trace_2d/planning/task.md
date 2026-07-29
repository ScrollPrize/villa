# Remote Lasagna Manifest Fetch Diagnostics

Improve error messages when `vc_fiber_trace_metric` or any Lasagna dataset
opener fails to fetch a remote `.lasagna.json` manifest.

Desired behavior:

- Include the original manifest location and the resolved HTTP/S3 request URL.
- Include HTTP status, content type, reported content length, received body byte
  count, and a bounded response-body excerpt when available.
- For S3 locations, include the region used and whether AWS SigV4 credentials
  were loaded, without printing credential values.
- Preserve the existing behavior for successful fetches and local manifests.

Out of scope for this task:

- do not change credential discovery or region selection;
- do not add network-dependent unit tests.
