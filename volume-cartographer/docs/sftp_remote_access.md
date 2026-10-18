# SFTP remote data access

## Implementation plan

- Extend the shared remote fetch interface with read-only SFTP v3 over a
  persistent system OpenSSH subprocess. Use Qt Core QProcess for portable
  process/pipe handling; this requires Qt Core even in headless builds.
- Reuse existing chunk queues, byte-progress accounting, metadata loading and
  the global disk-cache root. Do not introduce separate SFTP download workers.
- Normalize ssh:// to sftp://, retain user/port in cache identity, and support
  directory browsing in the existing attachment browser.
- Test protocol reads/ranges/stat/listing, connection reuse, failure handling,
  URL/cache identity and browser acceptance. Build the application and run
  existing remote-cache tests to protect HTTP/S3 behavior.

## Connection and security

Use `sftp://[user@]host[:port]/absolute/path/to/volume.zarr` (or `ssh://`).
The host can be an OpenSSH Host alias. OpenSSH reads its normal configuration,
including HostName, User, Port, IdentityFile and ProxyJump. An explicit URL user
or port overrides the corresponding SSH configuration. An omitted user/port
leaves that decision to SSH. Paths are absolute; percent-encode reserved URL
characters. Passwords, query strings and remote shell commands are not accepted.

Use **Attach Volume** or **Attach Remote Zarr**, enter the SFTP directory URL,
then navigate the remote listing and select the Zarr root. **Attach Remote
Lasagna Manifest** uses the same browser for manifests. Projects retain the
remote locator, not a replacement local-data path.

The same transport is available through the C++ remote-volume APIs and the
`vc` Python bindings, not only the GUI:

```python
import vc

volume = vc.Volume.open_url("sftp://my-ssh-alias/data/scan.zarr")
voxels = volume.read_zyx((0, 0, 0), (64, 64, 64))
```

Prefetch, coordinate sampling, pyramid levels and the process-wide persistent
cache use the existing volume APIs. This does not add an SFTP implementation to
independent Python Zarr/fsspec clients.

The system `ssh` executable must be on PATH. BatchMode and strict host-key
verification are enforced: configure keys/agent and known_hosts outside VC3D,
using a terminal. VC3D does not start an agent or prompt for passwords/trust.
Only an explicit SFTP no-such-file response is treated as missing data.

Each worker retains up to four SSH/SFTP sessions. Requests reuse these sessions;
individual files use pipelined 32 KiB reads. Connections are discarded after
protocol/transport failure and re-established on retry. Shutdown cancellation
and inactivity timeouts are checked while waiting for process I/O.

The existing shared fetch response model uses HTTP-shaped status values for
SFTP: 200/206 success, 404 missing, 403 permission denied and errors otherwise.
This is an internal adapter, not an HTTP server. SFTP uploads are not supported.
No cache directory override is added. Changing an SSH alias to refer to another
server requires clearing its old cache, as does changing data behind an HTTP URL.

## Validation log

- Implemented shared transport, URL/cache identities, Zarr and manifest access,
  browser listings and attachment. Existing HTTP/S3 behavior is retained.
- VC3D and focused tests built in the existing build directory. No dependencies
  were installed. The old script mentioned in the root playbook,
  `scripts/build_dependencies.sh`, is absent on this branch; the existing CMake
  build and current README are the available build entrypoints.
- Protocol fixture verifies persistent reuse across reads and missing files,
  2 MiB pipelined reads, ranges, out-of-order replies, short reads, permission
  errors, malformed packets, reconnects, in-flight cancellation, and explicit
  URL user/port versus leaving those options to SSH config.
- The installed `/usr/lib/ssh/sftp-server` also passed real reads, ranges,
  listings and Zarr pyramid metadata discovery through local process pipes.
- Existing URL/cache tests and offscreen browser tests passed. The latter cover
  SFTP/SSH file selection, directory selection and preservation of query strings
  and level selectors on existing HTTP/S3 paths.
- Python binding integration exercises `Volume.open_url` with both schemes,
  percent-encoded paths, uint8/uint16 reads, sparse missing chunks, multiscale
  reads, base-level selection, prefetch, coordinate sampling and invalid URLs.
  A second Python process reads the disk cache after source payloads are removed.
  The test uses the built extension, excludes user-site editable installations,
  and starts the installed OpenSSH SFTP server through the existing C++ fixture.
  No Qt application or separate Python transport is required.
- This test exposed a cache-format detection issue: a mirror's logical decoded
  payloads and empty markers were mistaken for a legacy cache on reopening.
  Native Zarr metadata now takes precedence over those auxiliary files; caches
  without native metadata retain legacy detection.
- GUI functionality was manually tested by the user.
- No real remote SSH login, ProxyJump network route, desktop-agent prompt, macOS
  or Windows runtime has been tested here. These still need deployment testing.
  The protocol fixture is portable; the installed-server check is optional and
  POSIX-only. Uploads are deliberately outside this read-only feature.

```sh
cmake --build volume-cartographer/build --target VC3D vc_volume test_sftp_fetch test_remote_url test_remote_file_cache test_unified_browser_dialog test_chunk_cache test_chunk_cache_persist -j8
ctest --test-dir volume-cartographer/build -R '^(test_(sftp_fetch|remote_url|remote_file_cache|chunk_cache|chunk_cache_persist)|python_sftp_volume|unified_browser_dialog)$' --output-on-failure
```

The Python test is registered when `VC_BUILD_PYTHON=ON` and requires NumPy.
It reports a CTest skip on Windows or when no POSIX OpenSSH SFTP server is
installed; the protocol-level C++ fixture does not require that server.

Protocol reference: [SFTP v3 specification](https://www.openssh.org/txt/draft-ietf-secsh-filexfer-02.txt).
