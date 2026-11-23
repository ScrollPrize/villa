# SFTP remote data access

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

The transport is vc_reader's (`vc_reader/sftp.hpp`), so remote volumes opened by the
CLI tools use it too. Independent Python Zarr/fsspec clients do not.


The system `ssh` executable must be on PATH. BatchMode and strict host-key
verification are enforced: configure keys/agent and known_hosts outside VC3D,
using a terminal. VC3D does not start an agent or prompt for passwords/trust.
Only an explicit SFTP no-such-file response is treated as missing data.

SFTP uploads are not supported. Changing an SSH alias to refer to another
server requires clearing its old cache, as does changing data behind an HTTP URL.
