"""Verified, independent input snapshots with opportunistic filesystem clones."""
import errno
import hashlib
import os
from pathlib import Path
import shutil
import sys

from input_publication import fingerprint
from service_http import ApiError, TRANSFER_CHUNK_BYTES, sha256_file


def _clone_file(source, destination):
    """Return False only when cloning is unsupported; never use hard links."""
    unsupported = {errno.EXDEV, errno.EINVAL, errno.ENOSYS, errno.ENOTTY,
                   errno.EOPNOTSUPP, errno.ENOTSUP}
    try:
        if sys.platform == 'linux':
            import fcntl
            with open(source, 'rb') as src, open(destination, 'xb') as dst:
                try:
                    # linux/fs.h: FICLONE, identical on amd64 and arm64.
                    fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())
                except OSError:
                    Path(destination).unlink()
                    raise
        elif sys.platform == 'darwin':
            import ctypes
            library = ctypes.CDLL(None, use_errno=True)
            clonefile = getattr(library, 'clonefile', None)
            if clonefile is None:
                return False
            clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            clonefile.restype = ctypes.c_int
            if clonefile(os.fsencode(source), os.fsencode(destination), 0) != 0:
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code), str(source))
        else:
            return False
    except OSError as exc:
        if exc.errno in unsupported:
            return False
        raise
    return True


def _copy_file(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ApiError(409, f'Managed input is not a regular file: {source}')
    if _clone_file(source, destination):
        # The new, private clone is the snapshot; no data copy or source
        # pre-hash is needed. The full source comparison below still runs.
        digest = sha256_file(destination)
    else:
        digest = hashlib.sha256()
        with source.open('rb') as src, destination.open('xb') as dst:
            while block := src.read(TRANSFER_CHUNK_BYTES):
                dst.write(block)
                digest.update(block)
        digest = digest.hexdigest()
        # Verify the bytes on the destination, not just the write buffer.
        if sha256_file(destination) != digest:
            raise ApiError(409, 'Input snapshot copy failed verification')
    shutil.copystat(source, destination)
    return digest


def snapshot_input(source, destination):
    """Copy a tree, then compare its verified fingerprint with the live source.

    Ordinary copies hash during copying and verify the destination (three
    reads including the final source check). Clones need only two reads.
    Directory ordering, empty directories and digest format match fingerprint.
    """
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.exists():
        raise ApiError(409, f'Managed input is missing or a symlink: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    digests = {}

    def copy_file(src, dst):
        digests[Path(dst)] = _copy_file(src, dst)
        return dst

    if source.is_dir():
        # Preserve links as links temporarily so copytree cannot follow them;
        # fingerprint rejects them before the snapshot can be accepted.
        shutil.copytree(source, destination, symlinks=True, copy_function=copy_file)
    else:
        copy_file(source, destination)
    captured = fingerprint(destination, file_digest=digests.__getitem__)
    if fingerprint(source) != captured:
        raise ApiError(409, 'Input source changed while taking a snapshot')
    return captured
