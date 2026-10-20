"""Shared vc_reader volume access and authentication for datasets and label caches."""
from vc_reader.reader import S3Credentials, load_basic_auth, open_zarr


def open_volume(path, level=None, auth=None, s3_storage_options=None):
    credentials = None
    if str(path).startswith("s3://"):
        options = dict(s3_storage_options or {})
        if options.get("key") and options.get("secret"):
            credentials = S3Credentials(options["key"], options["secret"], options.get("token"))
        elif not options.get("anon", True):
            credentials = "aws"
    elif str(path).startswith(("http://", "https://")) and auth:
        credentials = load_basic_auth(auth)
    return open_zarr(str(path), level=level, auth=credentials, threads=1, memory_cache=0)
