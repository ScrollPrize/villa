_PUBLIC_S3_ANON_BUCKETS = frozenset({"vesuvius-challenge-open-data"})


def s3_storage_options_for_path(path) -> dict:
    """Return fsspec/zarr S3 options for a path.

    Only the public Vesuvius Challenge open-data bucket is opened anonymously;
    other S3 buckets keep the authenticated default.
    """
    path = str(path)
    if not path.startswith("s3://"):
        return {}
    bucket = path[len("s3://") :].split("/", 1)[0]
    return {"anon": bucket in _PUBLIC_S3_ANON_BUCKETS}
