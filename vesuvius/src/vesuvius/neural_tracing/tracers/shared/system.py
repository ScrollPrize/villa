"""Process limits."""
import sys


def raise_open_file_limit():
    """Raise the soft descriptor limit before opening mmap caches or workers.

    Workers and collector subprocesses inherit this limit. Keep the macOS
    cap and leave the system hard limit unchanged.
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else max(soft, 1 << 20)
        if sys.platform == 'darwin':
            target = min(target, 10240)
        if soft == resource.RLIM_INFINITY or soft >= target:
            return
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ImportError, ValueError, OSError):
        pass
