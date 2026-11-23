"""SQLite allocator options are explicit process-startup API calls."""
import subprocess
import sys


def test_python_import_does_not_prevent_explicit_startup_configuration():
    result=subprocess.run([sys.executable,'-c',
        "from vc_object_store import configure_sqlite_runtime; "
        "configure_sqlite_runtime(memory_statistics=False,pagecache_bulk_bytes=1048576); "
        "print('configured')"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='configured'
