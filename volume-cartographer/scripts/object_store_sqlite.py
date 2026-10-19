"""SQLite diagnostics isolated from the native object-store library's locks."""
import json
import subprocess
import sys


def sqlite_rows(path, statement, parameters=()):
    # Python and vc.objects can link different SQLite builds. Closing one
    # build's WAL connection in-process can invalidate the other's POSIX locks.
    code = "import json,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); r=c.execute(sys.argv[2],json.loads(sys.argv[3])).fetchall(); c.commit(); print(json.dumps(r)); c.close()"
    return json.loads(subprocess.check_output(
        [sys.executable, '-c', code, str(path), statement, json.dumps(parameters)], text=True))
