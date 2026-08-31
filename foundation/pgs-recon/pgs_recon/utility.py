import subprocess
import sys
from datetime import datetime as dt, timezone as tz
from typing import List


def current_timestamp() -> str:
    return dt.now(tz.utc).strftime("%m/%d/%Y, %H:%M:%S.%f %Z")


def run_command(cmd: List[str], cwd=None):
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except OSError as e:
        print(f'Error: Failed to start command: {" ".join(cmd)}')
        sys.exit(f'{e.args}')
    except subprocess.SubprocessError as e:
        print(f'Error: Command failed: {" ".join(cmd)}')
        sys.exit(f'{e.args}')
    except:
        sys.exit(f'Unexpected error: {sys.exc_info()[0]}')
