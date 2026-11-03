import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LINKED = ["real"]
PHASE = re.compile(r"^(?P<phase>\S.*?)\s{2,}\d+ frames .*?query elapsed\s+(?P<query>[\d.]+) ms/frame\s+\| main thread\s+(?P<busy>\d+)%.*?"
                   r"\| caught up\s+(?P<caught>[\d.]+) s")
SETTLE = re.compile(r"^first settle: (?P<seconds>[\d.]+) s\s+\{.*?'wanted': (?P<wanted>\d+).*?'residentMiB': (?P<resident>\d+)")
MEASURES = [("query", "query ms/frame"), ("busy", "main thread %"), ("caught", "caught up s")]

parser = argparse.ArgumentParser(description="Speed of the working tree against HEAD")
parser.add_argument("hash", help="viewer hash, as tools/perf.py takes it")
parser.add_argument("--runs", type=int, default=3, help="runs of each tree")
parser.add_argument("--seconds", type=float, default=4.0, help="length of each phase")
args = parser.parse_args()


def run(tree):
    result = subprocess.run([sys.executable, str(tree / "tools/perf.py"), args.hash, "--seconds", str(args.seconds)],
                            capture_output=True, text=True, cwd=tree)
    if result.returncode != 0:
        raise SystemExit(f"{tree}: tools/perf.py stopped with {result.returncode}\n{result.stderr.strip()}")
    return result.stdout


def parse(output):
    rows = {}
    phase = None
    for line in output.splitlines():
        if (match := SETTLE.match(line)) is not None:
            rows["settle"] = {"seconds": match["seconds"], "wanted": match["wanted"], "resident MiB": match["resident"]}
        elif (match := PHASE.match(line)) is not None:
            phase = match["phase"].strip()
            rows[phase] = {label: match[key] for key, label in MEASURES}
    return rows


head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
worktree = REPO / "outputs" / f"head-{head}"
subprocess.run(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], check=True, cwd=REPO, stdout=subprocess.DEVNULL)
try:
    for name in LINKED:
        if (REPO / name).exists():
            os.symlink(REPO / name, worktree / name)
    trees = {f"HEAD {head}": worktree, "working tree": REPO}
    results = {name: [] for name in trees}
    for i in range(args.runs):
        for name, tree in trees.items():
            print(f"{name} run {i + 1}...", flush=True)
            results[name].append(parse(run(tree)))
finally:
    subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], check=True, cwd=REPO)

phases = []
for runs in results.values():
    for rows in runs:
        phases += [phase for phase in rows if phase not in phases]
width = max(len(label) for runs in results.values() for rows in runs for row in rows.values() for label in row) + 2
columns = {name: max(7 * args.runs, len(name) + 2) for name in trees}
print(f"\n{'':18}{'':{width}}" + "".join(f"{name:<{columns[name]}}" for name in trees))
for phase in phases:
    labels = []
    for runs in results.values():
        for rows in runs:
            labels += [label for label in rows[phase] if label not in labels]
    for j, label in enumerate(labels):
        cells = "".join(f"{''.join(f'{rows[phase][label]:<7}' for rows in runs):<{columns[name]}}" for name, runs in results.items())
        print(f"{phase if j == 0 else '':18}{label:{width}}{cells}")
