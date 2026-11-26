#!/usr/bin/env bash
# Assemble the Python tools' release files from built wheels: the villa source tarball, one hash-pinned requirements file
# per environment, and the installer with their checksums filled in. Pins come from the environments VC3D's features
# were tested with (lock-inputs/*.in); villa's own packages resolve from the wheels.
#   make-assets.sh VILLA_CHECKOUT WHEELS_DIR OUT REPOSITORY MIN_MACOS
# REPOSITORY is the GitHub repository whose releases carry these files, such as ScrollPrize/villa (the workflow passes
# github.repository). The installer finds the release by its source tarball, which is named for the commit and for a
# hash of this build's wheels and locks, so the files can be published under any tag, and another build of the same
# commit (a release or a stable rebuild) is never mistaken for this one
# MIN_MACOS (such as 15.0 or 27.2) is the oldest macOS the wheels run on. The locks resolve for it, so every pinned
# package has a wheel that installs there, and the installer refuses anything older
# OUT must be new or empty, so the installer lists exactly this run's files. The wheels hold only native extensions and
# packaging, so app-only commits on top of the tree they were built from don't need a rebuild
# The source tarball is `git archive` of the checkout's HEAD without the website, deprecated and foundation trees
set -euo pipefail
villa=$(cd "$1" && pwd); wheels=$(cd "$2" && pwd); out=$3; repository=$4; min_macos=$5
[[ $repository =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo "REPOSITORY must look like owner/name, not $repository" >&2; exit 1; }
[[ $min_macos =~ ^[0-9]+\.[0-9]+$ ]] || { echo "MIN_MACOS must look like 15.0, not $min_macos" >&2; exit 1; }
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYV=3.14.10
mkdir -p "$out"; out=$(cd "$out" && pwd)
[ -z "$(ls -A "$out")" ] || { echo "$out is not empty" >&2; exit 1; }
commit=$(git -C "$villa" rev-parse --short=9 HEAD)
[ -z "$(git -C "$villa" status --porcelain --untracked-files=no)" ] || { echo "uncommitted changes in $villa" >&2; exit 1; }
prefix=VC3D-macos-python-tools
tarball=$prefix-$commit.tar.gz
git -C "$villa" archive --format=tar.gz --prefix=villa/ -o "$out/$tarball" HEAD -- . \
  ':(exclude)scrollprize.org' ':(exclude)deprecated' ':(exclude)foundation'
echo "$tarball: $(du -h "$out/$tarball" | cut -f1)"

cp "$wheels"/*.whl "$out/"
uv python install "$PYV"
PY=$(uv python find --python-preference only-managed "$PYV")
for env in spiral lasagna vesuvius; do
  MACOSX_DEPLOYMENT_TARGET=$min_macos uv pip compile --quiet --python "$PY" --python-platform aarch64-apple-darwin \
    --no-build --generate-hashes --find-links "$out" --index-strategy unsafe-best-match \
    --custom-compile-command ".github/mac-app/make-assets.sh" \
    -o "$out/$prefix-requirements-$env.txt" "$here/lock-inputs/$env.in"
  echo "$prefix-requirements-$env.txt: $(grep -c '==' "$out/$prefix-requirements-$env.txt") packages"
done
# each of villa's packages must resolve to the wheel built here: PyPI hosts nothing under most of these names, so a
# package published there later with the same name and version must not get its hash into a lock
python3 -I "$here/fix_wheel.py" check-locks "$out"

# this build: wheels from another run of the same commit differ byte for byte
build=$(cd "$out" && shasum -a 256 ./*.whl "$prefix"-requirements-*.txt | shasum -a 256 | cut -c1-12)
mv "$out/$tarball" "$out/$prefix-$commit-$build.tar.gz"
tarball=$prefix-$commit-$build.tar.gz

assets=$(cd "$out" && shasum -a 256 ./*.whl "$prefix"-requirements-*.txt "$tarball" | sed 's| \./| |; s|  | |')
python3 -I - "$here/install-python-tools.sh.in" "$out/$prefix.sh" "$repository" "$commit" "$build" "$assets" \
  "$min_macos" <<'PY'
import sys
from pathlib import Path
src, dst, repository, commit, build, assets, min_macos = sys.argv[1:]
text = Path(src).read_text()
for key, value in (("@REPOSITORY@", repository), ("@VILLA_COMMIT@", commit), ("@TOOLS_BUILD@", build),
                   ("@ASSETS@", assets), ("@MIN_MACOS@", min_macos)):
    assert key in text, key
    text = text.replace(key, value)
Path(dst).write_text(text)
PY
chmod +x "$out/$prefix.sh"
bash -n "$out/$prefix.sh"
echo "Python tools for villa $commit (build $build) in $out"
