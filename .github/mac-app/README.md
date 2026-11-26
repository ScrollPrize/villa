# VC3D macOS Python tools

Spiral, Lasagna, sheet strips, fibre follow, fibre collections and the neural tracer run as Python services that need
villa's native packages. The [VC3D macOS App workflow](../workflows/vc3d-macos.yml) builds those packages as prebuilt
wheels next to VC3D.app, so nobody on a Mac compiles them.

## What the workflow adds

- The `python-tools` job builds the cp314 macOS arm64 wheels of vc-reader, vc-object-store, spiral, volume-cartographer,
  vesuvius-lasagna and villa-gpu-manager, plus posix-ipc, which has no such wheel on PyPI. It also writes one
  hash-pinned requirements file per environment (`spiral`, `lasagna`, `vesuvius`), the villa sources the services run
  from, and the installer `VC3D-macos-python-tools.sh` with every file's sha256 written into it. It then runs that
  installer as a new user would (`env -i`, a fresh home, Homebrew moved aside) and uploads everything as the
  `VC3D-macos-python-tools` artifact.
- `build-macos` builds the app with that installer in `VC3D.app/Contents/Resources/install-python-tools.sh`
  (`VC_MACOS_PYTHON_TOOLS_INSTALLER`), where the app's signature covers it. VC3D runs it, after asking, the first time
  a feature needs its Python environment, and from Help > Install or Update Python Tools. The installer needs no
  password, compiler, Xcode, Homebrew or conda: it fetches a pinned uv, which installs Python 3.14 and the three
  environments into `~/Library/Application Support/Vesuvius Challenge/VC3D/python-tools`.
- The publish workflows (`vc3d-publish-latest.yml`, `vc3d-promote-stable.yml`, and `release-macos` for a published
  release) attach the Python tools from the same run as the app, so a release's app and wheels always match.
- A failed `python-tools` job doesn't fail the run or hold back the app. `build-macos` then packages the app without
  the installer, so it offers no install and shows no Python tools menu items, and the publish workflows publish it
  without the Python tools, with a warning.

The installer finds its files by the source tarball, `VC3D-macos-python-tools-<commit>-<build>.tar.gz`, which is named
for the commit and for a hash of that build's wheels and locks: two builds of one commit (a published release, a stable
rebuild) hold different bytes, and each app's installer has only its own build's checksums. It looks under the `latest` release, then `stable`, then searches the repository's releases
through the GitHub API. That last step finds the `vc3d-archive-*` release that `latest` or `stable` became when its tag
moved on, so an app keeps installing its own wheels after newer builds are published. Every file is checked against
its sha256 before use. Release files in the same folder as the installer, or in `VC3D_TOOLS_ASSETS`, are used instead
of downloading them; `VC3D_PYTHON_TOOLS` overrides the install folder.

## Scripts

- `build-opencv.sh PREFIX [JOBS]`: a static OpenCV core and imgproc for the vc-reader, vc-object-store and spiral
  wheels, so they bundle no OpenCV dylibs.
- `build-wheels.sh SRC OUT [JOBS]`: builds the wheels, delocates them, and points every libomp reference at torch's
  copy, so a process holds one OpenMP runtime (`fix_wheel.py` does the rewrite and the leak checks).
- `make-assets.sh VILLA WHEELS OUT REPOSITORY MIN_MACOS`: the source tarball, the locks (resolved from
  `lock-inputs/*.in` for `MIN_MACOS`) and the filled-in installer.
- `install-python-tools.sh.in`: the installer template.
- `omp_check.py MODULE...`: imports the modules and torch, and fails unless exactly one OpenMP runtime (torch's) is
  loaded and it keeps several cores busy.

To build the Python tools on a Mac by hand and check an install as a new user gets it:

```sh
R=.github/mac-app
$R/build-opencv.sh /tmp/opencv
OCV=/tmp/opencv TORCH_LIBOMPS="<torch/lib/libomp.dylib of each torch the locks pin>" \
  MACOSX_DEPLOYMENT_TARGET=15.0 $R/build-wheels.sh <git archive of HEAD> /tmp/wheels
$R/make-assets.sh . /tmp/wheels/wheels /tmp/python-tools ScrollPrize/villa 15.0
home=$(mktemp -d)
env -i HOME="$home" PATH=/usr/bin:/bin:/usr/sbin:/sbin bash /tmp/python-tools/VC3D-macos-python-tools.sh
```
