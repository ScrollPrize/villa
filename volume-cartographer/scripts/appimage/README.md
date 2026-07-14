# Linux AppImage packaging

Bundles VC3D (the GUI) plus all ~35 `vc_*` command-line tools into a single
relocatable `VC3D-<version>-x86_64.AppImage`, where `<version>` is the git
`<sha7>-<commit-date>` (matching the macOS/Windows package names). Ad-hoc builds
with no git info produce an unversioned `VC3D-x86_64.AppImage`.

**Runs on old distros.** The build image (Ubuntu 26.04) ships a very new glibc
(2.43). glibc is backward- but not forward-compatible, so a binary built there
would refuse to start on anything older (`version 'GLIBC_2.43' not found`). To
avoid that, the packaging **bundles glibc + the dynamic loader** (plus
`libstdc++`, `gconv` charset modules, and NSS libs) and launches the app through
the bundled loader, so the host's glibc is never used. The result runs on
essentially any modern Linux (validated on glibc 2.42; works well below that).

## Files

| File | Purpose |
| --- | --- |
| `../build_appimage.sh` | Core packaging logic (shared source of truth): install the `vc_runtime` component → `linuxdeploy` + `linuxdeploy-plugin-qt` (Qt libs, platform/xcb plugin, all shared-lib deps) → bundle glibc/loader → `appimagetool`. |
| `../Dockerfile.appimage` | Clean-room reproducible build on the CI builder image. |
| `VC3D.desktop` | Desktop entry (menu name, icon, categories). |
| `AppRun` | Entry point. Sets `QT_PLUGIN_PATH`, launches via the bundled loader, and dispatches the GUI or a CLI tool by symlink name (see below). |

The AppImage icon reuses the GUI window icon (`apps/VC3D/logo.png`).

> Note on tooling: `linuxdeploy-plugin-qt` handles the Qt bundle because the Qt
> platform plugin (`libqxcb.so`) and its X11/xcb dependency tree are `dlopen`ed
> at runtime — a link-scan bundler (e.g. sharun) misses them. The glibc/loader
> bundling on top is the same technique sharun uses, applied to that AppDir.

## Build

### Reproducible (Docker, recommended for releases)

Writes `dist/VC3D-<version>-x86_64.AppImage` on the host. The build context is
only the `volume-cartographer/` subtree (no `.git`), so pass the revision in for
a versioned name; without it the binaries and file name fall back to
`Untracked`:

```bash
docker build -o type=local,dest=dist -f scripts/Dockerfile.appimage \
  --build-arg VC_GIT_SHA1="$(git rev-parse HEAD)" \
  --build-arg VC_GIT_COMMIT_DATE="$(git log -1 --format=%cs HEAD)" .
# against the local dev builder image instead of ghcr:
docker build -o type=local,dest=dist -f scripts/Dockerfile.appimage \
  --build-arg BUILDER_IMAGE=vc-builder:dev .
```

### From an existing local build

Needs a completed build (default preset `ci-release-gcc`) plus `qt6-base-dev`
(`qmake6`) and `patchelf`:

```bash
cmake --preset ci-release-gcc
cmake --build --preset ci-release-gcc
scripts/build_appimage.sh          # -> dist/VC3D-<version>-x86_64.AppImage
```

Here the version is read from git in the source tree automatically (override
with `VERSION=`, or `GIT_SHA1=`/`GIT_COMMIT_DATE=`).

Set `BUNDLE_GLIBC=0` to skip glibc bundling (smaller image, but then it only
runs on hosts whose glibc is at least as new as the build machine's).

## Running

The AppImage is a multitool: the first argument selects the GUI (`VC3D`) or any
bundled CLI tool. The examples below write the file name as `VC3D.AppImage` for
brevity — substitute the actual `VC3D-<version>-x86_64.AppImage` (or rename it).

```bash
./VC3D.AppImage VC3D                   # launch the GUI
./VC3D.AppImage VC3D --cache-size 20   # GUI with options
./VC3D.AppImage vc_obj2tifxyz in.obj out/   # run a CLI tool
./VC3D.AppImage list                   # list the bundled tools
./VC3D.AppImage help                   # usage
```

A short `vc` symlink makes it ergonomic; each tool can also be a symlink of its
own:

```bash
ln -s VC3D.AppImage vc
./vc vc_obj2tifxyz in.obj out/
./vc VC3D                                     # GUI

ln -s VC3D.AppImage vc_obj2tifxyz      # named-link dispatch
./vc_obj2tifxyz in.obj out/
```

Selection is explicit: a bare option (`./VC3D.AppImage --cache-size 20`)
is rejected — pass it through a tool (`... VC3D --cache-size 20`). Running the
AppImage with no arguments shows this usage in a terminal, and launches the GUI
when double-clicked from a desktop (no terminal attached).

On systems without FUSE, prefix with `APPIMAGE_EXTRACT_AND_RUN=1` or install
`libfuse2`.

## Prerequisites (what the host must provide)

The AppImage bundles glibc + the loader, Qt, OpenCV, Ceres, and the whole
`vc_*` closure. It deliberately does **not** bundle the GPU/graphics stack or
the font libraries (linuxdeploy's excludelist) — those are driver-coupled or
version-sensitive and are present on every desktop. The host must provide:

- **OpenGL (glvnd + a driver)** — `libGLX.so.0`, `libEGL.so.1`, `libOpenGL.so.0`
  (Debian/Ubuntu: `libglvnd0 libglx0 libegl1 libopengl0 libgl1`) plus a driver
  (`libgl1-mesa-dri libglx-mesa0`, or the vendor driver).
- **X11 client libs** — `libX11.so.6`, `libX11-xcb.so.1`, `libxcb.so.1`,
  `libICE.so.6`, `libSM.so.6` (`libx11-6 libx11-xcb1 libxcb1 libice6 libsm6`).
- **Fonts / text shaping** — `libfontconfig.so.1`, `libfreetype.so.6`,
  `libharfbuzz.so.0` (`libfontconfig1 libfreetype6 libharfbuzz0b`).
- **Ubiquitous base libs** (already on essentially every system) —
  `libexpat1 zlib1g libuuid1 libcom-err2 libgmp10 libgpg-error0`.

On any normal Linux **desktop** these are all already installed; the list
matters only for minimal/headless/container environments.

### CLI tools vs. the GUI

`vc_core` links Qt6::Gui, so every binary (CLI included) needs the libraries
above to be *present* — but the CLI tools don't open a display and tolerate old
versions. The GUI additionally needs a running display server (X, or Wayland via
XWayland/xcb) and a **recent FreeType** (`FT_Get_Paint`, FreeType ≥ 2.11).

Tested (glibc is bundled, so it's never the limit):

| Distro | glibc | FreeType | CLI tools | GUI |
| --- | --- | --- | --- | --- |
| Ubuntu 24.04 | 2.39 | 2.13 | ✅ | ✅ |
| Ubuntu 22.04 | 2.35 | 2.11 | ✅ | ✅ |
| Ubuntu 20.04 | 2.31 | 2.10 | ✅ | ❌ `undefined symbol: FT_Get_Paint` |

**Effective floor:** CLI → glibc 2.31+ (Ubuntu 20.04+); GUI → FreeType 2.11+
(Ubuntu 22.04+, Debian 12+, Fedora 36+). Bundling `libfreetype`+`libharfbuzz`
would lower the GUI floor to match the CLI, at the cost of using host font
config — left to the host by default here.

## Size

The Release build carries full DWARF debug info (VC3D alone is ~490 MB
unstripped), so packaging **strips** the binaries and, by default, extracts the
symbols into a separate companion. Two artifacts result:

| Artifact | Size | Notes |
| --- | --- | --- |
| `VC3D-<version>-x86_64.AppImage` | **~116 MB** | Minified download. |
| `VC3D-<version>-x86_64-debug.tar.zst` | ~375 MB | Split-out debug symbols, linked to the stripped binaries via `.gnu_debuglink` — keep it around to symbolize crash reports / run under gdb. |

Set `DEBUG_BUNDLE=0` to skip the companion, or `STRIP_FLAGS=` to ship an
unstripped AppImage instead.

Two size levers are applied by default:
- **Strip + split debug info.** The Release build carries full DWARF (VC3D alone
  is ~490 MB unstripped); stripping reclaims it into the companion above.
- **`EXCLUDE_TOOLS`.** `vc_render_video` and `vc_diffuse_winding` are dropped from
  the AppImage — they are the only binaries that pull the ~87 MB
  `opencv_videoio` → ffmpeg codec chain (x265/codec2/aom/SvtAv1/…), and neither
  is used by any pipeline. They remain in the Docker image. Set `EXCLUDE_TOOLS=`
  to keep them (adds ~45 MB to the download).

Where the ~116 MB goes (uncompressed AppDir ~354 MB, zstd-squashed): binaries
are ~25 MB after stripping — the rest is genuinely-used shared libraries. The
largest are all real dependencies: OpenBLAS (numerics), the OpenCV **GDAL** chain
(`libopencv_imgcodecs` hard-links `libgdal` for image I/O), ICU (Qt), and Qt
itself. Compression is already zstd and near its floor (xz is unsupported by the
bundled mksquashfs and too slow to mount anyway; zstd level 22 saves <1 MB).

The one sizeable lever left needs a build-level change, not a packaging tweak:
rebuild OpenCV with `-DWITH_GDAL=OFF` → drops the ~56 MB GDAL chain (VC only
needs tiff/png/jpg, which imgcodecs handles without GDAL). That requires a
source OpenCV build in the builder image, so it's out of scope here.
