# Linux AppImage packaging

Bundles VC3D (the GUI) plus all ~35 `vc_*` command-line tools into a single
relocatable `VC3D-x86_64.AppImage`.

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

Writes `dist/VC3D-x86_64.AppImage` on the host:

```bash
docker build -o type=local,dest=dist -f scripts/Dockerfile.appimage .
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
scripts/build_appimage.sh          # -> dist/VC3D-x86_64.AppImage
```

Set `BUNDLE_GLIBC=0` to skip glibc bundling (smaller image, but then it only
runs on hosts whose glibc is at least as new as the build machine's).

## Running

```bash
./VC3D-x86_64.AppImage             # launches the GUI

# Run any bundled CLI tool by symlinking the AppImage to its name:
ln -s VC3D-x86_64.AppImage vc_obj2tifxyz
./vc_obj2tifxyz in.obj out/        # runs the vc_obj2tifxyz binary
```

On systems without FUSE, prefix with `APPIMAGE_EXTRACT_AND_RUN=1` or install
`libfuse2`.

## Size

The Release build carries full DWARF debug info (VC3D alone is ~490 MB
unstripped), so packaging **strips** the binaries and, by default, extracts the
symbols into a separate companion. Two artifacts result:

| Artifact | Size | Notes |
| --- | --- | --- |
| `VC3D-x86_64.AppImage` | **~116 MB** | Minified download. |
| `VC3D-x86_64-debug.tar.zst` | ~375 MB | Split-out debug symbols, linked to the stripped binaries via `.gnu_debuglink` — keep it around to symbolize crash reports / run under gdb. |

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
