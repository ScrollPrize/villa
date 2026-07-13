#!/usr/bin/env bash
# build_appimage.sh — package an already-built VC3D tree into a Linux AppImage.
#
# Single source of truth for AppImage packaging, shared by
# scripts/Dockerfile.appimage (clean-room reproducible build) and by anyone
# packaging a local build tree.
#
# Pipeline:
#   1. install the vc_runtime component into an AppDir
#   2. add the desktop entry + icon
#   3. linuxdeploy + linuxdeploy-plugin-qt: bundle Qt (libs, platform/xcb
#      plugin, imageformats, ...) and every other shared-lib dependency
#   4. (default) bundle glibc + the dynamic loader + libstdc++/libgcc + gconv +
#      NSS, and launch through the bundled loader (see scripts/appimage/AppRun).
#      This is what lets the AppImage run on distros OLDER than the build box —
#      glibc is backward- but not forward-compatible, and the build image
#      (Ubuntu 26.04) ships a very new glibc. Disable with BUNDLE_GLIBC=0.
#   5. appimagetool: squash the AppDir into VC3D-<arch>.AppImage
#
# Prereqs: a completed build (default preset ci-release-gcc), qt6-base-dev
# (qmake6), and patchelf. Tools (linuxdeploy, the qt plugin, appimagetool) are
# downloaded on first run and cached under $TOOLS_DIR.
#
# Usage:
#   scripts/build_appimage.sh
#   BUILD_DIR=build/ci-release-gcc OUT_DIR=dist scripts/build_appimage.sh
#
# Env overrides:
#   BUILD_DIR     CMake build tree to install from (default build/ci-release-gcc)
#   OUT_DIR       Where the .AppImage is written                (default dist)
#   APPDIR        Staging AppDir                     (default $BUILD_DIR/AppDir)
#   TOOLS_DIR     Tool cache                        (default $BUILD_DIR/.appimage-tools)
#   STRIP_FLAGS   strip mode for VC's own binaries/libs   (default --strip-debug;
#                 the Release build still carries DWARF — this reclaims ~490MB
#                 from VC3D alone. --strip-debug keeps .symtab so the crash
#                 handler can still symbolize; use '--strip-unneeded' for the
#                 smallest result, or STRIP_FLAGS= to skip.)
#   DEBUG_BUNDLE  Also emit a VC3D-<arch>-debug.tar.* with the split-out debug
#                 symbols (linked to the stripped binaries via debuglink, so
#                 crash reports stay symbolizable). 1/0, default 1. Ignored when
#                 STRIP_FLAGS is empty.
#   QMAKE         Path to qmake6                              (default: autodetect)
#   ARCH          Target architecture                        (default: uname -m)
#   BUNDLE_GLIBC  Bundle glibc for old-distro support (1/0)   (default 1)
#   GLIBC_SRCDIR  Where to copy glibc libs from  (default /usr/lib/<arch>-linux-gnu)
#   EXCLUDE_TOOLS Space-separated executables to drop from the AppImage before
#                 dependency bundling. Default drops the two video tools
#                 (vc_render_video, vc_diffuse_winding) — the only things that
#                 pull the ~87MB opencv_videoio/ffmpeg chain. Set EXCLUDE_TOOLS=
#                 to keep everything. (Excluded tools remain in the Docker image.)

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"

ARCH="${ARCH:-$(uname -m)}"
BUILD_DIR="${BUILD_DIR:-$repo_root/build/ci-release-gcc}"
OUT_DIR="${OUT_DIR:-$repo_root/dist}"
APPDIR="${APPDIR:-$BUILD_DIR/AppDir}"
TOOLS_DIR="${TOOLS_DIR:-$BUILD_DIR/.appimage-tools}"
QMAKE="${QMAKE:-$(command -v qmake6 || command -v qmake || true)}"
BUNDLE_GLIBC="${BUNDLE_GLIBC:-1}"
GLIBC_SRCDIR="${GLIBC_SRCDIR:-/usr/lib/${ARCH}-linux-gnu}"
EXCLUDE_TOOLS="${EXCLUDE_TOOLS-vc_render_video vc_diffuse_winding}"
STRIP_FLAGS="${STRIP_FLAGS---strip-debug}"   # unset (STRIP_FLAGS=) to skip
DEBUG_BUNDLE="${DEBUG_BUNDLE:-1}"            # also emit VC3D-<arch>-debug.tar.* companion

[ -d "$BUILD_DIR" ] || { echo "error: build dir '$BUILD_DIR' not found — configure & build first" >&2; exit 1; }
[ -n "$QMAKE" ]     || { echo "error: qmake6 not found — install qt6-base-dev (or set QMAKE=...)" >&2; exit 1; }
command -v patchelf >/dev/null || { echo "error: patchelf not found (apt install patchelf)" >&2; exit 1; }
if [ "$BUNDLE_GLIBC" = 1 ] && [ "$ARCH" != x86_64 ]; then
    echo "error: glibc bundling here only handles x86_64 (set BUNDLE_GLIBC=0)" >&2; exit 1
fi

# linuxdeploy / appimagetool can't use FUSE in most containers; extract-and-run
# makes every nested AppImage self-extract.
export APPIMAGE_EXTRACT_AND_RUN=1
export QMAKE

echo "==> Staging install tree into $APPDIR"
rm -rf "$APPDIR"
cmake --install "$BUILD_DIR" --prefix "$APPDIR/usr" --component vc_runtime

if [ -n "$EXCLUDE_TOOLS" ]; then
    echo "==> Excluding tools from the AppImage: $EXCLUDE_TOOLS"
    for t in $EXCLUDE_TOOLS; do
        rm -fv "$APPDIR/usr/bin/$t"
    done
fi

echo "==> Adding desktop entry and icon"
install -Dm644 "$here/appimage/VC3D.desktop" "$APPDIR/usr/share/applications/VC3D.desktop"
# The GUI's window icon (apps/VC3D/logo.png, 128x128) doubles as the AppImage
# icon; its basename must match the desktop file's Icon= key (VC3D).
install -Dm644 "$repo_root/apps/VC3D/logo.png" \
    "$APPDIR/usr/share/icons/hicolor/128x128/apps/VC3D.png"

echo "==> Fetching packaging tools into $TOOLS_DIR"
mkdir -p "$TOOLS_DIR"
fetch() { # url dest
    if [ ! -x "$TOOLS_DIR/$2" ]; then
        curl -fsSL -o "$TOOLS_DIR/$2" "$1"
        chmod +x "$TOOLS_DIR/$2"
    fi
}
ldbase="https://github.com/linuxdeploy"
fetch "$ldbase/linuxdeploy/releases/download/continuous/linuxdeploy-$ARCH.AppImage" "linuxdeploy-$ARCH.AppImage"
fetch "$ldbase/linuxdeploy-plugin-qt/releases/download/continuous/linuxdeploy-plugin-qt-$ARCH.AppImage" "linuxdeploy-plugin-qt-$ARCH.AppImage"
fetch "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage" "appimagetool-$ARCH.AppImage"
# linuxdeploy discovers plugins named `linuxdeploy-plugin-<name>` on PATH.
export PATH="$TOOLS_DIR:$PATH"

echo "==> Bundling Qt + shared-lib dependencies (linuxdeploy)"
"$TOOLS_DIR/linuxdeploy-$ARCH.AppImage" --appdir "$APPDIR" --plugin qt

if [ "$BUNDLE_GLIBC" = 1 ]; then
    echo "==> Bundling glibc + loader (for old-distro compatibility)"
    libdir="$APPDIR/usr/lib"
    loader="$(patchelf --print-interpreter "$APPDIR/usr/bin/VC3D")"
    cp -Lv "$loader" "$libdir/"
    # glibc runtime family + libstdc++/libgcc (all excluded by linuxdeploy),
    # plus the NSS modules glibc dlopens for host/user lookups.
    for l in libc.so.6 libm.so.6 libmvec.so.1 libdl.so.2 libpthread.so.0 \
             librt.so.1 libresolv.so.2 libutil.so.1 libanl.so.1 \
             libBrokenLocale.so.1 libnss_files.so.2 libnss_dns.so.2 \
             libnss_compat.so.2 libstdc++.so.6 libgcc_s.so.1; do
        if [ -e "$GLIBC_SRCDIR/$l" ]; then
            cp -Lv "$GLIBC_SRCDIR/$l" "$libdir/"
        else
            echo "   (skip missing $l)"
        fi
    done
    # charset conversion modules (iconv) — glibc loads these from GCONV_PATH.
    [ -d "$GLIBC_SRCDIR/gconv" ] && cp -a "$GLIBC_SRCDIR/gconv" "$libdir/gconv"
    # offscreen QPA plugin — lets headless CLI tools / CI init Qt without a display.
    qpd="$("$QMAKE" -query QT_INSTALL_PLUGINS)"
    [ -e "$qpd/platforms/libqoffscreen.so" ] && \
        cp -Lv "$qpd/platforms/libqoffscreen.so" "$APPDIR/usr/plugins/platforms/"
fi

if [ -n "$STRIP_FLAGS" ]; then
    before=$(du -sm "$APPDIR" | cut -f1)
    dbgstage=""
    if [ "$DEBUG_BUNDLE" = 1 ]; then
        echo "==> Stripping ($STRIP_FLAGS) + extracting debug symbols into a companion bundle"
        dbgstage="$BUILD_DIR/.appimage-debug"; rm -rf "$dbgstage"
    else
        echo "==> Stripping binaries and libraries ($STRIP_FLAGS)"
    fi
    # Everything except the bundled dynamic loader (never strip ld-linux).
    while IFS= read -r -d '' f; do
        case "$(basename "$f")" in ld-linux-*) continue ;; esac
        # Split debug info out to a companion .debug (linked via debuglink) only
        # for files that actually carry it — skip already-stripped system libs.
        if [ -n "$dbgstage" ] && readelf -SW "$f" 2>/dev/null | grep -q '\.debug_info'; then
            rel="${f#"$APPDIR"/}"; dbg="$dbgstage/$rel.debug"
            mkdir -p "$(dirname "$dbg")"
            objcopy --only-keep-debug "$f" "$dbg" 2>/dev/null || true
            strip $STRIP_FLAGS "$f" 2>/dev/null || true
            objcopy --add-gnu-debuglink="$dbg" "$f" 2>/dev/null || true
        else
            strip $STRIP_FLAGS "$f" 2>/dev/null || true
        fi
    done < <(find "$APPDIR/usr/bin" "$APPDIR/usr/lib" "$APPDIR/usr/plugins" \
                  -type f \( -name '*.so*' -o -perm -u+x \) -print0)
    after=$(du -sm "$APPDIR" | cut -f1)
    echo "    AppDir: ${before} MB -> ${after} MB"

    if [ -n "$dbgstage" ] && [ -d "$dbgstage" ]; then
        mkdir -p "$OUT_DIR"
        if command -v zstd >/dev/null 2>&1; then
            dbgout="$OUT_DIR/VC3D-$ARCH-debug.tar.zst"; tar -C "$dbgstage" --zstd -cf "$dbgout" .
        else
            dbgout="$OUT_DIR/VC3D-$ARCH-debug.tar.gz";  tar -C "$dbgstage" -czf "$dbgout" .
        fi
        echo "    Debug bundle: $dbgout ($(du -sh "$dbgout" | cut -f1))"
    fi
fi

echo "==> Installing AppRun"
install -m755 "$here/appimage/AppRun" "$APPDIR/AppRun"

echo "==> Squashing AppImage (appimagetool)"
mkdir -p "$OUT_DIR"
out="$OUT_DIR/VC3D-$ARCH.AppImage"
ARCH="$ARCH" "$TOOLS_DIR/appimagetool-$ARCH.AppImage" "$APPDIR" "$out"
echo "==> Done: $out"
