# Release-only variant of the stock x64-windows triplet (dynamic CRT + dynamic
# libs, matching Qt/OpenCV's requirements). Dropping the debug build of every
# dependency roughly halves the vcpkg build time and on-disk footprint, which
# matters a lot for the Qt6 + OpenCV(contrib) + CGAL + Ceres closure.
#
# The VC3D app is therefore built as Release/RelWithDebInfo (release CRT, /MD)
# so it links cleanly against these release-only dependencies. A Debug (/MDd)
# app build would mismatch the CRT — use RelWithDebInfo for debugging instead.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)
