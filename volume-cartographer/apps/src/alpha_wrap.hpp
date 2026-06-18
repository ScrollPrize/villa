// CGAL firewall for vc_add_ignore_label. The 3D alpha-wrap + point-in-mesh
// classification is the heaviest TU in the build (~120s, all CGAL template
// instantiation). Keeping the CGAL include behind this CGAL-free boundary
// confines that cost to alpha_wrap.cpp so edits to the main tool stay fast.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

using Shape3 = std::array<std::size_t, 3>;

struct Box3 {
    Shape3 origin = {0, 0, 0};
    Shape3 shape = {0, 0, 0};
};

// Wrap the non-zero voxels of `halo` (in haloBox coords) with CGAL's
// alpha_wrap_3, then mark every core voxel that falls outside the wrap.
// Returns a coreBox-shaped mask (255 = ignore, 0 = keep).
std::vector<uint8_t> classifyOuterAlphaWrap(const std::vector<uint8_t>& halo,
                                            const Box3& haloBox,
                                            const Box3& coreBox,
                                            double alpha,
                                            double offset);
