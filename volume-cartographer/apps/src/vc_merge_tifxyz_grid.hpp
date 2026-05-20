// Grid-resolution helpers for vc_merge_tifxyz.
//
// Exposes the merge.json parser and edge-graph connectivity check so that
// they can be unit-tested without spawning the full binary. The merge tool
// itself includes this header and uses these symbols directly.

#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vc::merge {

struct GMSurfaceSpec {
    std::string name;
    std::filesystem::path path;
};

struct GMEdgeSpec {
    std::string a;
    std::string b;
};

// Parse a row-major grid of tifxyz directory names from merge.json and
// emit deduplicated surfaces + horizontal/vertical adjacency edges. The
// grid is loaded from `merge_path`; cell strings are resolved as
// `paths_dir / name` and validated to be directories. Throws
// std::runtime_error on malformed input, missing directories, or a grid
// containing fewer than 2 distinct surfaces.
void gmResolveGrid(const std::filesystem::path& merge_path,
                   const std::filesystem::path& paths_dir,
                   std::vector<GMSurfaceSpec>& surfaces,
                   std::vector<GMEdgeSpec>& edges);

// Verify that `edges` connects every surface in `surfaces`. Throws
// std::runtime_error listing the unreachable surfaces if not.
void gmCheckConnected(const std::vector<GMSurfaceSpec>& surfaces,
                      const std::vector<GMEdgeSpec>& edges);

} // namespace vc::merge
