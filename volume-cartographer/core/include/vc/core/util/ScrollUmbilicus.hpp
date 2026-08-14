#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "vc/core/util/Umbilicus.hpp"

class VolumePkg;

namespace vc::core::util {

    struct ScrollUmbilicusResolution {
        // The chosen file; empty when nothing could be resolved.
        std::filesystem::path path;
        // Contents of `path`, when it is non-empty.
        UmbilicusFileInfo info;
        // The distinct candidates found when a directory search turned up more
        // than one, in which case `path` is left empty.
        std::vector<std::filesystem::path> ambiguous;
        // Human-readable reason why `path` is empty.
        std::string error;
    };

    // Resolves the umbilicus a package's consumers should use.
    //
    // The project's "umbilicus" field is authoritative: when it is set, that
    // file is the answer and no directory search happens. Otherwise the
    // package root and the parent of every attached segments entry are
    // searched for umbilicus.json, then estimated_umbilicus.json, and the
    // result is only accepted when all roots agree on a single file.
    //
    // Frame metadata is reported as the file declares it; unstamped files are
    // returned as such and callers decide what to do about them.
    [[nodiscard]] ScrollUmbilicusResolution resolveScrollUmbilicus(
        const VolumePkg& pkg);

} // namespace vc::core::util
