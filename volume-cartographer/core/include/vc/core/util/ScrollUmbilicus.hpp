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
    // The project's "umbilicus" field is authoritative: when it is set at all,
    // that location is the answer and no directory search ever happens — not
    // even when the location cannot be used. A configured location that
    // umbilicusPath() cannot turn into a local file (s3://, http://, ...) is
    // reported as an unsupported-location error, distinct from the
    // does-not-exist error a missing local file gets.
    //
    // Only an empty field proceeds to discovery: the package root, plus both
    // the parent and the grandparent of every attached segments entry, are
    // searched for umbilicus.json, then estimated_umbilicus.json. Searching
    // grandparents covers the <volpkg>/paths/<segment> layout, whose
    // <volpkg>/umbilicus.json a parent-only search misses. The result is only
    // accepted when all roots agree on a single file; one file reachable
    // through several roots is deduplicated and stays unambiguous.
    //
    // Frame metadata is reported as the file declares it, and unstamped files
    // are returned as such for callers to decide about — but a file whose
    // metadata is present and malformed (UmbilicusFileInfo::metadataErrors) is
    // refused outright, with the errors listed in `error`, so that a typo can
    // never be mistaken downstream for a legacy unstamped file.
    [[nodiscard]] ScrollUmbilicusResolution resolveScrollUmbilicus(
        const VolumePkg& pkg);

} // namespace vc::core::util
