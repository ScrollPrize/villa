#include "vc/core/util/ScrollUmbilicus.hpp"

#include <algorithm>
#include <optional>
#include <limits>
#include <cmath>
#include <array>
#include <vector>
#include <exception>
#include <system_error>

#include "vc/core/types/VolumePkg.hpp"

namespace fs = std::filesystem;

namespace {

// A tifxyz surface directory carries all three coordinate planes; a directory of
// such surfaces carries none of its own. Checking the payload rather than
// meta.json keeps this from matching directories that happen to hold a file by
// that name for unrelated reasons.
bool isTifxyzSegmentDir(const fs::path& dir)
{
    std::error_code ec;
    for (const char* plane : {"x.tif", "y.tif", "z.tif"}) {
        if (!fs::exists(dir / plane, ec) || ec) {
            return false;
        }
    }
    return true;
}


fs::path canonicalize(const fs::path& path)
{
    std::error_code ec;
    auto canonical = fs::weakly_canonical(path, ec);
    if (ec) {
        return path.lexically_normal();
    }
    return canonical;
}

std::string joinPaths(const std::vector<fs::path>& paths)
{
    std::string joined;
    for (const auto& path : paths) {
        if (!joined.empty()) {
            joined += ", ";
        }
        joined += path.string();
    }
    return joined;
}

// Malformed frame metadata is a hard failure here rather than in every
// consumer: this function is the one place that answers "which umbilicus, in
// which frame", so a typo must not be allowed to look like an unstamped file.
std::string describeMetadataErrors(const fs::path& path,
                                   const std::vector<std::string>& errors)
{
    std::string message =
        "the umbilicus file " + path.string() + " declares malformed metadata: ";
    for (std::size_t i = 0; i < errors.size(); ++i) {
        if (i != 0) {
            message += "; ";
        }
        message += errors[i];
    }
    return message;
}

// The two names a search recognises, in priority order.
constexpr const char* kUmbilicusFileNames[] = {"umbilicus.json",
                                               "estimated_umbilicus.json"};

} // namespace

namespace vc::core::util {

// Directories a search looks in, in priority order and deduplicated.
std::vector<fs::path> umbilicusSearchRoots(const VolumePkg& pkg)
{
    std::vector<fs::path> roots;
    const auto addRoot = [&roots](const fs::path& root) {
        if (root.empty()) {
            return;
        }
        if (std::find(roots.begin(), roots.end(), root) != roots.end()) {
            return;
        }
        roots.push_back(root);
    };

    addRoot(pkg.path().empty() ? fs::path(pkg.getVolpkgDirectory())
                               : pkg.path().parent_path());
    for (const auto& segment : pkg.availableSegmentPaths()) {
        // The entry may be the segments directory itself, whose parent is the
        // volpkg, or a single segment inside it, whose grandparent is. Only the
        // latter earns the extra level: taking the grandparent of a segments
        // directory would search whatever holds the volpkg, where an unrelated
        // umbilicus could win or force a false ambiguity.
        addRoot(segment.parent_path());
        if (isTifxyzSegmentDir(segment)) {
            addRoot(segment.parent_path().parent_path());
        }
    }
    return roots;
}

std::vector<UmbilicusCandidate> scanUmbilicusCandidates(const VolumePkg& pkg)
{
    std::vector<UmbilicusCandidate> candidates;

    // The project's field short circuits the search entirely, so it is the whole
    // dependency: no discovered file can change what the resolver answers while it
    // is set. Included even when it does not exist, because its appearing does.
    if (const auto configured = pkg.umbilicus(); !configured.empty()) {
        const auto declared = pkg.umbilicusPath();
        if (declared.empty()) {
            // Remote or otherwise unsupported: the resolver errors without
            // touching the filesystem, so nothing on disk is a dependency.
            return candidates;
        }
        std::error_code ec;
        UmbilicusCandidate candidate;
        candidate.path = declared;
        candidate.exists = fs::exists(declared, ec) && !ec;
        candidate.decidesResolution = true;
        candidates.push_back(std::move(candidate));
        return candidates;
    }

    std::vector<fs::path> canonical;
    for (const auto& root : umbilicusSearchRoots(pkg)) {
        for (const char* name : kUmbilicusFileNames) {
            const fs::path path = root / name;
            // Canonical dedup, matching the search: two roots reaching one file
            // through a symlink must count once, or a caller comparing these would
            // see a change that did not happen. Deliberately `continue` and not
            // `break`: skipping a duplicate must not end this root's own priority
            // walk, or a root whose umbilicus.json is an alias of an earlier hit
            // would go on to offer its distinct estimated_umbilicus.json.
            const auto resolvedPath = canonicalize(path);
            const bool duplicate =
                std::find(canonical.begin(), canonical.end(), resolvedPath) !=
                canonical.end();
            std::error_code ec;
            const bool exists = fs::exists(path, ec) && !ec;
            if (!duplicate) {
                canonical.push_back(resolvedPath);
                UmbilicusCandidate candidate;
                candidate.path = path;
                candidate.exists = exists;
                // Every hit is one the resolver would open — a second one makes the
                // resolution ambiguous rather than being ignored.
                candidate.decidesResolution = exists;
                candidates.push_back(std::move(candidate));
            }
            if (exists) {
                // The search stops at the first existing name in this root, so no
                // lower-priority name here can affect the answer.
                break;
            }
        }
    }
    return candidates;
}

std::vector<fs::path> umbilicusCandidatePaths(const VolumePkg& pkg)
{
    std::vector<fs::path> paths;
    for (auto& candidate : scanUmbilicusCandidates(pkg)) {
        paths.push_back(std::move(candidate.path));
    }
    return paths;
}

ScrollUmbilicusResolution resolveScrollUmbilicus(const VolumePkg& pkg)
{
    ScrollUmbilicusResolution resolution;

    // The raw configured location, not umbilicusPath(), decides whether the
    // field is set: umbilicusPath() blanks remote locations, and a blank there
    // must not be mistaken for "nothing configured" and trigger a search.
    if (const auto configured = pkg.umbilicus(); !configured.empty()) {
        const auto declared = pkg.umbilicusPath();
        if (declared.empty()) {
            resolution.error =
                "the project's \"umbilicus\" field is set to " + configured +
                ", a remote or otherwise unsupported location that cannot be "
                "used as an umbilicus file";
            return resolution;
        }
        std::error_code ec;
        if (!fs::exists(declared, ec) || ec) {
            resolution.error =
                "the project's \"umbilicus\" field points at " +
                declared.string() + ", which does not exist";
            return resolution;
        }
        try {
            auto info = Umbilicus::LoadFileInfo(declared);
            if (!info.metadataErrors.empty()) {
                resolution.error =
                    describeMetadataErrors(declared, info.metadataErrors);
                return resolution;
            }
            resolution.info = std::move(info);
            resolution.path = declared;
        } catch (const std::exception& e) {
            resolution.error =
                "the project's \"umbilicus\" field points at " +
                declared.string() + ", which could not be loaded: " + e.what();
        }
        return resolution;
    }

    const auto roots = umbilicusSearchRoots(pkg);

    // One scan, shared with the dependency listing: the priority walk, the
    // per-root stop and the canonical dedup live there and nowhere else.
    std::vector<fs::path> hits;
    for (auto& candidate : scanUmbilicusCandidates(pkg)) {
        if (candidate.decidesResolution) {
            hits.push_back(std::move(candidate.path));
        }
    }

    if (hits.empty()) {
        resolution.error =
            "no umbilicus.json or estimated_umbilicus.json found; searched: " +
            (roots.empty() ? std::string("(no candidate roots)")
                           : joinPaths(roots));
        return resolution;
    }

    if (hits.size() > 1) {
        resolution.ambiguous = hits;
        resolution.error =
            "found several umbilicus files (" + joinPaths(hits) +
            "); set the project's \"umbilicus\" field to pick one";
        return resolution;
    }

    try {
        auto info = Umbilicus::LoadFileInfo(hits.front());
        if (!info.metadataErrors.empty()) {
            resolution.error =
                describeMetadataErrors(hits.front(), info.metadataErrors);
            return resolution;
        }
        resolution.info = std::move(info);
        resolution.path = hits.front();
    } catch (const std::exception& e) {
        resolution.error = "failed to load " + hits.front().string() + ": " +
                           e.what();
    }
    return resolution;
}

} // namespace vc::core::util
