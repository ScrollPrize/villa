#include "vc/core/util/ScrollUmbilicus.hpp"

#include <algorithm>
#include <exception>
#include <system_error>

#include "vc/core/types/VolumePkg.hpp"

namespace fs = std::filesystem;

namespace {

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

} // namespace

namespace vc::core::util {

ScrollUmbilicusResolution resolveScrollUmbilicus(const VolumePkg& pkg)
{
    ScrollUmbilicusResolution resolution;

    if (const auto declared = pkg.umbilicusPath(); !declared.empty()) {
        std::error_code ec;
        if (!fs::exists(declared, ec) || ec) {
            resolution.error =
                "the project's \"umbilicus\" field points at " +
                declared.string() + ", which does not exist";
            return resolution;
        }
        try {
            resolution.info = Umbilicus::LoadFileInfo(declared);
            resolution.path = declared;
        } catch (const std::exception& e) {
            resolution.error =
                "the project's \"umbilicus\" field points at " +
                declared.string() + ", which could not be loaded: " + e.what();
        }
        return resolution;
    }

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
        addRoot(segment.parent_path());
    }

    std::vector<fs::path> hits;
    std::vector<fs::path> canonicalHits;
    for (const auto& root : roots) {
        for (const char* name : {"umbilicus.json", "estimated_umbilicus.json"}) {
            const fs::path candidate = root / name;
            std::error_code ec;
            if (!fs::exists(candidate, ec) || ec) {
                continue;
            }
            const auto canonical = canonicalize(candidate);
            if (std::find(canonicalHits.begin(), canonicalHits.end(), canonical) ==
                canonicalHits.end()) {
                canonicalHits.push_back(canonical);
                hits.push_back(candidate);
            }
            break;
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
        resolution.info = Umbilicus::LoadFileInfo(hits.front());
        resolution.path = hits.front();
    } catch (const std::exception& e) {
        resolution.error = "failed to load " + hits.front().string() + ": " +
                           e.what();
    }
    return resolution;
}

} // namespace vc::core::util
