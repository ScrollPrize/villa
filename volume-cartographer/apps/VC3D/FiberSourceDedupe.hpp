#pragma once

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace vc3d {

// One loaded fiber file as seen by the cross-source dedupe.
struct FiberDedupeEntry {
    std::filesystem::path sourceRoot;
    std::string fileName;
    // Exact geometry fingerprint. Empty means "no content identity": the
    // entry is never merged with another by content.
    std::string contentKey;
    // fileName equals the canonical <username>_<startedAt>_<sequence>.json
    // derived from the fiber's own metadata (as opposed to e.g. a service
    // copy named after a runtime id).
    bool canonicalName = false;
};

struct FiberDedupeResult {
    // Indices of the surviving entries, in input order.
    std::vector<std::size_t> kept;
    // (dropped sourceRoot/fileName key) -> (surviving key), so branch links
    // written against a dropped copy still resolve to the survivor.
    std::unordered_map<std::string, std::string> linkAliases;
};

inline std::string fiberSourceFileKey(const std::filesystem::path& sourceRoot,
                                      const std::string& fileName)
{
    return (sourceRoot / fileName).lexically_normal().string();
}

// Collapse fibers that are the same fiber seen more than once: the same
// filename in several registered sources (the volpkg fibers directory plus a
// Spiral session's paths.fibers, where the service commits copies of uploaded
// fibers), or identical geometry under different names (copies committed
// under a stale numeric id). Groups are transitive. The survivor of a group
// is chosen by canonical filename first, then the earliest source in
// `sourcePreference` (sources not listed rank last), then input order.
inline FiberDedupeResult dedupeFiberSources(
    const std::vector<FiberDedupeEntry>& entries,
    const std::vector<std::filesystem::path>& sourcePreference)
{
    const std::size_t n = entries.size();
    std::vector<std::size_t> parent(n);
    std::iota(parent.begin(), parent.end(), std::size_t{0});
    auto find = [&](std::size_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    auto unite = [&](std::size_t a, std::size_t b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
    };

    std::unordered_map<std::string, std::size_t> firstByFileName;
    std::unordered_map<std::string, std::size_t> firstByContent;
    for (std::size_t i = 0; i < n; ++i) {
        if (!entries[i].fileName.empty()) {
            auto [it, inserted] = firstByFileName.emplace(entries[i].fileName, i);
            if (!inserted) unite(it->second, i);
        }
        if (!entries[i].contentKey.empty()) {
            auto [it, inserted] = firstByContent.emplace(entries[i].contentKey, i);
            if (!inserted) unite(it->second, i);
        }
    }

    auto sourceRank = [&](const std::filesystem::path& source) {
        for (std::size_t r = 0; r < sourcePreference.size(); ++r) {
            if (sourcePreference[r] == source) return r;
        }
        return sourcePreference.size();
    };
    auto rank = [&](std::size_t i) {
        return std::make_tuple(entries[i].canonicalName ? 0 : 1,
                               sourceRank(entries[i].sourceRoot), i);
    };

    std::unordered_map<std::size_t, std::size_t> survivorByGroup;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t group = find(i);
        auto [it, inserted] = survivorByGroup.emplace(group, i);
        if (!inserted && rank(i) < rank(it->second)) it->second = i;
    }

    FiberDedupeResult result;
    result.kept.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t survivor = survivorByGroup.at(find(i));
        if (survivor == i) {
            result.kept.push_back(i);
            continue;
        }
        result.linkAliases.emplace(
            fiberSourceFileKey(entries[i].sourceRoot, entries[i].fileName),
            fiberSourceFileKey(entries[survivor].sourceRoot,
                               entries[survivor].fileName));
    }
    return result;
}

}  // namespace vc3d
