#include "vc/fiber_graph/ScaffoldCache.hpp"

#include <vc_object_store/objects/StoreReader.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace vc::fiber_graph {
namespace {
// A readable `name=value;` statement. Numbers are written with every digit
// that distinguishes them, so two different parameters never read the same.
struct Statement {
    std::string text;
    void number(std::string_view name, double v)
    {
        char buffer[40];
        std::snprintf(buffer, sizeof buffer, "%.17g", v);
        text.append(name).append("=").append(buffer).append(";");
    }
    void integer(std::string_view name, std::int64_t v) { text.append(name).append("=").append(std::to_string(v)).append(";"); }
    void flag(std::string_view name, bool v) { integer(name, v ? 1 : 0); }
    void words(std::string_view name, std::string_view v) { text.append(name).append("=").append(v).append(";"); }
};
} // namespace

std::string scaffoldNormalInputs(const ScaffoldNormalOptions& options, const std::string& frame, bool fused,
                                 std::span<const std::string> anchors)
{
    Statement s;
    s.words("normals", "1");
    s.words("frame", frame);
    s.number("twist", options.strip.twistWeight);
    s.number("lateral", options.strip.lateralWeight);
    s.number("sigma", options.strip.smoothingSigma);
    s.number("spacing", options.strip.resampleSpacing);
    s.number("axis", options.strip.axisWeight);
    s.integer("iterations", options.strip.maxIterations);
    s.number("anchor", options.anchorWeight);
    if (fused) {
        s.words("prediction", options.predictionIdentity);
        s.number("presence_floor", options.presenceFloor);
        s.number("presence_full", options.presenceFull);
    }
    // Which records may anchor is decided by these; which did is `anchors`.
    for (const auto& anchor : anchors) s.words("by", anchor);
    return s.text;
}

std::string scaffoldCrossingInputs(const ScaffoldEventOptions& options)
{
    Statement s;
    s.words("crossings", "1");
    s.number("spacing", options.resampleSpacing);
    s.number("distance", options.maximumDistance);
    s.number("tangent", options.tangentWindow);
    s.flag("winding", options.measureWindingDistances);
    s.number("step", options.windingIntegrationStep);
    s.number("max_winding", options.maximumWindingDistance);
    s.flag("enforce_max", options.enforceMaximumWindingDistance);
    s.flag("relax_ends", options.relaxEndpointCertification);
    return s.text;
}

ScaffoldCache::ScaffoldCache(std::span<const objects::FiberScaffoldNormalsRow> normals,
                             std::span<const objects::FiberScaffoldCrossingRow> crossings)
{
    for (const auto& row : normals) _normals.insert_or_assign(NormalKey{row.fiberUuid, row.fiberVersion, row.inputs}, row);
    for (const auto& row : crossings)
        _crossings[PairKey{row.fiberA, row.fiberB, row.versionA, row.versionB, row.normalsA, row.normalsB, row.inputs}].push_back(row);
    sortPairs();
}

ScaffoldCache::ScaffoldCache(const objects::StoreReader& reader)
{
    for (auto& row : objects::readFiberScaffoldNormals(reader)) {
        NormalKey key{row.fiberUuid, row.fiberVersion, row.inputs};
        _normals.insert_or_assign(std::move(key), std::move(row));
    }
    for (auto& row : objects::readFiberScaffoldCrossings(reader)) {
        PairKey key{row.fiberA, row.fiberB, row.versionA, row.versionB, row.normalsA, row.normalsB, row.inputs};
        _crossings[key].push_back(std::move(row));
    }
    sortPairs();
}

void ScaffoldCache::sortPairs()
{
    for (auto& [key, rows] : _crossings) {
        std::stable_sort(rows.begin(), rows.end(),
                         [](const auto& x, const auto& y) { return x.eventIndex < y.eventIndex; });
        // The "no crossings" marker stands alone; a pair that also carries
        // events is a contradiction, and the events are what was measured.
        if (rows.size() > 1 && rows.front().eventIndex == objects::kFiberScaffoldNoCrossings)
            rows.erase(rows.begin());
    }
}

const objects::FiberScaffoldNormalsRow* ScaffoldCache::normals(const std::string& fiberUuid,
                                                               std::int64_t fiberVersion,
                                                               const std::string& inputs) const
{
    const auto found = _normals.find(NormalKey{fiberUuid, fiberVersion, inputs});
    return found == _normals.end() ? nullptr : &found->second;
}

std::optional<std::span<const objects::FiberScaffoldCrossingRow>> ScaffoldCache::crossings(
    const std::string& fiberA, std::int64_t versionA, const std::string& normalsA, const std::string& fiberB,
    std::int64_t versionB, const std::string& normalsB, const std::string& inputs) const
{
    const bool ordered = fiberA < fiberB;
    const PairKey key{ordered ? fiberA : fiberB,     ordered ? fiberB : fiberA,     ordered ? versionA : versionB,
                      ordered ? versionB : versionA, ordered ? normalsA : normalsB, ordered ? normalsB : normalsA,
                      inputs};
    const auto found = _crossings.find(key);
    if (found == _crossings.end()) return std::nullopt;
    if (found->second.size() == 1 && found->second.front().eventIndex == objects::kFiberScaffoldNoCrossings)
        return std::span<const objects::FiberScaffoldCrossingRow>{};
    return std::span<const objects::FiberScaffoldCrossingRow>(found->second);
}

} // namespace vc::fiber_graph
