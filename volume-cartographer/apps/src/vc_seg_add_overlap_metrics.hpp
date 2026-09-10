#pragma once

#include "utils/Json.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace vc_seg_overlap {

struct SurfaceInfo {
    std::string key;
    std::string id;
    std::string relativePath;
};

struct SourceMatches {
    SurfaceInfo source;
    std::size_t validPoints = 0;
    std::size_t sampledPoints = 0;
    std::map<std::string, std::size_t> matchedPointsByTarget;
};

inline utils::Json buildReport(std::vector<SourceMatches> sources,
                               std::vector<SurfaceInfo> targets,
                               std::size_t pointStride,
                               float tolerance)
{
    auto byPath = [](const auto& a, const auto& b) {
        return std::tie(a.relativePath, a.key) <
               std::tie(b.relativePath, b.key);
    };
    std::sort(targets.begin(), targets.end(), byPath);
    std::sort(sources.begin(), sources.end(), [](const SourceMatches& a,
                                                 const SourceMatches& b) {
        return std::tie(a.source.relativePath, a.source.key) <
               std::tie(b.source.relativePath, b.source.key);
    });

    utils::Json report = utils::Json::object();
    report["tool"] = "vc_seg_add_overlap";
    report["schema_version"] = 1;
    utils::Json metric = utils::Json::object();
    metric["name"] = "directed_source_point_surface_coverage";
    metric["definition"] = "matched_source_points / queried_source_points";
    metric["distance"] = "euclidean point-to-target-triangle";
    metric["comparison"] = "<=";
    metric["coordinate_units"] = "tifxyz coordinate units; usually voxels";
    report["metric"] = std::move(metric);
    utils::Json parameters = utils::Json::object();
    parameters["tolerance"] = tolerance;
    parameters["point_stride"] = static_cast<std::uint64_t>(pointStride);
    parameters["target_index_sampling_stride"] = 1;
    report["parameters"] = std::move(parameters);
    report["source_count"] = static_cast<std::uint64_t>(sources.size());
    report["target_count"] = static_cast<std::uint64_t>(targets.size());
    report["self_pairs_excluded"] = true;
    report["zero_match_pairs_omitted"] = true;

    utils::Json targetRows = utils::Json::array();
    for (const SurfaceInfo& target : targets) {
        utils::Json row = utils::Json::object();
        row["target_id"] = target.id;
        row["target_path"] = target.relativePath;
        targetRows.push_back(std::move(row));
    }
    report["targets"] = std::move(targetRows);

    utils::Json sourceRows = utils::Json::array();
    std::uint64_t overlapPairs = 0;
    for (const SourceMatches& source : sources) {
        if (source.sampledPoints > source.validPoints) {
            throw std::invalid_argument(
                "queried source points cannot exceed valid source points");
        }
        for (const auto& [targetKey, matched] :
             source.matchedPointsByTarget) {
            (void)targetKey;
            if (matched > source.sampledPoints) {
                throw std::invalid_argument(
                    "matched source points cannot exceed queried source points");
            }
        }

        utils::Json sourceRow = utils::Json::object();
        sourceRow["source_id"] = source.source.id;
        sourceRow["source_path"] = source.source.relativePath;
        sourceRow["valid_source_points"] =
            static_cast<std::uint64_t>(source.validPoints);
        sourceRow["queried_source_points"] =
            static_cast<std::uint64_t>(source.sampledPoints);
        utils::Json hits = utils::Json::array();
        for (const SurfaceInfo& target : targets) {
            if (source.source.key == target.key) {
                continue;
            }

            const auto hitIt = source.matchedPointsByTarget.find(target.key);
            if (hitIt == source.matchedPointsByTarget.end() || hitIt->second == 0) {
                continue;
            }
            const std::size_t matched = hitIt->second;
            const double fraction = source.sampledPoints == 0
                ? 0.0
                : static_cast<double>(matched) /
                      static_cast<double>(source.sampledPoints);

            utils::Json pair = utils::Json::object();
            pair["target_id"] = target.id;
            pair["target_path"] = target.relativePath;
            pair["matched_source_points"] = static_cast<std::uint64_t>(matched);
            pair["source_coverage_fraction"] = fraction;
            hits.push_back(std::move(pair));
            ++overlapPairs;
        }
        sourceRow["hits"] = std::move(hits);
        sourceRows.push_back(std::move(sourceRow));
    }
    report["directed_overlap_pair_count"] = overlapPairs;
    report["sources"] = std::move(sourceRows);
    return report;
}

} // namespace vc_seg_overlap
