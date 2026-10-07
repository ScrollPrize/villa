#pragma once
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "utils/zarr.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <nlohmann/json.hpp>

// Read-only diagnostic: coordinates are base XYZ; absent interpolation chunks
// are reported, never downloaded or silently treated as zero embeddings.
inline int queryCachedFiberSamples(int argc, char** argv, bool width = false)
{
    if (argc != 4) throw std::invalid_argument("sample-embeddings/sample-width MANIFEST POINTS_JSON");
    if (!std::filesystem::is_regular_file(argv[2]))
        throw std::invalid_argument("sample-embeddings requires a local manifest");
    std::ifstream input(argv[3]);
    const auto points = nlohmann::json::parse(input);
    if (!points.is_array() || points.size() > 200000)
        throw std::invalid_argument("expected at most 200000 base XYZ points");
    vc::render::processChunkCacheService()->configureDecodedByteCapacity(2ULL << 30);
    auto opened = vc::lasagna::LasagnaDataset::openLocation(argv[2], {});
    auto manifest = opened.manifest();
    manifest.workingToBaseScale = 1.0;
    for (const auto& group : manifest.groups)
        if (!std::filesystem::is_regular_file(group.zarrPath / ".zarray"))
            throw std::invalid_argument("sample-embeddings requires existing local channel arrays");
    const auto presence = vc::lasagna::bindLasagnaChannel(manifest, "presence");
    const auto channels = width
        ? std::vector<vc::lasagna::LasagnaChannelBinding>{vc::lasagna::bindLasagnaChannel(manifest, "width", true)}
        : vc::lasagna::bindLasagnaEmbeddingChannels(manifest, presence);
    auto coverageChannels = channels;
    if (width) {
        coverageChannels.clear();
        for (const auto& group : manifest.groups) {
            if (group.channels.empty()) throw std::invalid_argument("empty cached channel group");
            auto binding = vc::lasagna::bindLasagnaChannel(manifest, group.channels.front(), true);
            if (binding.shapeZYX != presence.shapeZYX || binding.spacing != presence.spacing)
                throw std::invalid_argument("cached width products disagree on lattice");
            coverageChannels.push_back(std::move(binding));
        }
    }
    nlohmann::json output = nlohmann::json::array();
    std::unordered_map<std::string, bool> coverage;
    auto covered = [&](const auto& binding, const cv::Vec3d& point) {
        auto request = vc::lasagna::prepareLasagnaCubeRequest(binding, point);
        if (!request.valid) return false;
        for (size_t k = 0; k < (request.singleChunk ? 1 : 8); ++k) {
            const auto& key = request.keys[k];
            const std::array<size_t,3> index{key.z, key.y, key.x};
            const auto path = binding.array->chunk_path(index).string();
            auto found = coverage.find(path);
            if (found == coverage.end())
                found = coverage.emplace(path, std::filesystem::is_regular_file(path)).first;
            if (!found->second) return false;
        }
        return true;
    };
    for (size_t begin = 0; begin < points.size(); begin += 512) {
        std::vector<cv::Vec3f> valid;
        std::vector<size_t> indices;
        for (size_t i = begin; i < std::min(points.size(), begin+512); ++i) {
            if (!points[i].is_array() || points[i].size() != 3)
                throw std::invalid_argument("expected XYZ triples");
            cv::Vec3f point{points[i][0].get<float>(), points[i][1].get<float>(), points[i][2].get<float>()};
            bool complete = covered(presence, cv::Vec3d(point));
            for (const auto& channel : coverageChannels) complete = complete && covered(channel, cv::Vec3d(point));
            output.push_back({{"status", complete ? (width ? "invalid_width" : "invalid_vector") : "missing_coverage"}});
            if (complete) { valid.push_back(point); indices.push_back(i); }
        }
        std::vector<std::vector<float>> embeddings;
        vc::lasagna::sampleLasagnaVectorBatch(channels, valid, embeddings, !width);
        vc::lasagna::LasagnaLocalChunkResolver resolver(presence);
        for (size_t j = 0; j < valid.size(); ++j) {
            if (embeddings[j].empty()) continue;
            auto request = vc::lasagna::prepareLasagnaCubeRequest(presence, cv::Vec3d(valid[j]));
            resolver.resolve(request);
            const auto value = vc::lasagna::sampleLasagnaChannel(presence, request);
            if (!value || !std::isfinite(*value)) continue;
            if (width) {
                if (embeddings[j].size() != 1 || !std::isfinite(embeddings[j][0]) || embeddings[j][0] <= 0) continue;
                output[indices[j]] = {{"status", "ok"}, {"width", embeddings[j][0]}, {"presence", *value/255.0}};
            } else {
                output[indices[j]] = {{"status", "ok"}, {"embedding", embeddings[j]}, {"presence", *value/255.0}};
            }
        }
    }
    std::cout << output.dump() << '\n';
    return 0;
}
