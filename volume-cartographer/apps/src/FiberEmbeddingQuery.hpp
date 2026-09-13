#pragma once
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "utils/zarr.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <nlohmann/json.hpp>

// Read-only diagnostic: coordinates are base XYZ; absent interpolation chunks
// are reported, never downloaded or silently treated as zero embeddings.
inline int queryCachedFiberEmbeddings(int argc, char** argv)
{
    if (argc != 4) throw std::invalid_argument("sample-embeddings MANIFEST POINTS_JSON");
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
    const auto channels = vc::lasagna::bindLasagnaEmbeddingChannels(manifest, presence);
    nlohmann::json output = nlohmann::json::array();
    auto covered = [](const auto& binding, const cv::Vec3d& point) {
        auto request = vc::lasagna::prepareLasagnaCubeRequest(binding, point);
        if (!request.valid) return false;
        for (size_t k = 0; k < (request.singleChunk ? 1 : 8); ++k) {
            const auto& key = request.keys[k];
            const std::array<size_t,3> index{key.z, key.y, key.x};
            if (!std::filesystem::is_regular_file(binding.array->chunk_path(index))) return false;
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
            for (const auto& channel : channels) complete = complete && covered(channel, cv::Vec3d(point));
            output.push_back({{"status", complete ? "invalid_vector" : "missing_coverage"}});
            if (complete) { valid.push_back(point); indices.push_back(i); }
        }
        std::vector<std::vector<float>> embeddings;
        vc::lasagna::sampleLasagnaVectorBatch(channels, valid, embeddings);
        vc::lasagna::LasagnaLocalChunkResolver resolver(presence);
        for (size_t j = 0; j < valid.size(); ++j) {
            if (embeddings[j].empty()) continue;
            auto request = vc::lasagna::prepareLasagnaCubeRequest(presence, cv::Vec3d(valid[j]));
            resolver.resolve(request);
            const auto value = vc::lasagna::sampleLasagnaChannel(presence, request);
            if (!value || !std::isfinite(*value)) continue;
            output[indices[j]] = {{"status", "ok"}, {"embedding", embeddings[j]}, {"presence", *value/255.0}};
        }
    }
    std::cout << output.dump() << '\n';
    return 0;
}
