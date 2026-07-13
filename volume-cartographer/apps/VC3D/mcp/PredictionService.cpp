#include "PredictionService.hpp"

#include <fastmcpp/exceptions.hpp>

#include <blosc.h>
#include <curl/curl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace vc::mcp
{
namespace
{

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw fastmcpp::ValidationError(message);
}

bool allowedUri(const std::string& uri)
{
    constexpr std::string_view s3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/";
    constexpr std::string_view ash = "https://dl.ash2txt.org/other/dev/scrolls/5/volumes/";
    return (uri.starts_with(s3) || uri.starts_with(ash)) && uri.find("..") == std::string::npos && uri.find('?') == std::string::npos &&
           uri.find('#') == std::string::npos;
}

std::size_t writeBytes(char* data, std::size_t size, std::size_t count, void* user)
{
    constexpr std::size_t kMaximumResponseBytes = 64ULL * 1024 * 1024;
    auto& output = *static_cast<std::vector<std::uint8_t>*>(user);
    const auto bytes = size * count;
    if (bytes > kMaximumResponseBytes || output.size() > kMaximumResponseBytes - bytes)
        return 0;
    output.insert(output.end(), reinterpret_cast<std::uint8_t*>(data), reinterpret_cast<std::uint8_t*>(data) + bytes);
    return bytes;
}

std::vector<std::uint8_t> fetch(const std::string& url)
{
    static const int initialized = [] { return curl_global_init(CURL_GLOBAL_DEFAULT); }();
    if (initialized != CURLE_OK)
        throw std::runtime_error("failed to initialize libcurl");
    CURL* curl = curl_easy_init();
    if (!curl)
        throw std::runtime_error("failed to create HTTP client");
    std::vector<std::uint8_t> bytes;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 90L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "volume-cartographer-mcp/0.1");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeBytes);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &bytes);
    const auto result = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(curl);
    if (result != CURLE_OK || status < 200 || status >= 300)
        throw std::runtime_error("HTTP read failed for " + url + " (status " + std::to_string(status) + ")");
    return bytes;
}

struct Metadata {
    std::string uri;
    std::array<int, 3> shape{};
    std::array<int, 3> chunks{};
    std::string dtype;
    std::string compressor;
};

Metadata metadata(const std::string& rawUri)
{
    require(allowedUri(rawUri), "prediction URI is outside the allowlist");
    std::string uri = rawUri;
    if (!uri.ends_with('/'))
        uri.push_back('/');
    const auto bytes = fetch(uri + "0/.zarray");
    const Json zarray = Json::parse(bytes.begin(), bytes.end());
    require(zarray.value("zarr_format", 0) == 2, "only Zarr v2 is supported");
    require(zarray.at("shape").size() == 3 && zarray.at("chunks").size() == 3, "prediction must be a rank-3 Zarr array");
    Metadata result;
    result.uri = std::move(uri);
    result.dtype = zarray.at("dtype").get<std::string>();
    require(result.dtype == "|u1", "prediction dtype must be uint8 (|u1)");
    for (int axis = 0; axis < 3; ++axis) {
        result.shape[axis] = zarray.at("shape").at(axis).get<int>();
        result.chunks[axis] = zarray.at("chunks").at(axis).get<int>();
        require(
            result.shape[axis] > 0 && result.shape[axis] <= 1000000 && result.chunks[axis] > 0 && result.chunks[axis] <= 512,
            "invalid or excessive Zarr shape or chunks");
    }
    const std::size_t chunkBytes = static_cast<std::size_t>(result.chunks[0]) * result.chunks[1] * result.chunks[2];
    require(chunkBytes <= 64ULL * 1024 * 1024, "uncompressed Zarr chunk exceeds 64 MiB");
    if (zarray.contains("compressor") && !zarray.at("compressor").is_null())
        result.compressor = zarray.at("compressor").value("id", "");
    require(result.compressor == "blosc", "only Blosc-compressed predictions are supported");
    return result;
}

std::vector<std::uint8_t> chunk(const Metadata& meta, int cz, int cy, int cx)
{
    const auto compressed = fetch(meta.uri + "0/" + std::to_string(cz) + "/" + std::to_string(cy) + "/" + std::to_string(cx));
    const std::size_t count = static_cast<std::size_t>(meta.chunks[0]) * meta.chunks[1] * meta.chunks[2];
    std::vector<std::uint8_t> output(count);
    const int decoded = blosc_decompress(compressed.data(), output.data(), output.size());
    if (decoded <= 0 || static_cast<std::size_t>(decoded) != output.size())
        throw std::runtime_error("failed to decompress prediction chunk");
    return output;
}

Json metadataJson(const Metadata& meta, const std::string& space)
{
    return {
        {"uri", meta.uri},
        {"space", space},
        {"shape_zyx", meta.shape},
        {"chunks_zyx", meta.chunks},
        {"dtype", meta.dtype},
        {"compressor", meta.compressor},
        {"axes", Json::array({"z", "y", "x"})}};
}

struct Candidate {
    int x{};
    int y{};
    int z{};
    int surface{};
    int ink{};
    double score{};
};

}  // namespace

Json inspectPrediction(const Json& request)
{
    require(request.contains("prediction_uri") && request.at("prediction_uri").is_string(), "prediction_uri is required");
    const std::string space = request.value("prediction_space", "ct_l2_xyz");
    require(space == "ct_l0_xyz" || space == "ct_l2_xyz", "unsupported prediction_space");
    return metadataJson(metadata(request.at("prediction_uri").get<std::string>()), space);
}

Json findSeedCandidates(const Json& request)
{
    const auto surface = metadata(request.at("prediction_uri").get<std::string>());
    const bool hasInk = request.contains("ink_prediction_uri");
    Metadata ink;
    if (hasInk) {
        ink = metadata(request.at("ink_prediction_uri").get<std::string>());
        require(
            ink.shape == surface.shape && ink.chunks == surface.chunks, "ink and surface predictions must have matching shape and chunks");
    }
    const std::string space = request.value("prediction_space", "ct_l2_xyz");
    require(space == "ct_l0_xyz" || space == "ct_l2_xyz", "unsupported prediction_space");
    const auto& region = request.at("region");
    const auto& center = region.at("center");
    const auto& radius = region.at("radius");
    const int centerX = center.at("x").get<int>();
    const int centerY = center.at("y").get<int>();
    const int centerZ = center.at("z").get<int>();
    const int radiusX = radius.at("x").get<int>();
    const int radiusY = radius.at("y").get<int>();
    const int radiusZ = radius.at("z").get<int>();
    require(
        radiusX >= 1 && radiusY >= 1 && radiusZ >= 1 && radiusX <= 192 && radiusY <= 192 && radiusZ <= 192,
        "candidate radius must be from 1 to 192 on each axis");
    const int threshold = request.value("surface_threshold", 128);
    const int inkThreshold = request.value("ink_threshold", 0);
    const double inkWeight = request.value("ink_weight", hasInk ? 0.5 : 0.0);
    const int maximum = request.value("max_candidates", 50);
    const int separation = request.value("minimum_separation_voxels", 24);
    require(threshold >= 0 && threshold <= 255 && inkThreshold >= 0 && inkThreshold <= 255, "thresholds must be from 0 to 255");
    require(std::isfinite(inkWeight) && inkWeight >= 0.0 && inkWeight <= 1.0, "ink_weight must be from 0 to 1");
    require(maximum >= 1 && maximum <= 100, "max_candidates must be from 1 to 100");
    require(separation >= 8 && separation <= 256, "minimum_separation_voxels must be from 8 to 256");

    const int minX = std::max(0, centerX - radiusX);
    const int minY = std::max(0, centerY - radiusY);
    const int minZ = std::max(0, centerZ - radiusZ);
    const int maxX = std::min(surface.shape[2] - 1, centerX + radiusX);
    const int maxY = std::min(surface.shape[1] - 1, centerY + radiusY);
    const int maxZ = std::min(surface.shape[0] - 1, centerZ + radiusZ);
    require(minX <= maxX && minY <= maxY && minZ <= maxZ, "candidate region is outside the prediction volume");
    const int minCx = minX / surface.chunks[2], maxCx = maxX / surface.chunks[2];
    const int minCy = minY / surface.chunks[1], maxCy = maxY / surface.chunks[1];
    const int minCz = minZ / surface.chunks[0], maxCz = maxZ / surface.chunks[0];
    const int chunkCount = (maxCx - minCx + 1) * (maxCy - minCy + 1) * (maxCz - minCz + 1);
    require(chunkCount <= 8, "candidate region touches more than 8 chunks");

    std::map<std::array<int, 3>, Candidate> buckets;
    std::size_t foreground = 0;
    for (int cz = minCz; cz <= maxCz; ++cz)
        for (int cy = minCy; cy <= maxCy; ++cy)
            for (int cx = minCx; cx <= maxCx; ++cx) {
                const auto surfaceData = chunk(surface, cz, cy, cx);
                const auto inkData = hasInk ? chunk(ink, cz, cy, cx) : std::vector<std::uint8_t>{};
                const int z0 = cz * surface.chunks[0];
                const int y0 = cy * surface.chunks[1];
                const int x0 = cx * surface.chunks[2];
                for (int z = std::max(minZ, z0); z <= std::min(maxZ, z0 + surface.chunks[0] - 1); ++z)
                    for (int y = std::max(minY, y0); y <= std::min(maxY, y0 + surface.chunks[1] - 1); ++y)
                        for (int x = std::max(minX, x0); x <= std::min(maxX, x0 + surface.chunks[2] - 1); ++x) {
                            const auto index = static_cast<std::size_t>(z - z0) * surface.chunks[1] * surface.chunks[2] +
                                               static_cast<std::size_t>(y - y0) * surface.chunks[2] + (x - x0);
                            const int surfaceValue = surfaceData[index];
                            const int inkValue = hasInk ? inkData[index] : 0;
                            if (surfaceValue < threshold || (hasInk && inkValue < inkThreshold))
                                continue;
                            ++foreground;
                            const double score = (1.0 - inkWeight) * surfaceValue / 255.0 + inkWeight * inkValue / 255.0;
                            const std::array<int, 3> key = {z / separation, y / separation, x / separation};
                            Candidate value{x, y, z, surfaceValue, inkValue, score};
                            auto found = buckets.find(key);
                            if (found == buckets.end() || score > found->second.score ||
                                (score == found->second.score &&
                                 std::tie(z, y, x) < std::tie(found->second.z, found->second.y, found->second.x)))
                                buckets[key] = value;
                        }
            }
    std::vector<Candidate> candidates;
    for (const auto& [key, value] : buckets)
        candidates.push_back(value);
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.score != b.score)
            return a.score > b.score;
        if (a.ink != b.ink)
            return a.ink > b.ink;
        if (a.surface != b.surface)
            return a.surface > b.surface;
        return std::tie(a.z, a.y, a.x) < std::tie(b.z, b.y, b.x);
    });
    if (candidates.size() > static_cast<std::size_t>(maximum))
        candidates.resize(maximum);
    const double l0Scale = space == "ct_l2_xyz" ? 4.0 : 1.0;
    Json output = Json::array();
    for (const auto& candidate : candidates)
        output.push_back(
            {{"coordinate", {{"x", candidate.x}, {"y", candidate.y}, {"z", candidate.z}, {"space", space}}},
             {"ct_l0_coordinate",
              {{"x", candidate.x * l0Scale}, {"y", candidate.y * l0Scale}, {"z", candidate.z * l0Scale}, {"space", "ct_l0_xyz"}}},
             {"surface_score", candidate.surface},
             {"ink_score", hasInk ? Json(candidate.ink) : Json(nullptr)},
             {"combined_score", candidate.score}});
    return {{"prediction", metadataJson(surface, space)}, {"ink_prediction", hasInk ? metadataJson(ink, space) : Json(nullptr)}, {"region", region}, {"foreground_voxels", foreground}, {"chunks_read", chunkCount * (hasInk ? 2 : 1)}, {"candidates", std::move(output)}};
}

}  // namespace vc::mcp
