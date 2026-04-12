#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"

#include <utils/zarr.hpp>

#include <cstdio>
#include <fstream>
#include <future>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include "utils/Json.hpp"

#include <utils/hash.hpp>
#include <utils/http_fetch.hpp>

namespace vc::cache {

// ---------------------------------------------------------------------------
// HttpClient factory — creates a configured client for S3 requests.
// Thread-safe: HttpClient uses an internal mutex.
// ---------------------------------------------------------------------------

static utils::HttpClient makeClient(const HttpAuth& auth, int timeoutSec = 30) {
    utils::HttpClient::Config cfg;
    cfg.aws_auth = auth;
    cfg.transfer_timeout = std::chrono::seconds{timeoutSec};
    cfg.connect_timeout = std::chrono::seconds{5};
    cfg.max_retries = 2;
    return utils::HttpClient(std::move(cfg));
}

// ---------------------------------------------------------------------------
// httpGetString — fetch URL body as string
// ---------------------------------------------------------------------------

std::string httpGetString(const std::string& url, const HttpAuth& auth)
{
    auto c = makeClient(auth, 30);
    auto resp = c.get(url);

    if (resp.ok())
        return std::string(resp.body_string());

    if (resp.status_code >= 400) {
        auto body = std::string(resp.body_string());

        bool isAuthErr = (resp.status_code == 401 || resp.status_code == 403);
        if (!isAuthErr && !body.empty()) {
            isAuthErr = body.find("ExpiredToken") != std::string::npos ||
                        body.find("AccessDenied") != std::string::npos ||
                        body.find("InvalidAccessKeyId") != std::string::npos ||
                        body.find("SignatureDoesNotMatch") != std::string::npos ||
                        body.find("TokenRefreshRequired") != std::string::npos ||
                        body.find("InvalidToken") != std::string::npos;
        }

        if (isAuthErr) {
            std::string errMsg = "Access denied (HTTP " + std::to_string(resp.status_code) + ")";
            auto msgStart = body.find("<Message>");
            auto msgEnd = body.find("</Message>");
            if (msgStart != std::string::npos && msgEnd != std::string::npos) {
                msgStart += 9;
                errMsg = body.substr(msgStart, msgEnd - msgStart) +
                         " (HTTP " + std::to_string(resp.status_code) + ")";
            }
            throw std::runtime_error(errMsg + ". Check your AWS credentials.");
        }

        if (resp.status_code >= 500) {
            throw std::runtime_error(
                "HTTP server error " + std::to_string(resp.status_code) + " fetching " + url);
        }

        return {};
    }

    return {};
}

// ---------------------------------------------------------------------------
// s3ListObjects — S3 ListObjectsV2 with delimiter
// ---------------------------------------------------------------------------

static std::vector<std::string> extractXmlTags(const std::string& xml, const std::string& tag)
{
    std::vector<std::string> results;
    const std::string openTag = "<" + tag + ">";
    const std::string closeTag = "</" + tag + ">";
    size_t pos = 0;
    while (true) {
        pos = xml.find(openTag, pos);
        if (pos == std::string::npos) break;
        pos += openTag.size();
        auto end = xml.find(closeTag, pos);
        if (end == std::string::npos) break;
        results.push_back(xml.substr(pos, end - pos));
        pos = end + closeTag.size();
    }
    return results;
}

static bool parseS3Url(const std::string& url, std::string& bucketHost, std::string& prefix)
{
    auto schemeEnd = url.find("://");
    if (schemeEnd == std::string::npos) return false;
    auto pathStart = url.find('/', schemeEnd + 3);
    if (pathStart == std::string::npos) {
        bucketHost = url;
        prefix = "";
    } else {
        bucketHost = url.substr(0, pathStart);
        prefix = url.substr(pathStart + 1);
    }
    return true;
}

S3ListResult s3ListObjects(const std::string& httpsBaseUrl, const HttpAuth& auth)
{
    S3ListResult result;

    std::string bucketHost, prefix;
    if (!parseS3Url(httpsBaseUrl, bucketHost, prefix))
        return result;

    // URL-encode the prefix manually (simple: just encode spaces and special chars)
    // For simplicity, build the URL directly — S3 prefixes are typically clean paths.
    std::string listUrl = bucketHost + "/?list-type=2&delimiter=/";
    if (!prefix.empty()) {
        listUrl += "&prefix=" + prefix;
    }

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[S3] ListObjects: %s\n", listUrl.c_str());

    auto client = makeClient(auth, 30);
    auto resp = client.get(listUrl);

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[S3] ListObjects HTTP %ld\n", resp.status_code);

    if (!resp.ok()) {
        auto body = std::string(resp.body_string());
        if (auto* log = cacheDebugLog()) {
            if (!body.empty())
                std::fprintf(log, "[S3] Response: %.500s\n", body.c_str());
        }

        if (resp.status_code == 400 || resp.status_code == 401 || resp.status_code == 403) {
            auto codes = extractXmlTags(body, "Code");
            for (const auto& code : codes) {
                if (code == "ExpiredToken" || code == "AccessDenied" ||
                    code == "InvalidAccessKeyId" || code == "SignatureDoesNotMatch" ||
                    code == "TokenRefreshRequired" || code == "InvalidToken") {
                    result.authError = true;
                    auto msgs = extractXmlTags(body, "Message");
                    if (!msgs.empty()) result.errorMessage = msgs.front();
                    break;
                }
            }
            if (!result.authError && (resp.status_code == 401 || resp.status_code == 403)) {
                result.authError = true;
                result.errorMessage = "HTTP " + std::to_string(resp.status_code);
            }
        }
        return result;
    }

    auto xml = std::string(resp.body_string());

    for (const auto& p : extractXmlTags(xml, "Prefix")) {
        if (prefix.empty() || p.rfind(prefix, 0) == 0) {
            std::string relative = p.substr(prefix.size());
            while (!relative.empty() && relative.back() == '/')
                relative.pop_back();
            if (!relative.empty())
                result.prefixes.push_back(relative);
        }
    }

    for (const auto& k : extractXmlTags(xml, "Key")) {
        if (prefix.empty() || k.rfind(prefix, 0) == 0) {
            std::string relative = k.substr(prefix.size());
            if (!relative.empty())
                result.objects.push_back(relative);
        }
    }

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[S3] Found %zu prefixes, %zu objects\n",
                     result.prefixes.size(), result.objects.size());

    return result;
}

// ---------------------------------------------------------------------------
// httpDownloadFile — download URL to local file atomically
// ---------------------------------------------------------------------------

bool httpDownloadFile(const std::string& url, const std::filesystem::path& dest, const HttpAuth& auth)
{
    namespace fs = std::filesystem;

    auto client = makeClient(auth, 120);
    auto resp = client.get(url);

    if (!resp.ok())
        return false;

    auto tempPath = dest;
    tempPath += ".tmp";
    fs::create_directories(dest.parent_path());

    std::FILE* fp = std::fopen(tempPath.c_str(), "wb");
    if (!fp) return false;
    std::fwrite(resp.body.data(), 1, resp.body.size(), fp);
    std::fclose(fp);

    std::error_code ec;
    fs::rename(tempPath, dest, ec);
    return !ec;
}

// ---------------------------------------------------------------------------
// zarr v3 helpers (unchanged)
// ---------------------------------------------------------------------------

static std::string synthesize_v2_metadata(const utils::Json& v3)
{
    utils::Json v2;
    v2["zarr_format"] = 2;
    v2["shape"] = v3["shape"];

    utils::Json chunkShape;
    if (v3.contains("chunk_grid") && v3["chunk_grid"].contains("configuration")) {
        auto gridCfg = v3["chunk_grid"]["configuration"];
        if (gridCfg.contains("chunk_shape"))
            chunkShape = gridCfg["chunk_shape"];
    }
    if (v3.contains("codecs") && v3["codecs"].is_array()) {
        for (auto& codec : v3["codecs"]) {
            if (codec.value("name", std::string("")) == "sharding_indexed") {
                if (codec.contains("configuration")) {
                    auto shardCfg = codec["configuration"];
                    if (shardCfg.contains("chunk_shape"))
                        chunkShape = shardCfg["chunk_shape"];
                }
                break;
            }
        }
    }
    v2["chunks"] = chunkShape;

    std::string dtype = v3["data_type"].get_string();
    if (dtype == "uint8")       v2["dtype"] = "|u1";
    else if (dtype == "uint16") v2["dtype"] = "<u2";
    else                        v2["dtype"] = dtype;

    v2["fill_value"] = v3.value("fill_value", 0);
    v2["order"] = "C";
    v2["dimension_separator"] = "/";

    v2["compressor"] = nullptr;
    if (v3.contains("codecs") && v3["codecs"].is_array()) {
        for (auto& codec : v3["codecs"]) {
            std::string name = codec.value("name", std::string(""));
            if (name == "blosc") {
                auto& cfg = codec["configuration"];
                utils::Json comp;
                comp["id"] = "blosc";
                comp["cname"] = cfg.value("cname", std::string("lz4"));
                comp["clevel"] = cfg.value("clevel", 5);
                comp["shuffle"] = cfg.value("shuffle", 1);
                if (cfg.contains("typesize"))
                    comp["typesize"] = cfg["typesize"];
                if (cfg.contains("blocksize"))
                    comp["blocksize"] = cfg["blocksize"];
                v2["compressor"] = comp;
                break;
            } else if (name == "zstd") {
                utils::Json comp;
                comp["id"] = "zstd";
                comp["level"] = codec["configuration"].value("level", 3);
                v2["compressor"] = comp;
                break;
            } else if (name == "gzip" || name == "zlib") {
                utils::Json comp;
                comp["id"] = "gzip";
                comp["level"] = codec["configuration"].value("level", 5);
                v2["compressor"] = comp;
                break;
            } else if (name == "lz4") {
                utils::Json comp;
                comp["id"] = "lz4";
                comp["acceleration"] = codec["configuration"].value("acceleration", 1);
                v2["compressor"] = comp;
                break;
            }
        }
    }

    v2["filters"] = nullptr;
    return v2.dump(2);
}

static ShardConfig parse_v3_shard_config(
    const utils::Json& v3,
    const std::array<int, 3>& chunkShape)
{
    ShardConfig config;

    auto tryParseShard = [&](const utils::Json& cfg) -> bool {
        if (cfg.contains("chunks_per_shard") && cfg["chunks_per_shard"].is_array() &&
            cfg["chunks_per_shard"].size() >= 3) {
            int cz = cfg["chunks_per_shard"][0].get_int();
            int cy = cfg["chunks_per_shard"][1].get_int();
            int cx = cfg["chunks_per_shard"][2].get_int();
            config.enabled = true;
            config.shardShape = {
                cz * chunkShape[0],
                cy * chunkShape[1],
                cx * chunkShape[2]
            };
            return true;
        }
        if (cfg.contains("chunk_shape") && cfg["chunk_shape"].is_array() &&
            cfg["chunk_shape"].size() >= 3) {
            if (v3.contains("chunk_grid") && v3["chunk_grid"].contains("configuration")) {
                auto gridCfg = v3["chunk_grid"]["configuration"];
                if (gridCfg.contains("chunk_shape") && gridCfg["chunk_shape"].is_array()) {
                    config.enabled = true;
                    config.shardShape = {
                        gridCfg["chunk_shape"][0].get_int(),
                        gridCfg["chunk_shape"][1].get_int(),
                        gridCfg["chunk_shape"][2].get_int()
                    };
                    return true;
                }
            }
        }
        return false;
    };

    if (v3.contains("storage_transformers") && v3["storage_transformers"].is_array()) {
        for (auto& t : v3["storage_transformers"]) {
            std::string name = t.value("name", std::string(""));
            if (name == "chunk_manifest_sharding" && t.contains("configuration")) {
                if (tryParseShard(t["configuration"])) return config;
            }
        }
    }

    if (v3.contains("codecs") && v3["codecs"].is_array()) {
        for (auto& codec : v3["codecs"]) {
            std::string name = codec.value("name", std::string(""));
            if (name == "sharding_indexed" && codec.contains("configuration")) {
                if (tryParseShard(codec["configuration"])) return config;
            }
        }
    }

    return config;
}

// ---------------------------------------------------------------------------
// metadata fetcher
// ---------------------------------------------------------------------------

static constexpr const char* kRemoteSourceFile = ".remote_source.json";

std::string normalizeRemoteUrl(const std::string& url)
{
    std::string u = url;
    while (!u.empty() && u.back() == '/') u.pop_back();
    return u;
}

static std::string deriveRemoteVolumeName(const std::string& url)
{
    std::string u = normalizeRemoteUrl(url);
    auto pos = u.rfind('/');
    if (pos != std::string::npos) return u.substr(pos + 1);
    return u;
}

std::string deriveRemoteVolumeId(const std::string& url)
{
    const auto normalized = normalizeRemoteUrl(url);
    const auto name = deriveRemoteVolumeName(normalized);
    const auto hash = utils::fnv1a(std::string_view(normalized));

    std::ostringstream oss;
    oss << name << "-" << std::hex << std::nouppercase << std::setw(16)
        << std::setfill('0') << hash;
    return oss.str();
}

static void writeFile(const std::filesystem::path& path, const std::string& content)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
}

static std::string readFile(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

static std::optional<std::string> readRemoteSourceMarker(
    const std::filesystem::path& stagingDir)
{
    auto markerJson = readFile(stagingDir / kRemoteSourceFile);
    if (markerJson.empty()) return std::nullopt;

    try {
        auto marker = utils::Json::parse(markerJson);
        if (!marker.contains("url") || !marker["url"].is_string()) return std::nullopt;
        return normalizeRemoteUrl(marker["url"].get_string());
    } catch (...) {
        return std::nullopt;
    }
}

static void writeRemoteSourceMarker(
    const std::filesystem::path& stagingDir,
    const std::string& baseUrl)
{
    utils::Json marker;
    marker["url"] = baseUrl;
    writeFile(stagingDir / kRemoteSourceFile, marker.dump(2));
}

static std::optional<RemoteZarrInfo> tryLoadCachedMetadata(
    const std::string& baseUrl,
    const std::filesystem::path& stagingDir)
{
    namespace fs = std::filesystem;

    if (auto markerUrl = readRemoteSourceMarker(stagingDir)) {
        if (*markerUrl != baseUrl) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log,
                             "[REMOTE] Rejecting cached metadata from %s: marker URL %s does not match %s\n",
                             stagingDir.c_str(), markerUrl->c_str(), baseUrl.c_str());
            return std::nullopt;
        }
    }

    auto metaPath = stagingDir / "meta.json";
    auto level0Zarray = stagingDir / "0" / ".zarray";
    if (!fs::exists(metaPath) || !fs::exists(level0Zarray))
        return std::nullopt;

    int numLevels = 0;
    for (int lvl = 0; lvl < 20; lvl++) {
        if (fs::exists(stagingDir / std::to_string(lvl) / ".zarray"))
            numLevels++;
        else
            break;
    }

    if (numLevels == 0) return std::nullopt;

    std::string delimiter = ".";
    auto zarray0 = readFile(level0Zarray);
    if (!zarray0.empty()) {
        try {
            auto j = utils::Json::parse(zarray0);
            if (j.contains("dimension_separator"))
                delimiter = j["dimension_separator"].get_string();
        } catch (...) {}
    }

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[REMOTE] Using cached metadata from %s (%d levels)\n",
                     stagingDir.c_str(), numLevels);

    return RemoteZarrInfo{
        .url = baseUrl,
        .stagingDir = stagingDir,
        .delimiter = delimiter,
        .numLevels = numLevels
    };
}

RemoteZarrInfo fetchRemoteZarrMetadata(
    const std::string& url,
    const std::filesystem::path& stagingRoot,
    const HttpAuth& auth)
{
    const std::string baseUrl = normalizeRemoteUrl(url);
    const std::string volumeName = deriveRemoteVolumeName(baseUrl);
    const std::string volumeId = deriveRemoteVolumeId(baseUrl);
    auto stagingDir = stagingRoot / volumeId;

    {
        auto staleChunks = stagingRoot / (volumeId + ".chunks");
        std::error_code ec;
        if (std::filesystem::exists(staleChunks, ec))
            std::filesystem::remove_all(staleChunks, ec);
    }

    if (auto cached = tryLoadCachedMetadata(baseUrl, stagingDir)) {
        writeRemoteSourceMarker(stagingDir, baseUrl);
        return *cached;
    }

    std::filesystem::create_directories(stagingDir);

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[REMOTE] Fetching metadata for %s -> %s\n",
                     baseUrl.c_str(), stagingDir.c_str());

    auto zgroup = httpGetString(baseUrl + "/.zgroup", auth);
    if (!zgroup.empty())
        writeFile(stagingDir / ".zgroup", zgroup);
    else
        writeFile(stagingDir / ".zgroup", R"({"zarr_format":2})");

    auto zattrs = httpGetString(baseUrl + "/.zattrs", auth);
    if (!zattrs.empty())
        writeFile(stagingDir / ".zattrs", zattrs);

    std::string delimiter = ".";
    int numLevels = 0;
    utils::Json level0Meta;
    ShardConfig shardConfig;
    bool isV3 = false;

    constexpr int kBatchSize = 8;
    constexpr int kMaxLevels = 20;

    auto probeLevels = [&](const char* urlSuffix, auto&& onLevel) -> int {
        int found = 0;
        for (int batchStart = 0; batchStart < kMaxLevels; batchStart += kBatchSize) {
            int batchEnd = std::min(batchStart + kBatchSize, kMaxLevels);

            std::vector<std::future<std::string>> futures;
            futures.reserve(batchEnd - batchStart);
            for (int lvl = batchStart; lvl < batchEnd; lvl++) {
                std::string lvlUrl = baseUrl + "/" + std::to_string(lvl) + "/" + urlSuffix;
                futures.push_back(std::async(std::launch::async,
                    [lvlUrl, &auth]() { return httpGetString(lvlUrl, auth); }));
            }

            bool batchHadMiss = false;
            for (int i = 0; i < static_cast<int>(futures.size()); i++) {
                int lvl = batchStart + i;
                auto response = futures[i].get();

                if (response.empty()) {
                    if (auto* log = cacheDebugLog())
                        std::fprintf(log, "[REMOTE] Level %d: no %s\n", lvl, urlSuffix);
                    batchHadMiss = true;
                    break;
                }

                if (!onLevel(lvl, std::move(response))) {
                    batchHadMiss = true;
                    break;
                }
                found++;
            }

            if (batchHadMiss) break;
        }
        return found;
    };

    numLevels = probeLevels(".zarray", [&](int lvl, std::string zarray) -> bool {
        auto levelDir = stagingDir / std::to_string(lvl);
        writeFile(levelDir / ".zarray", zarray);

        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[REMOTE] Level %d: fetched .zarray (%zu bytes)\n",
                         lvl, zarray.size());

        if (lvl == 0) {
            try {
                level0Meta = utils::Json::parse(zarray);
                if (level0Meta.contains("dimension_separator"))
                    delimiter = level0Meta["dimension_separator"].get_string();
            } catch (const std::exception& e) {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[REMOTE] Warning: failed to parse level 0 .zarray: %s\n", e.what());
            }
        }
        return true;
    });

    if (numLevels == 0) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[REMOTE] No zarr v2 levels found, trying zarr v3...\n");

        numLevels = probeLevels("zarr.json", [&](int lvl, std::string zarrJson) -> bool {
            utils::Json v3;
            try {
                v3 = utils::Json::parse(zarrJson);
            } catch (const std::exception& e) {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[REMOTE] Level %d: failed to parse zarr.json: %s\n", lvl, e.what());
                return false;
            }

            if (v3.value("zarr_format", 0) != 3 ||
                v3.value("node_type", std::string("")) != "array") {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[REMOTE] Level %d: zarr.json is not zarr v3 array\n", lvl);
                return false;
            }

            std::string synthesized = synthesize_v2_metadata(v3);
            auto levelDir = stagingDir / std::to_string(lvl);
            writeFile(levelDir / ".zarray", synthesized);
            isV3 = true;

            if (auto* log = cacheDebugLog())
                std::fprintf(log, "[REMOTE] Level %d: fetched zarr.json (v3), synthesized .zarray (%zu bytes)\n",
                             lvl, synthesized.size());

            if (lvl == 0) {
                try {
                    level0Meta = utils::Json::parse(synthesized);
                    delimiter = "/";
                    if (level0Meta.contains("dimension_separator"))
                        delimiter = level0Meta["dimension_separator"].get_string();

                    std::array<int, 3> cs = {128, 128, 128};
                    if (v3.contains("chunk_grid")) {
                        auto grid = v3["chunk_grid"];
                        if (grid.contains("configuration")) {
                            auto gridCfg = grid["configuration"];
                            if (gridCfg.contains("chunk_shape") && gridCfg["chunk_shape"].is_array()) {
                                auto s = gridCfg["chunk_shape"];
                                cs = {s[0].get_int(), s[1].get_int(), s[2].get_int()};
                            }
                        }
                    }
                    shardConfig = parse_v3_shard_config(v3, cs);

                    if (shardConfig.enabled) {
                        if (auto* log = cacheDebugLog())
                            std::fprintf(log, "[REMOTE] Shard config: shape=[%d, %d, %d]\n",
                                         shardConfig.shardShape[0],
                                         shardConfig.shardShape[1],
                                         shardConfig.shardShape[2]);
                    }
                } catch (const std::exception& e) {
                    if (auto* log = cacheDebugLog())
                        std::fprintf(log, "[REMOTE] Warning: failed to parse level 0 synthesized .zarray: %s\n", e.what());
                }
            }
            return true;
        });
    }

    if (numLevels == 0) {
        throw std::runtime_error("No zarr levels found at " + baseUrl +
                                 " (tried both v2 .zarray and v3 zarr.json)");
    }

    int width = 0, height = 0, slices = 0;
    if (level0Meta.contains("shape") && level0Meta["shape"].is_array() &&
        level0Meta["shape"].size() >= 3) {
        slices = level0Meta["shape"][0].get_int();
        height = level0Meta["shape"][1].get_int();
        width  = level0Meta["shape"][2].get_int();
    }

    utils::Json meta;
    meta["uuid"] = volumeId;
    meta["name"] = volumeName;
    meta["type"] = "vol";
    meta["width"] = width;
    meta["height"] = height;
    meta["slices"] = slices;
    meta["format"] = "zarr";
    meta["voxelsize"] = 0;
    meta["min"] = 0;
    meta["max"] = 255;

    writeFile(stagingDir / "meta.json", meta.dump(2));
    writeRemoteSourceMarker(stagingDir, baseUrl);

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[REMOTE] Metadata complete: %d levels, shape=[%d, %d, %d] delimiter='%s'%s\n",
                     numLevels, slices, height, width, delimiter.c_str(),
                     isV3 ? " (zarr v3)" : "");

    return RemoteZarrInfo{
        .url = baseUrl,
        .stagingDir = stagingDir,
        .delimiter = delimiter,
        .numLevels = numLevels,
        .shardConfig = shardConfig
    };
}

}  // namespace vc::cache
