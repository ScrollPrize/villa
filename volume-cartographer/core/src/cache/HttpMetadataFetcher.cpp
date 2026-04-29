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

// Read any persisted source shard config from the marker. Only returns a
// config when this cache was created for a v3-sharded source — v2 sources
// write enabled=false (or omit the field) so we never mistake them.
static std::optional<ShardConfig> readRemoteShardConfig(
    const std::filesystem::path& stagingDir)
{
    auto markerJson = readFile(stagingDir / kRemoteSourceFile);
    if (markerJson.empty()) return std::nullopt;
    try {
        auto marker = utils::Json::parse(markerJson);
        auto* sc = json_find(marker, "shard_config");
        if (!sc || !sc->is_object()) return std::nullopt;
        ShardConfig cfg;
        auto* en = json_find(*sc, "enabled");
        if (en && en->is_boolean()) cfg.enabled = en->get_bool();
        if (!cfg.enabled) return cfg;
        auto* shape = json_find(*sc, "shape");
        if (!shape || !shape->is_array()) return std::nullopt;
        if (shape->size() != 3) return std::nullopt;
        cfg.shardShape = {
            int((*shape)[0].get_int()),
            int((*shape)[1].get_int()),
            int((*shape)[2].get_int()),
        };
        return cfg;
    } catch (...) {
        return std::nullopt;
    }
}

static void writeRemoteSourceMarker(
    const std::filesystem::path& stagingDir,
    const std::string& baseUrl,
    const ShardConfig* shardConfig = nullptr)
{
    utils::Json marker;
    marker["url"] = baseUrl;
    if (shardConfig) {
        utils::Json sc;
        sc["enabled"] = shardConfig->enabled;
        if (shardConfig->enabled) {
            utils::JsonArray shape;
            shape.push_back(utils::Json(shardConfig->shardShape[0]));
            shape.push_back(utils::Json(shardConfig->shardShape[1]));
            shape.push_back(utils::Json(shardConfig->shardShape[2]));
            sc["shape"] = utils::Json(std::move(shape));
        }
        marker["shard_config"] = std::move(sc);
    }
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

    // Recover the source shard config from the marker file we wrote during
    // the live metadata fetch. Reading the local `0/zarr.json` is WRONG
    // here — that's our own canonical output format, not the source's
    // metadata; doing so flipped v2-chunked sources into sharded mode and
    // made every subsequent chunk fetch 404 against the real S3 layout.
    //
    // If the marker is missing the shard_config field entirely (pre-fix
    // staging dir), we can't trust the cached metadata: a v3-sharded
    // source would silently come back up in non-sharded mode and every
    // chunk fetch would 404. Force a live re-fetch in that case.
    auto recovered = readRemoteShardConfig(stagingDir);
    if (!recovered) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log,
                "[REMOTE] Marker at %s lacks shard_config (pre-upgrade cache); re-fetching\n",
                stagingDir.c_str());
        return std::nullopt;
    }
    ShardConfig shardConfig = *recovered;
    if (auto* log = cacheDebugLog()) {
        if (shardConfig.enabled)
            std::fprintf(log,
                "[REMOTE] Recovered shard config from marker: shape=[%d, %d, %d]\n",
                shardConfig.shardShape[0],
                shardConfig.shardShape[1],
                shardConfig.shardShape[2]);
        else
            std::fprintf(log,
                "[REMOTE] Marker reports source is not sharded\n");
    }

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[REMOTE] Using cached metadata from %s (%d levels)\n",
                     stagingDir.c_str(), numLevels);

    std::vector<std::array<int, 3>> sourceChunkShapes;
    sourceChunkShapes.reserve(numLevels);
    for (int lvl = 0; lvl < numLevels; ++lvl) {
        auto body = readFile(stagingDir / std::to_string(lvl) / ".zarray");
        std::array<int, 3> cs{0, 0, 0};
        if (!body.empty()) {
            try {
                auto j = utils::Json::parse(body);
                if (j.contains("chunks") && j["chunks"].is_array() && j["chunks"].size() >= 3) {
                    cs = {
                        int(j["chunks"][0].get_int()),
                        int(j["chunks"][1].get_int()),
                        int(j["chunks"][2].get_int()),
                    };
                }
            } catch (...) {}
        }
        sourceChunkShapes.push_back(cs);
    }

    return RemoteZarrInfo{
        .url = baseUrl,
        .stagingDir = stagingDir,
        .delimiter = delimiter,
        .numLevels = numLevels,
        .shardConfig = shardConfig,
        .sourceChunkShapes = std::move(sourceChunkShapes),
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
        // Preserve the existing shard config entry so we don't downgrade
        // the marker back to "unknown" on the rewrite.
        writeRemoteSourceMarker(stagingDir, baseUrl, &cached->shardConfig);
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
    std::vector<std::array<int, 3>> sourceChunkShapes;

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

        try {
            auto meta = utils::Json::parse(zarray);
            if (lvl == 0) {
                level0Meta = meta;
                if (meta.contains("dimension_separator"))
                    delimiter = meta["dimension_separator"].get_string();
            }
            if (meta.contains("chunks") && meta["chunks"].is_array() && meta["chunks"].size() >= 3) {
                sourceChunkShapes.push_back({
                    int(meta["chunks"][0].get_int()),
                    int(meta["chunks"][1].get_int()),
                    int(meta["chunks"][2].get_int()),
                });
            } else {
                sourceChunkShapes.push_back({0, 0, 0});
            }
        } catch (const std::exception& e) {
            if (auto* log = cacheDebugLog())
                std::fprintf(log, "[REMOTE] Warning: failed to parse level %d .zarray: %s\n", lvl, e.what());
            sourceChunkShapes.push_back({0, 0, 0});
        }
        return true;
    });

    if (numLevels == 0) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[REMOTE] No zarr v2 levels found, trying zarr v3...\n");

        numLevels = probeLevels("zarr.json", [&](int lvl, std::string zarrJson) -> bool {
            utils::ZarrMetadata meta;
            try {
                meta = utils::detail::parse_zarr_json(zarrJson);
            } catch (const std::exception& e) {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[REMOTE] Level %d: parse_zarr_json failed: %s\n", lvl, e.what());
                return false;
            }

            if (meta.version != utils::ZarrVersion::v3 || meta.node_type != "array") {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[REMOTE] Level %d: zarr.json is not zarr v3 array\n", lvl);
                return false;
            }

            // v2 emission for the VcDataset reader. For sharded v3 the v2
            // "chunks" is the finest granularity (inner chunk shape).
            utils::ZarrMetadata v2meta = meta;
            v2meta.version = utils::ZarrVersion::v2;
            if (meta.shard_config && meta.shard_config->sub_chunks.size() >= 3) {
                v2meta.chunks = meta.shard_config->sub_chunks;
            }
            std::string synthesized = utils::detail::serialize_zarray(v2meta);
            auto levelDir = stagingDir / std::to_string(lvl);
            writeFile(levelDir / ".zarray", synthesized);
            isV3 = true;

            if (meta.shard_config && meta.shard_config->sub_chunks.size() >= 3) {
                sourceChunkShapes.push_back({
                    int(meta.shard_config->sub_chunks[0]),
                    int(meta.shard_config->sub_chunks[1]),
                    int(meta.shard_config->sub_chunks[2]),
                });
            } else if (meta.chunks.size() >= 3) {
                sourceChunkShapes.push_back({
                    int(meta.chunks[0]), int(meta.chunks[1]), int(meta.chunks[2]),
                });
            } else {
                sourceChunkShapes.push_back({0, 0, 0});
            }

            if (auto* log = cacheDebugLog())
                std::fprintf(log, "[REMOTE] Level %d: fetched zarr.json (v3), synthesized .zarray (%zu bytes)\n",
                             lvl, synthesized.size());

            if (lvl == 0) {
                level0Meta = utils::Json::parse(synthesized);
                delimiter = meta.dimension_separator.empty() ? "/" : meta.dimension_separator;
                if (meta.shard_config && meta.chunks.size() >= 3) {
                    shardConfig.enabled = true;
                    shardConfig.shardShape = {
                        int(meta.chunks[0]), int(meta.chunks[1]), int(meta.chunks[2])
                    };
                    if (auto* log = cacheDebugLog())
                        std::fprintf(log, "[REMOTE] Shard config: shape=[%d, %d, %d]\n",
                                     shardConfig.shardShape[0],
                                     shardConfig.shardShape[1],
                                     shardConfig.shardShape[2]);
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
    writeRemoteSourceMarker(stagingDir, baseUrl, &shardConfig);

    if (auto* log = cacheDebugLog())
        std::fprintf(log, "[REMOTE] Metadata complete: %d levels, shape=[%d, %d, %d] delimiter='%s'%s\n",
                     numLevels, slices, height, width, delimiter.c_str(),
                     isV3 ? " (zarr v3)" : "");

    return RemoteZarrInfo{
        .url = baseUrl,
        .stagingDir = stagingDir,
        .delimiter = delimiter,
        .numLevels = numLevels,
        .shardConfig = shardConfig,
        .sourceChunkShapes = std::move(sourceChunkShapes),
    };
}

}  // namespace vc::cache
