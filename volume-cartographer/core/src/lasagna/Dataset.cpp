#include "vc/lasagna/Dataset.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "utils/http_fetch.hpp"
#include "utils/zarr.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <span>

#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace vc::lasagna {
namespace {

[[nodiscard]] bool startsWithNoCase(std::string_view value, std::string_view prefix)
{
    if (value.size() < prefix.size())
        return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(value[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool isRemoteLocation(std::string_view value)
{
    return startsWithNoCase(value, "http://") ||
           startsWithNoCase(value, "https://") ||
           startsWithNoCase(value, "s3://") ||
           (startsWithNoCase(value, "s3+") &&
            value.find("://") != std::string_view::npos);
}

[[nodiscard]] bool isAbsoluteLocalPathString(const std::string& value)
{
    return std::filesystem::path(value).is_absolute();
}

[[nodiscard]] std::string trimRemotePathTrailingSlashes(std::string url)
{
    const auto query = url.find('?');
    const auto pathEnd = query == std::string::npos ? url.size() : query;
    auto trimmedEnd = pathEnd;
    while (trimmedEnd > 0 && url[trimmedEnd - 1] == '/') {
        --trimmedEnd;
    }
    if (trimmedEnd != pathEnd)
        url.erase(trimmedEnd, pathEnd - trimmedEnd);
    return url;
}

[[nodiscard]] vc::ResolvedUrl resolveRemoteEndpoint(const std::string& location)
{
    auto resolved = vc::resolveRemoteUrl(location);
    resolved.httpsUrl = trimRemotePathTrailingSlashes(std::move(resolved.httpsUrl));
    return resolved;
}

struct RemoteClientDetails {
    utils::HttpClient::Config config;
    bool isS3 = false;
    bool awsCredentialsLoaded = false;
    std::string awsRegion;
};

[[nodiscard]] RemoteClientDetails makeRemoteClientDetails(const vc::ResolvedUrl& endpoint)
{
    RemoteClientDetails details;
    details.config.transfer_timeout = std::chrono::seconds{60};
    if (endpoint.useAwsSigv4) {
        details.isS3 = true;
        details.config.aws_auth = utils::AwsAuth::load();
        if (!endpoint.awsRegion.empty())
            details.config.aws_auth.region = endpoint.awsRegion;
        details.awsCredentialsLoaded = !details.config.aws_auth.empty();
        details.awsRegion = details.config.aws_auth.region;
    }
    return details;
}

[[nodiscard]] utils::HttpClient makeRemoteClient(const vc::ResolvedUrl& endpoint)
{
    auto details = makeRemoteClientDetails(endpoint);
    return utils::HttpClient(std::move(details.config));
}

[[nodiscard]] std::string redactedUrlForDiagnostics(const std::string& url)
{
    const auto query = url.find('?');
    if (query == std::string::npos)
        return url;
    return url.substr(0, query) + "?<redacted>";
}

[[nodiscard]] std::string escapedDiagnosticString(std::string_view value)
{
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        if (ch == '\\' || ch == '"')
            out.push_back('\\');
        out.push_back(ch);
    }
    return out;
}

[[nodiscard]] std::string compactBodyExcerpt(std::string_view body)
{
    constexpr size_t kMaxExcerptBytes = 1024;
    const size_t take = std::min(body.size(), kMaxExcerptBytes);
    std::string out;
    out.reserve(take);
    bool previousWhitespace = false;
    for (size_t index = 0; index < take; ++index) {
        const unsigned char ch = static_cast<unsigned char>(body[index]);
        if (std::isspace(ch)) {
            if (!previousWhitespace && !out.empty()) {
                out.push_back(' ');
                previousWhitespace = true;
            }
            continue;
        }
        previousWhitespace = false;
        out.push_back(std::isprint(ch) ? static_cast<char>(ch) : '.');
    }
    while (!out.empty() && out.back() == ' ')
        out.pop_back();
    if (body.size() > kMaxExcerptBytes)
        out += "...";
    return out;
}

[[nodiscard]] std::string xmlTagValue(std::string_view body, std::string_view tag)
{
    const std::string open = "<" + std::string(tag) + ">";
    const std::string close = "</" + std::string(tag) + ">";
    const auto start = body.find(open);
    if (start == std::string_view::npos)
        return {};
    const auto valueStart = start + open.size();
    const auto end = body.find(close, valueStart);
    if (end == std::string_view::npos)
        return {};
    return compactBodyExcerpt(body.substr(valueStart, end - valueStart));
}

[[nodiscard]] std::string remoteManifestFetchFailureMessage(
    const std::string& location,
    const vc::ResolvedUrl& endpoint,
    const RemoteClientDetails& clientDetails,
    const utils::HttpResponse& response)
{
    std::string message =
        "Failed to fetch remote Lasagna manifest: " + location +
        " request_url=" + redactedUrlForDiagnostics(endpoint.httpsUrl);
    if (response.status_code > 0) {
        message += " HTTP " + std::to_string(response.status_code);
    } else {
        message += " no_http_response";
    }
    if (!response.content_type.empty())
        message += " content_type=\"" + escapedDiagnosticString(response.content_type) + "\"";
    if (response.content_length > 0)
        message += " content_length=" + std::to_string(response.content_length);
    message += " received_bytes=" + std::to_string(response.body.size());
    if (clientDetails.isS3) {
        message += " s3_region=" +
            (clientDetails.awsRegion.empty() ? std::string{"<unset>"} : clientDetails.awsRegion);
        message += " aws_sigv4_credentials=" +
            std::string(clientDetails.awsCredentialsLoaded ? "loaded" : "missing");
    }

    const std::string body(response.body_string());
    if (!body.empty()) {
        const auto s3Code = xmlTagValue(body, "Code");
        const auto s3Message = xmlTagValue(body, "Message");
        const auto s3Region = xmlTagValue(body, "Region");
        if (!s3Code.empty())
            message += " s3_code=\"" + escapedDiagnosticString(s3Code) + "\"";
        if (!s3Message.empty())
            message += " s3_message=\"" + escapedDiagnosticString(s3Message) + "\"";
        if (!s3Region.empty())
            message += " s3_response_region=\"" + escapedDiagnosticString(s3Region) + "\"";
        message += " response_body=\"" +
            escapedDiagnosticString(compactBodyExcerpt(body)) + "\"";
    }
    if (clientDetails.isS3) {
        message +=
            " hint=\"For s3:// manifests, verify bucket/key, region "
            "(use s3+REGION:// when needed), AWS login, and s3:GetObject permissions.\"";
    }
    return message;
}

[[nodiscard]] std::string remoteParentUrl(const std::string& normalizedRemoteUrl)
{
    const auto query = normalizedRemoteUrl.find('?');
    const std::string path = query == std::string::npos
        ? normalizedRemoteUrl
        : normalizedRemoteUrl.substr(0, query);
    const std::string suffix = query == std::string::npos
        ? std::string{}
        : normalizedRemoteUrl.substr(query);

    const auto schemeEnd = path.find("://");
    const auto authorityStart = schemeEnd == std::string::npos
        ? std::string::npos
        : schemeEnd + 3;
    const auto slash = path.rfind('/');
    if (slash == std::string::npos ||
        (authorityStart != std::string::npos && slash < authorityStart)) {
        return path + suffix;
    }
    return path.substr(0, slash) + suffix;
}

[[nodiscard]] std::string normalizedRelativeRemoteKey(const std::string& rawPath)
{
    const std::filesystem::path path(rawPath);
    const auto normalized = path.lexically_normal();
    if (rawPath.empty() || path.is_absolute() || normalized.empty() ||
        *normalized.begin() == "..") {
        throw std::runtime_error(
            "Remote Lasagna group path must remain inside the artifact: " +
            rawPath);
    }
    return normalized.generic_string();
}

[[nodiscard]] std::string hexEncode(std::string_view value)
{
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() * 2);
    for (const unsigned char ch : value) {
        out.push_back(kHex[ch >> 4]);
        out.push_back(kHex[ch & 0x0f]);
    }
    return out;
}

[[nodiscard]] std::filesystem::path collisionFreeRemoteCachePath(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& normalizedRemoteLocation)
{
    auto path = remoteCacheRoot / "remote_lasagna" / "url_hex";
    const auto hex = hexEncode(normalizedRemoteLocation);
    constexpr size_t kSegmentChars = 96;
    for (size_t pos = 0; pos < hex.size(); pos += kSegmentChars) {
        path /= hex.substr(pos, std::min(kSegmentChars, hex.size() - pos));
    }
    return path;
}

[[nodiscard]] std::string fetchRemoteText(const std::string& location)
{
    const auto endpoint = resolveRemoteEndpoint(location);
    auto clientDetails = makeRemoteClientDetails(endpoint);
    auto client = utils::HttpClient(std::move(clientDetails.config));
    const auto response = client.get(endpoint.httpsUrl);
    if (!response.ok()) {
        throw std::runtime_error(remoteManifestFetchFailureMessage(
            location,
            endpoint,
            clientDetails,
            response));
    }
    if (response.body.empty()) {
        throw std::runtime_error(
            "Remote Lasagna manifest is empty: " + location);
    }
    return std::string(reinterpret_cast<const char*>(response.body.data()),
                       response.body.size());
}

void applyRemoteGroup(
    LasagnaChannelGroup& group,
    std::string remoteBaseUrl,
    std::string remoteKey,
    std::filesystem::path remoteCacheRoot)
{
    if (remoteCacheRoot.empty()) {
        throw std::runtime_error(
            "Remote Lasagna group requires a remote cache directory: " +
            group.relativeZarrKey);
    }
    const auto endpoint = resolveRemoteEndpoint(remoteBaseUrl);
    group.remoteZarrBaseUrl = endpoint.httpsUrl;
    group.remoteZarrKey = std::move(remoteKey);
    group.remoteCacheRoot = std::move(remoteCacheRoot);
}

void resolveGroupLocations(
    LasagnaDatasetManifest& manifest,
    const LasagnaDatasetOpenOptions& options)
{
    const std::filesystem::path optionRemoteCacheRoot =
        options.remoteCacheRoot.empty()
            ? std::filesystem::path{}
            : std::filesystem::absolute(options.remoteCacheRoot).lexically_normal();
    const std::filesystem::path defaultRemoteCacheRoot =
        manifest.remoteCacheRoot.empty()
            ? optionRemoteCacheRoot
            : manifest.remoteCacheRoot;

    for (auto& group : manifest.groups) {
        group.remoteZarrBaseUrl.clear();
        group.remoteZarrKey.clear();
        group.remoteCacheRoot.clear();

        if (isRemoteLocation(group.relativeZarrKey)) {
            const auto endpoint = resolveRemoteEndpoint(group.relativeZarrKey);
            if (defaultRemoteCacheRoot.empty()) {
                throw std::runtime_error(
                    "Remote Lasagna group requires a remote cache directory: " +
                    group.relativeZarrKey);
            }
            const auto cacheRoot =
                collisionFreeRemoteCachePath(defaultRemoteCacheRoot, endpoint.httpsUrl);
            applyRemoteGroup(group, endpoint.httpsUrl, {}, cacheRoot);
            continue;
        }

        if (isAbsoluteLocalPathString(group.relativeZarrKey)) {
            group.zarrPath = std::filesystem::absolute(
                std::filesystem::path(group.relativeZarrKey)).lexically_normal();
            continue;
        }

        if (!manifest.remoteBaseUrl.empty()) {
            const auto key = normalizedRelativeRemoteKey(group.relativeZarrKey);
            applyRemoteGroup(group, manifest.remoteBaseUrl, key,
                             manifest.remoteCacheRoot);
            continue;
        }

        group.zarrPath = std::filesystem::absolute(
            manifest.baseDirectory / group.relativeZarrKey).lexically_normal();
    }
}

// Lasagna values drive geometry and must never pass through the configurable
// lossy volume-cache encoder. This store is deliberately an object-for-object
// read-through cache: publish() persists the exact bytes returned by the Zarr
// origin, whose own compressor remains responsible for lossless decoding.
class PersistentHttpStore final : public utils::Store {
public:
    PersistentHttpStore(std::string baseUrl, std::filesystem::path cacheRoot)
        : cacheRoot_(std::move(cacheRoot)),
          budget_(vc::render::PersistentZarrCacheBudget::findForPath(cacheRoot_))
    {
        const auto endpoint = resolveRemoteEndpoint(baseUrl);
        baseUrl_ = endpoint.httpsUrl;
        client_ = std::make_unique<utils::HttpClient>(makeRemoteClient(endpoint));
    }

    bool exists(const std::string& key) const override
    {
        const auto relative = checkedRelativePath(key);
        const auto filename = relative.filename().string();
        if (filename == "zarr.json" &&
            metadataExists(relative.parent_path() / ".zarray")) {
            return false;
        }
        if (filename == ".zarray" &&
            metadataExists(relative.parent_path() / "zarr.json")) {
            return false;
        }
        const auto path = cachePath(relative);
        if (std::filesystem::is_regular_file(path))
            return true;
        if (isMetadataPath(relative) &&
            std::filesystem::is_regular_file(cacheRoot_ / relative)) {
            return true;
        }
        const auto response = client_->head(makeUrl(key));
        return response.ok();
    }

    std::vector<std::byte> get(const std::string& key) const override
    {
        auto bytes = get_if_exists(key);
        if (!bytes)
            throw std::runtime_error("Remote Lasagna Zarr key not found: " + key);
        return std::move(*bytes);
    }

    std::optional<std::vector<std::byte>> get_if_exists(const std::string& key) const override
    {
        const auto relative = checkedRelativePath(key);
        const auto path = cachePath(relative);
        if (auto bytes = readIfExists(path, !isMetadataPath(relative)))
            return bytes;
        if (isMetadataPath(relative)) {
            const auto legacyPath = cacheRoot_ / relative;
            if (auto bytes = readIfExists(legacyPath, false)) {
                publish(path, *bytes, false);
                return bytes;
            }
        }

        const std::string artifactKey =
            baseUrl_ + '\n' + cacheRoot_.lexically_normal().string();
        const std::string requestKey = artifactKey + '\n' + key;
        std::shared_ptr<InFlightRequest> request;
        bool ownsRequest = false;
        bool announceStreaming = false;
        {
            std::lock_guard<std::mutex> lock(inFlightMutex_);
            if (auto bytes = readIfExists(path, !isMetadataPath(relative)))
                return bytes;
            if (auto it = inFlight_.find(requestKey); it != inFlight_.end()) {
                request = it->second;
            } else {
                request = std::make_shared<InFlightRequest>();
                inFlight_.emplace(requestKey, request);
                ownsRequest = true;
                announceStreaming = announcedArtifacts_.insert(artifactKey).second;
            }
        }

        if (announceStreaming) {
            std::clog << "[lasagna] streaming uncached data into "
                      << cacheRoot_.string() << std::endl;
        }

        if (!ownsRequest) {
            std::unique_lock<std::mutex> lock(inFlightMutex_);
            request->finished.wait(lock, [&]() { return request->done; });
            const auto error = request->error;
            const bool found = request->found;
            auto sharedBytes = request->bytes;
            lock.unlock();
            if (error)
                std::rethrow_exception(error);
            return found ? std::move(sharedBytes) : std::nullopt;
        }

        std::optional<std::vector<std::byte>> bytes;
        std::exception_ptr error;
        try {
            const auto response = client_->get(makeUrl(key));
            if (response.ok()) {
                bytes = std::move(response.body);
            } else if (!response.not_found()) {
                throw std::runtime_error(
                    "Remote Lasagna Zarr fetch failed HTTP " +
                    std::to_string(response.status_code) + ": " + key);
            }
            if (bytes)
                // Preserve the source object byte-for-byte. Do not decode and
                // re-encode it through the remote volume cache.
                publish(path, *bytes, !isMetadataPath(relative));
        } catch (...) {
            error = std::current_exception();
        }

        size_t cachedCount = 0;
        {
            std::lock_guard<std::mutex> lock(inFlightMutex_);
            request->found = bytes.has_value();
            request->bytes = bytes;
            request->error = error;
            request->done = true;
            inFlight_.erase(requestKey);
            if (bytes)
                cachedCount = ++cachedObjectCounts_[artifactKey];
        }
        request->finished.notify_all();
        if (cachedCount != 0 && cachedCount % 64 == 0) {
            std::clog << "[lasagna] cached " << cachedCount
                      << " remote objects in " << cacheRoot_.string() << std::endl;
        }
        if (error)
            std::rethrow_exception(error);
        return bytes;
    }

    std::optional<std::vector<std::byte>> get_partial(
        const std::string& key, std::size_t offset, std::size_t length) const override
    {
        auto bytes = get_if_exists(key);
        if (!bytes || offset > bytes->size())
            return std::nullopt;
        const auto count = std::min(length, bytes->size() - offset);
        return std::vector<std::byte>(bytes->begin() + static_cast<std::ptrdiff_t>(offset),
                                      bytes->begin() + static_cast<std::ptrdiff_t>(offset + count));
    }

    void set(const std::string&, std::span<const std::byte>) override
    {
        throw std::runtime_error("Remote Lasagna cache store is read-only");
    }
    void erase(const std::string&) override
    {
        throw std::runtime_error("Remote Lasagna cache store is read-only");
    }

private:
    struct InFlightRequest {
        std::condition_variable finished;
        bool done = false;
        bool found = false;
        std::optional<std::vector<std::byte>> bytes;
        std::exception_ptr error;
    };

    [[nodiscard]] static bool isMetadataPath(const std::filesystem::path& relative)
    {
        const auto filename = relative.filename().string();
        return filename == ".zarray" || filename == ".zattrs" ||
               filename == ".zgroup" || filename == "zarr.json";
    }

    [[nodiscard]] std::filesystem::path checkedRelativePath(const std::string& key) const
    {
        const std::filesystem::path relative(key);
        if (relative.empty() || relative.is_absolute())
            throw std::runtime_error("Invalid remote Lasagna Zarr key: " + key);
        const auto normalized = relative.lexically_normal();
        if (normalized.empty() || *normalized.begin() == "..")
            throw std::runtime_error("Remote Lasagna Zarr key escapes cache root: " + key);
        return normalized;
    }

    [[nodiscard]] std::string makeUrl(const std::string& key) const
    {
        return vc::joinRemoteUrlPath(baseUrl_, key);
    }

    [[nodiscard]] std::filesystem::path metadataPath(
        const std::filesystem::path& relative) const
    {
        return cacheRoot_ / ".lasagna-zarr-metadata" / relative;
    }

    [[nodiscard]] std::filesystem::path cachePath(
        const std::filesystem::path& relative) const
    {
        return isMetadataPath(relative) ? metadataPath(relative)
                                        : cacheRoot_ / relative;
    }

    [[nodiscard]] bool metadataExists(const std::filesystem::path& relative) const
    {
        return std::filesystem::is_regular_file(metadataPath(relative)) ||
               std::filesystem::is_regular_file(cacheRoot_ / relative);
    }

    std::optional<std::vector<std::byte>> readIfExists(
        const std::filesystem::path& path, bool managed) const
    {
        auto pin = managed && budget_
            ? budget_->pinRead(path)
            : vc::render::PersistentZarrCacheBudget::ReadPin{};
        if (!std::filesystem::is_regular_file(path))
            return std::nullopt;
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("Failed to read cached Lasagna object: " + path.string());
        in.seekg(0, std::ios::end);
        const auto size = in.tellg();
        in.seekg(0);
        std::vector<std::byte> out(static_cast<std::size_t>(size));
        if (!out.empty())
            in.read(reinterpret_cast<char*>(out.data()),
                    static_cast<std::streamsize>(size));
        pin.complete(true);
        return out;
    }

    bool publish(const std::filesystem::path& path,
                 std::span<const std::byte> bytes,
                 bool managed) const
    {
        auto reservation = managed && budget_
            ? budget_->reserveWrite(path, bytes.size())
            : vc::render::PersistentZarrCacheBudget::WriteReservation{};
        if (managed && budget_ && !reservation)
            return false;
        static std::atomic<std::uint64_t> serial{0};
        std::filesystem::create_directories(path.parent_path());
        const auto tmp = std::filesystem::path(
            path.string() + ".tmp-" + std::to_string(serial.fetch_add(1)));
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out)
                throw std::runtime_error("Failed to create Lasagna cache file: " + tmp.string());
            if (!bytes.empty())
                out.write(reinterpret_cast<const char*>(bytes.data()),
                          static_cast<std::streamsize>(bytes.size()));
            out.close();
            if (!out)
                throw std::runtime_error("Failed to write Lasagna cache file: " + tmp.string());
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec && !std::filesystem::is_regular_file(path)) {
            std::filesystem::remove(tmp);
            throw std::runtime_error("Failed to publish Lasagna cache file: " + ec.message());
        }
        std::filesystem::remove(tmp, ec);
        reservation.commit();
        return true;
    }

    std::unique_ptr<utils::HttpClient> client_;
    std::string baseUrl_;
    std::filesystem::path cacheRoot_;
    std::shared_ptr<vc::render::PersistentZarrCacheBudget> budget_;
    static std::mutex inFlightMutex_;
    static std::unordered_map<std::string, std::shared_ptr<InFlightRequest>> inFlight_;
    static std::unordered_set<std::string> announcedArtifacts_;
    static std::unordered_map<std::string, size_t> cachedObjectCounts_;
};

std::mutex PersistentHttpStore::inFlightMutex_;
std::unordered_map<std::string, std::shared_ptr<PersistentHttpStore::InFlightRequest>>
    PersistentHttpStore::inFlight_;
std::unordered_set<std::string> PersistentHttpStore::announcedArtifacts_;
std::unordered_map<std::string, size_t> PersistentHttpStore::cachedObjectCounts_;

void loadRemoteMarker(LasagnaDatasetManifest& manifest)
{
    const auto markerPath = manifest.baseDirectory / kLasagnaRemoteMarker;
    if (!std::filesystem::is_regular_file(markerPath))
        return;
    const auto marker = nlohmann::json::parse(std::ifstream(markerPath));
    const auto url = marker.value("artifact_url", std::string{});
    if (url.empty())
        throw std::runtime_error("Lasagna remote marker has no artifact_url");
    const auto manifestFile = marker.value("manifest_file", std::string{});
    if (manifestFile.empty() ||
        std::filesystem::absolute(manifest.baseDirectory / manifestFile).lexically_normal() !=
            std::filesystem::absolute(manifest.manifestPath).lexically_normal()) {
        throw std::runtime_error(
            "Lasagna remote marker does not identify the opened manifest");
    }
    manifest.remoteBaseUrl = resolveRemoteEndpoint(url).httpsUrl;
    manifest.remoteCacheRoot = manifest.baseDirectory;
}

} // namespace

LasagnaDataset::LasagnaDataset(LasagnaDatasetManifest manifest)
    : manifest_(std::move(manifest))
{
}

bool isRemoteLasagnaLocation(std::string_view location)
{
    return isRemoteLocation(location);
}

namespace {

void validateOpenOptions(const LasagnaDatasetOpenOptions& options)
{
    if (!(options.workingToBaseScale > 0.0) ||
        !std::isfinite(options.workingToBaseScale)) {
        throw std::runtime_error("Lasagna working-to-base scale must be positive");
    }
}

} // namespace

LasagnaDataset LasagnaDataset::open(const std::filesystem::path& manifestPath,
                                    LasagnaDatasetOpenOptions options)
{
    validateOpenOptions(options);
    auto manifest = LasagnaDatasetManifest::parseFile(manifestPath);
    manifest.manifestLocation = manifest.manifestPath.string();
    manifest.manifestIsRemote = false;
    manifest.workingToBaseScale = options.workingToBaseScale;
    loadRemoteMarker(manifest);
    resolveGroupLocations(manifest, options);
    return LasagnaDataset(std::move(manifest));
}

LasagnaDataset LasagnaDataset::openLocation(
    const std::string& manifestLocation,
    LasagnaDatasetOpenOptions options)
{
    validateOpenOptions(options);
    if (!isRemoteLocation(manifestLocation)) {
        return open(std::filesystem::path(manifestLocation), std::move(options));
    }
    if (options.remoteCacheRoot.empty()) {
        throw std::runtime_error(
            "Remote Lasagna manifest requires --remote-cache-dir: " +
            manifestLocation);
    }

    const auto endpoint = resolveRemoteEndpoint(manifestLocation);
    auto manifest = LasagnaDatasetManifest::parseText(
        fetchRemoteText(manifestLocation));
    manifest.manifestLocation = endpoint.httpsUrl;
    manifest.manifestIsRemote = true;
    manifest.workingToBaseScale = options.workingToBaseScale;
    manifest.remoteBaseUrl = remoteParentUrl(endpoint.httpsUrl);
    const auto root = std::filesystem::absolute(
        options.remoteCacheRoot).lexically_normal();
    manifest.remoteCacheRoot =
        collisionFreeRemoteCachePath(root, endpoint.httpsUrl);
    resolveGroupLocations(manifest, options);
    return LasagnaDataset(std::move(manifest));
}

const LasagnaDatasetManifest& LasagnaDataset::manifest() const noexcept
{
    return manifest_;
}

bool LasagnaDataset::hasNormalSource() const noexcept
{
    return manifest_.hasNormalSource();
}

const std::filesystem::path& LasagnaDataset::normalSourcePath() const
{
    if (!manifest_.normalPath.has_value()) {
        throw std::runtime_error("Lasagna dataset manifest has no normal source path");
    }
    return *manifest_.normalPath;
}

utils::ZarrArray openLasagnaChannelArray(
    const LasagnaDatasetManifest& manifest,
    const LasagnaChannelGroup& group,
    std::size_t dtypeSize)
{
    (void)manifest;
    auto registry = vc::buildZarrCodecRegistry(dtypeSize);
    if (group.isRemote()) {
        auto store = std::make_shared<PersistentHttpStore>(
            group.remoteZarrBaseUrl, group.remoteCacheRoot);
        return utils::ZarrArray::open(
            std::move(store), group.remoteZarrKey, std::move(registry));
    }
    return utils::ZarrArray::open(group.zarrPath, std::move(registry));
}

} // namespace vc::lasagna
