#include "vc/core/util/RemoteFileCache.hpp"
#include "vc_reader/native/http.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_reader/sftp.hpp"


#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

namespace vc::core::util
{
namespace
{

constexpr int kSidecarVersion = 2;
constexpr std::string_view kSidecarSuffix = ".vc-remote-file.json";

struct InFlight {
    std::condition_variable finished;
    bool done = false;
    std::exception_ptr error;
};

std::mutex gInFlightMutex;
std::unordered_map<std::string, std::shared_ptr<InFlight>> gInFlight;
std::atomic<std::uint64_t> gTemporarySerial{0};

std::filesystem::path sidecarPath(const std::filesystem::path& payload)
{
    return std::filesystem::path(payload.string() + std::string(kSidecarSuffix));
}

std::filesystem::path checkedDestination(const RemoteFileCacheOptions& options)
{
    if (options.cacheRoot.empty())
        throw std::invalid_argument("remote file cache root is required");
    if (options.destination.empty() || options.destination.is_absolute()) {
        throw std::invalid_argument("remote file cache destination must be a non-empty relative path");
    }
    const auto relative = options.destination.lexically_normal();
    if (relative.empty() || *relative.begin() == "..")
        throw std::invalid_argument("remote file cache destination escapes cache root");

    auto root = std::filesystem::absolute(options.cacheRoot).lexically_normal();
    // A root written with a trailing separator ("/cache/") ends in an empty
    // element that no payload path contains; compare against the directory.
    if (!root.has_filename() && root.has_relative_path())
        root = root.parent_path();
    const auto payload = (root / relative).lexically_normal();
    auto mismatch = std::mismatch(root.begin(), root.end(), payload.begin(), payload.end());
    if (mismatch.first != root.end())
        throw std::invalid_argument("remote file cache destination escapes cache root");
    return payload;
}

bool validCacheHit(const std::filesystem::path& payload, std::string_view source)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(payload, ec) || ec)
        return false;
    const auto sidecar = sidecarPath(payload);
    if (!std::filesystem::is_regular_file(sidecar, ec) || ec)
        return false;
    try {
        std::ifstream input(sidecar);
        const auto metadata = nlohmann::json::parse(input);
        if (metadata.value("version", 0) != kSidecarVersion || metadata.value("source", std::string{}) != source ||
            !metadata.contains("size") || !metadata.at("size").is_number_unsigned()) {
            return false;
        }
        return std::filesystem::file_size(payload) == metadata.at("size").get<std::uintmax_t>();
    } catch (...) {
        return false;
    }
}

void renameReplacing(const std::filesystem::path& source, const std::filesystem::path& destination)
{
    std::error_code ec;
    std::filesystem::rename(source, destination, ec);
    if (!ec)
        return;
    // std::filesystem::rename does not replace an existing file on Windows.
    std::filesystem::remove(destination, ec);
    ec.clear();
    std::filesystem::rename(source, destination, ec);
    if (ec)
        throw std::filesystem::filesystem_error("cannot publish remote cache file", source, destination, ec);
}

std::filesystem::path writeTemporarySidecar(
    const std::filesystem::path& payload, const std::filesystem::path& temporaryPayload, std::string_view source)
{
    const auto sidecar = sidecarPath(payload);
    const auto tmp = std::filesystem::path(sidecar.string() + ".tmp-" + std::to_string(gTemporarySerial.fetch_add(1)));
    const nlohmann::json metadata{
        {"version", kSidecarVersion},
        {"source", source},
        {"size", std::filesystem::file_size(temporaryPayload)},
    };
    {
        std::ofstream output(tmp, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("cannot create remote cache sidecar: " + tmp.string());
        output << metadata.dump(2);
        output.close();
        if (!output)
            throw std::runtime_error("cannot write remote cache sidecar: " + tmp.string());
    }
    return tmp;
}

RemoteFileCacheResult makeResult(const std::filesystem::path& payload, const std::string& normalized, bool cacheHit)
{
    return {payload, normalized, cacheHit};
}

void defaultFetch(const std::string& sourceLocation,
                  const std::filesystem::path& temporaryPath,
                  const vc::HttpAuth& explicitAuth,
                  bool discoverAwsCredentials)
{
    const auto endpoint = vc::resolveRemoteUrl(sourceLocation);
    std::vector<std::uint8_t> body;
    const auto fetch = [&](const vc::HttpAuth& auth) {
        vcr::HttpOptions options;
        options.auth = vc::readerAuth(auth);
        vcr::HttpClient client(options);
        return client.get(endpoint.httpsUrl, body);
    };
    const auto failure = [&](const std::string& reason) {
        return std::runtime_error(
            "failed to fetch remote file source=" + redactedRemoteLocation(sourceLocation) +
            " request_url=" + redactedRemoteLocation(endpoint.httpsUrl) + ": " + reason);
    };
    bool found = false;
    try {
        // Recognized S3 sources are tried anonymously first; credentials are
        // only consulted when that is denied.
        try {
            found = fetch(endpoint.useAwsSigv4 ? vc::HttpAuth{} : explicitAuth);
        } catch (const vcr::HttpDenied&) {
            if (!endpoint.useAwsSigv4)
                throw;
            auto auth = explicitAuth;
            if (auth.empty() && discoverAwsCredentials)
                auth = vc::loadAwsCredentials();
            if (auth.empty())
                throw;
            if (auth.region.empty())
                auth.region = endpoint.awsRegion;
            found = fetch(auth);
        }
    } catch (const std::exception& error) {
        throw failure(error.what());
    }
    if (!found)
        throw failure("HTTP 404");
    std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot create remote cache temporary file");
    if (!body.empty())
        output.write(reinterpret_cast<const char*>(body.data()), static_cast<std::streamsize>(body.size()));
    output.close();
    if (!output)
        throw std::runtime_error("cannot write remote cache temporary file");
}

}  // namespace

std::string normalizeRemoteFileLocation(const std::string& sourceLocation)
{
    auto endpoint = vc::resolveRemoteUrl(sourceLocation).httpsUrl;
    const auto query = endpoint.find('?');
    const auto pathEnd = query == std::string::npos ? endpoint.size() : query;
    auto trimmedEnd = pathEnd;
    while (trimmedEnd > 0 && endpoint[trimmedEnd - 1] == '/')
        --trimmedEnd;
    if (trimmedEnd != pathEnd)
        endpoint.erase(trimmedEnd, pathEnd - trimmedEnd);
    return endpoint;
}

std::string redactedRemoteLocation(std::string_view location)
{
    const auto query = location.find('?');
    return query == std::string_view::npos ? std::string(location) : std::string(location.substr(0, query)) + "?<redacted>";
}

std::string remoteFileCacheSource(std::string_view sourceLocation)
{
    const auto suffix = sourceLocation.find_first_of("?#");
    auto source = std::string(sourceLocation.substr(0, suffix));
    if (vcr::sftp::isSftpUrl(source)) source = vcr::sftp::parseUrl(source).str();
    while (!source.empty() && source.back() == '/')
        source.pop_back();
    if (source.empty())
        throw std::invalid_argument("remote file cache source is empty");
    return source;
}

std::filesystem::path remoteFileCachePath(std::string_view sourceLocation)
{
    const auto source = remoteFileCacheSource(sourceLocation);
    const auto schemeEnd = source.find("://");
    if (schemeEnd == std::string::npos || schemeEnd == 0)
        throw std::invalid_argument("remote file cache source requires a URL scheme");

    auto validComponent = [](std::string_view component) {
        if (component.empty() || component == "." || component == "..")
            return false;
        return std::none_of(component.begin(), component.end(), [](unsigned char ch) {
            return ch < 0x20 || ch == 0x7f || ch == '<' || ch == '>' ||
                   ch == ':' || ch == '"' || ch == '\\' || ch == '|' ||
                   ch == '?' || ch == '*';
        });
    };
    const auto scheme = std::string_view(source).substr(0, schemeEnd);
    if (!std::all_of(scheme.begin(), scheme.end(), [](unsigned char ch) {
            return std::isalnum(ch) || ch == '+' || ch == '-' || ch == '.';
        }) || !validComponent(scheme)) {
        throw std::invalid_argument("remote file cache source has an invalid scheme");
    }
    const auto authorityStart = schemeEnd + 3;
    const auto pathStart = source.find('/', authorityStart);
    const auto authority = std::string_view(source).substr(
        authorityStart,
        pathStart == std::string::npos ? std::string::npos : pathStart - authorityStart);
    const auto cacheAuthority = vcr::sftp::isSftpUrl(source)
        ? vcr::sftp::cacheAuthority(source) : std::string(authority);
    if (!validComponent(cacheAuthority))
        throw std::invalid_argument("remote file cache source has an invalid authority");

    std::filesystem::path result = "remote_sources";
    result /= scheme;
    result /= cacheAuthority;
    if (pathStart == std::string::npos || pathStart + 1 == source.size())
        throw std::invalid_argument("remote file cache source requires an object path");
    std::string_view remaining(source.data() + pathStart + 1,
                               source.size() - pathStart - 1);
    while (!remaining.empty()) {
        const auto slash = remaining.find('/');
        const auto component = remaining.substr(0, slash);
        if (!validComponent(component))
            throw std::invalid_argument("remote file cache source has an invalid path component");
        result /= component;
        if (slash == std::string_view::npos)
            break;
        remaining.remove_prefix(slash + 1);
    }
    return result;
}

RemoteFileCacheResult cacheRemoteFile(const std::string& sourceLocation, const RemoteFileCacheOptions& options)
{
    const auto payload = checkedDestination(options);
    const auto normalized = normalizeRemoteFileLocation(sourceLocation);
    const auto source = remoteFileCacheSource(sourceLocation);
    if (options.policy == RemoteFileCachePolicy::CacheFirst && validCacheHit(payload, source)) {
        return makeResult(payload, normalized, true);
    }

    const std::string flightKey = payload.string() + '\n' + source;
    std::shared_ptr<InFlight> flight;
    bool owner = false;
    {
        std::lock_guard lock(gInFlightMutex);
        if (options.policy == RemoteFileCachePolicy::CacheFirst && validCacheHit(payload, source)) {
            return makeResult(payload, normalized, true);
        }
        auto [it, inserted] = gInFlight.try_emplace(flightKey, std::make_shared<InFlight>());
        flight = it->second;
        owner = inserted;
    }
    if (!owner) {
        std::unique_lock lock(gInFlightMutex);
        flight->finished.wait(lock, [&] { return flight->done; });
        if (flight->error)
            std::rethrow_exception(flight->error);
        if (!validCacheHit(payload, source))
            throw std::runtime_error("remote file cache publication did not produce a valid entry");
        return makeResult(payload, normalized, true);
    }

    std::exception_ptr error;
    try {
        std::filesystem::create_directories(payload.parent_path());
        const auto tmp = std::filesystem::path(payload.string() + ".tmp-" + std::to_string(gTemporarySerial.fetch_add(1)));
        std::filesystem::path temporarySidecar;
        const auto sidecar = sidecarPath(payload);
        const auto backupSuffix = ".backup-" + std::to_string(gTemporarySerial.fetch_add(1));
        const auto payloadBackup = std::filesystem::path(payload.string() + backupSuffix);
        const auto sidecarBackup = std::filesystem::path(sidecar.string() + backupSuffix);
        bool payloadBackedUp = false;
        bool sidecarBackedUp = false;
        try {
            if (options.fetcher)
                options.fetcher(sourceLocation, tmp);
            else
                defaultFetch(sourceLocation, tmp, options.auth,
                             options.discoverAwsCredentials);
            if (!std::filesystem::is_regular_file(tmp))
                throw std::runtime_error("remote file fetcher did not create its temporary file");
            temporarySidecar = writeTemporarySidecar(payload, tmp, source);
            std::error_code ec;
            if (std::filesystem::is_regular_file(payload, ec) && !ec) {
                std::filesystem::rename(payload, payloadBackup);
                payloadBackedUp = true;
            }
            ec.clear();
            if (std::filesystem::is_regular_file(sidecar, ec) && !ec) {
                std::filesystem::rename(sidecar, sidecarBackup);
                sidecarBackedUp = true;
            }
            renameReplacing(tmp, payload);
            renameReplacing(temporarySidecar, sidecar);
            std::filesystem::remove(payloadBackup, ec);
            std::filesystem::remove(sidecarBackup, ec);
        } catch (...) {
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            if (!temporarySidecar.empty())
                std::filesystem::remove(temporarySidecar, ec);
            if (payloadBackedUp) {
                std::filesystem::remove(payload, ec);
                std::filesystem::rename(payloadBackup, payload, ec);
            }
            if (sidecarBackedUp) {
                std::filesystem::remove(sidecar, ec);
                std::filesystem::rename(sidecarBackup, sidecar, ec);
            }
            throw;
        }
    } catch (...) {
        error = std::current_exception();
    }

    {
        std::lock_guard lock(gInFlightMutex);
        flight->error = error;
        flight->done = true;
        gInFlight.erase(flightKey);
    }
    flight->finished.notify_all();
    if (error)
        std::rethrow_exception(error);
    return makeResult(payload, normalized, false);
}

void invalidateRemoteFileCacheEntry(const RemoteFileCacheOptions& options)
{
    const auto payload = checkedDestination(options);
    std::error_code ec;
    std::filesystem::remove(payload, ec);
    std::filesystem::remove(sidecarPath(payload), ec);
}

}  // namespace vc::core::util

namespace vc {
HttpAuth loadAwsCredentials()
{
    const auto keys = vcr::zarr::awsCredentials();
    return keys ? HttpAuth{keys->user, keys->secret, keys->token, keys->region} : HttpAuth{};
}
} // namespace vc
