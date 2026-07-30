#include "vc/core/util/RemoteFileCache.hpp"

#include "utils/http_fetch.hpp"

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

constexpr int kSidecarVersion = 1;
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

    const auto root = std::filesystem::absolute(options.cacheRoot).lexically_normal();
    const auto payload = (root / relative).lexically_normal();
    auto mismatch = std::mismatch(root.begin(), root.end(), payload.begin(), payload.end());
    if (mismatch.first != root.end())
        throw std::invalid_argument("remote file cache destination escapes cache root");
    return payload;
}

bool validCacheHit(const std::filesystem::path& payload, std::string_view identityHex, RemoteFileCacheAccounting accounting)
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
        if (metadata.value("version", 0) != kSidecarVersion || metadata.value("source_identity_hex", std::string{}) != identityHex ||
            metadata.value("accounting", std::string{}) != (accounting == RemoteFileCacheAccounting::Managed ? "managed" : "unmanaged") ||
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
    const std::filesystem::path& payload, const std::filesystem::path& temporaryPayload, std::string_view identityHex, RemoteFileCacheAccounting accounting)
{
    const auto sidecar = sidecarPath(payload);
    const auto tmp = std::filesystem::path(sidecar.string() + ".tmp-" + std::to_string(gTemporarySerial.fetch_add(1)));
    const nlohmann::json metadata{
        {"version", kSidecarVersion},
        {"source_identity_hex", identityHex},
        {"size", std::filesystem::file_size(temporaryPayload)},
        {"accounting", accounting == RemoteFileCacheAccounting::Managed ? "managed" : "unmanaged"},
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

std::optional<vc::render::PersistentZarrCacheBudget::ReadPin> readPinFor(const std::filesystem::path& payload, RemoteFileCacheAccounting accounting)
{
    if (accounting != RemoteFileCacheAccounting::Managed)
        return std::nullopt;
    const auto budget = vc::render::PersistentZarrCacheBudget::findForPath(payload);
    if (!budget)
        return std::nullopt;
    return budget->pinRead(payload);
}

RemoteFileCacheResult makeResult(const std::filesystem::path& payload, const std::string& normalized, bool cacheHit, RemoteFileCacheAccounting accounting)
{
    return {payload, normalized, cacheHit, readPinFor(payload, accounting)};
}

struct RemoteClientDetails {
    utils::HttpClient::Config config;
    bool isS3 = false;
    bool credentialsLoaded = false;
    std::string region;
};

RemoteClientDetails makeRemoteClientDetails(const vc::ResolvedUrl& endpoint, const vc::HttpAuth& explicitAuth)
{
    RemoteClientDetails details;
    details.config.transfer_timeout = std::chrono::seconds{60};
    details.config.connect_timeout = std::chrono::seconds{10};
    details.config.max_retries = 2;
    details.config.aws_auth = explicitAuth;
    details.isS3 = endpoint.useAwsSigv4;
    if (details.isS3 && details.config.aws_auth.empty())
        details.config.aws_auth = vc::loadAwsCredentials();
    if (details.isS3 && details.config.aws_auth.region.empty())
        details.config.aws_auth.region = endpoint.awsRegion;
    details.credentialsLoaded = !details.config.aws_auth.empty();
    details.region = details.config.aws_auth.region;
    return details;
}

std::string compactBodyExcerpt(std::string_view body)
{
    constexpr std::size_t kMaxExcerptBytes = 1024;
    const auto take = std::min(body.size(), kMaxExcerptBytes);
    std::string out;
    out.reserve(take);
    bool previousWhitespace = false;
    for (std::size_t index = 0; index < take; ++index) {
        const auto ch = static_cast<unsigned char>(body[index]);
        if (std::isspace(ch)) {
            if (!previousWhitespace && !out.empty())
                out.push_back(' ');
            previousWhitespace = true;
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

std::string escapedDiagnosticString(std::string_view value)
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

std::string xmlTagValue(std::string_view body, std::string_view tag)
{
    const std::string open = "<" + std::string(tag) + ">";
    const std::string close = "</" + std::string(tag) + ">";
    const auto start = body.find(open);
    if (start == std::string_view::npos)
        return {};
    const auto valueStart = start + open.size();
    const auto end = body.find(close, valueStart);
    return end == std::string_view::npos ? std::string{} : compactBodyExcerpt(body.substr(valueStart, end - valueStart));
}

std::string fetchFailureMessage(const std::string& sourceLocation, const vc::ResolvedUrl& endpoint, const RemoteClientDetails& details, const utils::HttpResponse& response)
{
    std::string message = "failed to fetch remote file source=" + redactedRemoteLocation(sourceLocation) +
                          " request_url=" + redactedRemoteLocation(endpoint.httpsUrl);
    message += response.status_code > 0 ? " HTTP " + std::to_string(response.status_code) : " no_http_response";
    if (!response.content_type.empty())
        message += " content_type=\"" + escapedDiagnosticString(response.content_type) + "\"";
    if (response.content_length > 0)
        message += " content_length=" + std::to_string(response.content_length);
    message += " received_bytes=" + std::to_string(response.body.size());
    if (details.isS3) {
        message += " s3_region=" + (details.region.empty() ? std::string{"<unset>"} : details.region);
        message += " aws_sigv4_credentials=" + std::string(details.credentialsLoaded ? "loaded" : "missing");
    }
    const auto body = std::string(response.body_string());
    if (!body.empty()) {
        for (const auto tag : {std::string_view{"Code"}, std::string_view{"Message"}, std::string_view{"Region"}}) {
            const auto value = xmlTagValue(body, tag);
            if (!value.empty())
                message += " s3_" + std::string(tag) + "=\"" + escapedDiagnosticString(value) + "\"";
        }
        message += " response_body=\"" + escapedDiagnosticString(compactBodyExcerpt(body)) + "\"";
    }
    if (details.isS3) {
        message += " hint=\"Verify bucket/key, region, AWS login, and s3:GetObject permissions.\"";
    }
    return message;
}

void defaultFetch(const std::string& sourceLocation, const std::filesystem::path& temporaryPath, const vc::HttpAuth& explicitAuth)
{
    const auto endpoint = vc::resolveRemoteUrl(sourceLocation);
    auto details = makeRemoteClientDetails(endpoint, explicitAuth);
    utils::HttpClient client(std::move(details.config));
    const auto response = client.get(endpoint.httpsUrl);
    if (!response.ok())
        throw std::runtime_error(fetchFailureMessage(sourceLocation, endpoint, details, response));
    std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot create remote cache temporary file");
    if (!response.body.empty()) {
        output.write(reinterpret_cast<const char*>(response.body.data()), static_cast<std::streamsize>(response.body.size()));
    }
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

std::string remoteFileIdentityHex(std::string_view normalizedLocation)
{
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(normalizedLocation.size() * 2);
    for (const unsigned char ch : normalizedLocation) {
        out.push_back(kHex[ch >> 4]);
        out.push_back(kHex[ch & 0x0f]);
    }
    return out;
}

std::filesystem::path remoteFileIdentityPath(const std::filesystem::path& base, std::string_view normalizedLocation)
{
    auto path = base;
    const auto hex = remoteFileIdentityHex(normalizedLocation);
    constexpr std::size_t kSegmentChars = 96;
    for (std::size_t pos = 0; pos < hex.size(); pos += kSegmentChars)
        path /= hex.substr(pos, std::min(kSegmentChars, hex.size() - pos));
    return path;
}

RemoteFileCacheResult cacheRemoteFile(const std::string& sourceLocation, const RemoteFileCacheOptions& options)
{
    const auto payload = checkedDestination(options);
    const auto normalized = normalizeRemoteFileLocation(sourceLocation);
    const auto identityHex = remoteFileIdentityHex(normalized);
    if (options.policy == RemoteFileCachePolicy::CacheFirst && validCacheHit(payload, identityHex, options.accounting)) {
        return makeResult(payload, normalized, true, options.accounting);
    }

    const std::string flightKey = payload.string() + '\n' + identityHex;
    std::shared_ptr<InFlight> flight;
    bool owner = false;
    {
        std::lock_guard lock(gInFlightMutex);
        if (options.policy == RemoteFileCachePolicy::CacheFirst && validCacheHit(payload, identityHex, options.accounting)) {
            return makeResult(payload, normalized, true, options.accounting);
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
        if (!validCacheHit(payload, identityHex, options.accounting))
            throw std::runtime_error("remote file cache publication did not produce a valid entry");
        return makeResult(payload, normalized, true, options.accounting);
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
        std::optional<vc::render::PersistentZarrCacheBudget::WriteReservation> reservation;
        try {
            if (options.fetcher)
                options.fetcher(sourceLocation, tmp);
            else
                defaultFetch(sourceLocation, tmp, options.auth);
            if (!std::filesystem::is_regular_file(tmp))
                throw std::runtime_error("remote file fetcher did not create its temporary file");
            temporarySidecar = writeTemporarySidecar(payload, tmp, identityHex, options.accounting);
            if (options.accounting == RemoteFileCacheAccounting::Managed) {
                if (const auto budget = vc::render::PersistentZarrCacheBudget::findForPath(payload)) {
                    auto reserved = budget->reserveWrite(payload, std::filesystem::file_size(tmp));
                    if (!reserved)
                        throw std::runtime_error("remote file cache budget rejected the managed payload");
                    reservation.emplace(std::move(reserved));
                }
            }

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
            if (reservation)
                reservation->commit();
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
            if (reservation)
                reservation->cancel();
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
    return makeResult(payload, normalized, false, options.accounting);
}

void invalidateRemoteFileCacheEntry(const RemoteFileCacheOptions& options)
{
    const auto payload = checkedDestination(options);
    std::optional<vc::render::PersistentZarrCacheBudget::WriteReservation> reservation;
    if (options.accounting == RemoteFileCacheAccounting::Managed) {
        if (const auto budget = vc::render::PersistentZarrCacheBudget::findForPath(payload)) {
            auto reserved = budget->reserveWrite(payload, 0);
            if (reserved)
                reservation.emplace(std::move(reserved));
        }
    }
    std::error_code ec;
    std::filesystem::remove(payload, ec);
    std::filesystem::remove(sidecarPath(payload), ec);
    if (reservation)
        reservation->commit();
}

}  // namespace vc::core::util
