#pragma once

#include "utils/http_fetch.hpp"
#include <utility>

namespace utils
{

bool is_sftp_url(std::string_view url) noexcept;
// Validates and canonicalizes ssh:// to sftp://. Passwords/queries are rejected.
std::string canonical_sftp_url(std::string_view url);
// Filesystem-safe source identity, including explicit user and port.
std::string sftp_cache_authority(std::string_view url);

struct SftpDirectoryEntry {
    std::string name;
    std::string url;
    bool directory = false;
};

std::vector<SftpDirectoryEntry> list_sftp_directory(std::string_view url);
HttpResponse fetch_sftp(
    const HttpClient::Config& config,
    std::string_view url,
    bool metadataOnly,
    std::optional<std::pair<std::size_t, std::size_t>> range,
    const HttpClient::DownloadObserver& observer);

}  // namespace utils
