#include "vc/core/util/RemoteCacheSettings.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace fs = std::filesystem;

namespace vc::settings {
namespace {

fs::path pathFromUtf8(std::string_view value)
{
    std::u8string utf8;
    utf8.reserve(value.size());
    for (const unsigned char byte : value)
        utf8.push_back(static_cast<char8_t>(byte));
    return fs::path(utf8);
}

std::string pathToUtf8(const fs::path& path)
{
    const auto utf8 = path.u8string();
    std::string result;
    result.reserve(utf8.size());
    for (const char8_t byte : utf8)
        result.push_back(static_cast<char>(byte));
    return result;
}

std::string trim(std::string value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string decodeIniValue(std::string value)
{
    value = trim(std::move(value));
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
        value = value.substr(1, value.size() - 2);

    std::string decoded;
    decoded.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '\\' || i + 1 == value.size()) {
            decoded.push_back(value[i]);
            continue;
        }
        const char escaped = value[++i];
        switch (escaped) {
        case '\\': decoded.push_back('\\'); break;
        case '"': decoded.push_back('"'); break;
        case 'n': decoded.push_back('\n'); break;
        case 'r': decoded.push_back('\r'); break;
        case 't': decoded.push_back('\t'); break;
        default:
            decoded.push_back('\\');
            decoded.push_back(escaped);
            break;
        }
    }
    return decoded;
}

fs::path homeDirectory()
{
#ifdef _WIN32
    if (const wchar_t* profile = _wgetenv(L"USERPROFILE"); profile && *profile)
        return profile;
    const wchar_t* drive = _wgetenv(L"HOMEDRIVE");
    const wchar_t* path = _wgetenv(L"HOMEPATH");
    if (drive && *drive && path && *path)
        return std::wstring(drive) + path;
#endif
    if (const char* home = std::getenv("HOME"); home && *home)
        return pathFromUtf8(home);
    throw std::runtime_error("Cannot determine the user home directory for VC3D settings");
}

fs::path configuredRemoteCachePath(const fs::path& settingsPath)
{
    std::ifstream input(settingsPath);
    if (!input)
        return {};

    bool viewerSection = false;
    std::string line;
    while (std::getline(input, line)) {
        const std::string stripped = trim(line);
        if (stripped.empty() || stripped.front() == ';' || stripped.front() == '#')
            continue;
        if (stripped.front() == '[' && stripped.back() == ']') {
            viewerSection = stripped == "[viewer]";
            continue;
        }
        if (!viewerSection)
            continue;
        const auto separator = stripped.find('=');
        if (separator == std::string::npos ||
            trim(stripped.substr(0, separator)) != "remote_cache_dir") {
            continue;
        }
        const std::string configured = decodeIniValue(stripped.substr(separator + 1));
        return configured.empty() ? fs::path{} : pathFromUtf8(configured);
    }
    return {};
}

fs::path ensureDirectory(fs::path path)
{
    if (path.is_relative())
        path = fs::absolute(path);
    path = path.lexically_normal();
    std::error_code ec;
    fs::create_directories(path, ec);
    if (ec || !fs::is_directory(path)) {
        throw std::runtime_error(
            "Cannot create remote cache directory '" + pathToUtf8(path) + "': " +
            (ec ? ec.message() : "path is not a directory"));
    }

    return path;
}

fs::path ensureWritableDirectory(fs::path path)
{
    path = ensureDirectory(std::move(path));

    static std::atomic<std::uint64_t> probeSequence{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path probe = path / (
        ".vc3d-write-probe-" + std::to_string(stamp) + "-" +
        std::to_string(probeSequence.fetch_add(1, std::memory_order_relaxed)));

    {
        std::ofstream output(probe, std::ios::binary | std::ios::trunc);
        output.put('\0');
        output.close();
        if (!output) {
            std::error_code cleanupEc;
            fs::remove(probe, cleanupEc);
            throw std::runtime_error(
                "Remote cache directory '" + pathToUtf8(path) +
                "' is not writable by this user");
        }
    }

    std::error_code ec;
    fs::remove(probe, ec);
    if (ec) {
        throw std::runtime_error(
            "Cannot remove write probe from remote cache directory '" +
            pathToUtf8(path) + "': " + ec.message());
    }
    return path;
}

} // namespace

fs::path settingsFilePath()
{
#ifdef _WIN32
    if (const wchar_t* configured = _wgetenv(L"VC3D_CONFIG_DIR");
        configured && *configured) {
        return ensureDirectory(configured) / "VC3D.ini";
    }
#else
    if (const char* configured = std::getenv("VC3D_CONFIG_DIR");
        configured && *configured) {
        return ensureDirectory(pathFromUtf8(configured)) / "VC3D.ini";
    }
#endif
    return ensureDirectory(homeDirectory() / ".VC3D") / "VC3D.ini";
}

fs::path remoteCachePath()
{
    static const fs::path active = [] {
        if (auto configured = configuredRemoteCachePath(settingsFilePath());
            !configured.empty()) {
            return ensureWritableDirectory(std::move(configured));
        }

        for (const fs::path root : {fs::path("/volpkgs"), fs::path("/ephemeral")}) {
            std::error_code ec;
            if (!fs::exists(root, ec) || ec)
                continue;
            if (!fs::is_directory(root, ec) || ec) {
                throw std::runtime_error(
                    "Remote cache root '" + root.string() + "' is not a directory");
            }
            return ensureWritableDirectory(root / "remote_cache");
        }

        return ensureWritableDirectory(homeDirectory() / ".VC3D" / "remote_cache");
    }();
    return active;
}

} // namespace vc::settings
