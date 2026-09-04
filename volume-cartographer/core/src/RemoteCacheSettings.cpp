#include "vc/core/util/RemoteCacheSettings.hpp"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace fs = std::filesystem;

namespace vc::settings {
namespace {

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
    if (const char* profile = std::getenv("USERPROFILE"); profile && *profile)
        return profile;
    const char* drive = std::getenv("HOMEDRIVE");
    const char* path = std::getenv("HOMEPATH");
    if (drive && *drive && path && *path)
        return std::string(drive) + path;
#endif
    if (const char* home = std::getenv("HOME"); home && *home)
        return home;
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
        return configured.empty() ? fs::path{} : fs::path(configured);
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
            "Cannot create remote cache directory '" + path.string() + "': " +
            (ec ? ec.message() : "path is not a directory"));
    }
    return path;
}

} // namespace

fs::path settingsFilePath()
{
    if (const char* configured = std::getenv("VC3D_CONFIG_DIR");
        configured && *configured) {
        return ensureDirectory(configured) / "VC3D.ini";
    }
    return ensureDirectory(homeDirectory() / ".VC3D") / "VC3D.ini";
}

fs::path remoteCachePath()
{
    if (auto configured = configuredRemoteCachePath(settingsFilePath());
        !configured.empty()) {
        return ensureDirectory(std::move(configured));
    }

    for (const fs::path root : {fs::path("/volpkgs"), fs::path("/ephemeral")}) {
        std::error_code ec;
        if (!fs::exists(root, ec) || ec)
            continue;
        if (!fs::is_directory(root, ec) || ec) {
            throw std::runtime_error(
                "Remote cache root '" + root.string() + "' is not a directory");
        }
        return ensureDirectory(root / "remote_cache");
    }

    return ensureDirectory(homeDirectory() / ".VC3D" / "remote_cache");
}

} // namespace vc::settings
