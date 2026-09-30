#include "OpenDataVolumeOrientation.hpp"

#include <system_error>

namespace vc3d::opendata {

namespace {

std::optional<bool> booleanProperty(const nlohmann::json& properties, const char* key)
{
    if (!properties.is_object()) {
        return std::nullopt;
    }
    const auto it = properties.find(key);
    if (it == properties.end()) {
        return std::nullopt;
    }
    if (it->is_boolean()) {
        return it->get<bool>();
    }
    if (it->is_string()) {
        const std::string& text = it->get_ref<const std::string&>();
        if (text == "true") {
            return true;
        }
        if (text == "false") {
            return false;
        }
    }
    return std::nullopt;
}

} // namespace

std::optional<std::pair<std::string, std::string>> sampleAndVolumeOfCoordinateSpace(
    std::string_view coordinateSpace)
{
    const auto slash = coordinateSpace.find('/');
    if (slash == std::string_view::npos) {
        return std::nullopt;
    }
    const std::string_view sample = coordinateSpace.substr(0, slash);
    std::string_view volume = coordinateSpace.substr(slash + 1);
    if (const auto at = volume.find('@'); at != std::string_view::npos) {
        volume = volume.substr(0, at);
    }
    if (sample.empty() || volume.empty() || volume.find('/') != std::string_view::npos) {
        return std::nullopt;
    }
    return std::make_pair(std::string(sample), std::string(volume));
}

std::string catalogVolumeOfCoordinateSpace(std::string_view coordinateSpace)
{
    const auto ids = sampleAndVolumeOfCoordinateSpace(coordinateSpace);
    if (!ids) {
        return {};
    }
    return ids->first + '/' + ids->second;
}

VolumeOrientation volumeOrientationOf(const OpenDataVolume& volume)
{
    VolumeOrientation orientation;
    orientation.zTopToBottom =
        booleanProperty(volume.properties, "z_direction_is_top_to_bottom");
    orientation.leftHandedCoordinates =
        booleanProperty(volume.properties, "left_handed_coordinates");
    return orientation;
}

std::optional<VolumeOrientation> findVolumeOrientation(const OpenDataManifest& manifest,
                                                       std::string_view sampleId,
                                                       std::string_view volumeId)
{
    const OpenDataSample* sample = manifest.findSample(sampleId);
    if (sample == nullptr) {
        return std::nullopt;
    }
    for (const OpenDataVolume& volume : sample->volumes) {
        if (volume.id == volumeId) {
            return volumeOrientationOf(volume);
        }
    }
    return std::nullopt;
}

std::optional<int> windingChiralityOf(const VolumeOrientation& orientation)
{
    if (!orientation.zTopToBottom || !orientation.leftHandedCoordinates) {
        return std::nullopt;
    }
    return *orientation.zTopToBottom != *orientation.leftHandedCoordinates ? -1 : 1;
}

CatalogVolumeOrientationLookup::CatalogVolumeOrientationLookup(std::filesystem::path manifestPath)
    : _manifestPath(std::move(manifestPath))
{
}

std::optional<CatalogVolumeOrientationLookup::FileToken>
CatalogVolumeOrientationLookup::fileToken() const
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(_manifestPath, ec)) {
        return std::nullopt;
    }
    FileToken token;
    token.size = std::filesystem::file_size(_manifestPath, ec);
    if (ec) {
        return std::nullopt;
    }
    token.mtime = std::filesystem::last_write_time(_manifestPath, ec);
    if (ec) {
        return std::nullopt;
    }
    return token;
}

std::string CatalogVolumeOrientationLookup::tokenText(const std::optional<FileToken>& token)
{
    if (!token) {
        return "absent";
    }
    // The clock's tick type is the library's own (libc++ makes to_string
    // ambiguous on it); the count fits a long long on every platform here.
    return std::to_string(static_cast<unsigned long long>(token->size)) + ':' +
           std::to_string(
               static_cast<long long>(token->mtime.time_since_epoch().count()));
}

std::string CatalogVolumeOrientationLookup::manifestToken(std::string_view coordinateSpace) const
{
    if (coordinateSpace.empty()) {
        return {};
    }
    return tokenText(fileToken());
}

CatalogVolumeOrientationLookup::CatalogSense CatalogVolumeOrientationLookup::resolve(
    std::string_view coordinateSpace)
{
    CatalogSense answer;
    if (coordinateSpace.empty()) {
        return answer;
    }
    const auto ids = sampleAndVolumeOfCoordinateSpace(coordinateSpace);
    const std::lock_guard<std::mutex> lock(_mutex);
    // The file is measured under the lock, so no caller can memoize an
    // answer under a token another caller measured earlier.
    std::optional<FileToken> token = fileToken();
    if (!ids) {
        answer.manifestToken = tokenText(token);
        return answer;
    }
    if (_memo && _memo->coordinateSpace == coordinateSpace && _memo->token == token) {
        answer.orientation = _memo->value;
        answer.manifestToken = tokenText(token);
        return answer;
    }
    // A parse is bound to the version measured around it: the catalog
    // window replaces the file atomically, but a replacement between the
    // measurement and the read would pair the new bytes with the old token
    // (or, without atomic replacement, read a torn file). Re-measure after
    // the read and go again while the file moved; a file that keeps moving
    // is answered for now and not memoized.
    constexpr int kAttempts = 3;
    for (int attempt = 0; attempt < kAttempts; ++attempt) {
        std::optional<VolumeOrientation> value;
        if (token) {
            try {
                value = findVolumeOrientation(loadOpenDataManifestFile(_manifestPath),
                                              ids->first, ids->second);
            } catch (...) {
                value = std::nullopt;
            }
        }
        if (_afterParseHook) {
            _afterParseHook();
        }
        const std::optional<FileToken> after = fileToken();
        if (after == token) {
            _memo = Memo{std::string(coordinateSpace), token, value};
            answer.orientation = value;
            answer.manifestToken = tokenText(token);
            return answer;
        }
        token = after;
    }
    answer.manifestToken = "unstable";
    return answer;
}

void CatalogVolumeOrientationLookup::setAfterParseHookForTesting(std::function<void()> hook)
{
    const std::lock_guard<std::mutex> lock(_mutex);
    _afterParseHook = std::move(hook);
}

} // namespace vc3d::opendata
