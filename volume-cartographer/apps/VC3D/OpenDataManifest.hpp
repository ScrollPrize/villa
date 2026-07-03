#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>

namespace vc3d::opendata {

inline constexpr std::string_view kDefaultManifestUrl =
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json";

struct OpenDataAccessRoot {
    std::string type;
    std::string url;
    std::string usage;
    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json raw = nlohmann::json::object();
};

struct OpenDataOrigin {
    std::string path;
    std::vector<OpenDataAccessRoot> accessRoots;
    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json raw = nlohmann::json::object();
};

struct OpenDataArtifact {
    std::string type;
    std::vector<OpenDataOrigin> origins;
    nlohmann::json properties = nlohmann::json::object();
    nlohmann::json raw = nlohmann::json::object();

    // Preferred public origin selected from origins/access_roots.
    std::string sourcePath;
    std::string resolvedUrl;
    std::string accessUsage;

    [[nodiscard]] bool hasResolvedUrl() const noexcept;
};

struct OpenDataVolumeTransform {
    std::string toVolumeId;
    cv::Matx44d matrix = cv::Matx44d::eye();
    std::string derivationPath;
    nlohmann::json raw = nlohmann::json::object();
};

struct OpenDataVolumeTransformGroup {
    std::string fromVolumeId;
    std::vector<OpenDataVolumeTransform> transforms;
    nlohmann::json raw = nlohmann::json::object();
};

struct OpenDataScan {
    std::string id;
    std::string suffix;
    std::string createdAt;
    nlohmann::json properties = nlohmann::json::object();
    std::vector<OpenDataArtifact> artifacts;
    nlohmann::json raw = nlohmann::json::object();
};

struct OpenDataVolume {
    std::string id;
    std::string scanId;
    std::string suffix;
    std::optional<double> pixelSizeUm;
    std::optional<double> energyKeV;
    std::optional<double> detectorDistanceMm;
    std::string dataFormat;
    std::string createdAt;
    nlohmann::json properties = nlohmann::json::object();
    std::vector<OpenDataArtifact> artifacts;
    nlohmann::json raw = nlohmann::json::object();
};

struct OpenDataSegment {
    std::string id;
    std::string longId;
    std::string suffix;
    std::string originalVolumeId;
    std::optional<int> width;
    std::optional<int> height;
    std::string createdAt;
    nlohmann::json properties = nlohmann::json::object();
    std::vector<OpenDataArtifact> artifacts;
    nlohmann::json raw = nlohmann::json::object();

    [[nodiscard]] bool hasArtifactType(std::string_view type) const;
    [[nodiscard]] bool hasTifxyz() const;
    [[nodiscard]] bool hasInkDetection() const;
    [[nodiscard]] bool hasLayersZarr() const;
};

struct OpenDataSample {
    std::string id;
    std::string type;
    std::string description;
    nlohmann::json properties = nlohmann::json::object();
    std::vector<OpenDataArtifact> artifacts;
    std::vector<OpenDataScan> scans;
    std::vector<OpenDataVolume> volumes;
    std::vector<OpenDataSegment> segments;
    std::vector<OpenDataVolumeTransformGroup> volumeTransforms;
    nlohmann::json raw = nlohmann::json::object();

    [[nodiscard]] std::size_t scanCount() const noexcept;
    [[nodiscard]] std::size_t volumeCount() const noexcept;
    [[nodiscard]] std::size_t segmentCount() const noexcept;
    [[nodiscard]] std::size_t tifxyzSegmentCount() const;
    [[nodiscard]] std::size_t inkDetectionSegmentCount() const;
};

struct OpenDataModel {
    std::string id;
    nlohmann::json raw = nlohmann::json::object();
};

struct OpenDataManifest {
    std::string manifestUrl;
    std::vector<OpenDataSample> samples;
    std::vector<OpenDataModel> models;
    nlohmann::json raw = nlohmann::json::object();

    [[nodiscard]] const OpenDataSample* findSample(std::string_view id) const noexcept;
    [[nodiscard]] const OpenDataModel* findModel(std::string_view id) const noexcept;
};

[[nodiscard]] OpenDataManifest parseOpenDataManifest(
    std::string_view jsonText,
    std::string manifestUrl = std::string(kDefaultManifestUrl));

[[nodiscard]] OpenDataManifest loadOpenDataManifestFile(
    const std::filesystem::path& manifestPath,
    std::string manifestUrl = {});

[[nodiscard]] OpenDataManifest fetchOpenDataManifest(
    std::string manifestUrl = std::string(kDefaultManifestUrl));

[[nodiscard]] std::string resolveOpenDataUrl(std::string url);
[[nodiscard]] std::string joinOpenDataUrl(std::string root, std::string path);

[[nodiscard]] const OpenDataArtifact* findArtifact(
    const std::vector<OpenDataArtifact>& artifacts,
    std::string_view type) noexcept;

[[nodiscard]] const OpenDataArtifact* preferredTifxyzArtifact(
    const OpenDataSegment& segment) noexcept;

[[nodiscard]] const OpenDataArtifact* preferredVolumeArtifact(
    const OpenDataVolume& volume) noexcept;

[[nodiscard]] const OpenDataArtifact* preferredPhotoArtifact(
    const OpenDataSample& sample) noexcept;

[[nodiscard]] std::optional<cv::Matx44d> findSampleVolumeTransform(
    const OpenDataSample& sample,
    std::string_view fromVolumeId,
    std::string_view toVolumeId) noexcept;

} // namespace vc3d::opendata
