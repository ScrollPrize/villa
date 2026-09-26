#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Pure classification of a project's volumes and datasets into "sets": one
// raw scan (with its downsampled twins) plus the Lasagna, fiber-model and
// surface products published against it. The line annotation workspace
// selects a scan, greys out datasets of other scans and lists only the
// selected set's volumes under readable names. Everything here works from
// project tags and shapes; nothing opens a volume.
namespace vc3d::line_annotation
{

// Open-data project tags this reads (produced by OpenDataSampleProject /
// OpenDataLasagna / ProjectVolumes).
inline constexpr std::string_view kOpenDataVolumeIdTagPrefix = "vc-open-data-volume-id:";
inline constexpr std::string_view kOpenDataLevelTagPrefix = "vc-open-data-source-coordinate-level:";
inline constexpr std::string_view kOpenDataVoxelSizeTagPrefix = "vc-open-data-voxel-size-um:";
inline constexpr std::string_view kOpenDataVirtualSourceTag = "vc-open-data-virtual-source";
inline constexpr std::string_view kOpenDataLasagnaModelIdTagPrefix = "vc-open-data-lasagna-model-id:";
inline constexpr std::string_view kPredictionTag = "prediction";
inline constexpr std::string_view kLasagnaGroupTagPrefixForSets = "vc-lasagna-group:";
inline constexpr std::string_view kLasagnaManifestTagPrefixForSets = "vc-lasagna-manifest:";

inline std::optional<std::string> tagValue(const std::vector<std::string>& tags,
                                           std::string_view prefix)
{
    for (const auto& tag : tags) {
        if (tag.size() >= prefix.size() && tag.compare(0, prefix.size(), prefix) == 0) {
            return tag.substr(prefix.size());
        }
    }
    return std::nullopt;
}

inline bool hasTag(const std::vector<std::string>& tags, std::string_view tag)
{
    return std::find(tags.begin(), tags.end(), tag) != tags.end();
}

// Every value of a repeated tag, in tag order (a volume shared by two
// manifests carries one `vc-lasagna-manifest:` tag per manifest).
inline std::vector<std::string> tagValues(const std::vector<std::string>& tags,
                                          std::string_view prefix)
{
    std::vector<std::string> values;
    for (const auto& tag : tags) {
        if (tag.size() >= prefix.size() && tag.compare(0, prefix.size(), prefix) == 0) {
            values.push_back(tag.substr(prefix.size()));
        }
    }
    return values;
}

enum class ProjectVolumeKind { RawScan, Lasagna, Fiber, SurfacePrediction, Other };

// What the controller reads off a loaded project volume.
struct ProjectVolumeInfo {
    std::string id;
    std::string name;
    std::vector<std::string> tags;
    std::array<std::size_t, 3> shapeZYX{};
    // From the Volume's metadata; the open-data tag wins when present.
    double voxelSizeUm = 0.0;
    // Volume::baseScaleLevel(): the pyramid level the volume was opened at
    // (`#vc-base-scale=N` locations). The open-data level tag wins when
    // present; this places untagged twins.
    int openedLevel = 0;
};

struct ClassifiedVolume {
    std::string id;
    std::string name;
    ProjectVolumeKind kind = ProjectVolumeKind::RawScan;
    // The scan this volume is, or was published against. For a raw scan the
    // open-data scan id shared by its downsampled twins, else its own id.
    std::string scanKey;
    int level = 0;
    double voxelSizeUm = 0.0;
    bool virtualSource = false;
    // Lasagna / fiber: the channel (group name) and every manifest the
    // volume belongs to (attachment merges the tags of a zarr two manifests
    // share). `kind` is Fiber when every membership is a fiber dataset;
    // volumeSelectorOptions() re-roles a shared volume by the selected dataset.
    std::string channel;
    std::vector<std::string> manifestLocations;
    std::array<std::size_t, 3> shapeZYX{};
};

inline bool volumeBelongsToDataset(const ClassifiedVolume& volume,
                                   const std::string& manifestLocation)
{
    return !manifestLocation.empty() &&
           std::find(volume.manifestLocations.begin(), volume.manifestLocations.end(),
                     manifestLocation) != volume.manifestLocations.end();
}

inline ClassifiedVolume classifyProjectVolume(
    const ProjectVolumeInfo& info,
    const std::vector<std::string>& fiberManifestLocations)
{
    ClassifiedVolume out;
    out.id = info.id;
    out.name = info.name;
    out.shapeZYX = info.shapeZYX;
    out.voxelSizeUm = info.voxelSizeUm;
    out.level = std::max(0, info.openedLevel);
    if (const auto group = tagValue(info.tags, kLasagnaGroupTagPrefixForSets)) {
        out.channel = *group;
        out.manifestLocations = tagValues(info.tags, kLasagnaManifestTagPrefixForSets);
        const bool allFiber = !out.manifestLocations.empty() &&
            std::all_of(out.manifestLocations.begin(), out.manifestLocations.end(),
                        [&](const std::string& location) {
                            return std::find(fiberManifestLocations.begin(),
                                             fiberManifestLocations.end(),
                                             location) != fiberManifestLocations.end();
                        });
        out.kind = allFiber ? ProjectVolumeKind::Fiber : ProjectVolumeKind::Lasagna;
        out.scanKey = tagValue(info.tags, kOpenDataVolumeIdTagPrefix).value_or("");
        return out;
    }
    if (hasTag(info.tags, kPredictionTag)) {
        out.kind = ProjectVolumeKind::SurfacePrediction;
        out.scanKey = tagValue(info.tags, kOpenDataVolumeIdTagPrefix).value_or("");
        if (const auto level = tagValue(info.tags, kOpenDataLevelTagPrefix)) {
            out.level = std::atoi(level->c_str());
        }
        return out;
    }
    if (const auto scanId = tagValue(info.tags, kOpenDataVolumeIdTagPrefix)) {
        out.kind = ProjectVolumeKind::RawScan;
        out.scanKey = *scanId;
        if (const auto level = tagValue(info.tags, kOpenDataLevelTagPrefix)) {
            out.level = std::max(0, std::atoi(level->c_str()));
        }
        if (const auto voxel = tagValue(info.tags, kOpenDataVoxelSizeTagPrefix)) {
            out.voxelSizeUm = std::atof(voxel->c_str());
        }
        out.virtualSource = hasTag(info.tags, kOpenDataVirtualSourceTag);
        return out;
    }
    // Untagged (a local folder attached by hand): read the published file
    // names. Scans and their products are named "<14-digit scan id>-...":
    // "20251211183505-2.399um-0.2m-78keV-masked.zarr" is the scan,
    // "20251211183505-surface-20260413222639-...zarr" its surface prediction,
    // "20260411134726-ink3d-...zarr" another product. Anything else stays an
    // unclassified extra (listed under "All volumes" only), unless the project
    // has no scan at all, in which case classifyProjectVolumes() promotes it.
    const std::string& name = info.name.empty() ? info.id : info.name;
    const auto scanIdPrefix = [&]() -> std::string {
        if (name.size() > 15 && name[14] == '-' &&
            std::all_of(name.begin(), name.begin() + 14,
                        [](unsigned char c) { return std::isdigit(c) != 0; })) {
            return name.substr(0, 14);
        }
        return {};
    }();
    const std::string rest = scanIdPrefix.empty() ? std::string{} : name.substr(15);
    if (!scanIdPrefix.empty() && rest.rfind("surface-", 0) == 0) {
        out.kind = ProjectVolumeKind::SurfacePrediction;
        out.scanKey = scanIdPrefix;
        return out;
    }
    if (!scanIdPrefix.empty() && rest.find("um-") != std::string::npos &&
        rest.find("um-") < 12) {
        out.kind = ProjectVolumeKind::RawScan;
        out.scanKey = scanIdPrefix;
        if (!(out.voxelSizeUm > 0.0)) {
            out.voxelSizeUm = std::atof(rest.c_str());
        }
        return out;
    }
    out.kind = ProjectVolumeKind::Other;
    out.scanKey = scanIdPrefix;
    return out;
}

// Classifies every volume of a project and then resolves what single-volume
// tags cannot: Lasagna/fiber channels take the scan of their dataset, and a
// project whose files carry neither tags nor recognisable names (a custom
// local folder) treats every unclassified volume as a scan rather than
// offering none.
inline std::vector<ClassifiedVolume> classifyProjectVolumes(
    const std::vector<ProjectVolumeInfo>& infos,
    const std::vector<std::string>& fiberManifestLocations)
{
    std::vector<ClassifiedVolume> volumes;
    for (const auto& info : infos) {
        volumes.push_back(classifyProjectVolume(info, fiberManifestLocations));
    }
    const bool anyScan = std::any_of(volumes.begin(), volumes.end(), [](const auto& v) {
        return v.kind == ProjectVolumeKind::RawScan;
    });
    if (!anyScan) {
        for (auto& v : volumes) {
            if (v.kind == ProjectVolumeKind::Other) {
                v.kind = ProjectVolumeKind::RawScan;
                v.scanKey = v.id;
            }
        }
    }
    return volumes;
}

// "2.399", "4.798", "8.64", "9.596": three decimals, trailing zeros trimmed.
inline std::string formatVoxelSizeUm(double voxelSizeUm)
{
    if (!(voxelSizeUm > 0.0) || !std::isfinite(voxelSizeUm)) {
        return {};
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.3f", voxelSizeUm);
    std::string text(buffer);
    while (!text.empty() && text.back() == '0') {
        text.pop_back();
    }
    if (!text.empty() && text.back() == '.') {
        text.pop_back();
    }
    return text;
}

// One raw scan of the project with its downsampled twins.
struct RawScanOption {
    std::string scanKey;
    // "20260319101107, 2.399 µm" (or the volume name when no voxel size).
    std::string label;
    double voxelSizeUm = 0.0;
    // Level-0 frame, from the level-0 entry or a coarser twin scaled up: an
    // upper bound when only a coarser twin is attached (the finest attached
    // extent times 2^level rounds the ceiling division back up).
    std::array<std::size_t, 3> level0ShapeZYX{};
    // The finest attached level and its extent, for frame matching.
    int finestLevel = 0;
    std::array<std::size_t, 3> finestShapeZYX{};
    // (level, volume id), ascending level.
    std::vector<std::pair<int, std::string>> levels;
};

inline std::vector<RawScanOption> rawScanOptions(const std::vector<ClassifiedVolume>& volumes)
{
    std::map<std::string, RawScanOption> byKey;
    for (const auto& volume : volumes) {
        if (volume.kind != ProjectVolumeKind::RawScan) {
            continue;
        }
        auto& scan = byKey[volume.scanKey];
        scan.scanKey = volume.scanKey;
        scan.levels.emplace_back(volume.level, volume.id);
    }
    std::vector<RawScanOption> scans;
    for (auto& [key, scan] : byKey) {
        std::sort(scan.levels.begin(), scan.levels.end());
        const auto& finestId = scan.levels.front().second;
        const auto finest = std::find_if(volumes.begin(), volumes.end(),
                                         [&](const auto& v) { return v.id == finestId; });
        const int finestLevel = scan.levels.front().first;
        const double factor = static_cast<double>(std::size_t{1} << finestLevel);
        scan.voxelSizeUm = finest->voxelSizeUm > 0.0 ? finest->voxelSizeUm / factor : 0.0;
        scan.finestLevel = finestLevel;
        scan.finestShapeZYX = finest->shapeZYX;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            scan.level0ShapeZYX[axis] = finest->shapeZYX[axis] << finestLevel;
        }
        const std::string voxel = formatVoxelSizeUm(scan.voxelSizeUm);
        scan.label = voxel.empty()
            ? (finest->name.empty() ? scan.scanKey : finest->name)
            : scan.scanKey + ", " + voxel + " \xC2\xB5m";
        scans.push_back(std::move(scan));
    }
    // Finest first; unknown voxel sizes last; stable by key.
    std::stable_sort(scans.begin(), scans.end(), [](const auto& a, const auto& b) {
        const double av = a.voxelSizeUm > 0.0 ? a.voxelSizeUm : 1e30;
        const double bv = b.voxelSizeUm > 0.0 ? b.voxelSizeUm : 1e30;
        if (av != bv) return av < bv;
        return a.scanKey < b.scanKey;
    });
    return scans;
}

// A Lasagna or fiber dataset entry as the controller sees it.
struct DatasetInfo {
    std::string location;
    std::vector<std::string> tags;
    // From its manifest, when it could be parsed.
    std::optional<std::array<std::size_t, 3>> baseShapeZYX;
    // The manifest is remote and being fetched: its frame is not known YET,
    // as opposed to unreadable. Never chosen automatically; a selection made
    // meanwhile is revalidated once the frame arrives.
    bool framePending = false;
};

// Whether a manifest base extent (a level-0 count, or an inclusive maximum
// one smaller) is the scan's, judged at the scan's finest attached level:
// a level-L extent of `a` covers base counts (a-1)*2^L < b <= a*2^L, so
// only a level-0 twin pins the count exactly. One voxel of tolerance for
// the two recording conventions.
inline bool scanFrameMatchesBaseExtent(const RawScanOption& scan,
                                       const std::array<std::size_t, 3>& baseShapeZYX)
{
    if (scan.finestLevel < 0 || scan.finestLevel > 30) {
        return false;
    }
    const std::size_t stride = std::size_t{1} << scan.finestLevel;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const std::size_t a = scan.finestShapeZYX[axis];
        const std::size_t b = baseShapeZYX[axis];
        if (a == 0 || b == 0) {
            return false;
        }
        const std::size_t low = (a - 1) * stride;   // (a-1)*2^L < b, minus the tolerance
        const std::size_t high = a * stride + 1;    // b <= a*2^L, plus the tolerance
        if (b < low || b > high) {
            return false;
        }
    }
    return true;
}

// The scan a dataset was published against: its open-data scan id tag, else
// the scan(s) whose frame matches its manifest base frame (see
// scanFrameMatchesBaseExtent). The four outcomes are kept apart: a dataset
// whose frame matches no scan is positive evidence it belongs to none of
// them, unlike one whose manifest could not be read.
struct DatasetScanMatch {
    enum class Kind { Matched, Ambiguous, Unknown, Incompatible, Pending };
    Kind kind = Kind::Unknown;
    // Matched: the one scan. Ambiguous: every scan with that frame.
    std::vector<std::string> scanKeys;
};

inline DatasetScanMatch datasetScanMatch(const DatasetInfo& dataset,
                                         const std::vector<RawScanOption>& scans)
{
    DatasetScanMatch match;
    if (const auto id = tagValue(dataset.tags, kOpenDataVolumeIdTagPrefix)) {
        match.kind = DatasetScanMatch::Kind::Matched;
        match.scanKeys = {*id};
        return match;
    }
    if (!dataset.baseShapeZYX) {
        match.kind = dataset.framePending ? DatasetScanMatch::Kind::Pending
                                          : DatasetScanMatch::Kind::Unknown;
        return match;
    }
    for (const auto& scan : scans) {
        if (scanFrameMatchesBaseExtent(scan, *dataset.baseShapeZYX)) {
            match.scanKeys.push_back(scan.scanKey);
        }
    }
    match.kind = match.scanKeys.empty() ? DatasetScanMatch::Kind::Incompatible
               : match.scanKeys.size() == 1 ? DatasetScanMatch::Kind::Matched
                                            : DatasetScanMatch::Kind::Ambiguous;
    return match;
}

// The matched scan, empty for anything but a unique match.
inline std::string datasetScanKey(const DatasetInfo& dataset,
                                  const std::vector<RawScanOption>& scans)
{
    const auto match = datasetScanMatch(dataset, scans);
    return match.kind == DatasetScanMatch::Kind::Matched ? match.scanKeys.front() : std::string{};
}

// Whether a dataset can drive the selected scan: a matched or ambiguous
// match naming it, or a dataset nothing is known about. A dataset known to
// belong to another scan, or to no scan of the project, never applies.
// The manifest base frames a channel volume can be validated against for
// `scanKey`: one per membership whose dataset is placed under that scan
// (by tag or by frame) and whose manifest could be read, in Lasagna-then-
// fiber dataset order. Several memberships may disagree by the recording
// convention (20811 vs 20812); the caller tries each, so no membership can
// veto another's valid frame.
std::vector<std::array<std::size_t, 3>> channelManifestFrameCandidates(
    const ClassifiedVolume& channel,
    const std::vector<DatasetInfo>& lasagnaDatasets,
    const std::vector<DatasetInfo>& fiberDatasets,
    const std::vector<RawScanOption>& scans,
    const std::string& scanKey);

// A dataset whose frame is still being fetched applies too: a selection of
// it is usable meanwhile and revalidated once the frame arrives.
inline bool datasetAppliesToScan(const DatasetScanMatch& match, const std::string& scanKey)
{
    switch (match.kind) {
    case DatasetScanMatch::Kind::Unknown:
    case DatasetScanMatch::Kind::Pending:
        return true;
    case DatasetScanMatch::Kind::Incompatible:
        return false;
    case DatasetScanMatch::Kind::Matched:
    case DatasetScanMatch::Kind::Ambiguous:
        return std::find(match.scanKeys.begin(), match.scanKeys.end(), scanKey) !=
               match.scanKeys.end();
    }
    return false;
}

inline std::vector<std::array<std::size_t, 3>> channelManifestFrameCandidates(
    const ClassifiedVolume& channel,
    const std::vector<DatasetInfo>& lasagnaDatasets,
    const std::vector<DatasetInfo>& fiberDatasets,
    const std::vector<RawScanOption>& scans,
    const std::string& scanKey)
{
    std::vector<std::array<std::size_t, 3>> frames;
    for (const auto* datasets : {&lasagnaDatasets, &fiberDatasets}) {
        for (const auto& dataset : *datasets) {
            if (!volumeBelongsToDataset(channel, dataset.location) || !dataset.baseShapeZYX) {
                continue;
            }
            if (!datasetAppliesToScan(datasetScanMatch(dataset, scans), scanKey)) {
                continue;
            }
            frames.push_back(*dataset.baseShapeZYX);
        }
    }
    return frames;
}

// Lasagna and fiber channel volumes carry no scan of their own; they take the
// scan their dataset was published against, so "All volumes" groups them.
inline void assignDatasetScansToChannels(std::vector<ClassifiedVolume>& volumes,
                                         const std::vector<DatasetInfo>& datasets,
                                         const std::vector<RawScanOption>& scans)
{
    for (auto& v : volumes) {
        if (v.kind != ProjectVolumeKind::Lasagna && v.kind != ProjectVolumeKind::Fiber) {
            continue;
        }
        if (!v.scanKey.empty()) {
            continue;
        }
        // The first membership whose dataset resolves to one scan.
        for (const auto& dataset : datasets) {
            if (!volumeBelongsToDataset(v, dataset.location)) {
                continue;
            }
            const std::string key = datasetScanKey(dataset, scans);
            if (!key.empty()) {
                v.scanKey = key;
                break;
            }
        }
    }
}

// The dataset to select when switching scans: the newest applicable one by
// its open-data model id (a timestamp), else the last applicable entry.
inline std::optional<std::string> newestDatasetForScan(
    const std::vector<DatasetInfo>& datasets,
    const std::vector<RawScanOption>& scans,
    const std::string& scanKey)
{
    std::optional<std::string> best;
    std::string bestModelId;
    for (const auto& dataset : datasets) {
        // A frame still being fetched is not chosen automatically: it may
        // turn out to belong elsewhere.
        const auto match = datasetScanMatch(dataset, scans);
        if (match.kind == DatasetScanMatch::Kind::Pending ||
            !datasetAppliesToScan(match, scanKey)) {
            continue;
        }
        const std::string modelId =
            tagValue(dataset.tags, kOpenDataLasagnaModelIdTagPrefix).value_or("");
        if (!best || modelId >= bestModelId) {
            best = dataset.location;
            bestModelId = modelId;
        }
    }
    return best;
}

// The scan to work on when the project records none: the selected fiber
// dataset's scan, then the selected Lasagna dataset's, then the scan with the
// most datasets published against it, then the finest scan.
inline std::string defaultRawScanKey(const std::vector<RawScanOption>& scans,
                                     const std::string& fiberDatasetScanKey,
                                     const std::string& lasagnaDatasetScanKey,
                                     const std::vector<std::string>& allDatasetScanKeys)
{
    const auto known = [&](const std::string& key) {
        return !key.empty() && std::any_of(scans.begin(), scans.end(),
                                           [&](const auto& s) { return s.scanKey == key; });
    };
    if (known(fiberDatasetScanKey)) return fiberDatasetScanKey;
    if (known(lasagnaDatasetScanKey)) return lasagnaDatasetScanKey;
    std::string best;
    std::size_t bestCount = 0;
    for (const auto& scan : scans) {  // finest first breaks ties
        const auto count = static_cast<std::size_t>(std::count(
            allDatasetScanKeys.begin(), allDatasetScanKeys.end(), scan.scanKey));
        if (count > bestCount) {
            best = scan.scanKey;
            bestCount = count;
        }
    }
    if (!best.empty()) return best;
    return scans.empty() ? std::string{} : scans.front().scanKey;
}

// The scan's volume at exactly `level`, else empty.
inline std::string scanVolumeIdAtExactLevel(const RawScanOption& scan, int level)
{
    for (const auto& [l, id] : scan.levels) {
        if (l == level) return id;
    }
    return {};
}

// The scan's volume at `level`, else its finest level.
inline std::string scanVolumeIdAtLevel(const RawScanOption& scan, int level)
{
    const std::string exact = scanVolumeIdAtExactLevel(scan, level);
    if (!exact.empty()) return exact;
    return scan.levels.empty() ? std::string{} : scan.levels.front().second;
}

// The scan's pyramid level a raw-scan volume is opened at, when the volume
// is one of the scan's levels; nullopt for anything else.
inline std::optional<int> rawScanLevelOfVolume(const std::vector<ClassifiedVolume>& volumes,
                                               const std::string& volumeId)
{
    for (const auto& v : volumes) {
        if (v.id == volumeId) {
            return v.kind == ProjectVolumeKind::RawScan ? std::optional<int>(v.level) : std::nullopt;
        }
    }
    return std::nullopt;
}

// "Level 1 (4.798 µm)": the Raw scan submenu's level entries.
inline std::string rawScanLevelLabel(const RawScanOption& scan, int level)
{
    std::string label = "Level " + std::to_string(level);
    if (scan.voxelSizeUm > 0.0 && level >= 0 && level < 30) {
        const double voxel = scan.voxelSizeUm * static_cast<double>(std::size_t{1} << level);
        label += " (" + formatVoxelSizeUm(voxel) + " \xC2\xB5m)";
    }
    return label;
}

// "raw scan, 2.399 µm" / "lasagna, nx" / "fiber, presence" / "surface, <name>".
// A raw scan carries no level: the level is chosen in the Raw scan submenu and
// the selector lists each scan once.
inline std::string volumeLabel(const ClassifiedVolume& volume, double scanVoxelSizeUm)
{
    switch (volume.kind) {
    case ProjectVolumeKind::Lasagna:
        return "lasagna, " + volume.channel;
    case ProjectVolumeKind::Fiber:
        return "fiber, " + volume.channel;
    case ProjectVolumeKind::SurfacePrediction:
        return "surface";
    case ProjectVolumeKind::Other:
        return "other, " + (volume.name.empty() ? volume.id : volume.name);
    case ProjectVolumeKind::RawScan:
        break;
    }
    const std::string scanVoxel = formatVoxelSizeUm(scanVoxelSizeUm);
    std::string label = "raw scan";
    if (scanVoxel.empty()) {
        label += ", " + (volume.name.empty() ? volume.id : volume.name);
    } else {
        label += ", " + scanVoxel + " \xC2\xB5m";
    }
    return label;
}

// "20260413222639-surface-m7-L2-th0.2": a surface prediction's name without
// its scan id, for the Surface dataset submenu.
inline std::string surfaceMenuLabel(const ClassifiedVolume& volume)
{
    std::string name = volume.name.empty() ? volume.id : volume.name;
    const std::string prefix = volume.scanKey + "-surface-";
    if (!volume.scanKey.empty() && name.compare(0, prefix.size(), prefix) == 0) {
        name = name.substr(prefix.size());
    }
    return name;
}

// The surface prediction to list for a scan when the project records none:
// the newest by the run id that follows "-surface-" in its name (a
// timestamp), else the last one of that scan.
inline std::optional<std::string> defaultSurfaceVolumeId(
    const std::vector<ClassifiedVolume>& volumes, const std::string& scanKey)
{
    std::optional<std::string> best;
    std::string bestRun;
    for (const auto& v : volumes) {
        if (v.kind != ProjectVolumeKind::SurfacePrediction || v.scanKey != scanKey) {
            continue;
        }
        const std::string run = surfaceMenuLabel(v);
        if (!best || run >= bestRun) {
            best = v.id;
            bestRun = run;
        }
    }
    return best;
}

struct VolumeSelectorOption {
    std::string id;
    std::string label;
    std::string tooltip;
};

// Advanced mode: every project volume in project order, under its raw name
// (the file name VC3D derived), including the downsampled twins.
inline std::vector<VolumeSelectorOption> rawVolumeSelectorOptions(
    const std::vector<ClassifiedVolume>& volumes)
{
    std::vector<VolumeSelectorOption> options;
    for (const auto& v : volumes) {
        VolumeSelectorOption o;
        o.id = v.id;
        o.label = v.name.empty() ? v.id : v.name;
        o.tooltip = v.id;
        options.push_back(std::move(o));
    }
    return options;
}

// The top-bar volume list in the simple mode: the selected scan, the selected
// Lasagna dataset's channels, the selected fiber dataset's channels and the
// selected surface prediction, in that order, under readable labels with the
// file name in the tooltip. The scan is listed once, as its volume at
// `preferredLevel` (the level the workspace is on) or its finest level;
// levels are chosen in the Raw scan submenu, not here. The current volume is
// always listed (a raw one as its scan's representative), so the selector
// never shows nothing, and its scan is listed even outside the set.
inline std::vector<VolumeSelectorOption> volumeSelectorOptions(
    const std::vector<ClassifiedVolume>& classified,
    const std::vector<RawScanOption>& scans,
    const std::string& scanKey,
    const std::string& lasagnaManifestLocation,
    const std::string& fiberManifestLocation,
    const std::string& selectedSurfaceVolumeId,
    const std::string& currentVolumeId,
    int preferredLevel = 0)
{
    // A channel shared by two manifests is shown in the role of the selected
    // dataset it belongs to.
    std::vector<ClassifiedVolume> volumes = classified;
    for (auto& v : volumes) {
        if (v.kind != ProjectVolumeKind::Lasagna && v.kind != ProjectVolumeKind::Fiber) {
            continue;
        }
        if (volumeBelongsToDataset(v, lasagnaManifestLocation)) {
            v.kind = ProjectVolumeKind::Lasagna;
        } else if (volumeBelongsToDataset(v, fiberManifestLocation)) {
            v.kind = ProjectVolumeKind::Fiber;
        }
    }
    const auto currentIt = std::find_if(volumes.begin(), volumes.end(),
                                        [&](const auto& v) { return v.id == currentVolumeId; });
    const std::string currentScanKey =
        currentIt != volumes.end() ? currentIt->scanKey : std::string{};
    // One representative volume per scan: the current volume when it is one
    // of the scan's levels, else the scan at the preferred level.
    std::map<std::string, std::string> scanRepresentative;
    for (const auto& scan : scans) {
        scanRepresentative[scan.scanKey] = scanVolumeIdAtLevel(scan, preferredLevel);
    }
    if (currentIt != volumes.end() && currentIt->kind == ProjectVolumeKind::RawScan) {
        scanRepresentative[currentIt->scanKey] = currentIt->id;
    }
    const auto representsItsScan = [&](const ClassifiedVolume& v) {
        if (v.kind != ProjectVolumeKind::RawScan) return true;
        const auto it = scanRepresentative.find(v.scanKey);
        return it != scanRepresentative.end() && it->second == v.id;
    };
    const auto scanVoxel = [&](const std::string& key) {
        for (const auto& scan : scans) {
            if (scan.scanKey == key) return scan.voxelSizeUm;
        }
        return 0.0;
    };
    const auto inSet = [&](const ClassifiedVolume& v) {
        switch (v.kind) {
        case ProjectVolumeKind::RawScan:
            return v.scanKey == scanKey;
        case ProjectVolumeKind::SurfacePrediction:
            return v.scanKey == scanKey && v.id == selectedSurfaceVolumeId;
        case ProjectVolumeKind::Lasagna:
            return volumeBelongsToDataset(v, lasagnaManifestLocation);
        case ProjectVolumeKind::Fiber:
            return volumeBelongsToDataset(v, fiberManifestLocation);
        case ProjectVolumeKind::Other:
            return false;
        }
        return false;
    };
    const auto rank = [](const ClassifiedVolume& v) {
        switch (v.kind) {
        case ProjectVolumeKind::RawScan: return 0;
        case ProjectVolumeKind::Lasagna: return 1;
        case ProjectVolumeKind::Fiber: return 2;
        case ProjectVolumeKind::SurfacePrediction: return 3;
        case ProjectVolumeKind::Other: return 4;
        }
        return 5;
    };
    std::vector<const ClassifiedVolume*> chosen;
    for (const auto& v : volumes) {
        if (!representsItsScan(v)) {
            continue;
        }
        const bool currentsScan = v.kind == ProjectVolumeKind::RawScan &&
                                  !currentScanKey.empty() && v.scanKey == currentScanKey;
        if (inSet(v) || v.id == currentVolumeId || currentsScan) {
            chosen.push_back(&v);
        }
    }
    std::stable_sort(chosen.begin(), chosen.end(), [&](const auto* a, const auto* b) {
        const bool aIn = inSet(*a), bIn = inSet(*b);
        if (aIn != bIn) return aIn;  // the stray current volume goes last
        if (rank(*a) != rank(*b)) return rank(*a) < rank(*b);
        if (a->kind == ProjectVolumeKind::RawScan && a->level != b->level) return a->level < b->level;
        if (a->manifestLocations != b->manifestLocations) return a->manifestLocations < b->manifestLocations;
        if (a->channel != b->channel) return a->channel < b->channel;
        return a->id < b->id;
    });
    std::vector<VolumeSelectorOption> options;
    for (const auto* v : chosen) {
        VolumeSelectorOption o;
        o.id = v->id;
        o.label = volumeLabel(*v, scanVoxel(v->scanKey));
        o.tooltip = v->name.empty() ? v->id : v->name + " (" + v->id + ")";
        if (!inSet(*v)) {
            o.label += " (not in this set)";
        }
        options.push_back(std::move(o));
    }
    return options;
}

}  // namespace vc3d::line_annotation
