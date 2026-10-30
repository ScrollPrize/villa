#include "vc_reader/catalog.hpp"
#include "catalog_io.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <mutex>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace vcr::catalog {
namespace {

std::string lowerCopy(std::string_view value)
{
    std::string out(value);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

bool iequals(std::string_view a, std::string_view b)
{
    return lowerCopy(a) == lowerCopy(b);
}

bool containsInsensitive(std::string_view haystack, std::string_view needle)
{
    return lowerCopy(haystack).find(lowerCopy(needle)) != std::string::npos;
}


// Fields of published records, by their published keys only.
const nlohmann::json* member(const nlohmann::json& obj, const char* key)
{
    if (!obj.is_object()) return nullptr;
    const auto it = obj.find(key);
    return it == obj.end() || it->is_null() ? nullptr : &*it;
}

std::optional<std::string> stringValue(const nlohmann::json& obj, const char* key)
{
    const auto* value = member(obj, key);
    if (!value) return std::nullopt;
    if (value->is_string()) return value->get<std::string>();
    if (value->is_number_integer() || value->is_number_unsigned()) return std::to_string(value->get<long long>());
    return std::nullopt;
}

std::optional<double> numberValue(const nlohmann::json& obj, const char* key)
{
    const auto* value = member(obj, key);
    if (!value || !value->is_number()) return std::nullopt;
    return value->get<double>();
}

std::optional<int> intValue(const nlohmann::json& obj, const char* key)
{
    if (auto value = numberValue(obj, key); value && std::isfinite(*value) &&
        *value >= std::numeric_limits<int>::min() && *value <= std::numeric_limits<int>::max() && std::floor(*value)==*value) {
        return static_cast<int>(*value);
    }
    return std::nullopt;
}

std::optional<bool> boolValue(const nlohmann::json& obj, const char* key)
{
    const auto* value = member(obj, key);
    if (!value || !value->is_boolean()) return std::nullopt;
    return value->get<bool>();
}

std::optional<std::vector<std::string>> stringList(const nlohmann::json& obj, const char* key)
{
    const auto* value = member(obj, key);
    if (!value || !value->is_array()) return std::nullopt;
    std::vector<std::string> list;
    for (const auto& item : *value)
        if (item.is_string()) list.push_back(item.get<std::string>());
    return list;
}

std::optional<std::array<std::size_t, 3>> shapeValue(const nlohmann::json& obj)
{
    const auto* it = member(obj, "shape");
    if (!it || !it->is_array() || it->size() != 3)
        return std::nullopt;
    std::array<std::size_t, 3> shape{};
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (!it->at(i).is_number_unsigned() && !it->at(i).is_number_integer())
            return std::nullopt;
        const auto value = it->at(i).get<long long>();
        if (value <= 0) return std::nullopt;
        shape[i] = static_cast<std::size_t>(value);
    }
    return shape;
}

nlohmann::json objectOrEmpty(const nlohmann::json& obj, const char* key)
{
    const auto* value = member(obj, key);
    return value && value->is_object() ? *value : nlohmann::json::object();
}

std::optional<OpenDataBounds> boundsValue(const nlohmann::json& obj, const char* key)
{
    const auto* value = member(obj, key);
    if (!value || !value->is_array() || value->size() != 2) return std::nullopt;
    OpenDataBounds bounds;
    for (int corner = 0; corner < 2; ++corner) {
        const auto& point = (*value)[corner];
        if (!point.is_array() || point.size() != 3) return std::nullopt;
        auto& out = corner == 0 ? bounds.low : bounds.high;
        for (int axis = 0; axis < 3; ++axis) {
            if (!point[axis].is_number()) return std::nullopt;
            out[axis] = point[axis].get<double>();
        }
    }
    return bounds;
}

OpenDataCreation parseCreation(const nlohmann::json& creationJson)
{
    OpenDataCreation creation;
    creation.date = stringValue(creationJson, "date").value_or("");
    creation.process = stringValue(creationJson, "process").value_or("");
    const auto derived = objectOrEmpty(creationJson, "derived_from");
    const auto type = stringValue(derived, "type"), id = stringValue(derived, "id");
    if (type && id) creation.derivedFrom = OpenDataReference{*type, *id};
    return creation;
}

std::optional<Affine> matrixValue(const nlohmann::json& obj, const char* key)
{
    const auto* value = member(obj, key);
    if (!value) return std::nullopt;
    try { return Affine::fromJson(*value); }
    catch (const std::invalid_argument&) { return std::nullopt; }
}

std::vector<OpenDataVolumeTransformGroup> parseVolumeTransforms(const nlohmann::json& propertiesJson)
{
    std::vector<OpenDataVolumeTransformGroup> groups;
    const auto* groupsJson = member(propertiesJson, "volume_transforms");
    if (!groupsJson || !groupsJson->is_array()) return groups;
    groups.reserve(groupsJson->size());
    for (const auto& groupJson : *groupsJson) {
        if (!groupJson.is_object()) continue;
        OpenDataVolumeTransformGroup group;
        group.raw = groupJson;
        group.fromVolumeId = stringValue(groupJson, "from_volume_id").value_or("");
        const auto* transformsJson = member(groupJson, "transforms");
        if (group.fromVolumeId.empty() || !transformsJson || !transformsJson->is_array()) continue;
        group.transforms.reserve(transformsJson->size());
        for (const auto& transformJson : *transformsJson) {
            auto matrix = matrixValue(transformJson, "matrix");
            if (!matrix) continue;
            OpenDataVolumeTransform transform;
            transform.raw = transformJson;
            transform.toVolumeId = stringValue(transformJson, "to_volume_id").value_or("");
            transform.derivationPath = stringValue(transformJson, "derivation_path").value_or("");
            transform.matrix = *matrix;
            if (!transform.toVolumeId.empty()) group.transforms.push_back(std::move(transform));
        }
        if (!group.transforms.empty()) groups.push_back(std::move(group));
    }
    return groups;
}

bool startsWith(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

std::vector<OpenDataAccessRoot> parseAccessRoots(const nlohmann::json& rootsJson)
{
    std::vector<OpenDataAccessRoot> roots;
    if (!rootsJson.is_array()) return roots;
    roots.reserve(rootsJson.size());
    for (const auto& rootJson : rootsJson) {
        if (!rootJson.is_object()) continue;
        OpenDataAccessRoot root;
        root.raw = rootJson;
        root.properties = objectOrEmpty(rootJson, "properties");
        root.type = stringValue(rootJson, "type").value_or("");
        root.url = stringValue(rootJson, "url").value_or("");
        root.usage = stringValue(rootJson, "usage").value_or("");
        roots.push_back(std::move(root));
    }
    return roots;
}

std::vector<OpenDataOrigin> parseOrigins(const nlohmann::json& originsJson)
{
    std::vector<OpenDataOrigin> origins;
    if (!originsJson.is_array()) return origins;
    origins.reserve(originsJson.size());
    for (const auto& originJson : originsJson) {
        if (!originJson.is_object()) continue;
        OpenDataOrigin origin;
        origin.raw = originJson;
        origin.properties = objectOrEmpty(originJson, "properties");
        origin.path = stringValue(originJson, "path").value_or("");
        origin.copySourcePath = stringValue(objectOrEmpty(originJson, "copy_info"), "source_path").value_or("");
        if (const auto* roots = member(originJson, "access_roots")) origin.accessRoots = parseAccessRoots(*roots);
        origins.push_back(std::move(origin));
    }
    return origins;
}

void selectPreferredOrigin(OpenDataArtifact& artifact)
{
    for (bool primary : {true, false}) {
        for (const auto& origin : artifact.origins) {
            for (const auto& root : origin.accessRoots) {
                if (!root.usage.empty() && !iequals(root.usage, "public-read")) continue;
                if (root.url.empty()) continue;
                const bool openData = root.url.starts_with("s3://vesuvius-challenge-open-data") ||
                    root.url.starts_with("https://vesuvius-challenge-open-data.s3.");
                if (openData != primary) continue;
                artifact.sourcePath = joinOpenDataUrl(root.url, origin.path);
                artifact.resolvedUrl = resolveOpenDataUrl(artifact.sourcePath);
                artifact.accessUsage = root.usage;
                return;
            }
        }
    }
}

std::vector<OpenDataArtifact> parseArtifacts(const nlohmann::json& ownerJson)
{
    const auto* dataJson = member(ownerJson, "data");
    if (!dataJson || !dataJson->is_array()) return {};
    std::vector<OpenDataArtifact> artifacts;
    artifacts.reserve(dataJson->size());
    for (const auto& itemJson : *dataJson) {
        if (!itemJson.is_object()) continue;
        OpenDataArtifact artifact;
        artifact.raw = itemJson;
        artifact.parameters = objectOrEmpty(itemJson, "parameters");
        artifact.properties = objectOrEmpty(itemJson, "properties");
        artifact.creationInfo = objectOrEmpty(itemJson, "creation_info");
        artifact.type = stringValue(itemJson, "type").value_or("");
        artifact.modelId = stringValue(artifact.parameters, "model_id");
        artifact.lasagnaVersionPresent = artifact.creationInfo.contains("lasagna_version");
        artifact.lasagnaVersion = intValue(artifact.creationInfo, "lasagna_version");
        artifact.sourceToBasePresent = artifact.creationInfo.contains("source_to_base");
        artifact.sourceToBase = numberValue(artifact.creationInfo, "source_to_base");
        if (artifact.sourceToBase &&
            (!std::isfinite(*artifact.sourceToBase) || *artifact.sourceToBase <= 0.0)) {
            artifact.sourceToBase.reset();
        }
        if (const auto levelIt = artifact.parameters.find("level");
            levelIt != artifact.parameters.end()) {
            artifact.levelParameterPresent = true;
            if (levelIt->is_number_unsigned()) {
                const auto value = levelIt->get<unsigned long long>();
                if (value <= 5)
                    artifact.sourceCoordinateLevel = static_cast<int>(value);
            } else if (levelIt->is_number_integer()) {
                const auto value = levelIt->get<long long>();
                if (value >= 0 && value <= 5)
                    artifact.sourceCoordinateLevel = static_cast<int>(value);
            }
        }
        if (auto target = stringValue(artifact.parameters, "target_volume"); target && !target->empty())
            artifact.targetVolumeId = std::move(*target);
        artifact.renderLevel = intValue(artifact.parameters, "render_level");
        artifact.downscale = numberValue(artifact.parameters, "downscale");
        artifact.format = stringValue(artifact.parameters, "format").value_or("");
        artifact.compositeMode = stringValue(artifact.parameters, "composite_mode").value_or("");
        artifact.labelSet = stringValue(artifact.parameters, "label_set").value_or("");
        artifact.width = intValue(artifact.properties, "width");
        artifact.height = intValue(artifact.properties, "height");
        artifact.arrays = stringList(artifact.properties, "arrays").value_or(std::vector<std::string>{});
        if (auto single = stringValue(artifact.properties, "supersedes")) artifact.supersedes = {*single};
        else artifact.supersedes = stringList(artifact.properties, "supersedes").value_or(std::vector<std::string>{});
        artifact.registration = stringValue(artifact.properties, "registration").value_or("");
        artifact.annotatedOn = stringValue(artifact.properties, "annotated_on").value_or("");
        artifact.mappingCoverage = numberValue(artifact.properties, "mapping_coverage");
        artifact.creation = parseCreation(artifact.creationInfo);
        if (const auto* origins = member(itemJson, "origins")) artifact.origins = parseOrigins(*origins);
        selectPreferredOrigin(artifact);
        if (const auto at = ("/" + artifact.sourcePath).find("/predictions/"); at != std::string::npos) {
            const auto type = artifact.sourcePath.substr(at + 12);
            artifact.predictionType = type.substr(0, type.find('/'));
        }
        artifacts.push_back(std::move(artifact));
    }
    return artifacts;
}

OpenDataScan parseScan(std::string id, const nlohmann::json& scanJson)
{
    OpenDataScan scan;
    scan.id = stringValue(scanJson, "id").value_or(std::move(id));
    scan.longId = stringValue(scanJson, "long_id").value_or("");
    scan.suffix = stringValue(scanJson, "suffix").value_or("");
    const auto creationJson = objectOrEmpty(scanJson, "creation");
    scan.creation = parseCreation(creationJson);
    scan.createdAt = scan.creation.date;
    scan.properties = objectOrEmpty(scanJson, "properties");
    scan.pixelSizeUm = numberValue(scan.properties, "pixel_size_um");
    scan.energyKeV = numberValue(scan.properties, "energy_keV");
    scan.detectorDistanceMm = numberValue(scan.properties, "detector_distance_mm");
    const auto metadata = objectOrEmpty(creationJson, "metadata");
    scan.location = stringValue(metadata, "location").value_or("");
    const auto acquisition = objectOrEmpty(objectOrEmpty(objectOrEmpty(metadata, "metadata"), "tomo"), "acquisition");
    scan.name = stringValue(acquisition, "scanRadix").value_or(stringValue(metadata, "scan_name").value_or(""));
    scan.artifacts = parseArtifacts(scanJson);
    scan.raw = scanJson.is_object() ? scanJson : nlohmann::json::object();
    return scan;
}

OpenDataVolume parseVolume(std::string id, const nlohmann::json& volumeJson)
{
    OpenDataVolume volume;
    volume.id = stringValue(volumeJson, "id").value_or(std::move(id));
    volume.longId = stringValue(volumeJson, "long_id").value_or("");
    volume.scanId = stringValue(volumeJson, "scan_id").value_or("");
    volume.suffix = stringValue(volumeJson, "suffix").value_or("");
    const auto creationJson = objectOrEmpty(volumeJson, "creation");
    volume.creation = parseCreation(creationJson);
    volume.createdAt = volume.creation.date;
    volume.properties = objectOrEmpty(volumeJson, "properties");
    const auto& p = volume.properties;
    volume.pixelSizeUm = numberValue(p, "pixel_size_um");
    volume.energyKeV = numberValue(p, "energy_keV");
    volume.detectorDistanceMm = numberValue(p, "detector_distance_mm");
    volume.dataFormat = stringValue(p, "data_format").value_or("");
    volume.shapeZYX = shapeValue(p);
    volume.orientation = {boolValue(p, "z_direction_is_top_to_bottom"), boolValue(p, "left_handed_coordinates")};
    volume.textSide = stringValue(p, "text_side").value_or("");
    if (const auto superseded = objectOrEmpty(p, "superseded_by"); auto target = stringValue(superseded, "volume_id"))
        volume.supersededBy = OpenDataSupersession{*target, stringValue(superseded, "date").value_or(""),
                                                   stringValue(superseded, "reason").value_or("")};
    if (const auto license = objectOrEmpty(p, "license"); auto name = stringValue(license, "name"))
        volume.license = OpenDataLicense{*name, stringValue(license, "url").value_or("")};
    volume.renderLevel = intValue(p, "render_level").value_or(0);
    volume.renderSegments = boolValue(p, "render_segments");
    const auto metadata = objectOrEmpty(creationJson, "metadata");
    volume.windowU16Min = numberValue(metadata, "window_u16_min");
    volume.windowU16Max = numberValue(metadata, "window_u16_max");
    volume.artifacts = parseArtifacts(volumeJson);
    volume.raw = volumeJson.is_object() ? volumeJson : nlohmann::json::object();
    return volume;
}

OpenDataSegment parseSegment(std::string id, const nlohmann::json& segmentJson)
{
    OpenDataSegment segment;
    segment.id = stringValue(segmentJson, "id").value_or(id);
    segment.longId = stringValue(segmentJson, "long_id").value_or("");
    segment.suffix = stringValue(segmentJson, "suffix").value_or("");
    segment.originalVolumeId = stringValue(segmentJson, "original_volume_id").value_or("");
    const auto creationJson = objectOrEmpty(segmentJson, "creation");
    segment.creation = parseCreation(creationJson);
    segment.createdAt = segment.creation.date;
    segment.properties = objectOrEmpty(segmentJson, "properties");
    const auto& p = segment.properties;
    segment.width = intValue(p, "width");
    segment.height = intValue(p, "height");
    if (const auto downscale = numberValue(p, "original_volume_downscale"); downscale && std::isfinite(*downscale) && *downscale > 0)
        segment.originalVolumeDownscale = *downscale;
    if (const auto* render = member(p, "render")) segment.render = *render;
    if (const auto* coverage = member(p, "volume_coverage"); coverage && coverage->is_object()) {
        for (auto it = coverage->begin(); it != coverage->end(); ++it) {
            const auto bbox = boundsValue(it.value(), "bbox_transformed");
            if (!bbox) continue;
            segment.volumeCoverage.push_back({it.key(), *bbox, numberValue(it.value(), "overlap_ratio").value_or(0),
                                              stringValue(it.value(), "computed_at").value_or("")});
        }
    }
    const auto metadata = objectOrEmpty(creationJson, "metadata");
    if (stringValue(metadata, "format") == std::optional<std::string>("tifxyz")) {
        const auto* scale = member(metadata, "scale");
        if (scale && scale->is_array() && scale->size() == 2 && (*scale)[0].is_number() && (*scale)[1].is_number()) {
            OpenDataTifxyzInfo info;
            info.scale = {(*scale)[0].get<double>(), (*scale)[1].get<double>()};
            info.bbox = boundsValue(metadata, "bbox");
            if (const auto* dims = member(metadata, "tiff_dimensions");
                dims && dims->is_array() && dims->size() == 2 && (*dims)[0].is_number_integer() && (*dims)[1].is_number_integer())
                info.tiffDimensions = std::array<int64_t, 2>{(*dims)[0].get<int64_t>(), (*dims)[1].get<int64_t>()};
            segment.originalTifxyz = info;
        }
    }
    segment.artifacts = parseArtifacts(segmentJson);
    segment.raw = segmentJson.is_object() ? segmentJson : nlohmann::json::object();
    return segment;
}

template <typename ParseFn>
auto parseObjectMap(const nlohmann::json& ownerJson, const char* key, ParseFn parse)
{
    using Value = decltype(parse(std::string(), nlohmann::json::object()));
    std::vector<Value> values;
    const auto* items = member(ownerJson, key);
    if (!items || !items->is_object()) return values;
    values.reserve(items->size());
    for (auto item = items->begin(); item != items->end(); ++item) {
        values.push_back(parse(item.key(), item.value()));
    }
    return values;
}

OpenDataSample parseSample(std::string id, const nlohmann::json& sampleJson)
{
    OpenDataSample sample;
    const auto record = objectOrEmpty(sampleJson, "sample");
    sample.id = stringValue(record, "id").value_or(std::move(id));
    sample.description = stringValue(record, "description").value_or("");
    sample.properties = objectOrEmpty(record, "properties");
    sample.type = stringValue(sample.properties, "type").value_or("");
    if (const auto* mappings = member(sample.properties, "public_access_mappings")) sample.publicAccessMappings = *mappings;
    sample.volumeTransforms = parseVolumeTransforms(sample.properties);
    sample.artifacts = parseArtifacts(record);
    sample.scans = parseObjectMap(sampleJson, "scans", parseScan);
    sample.volumes = parseObjectMap(sampleJson, "volumes", parseVolume);
    for (auto& volume : sample.volumes) {
        if (volume.pixelSizeUm && *volume.pixelSizeUm > 0.0) {
            continue;
        }
        if (volume.scanId.empty()) {
            continue;
        }
        const auto scanIt = std::find_if(
            sample.scans.begin(), sample.scans.end(), [&](const OpenDataScan& scan) {
                return scan.id == volume.scanId;
            });
        if (scanIt != sample.scans.end() &&
            scanIt->pixelSizeUm &&
            *scanIt->pixelSizeUm > 0.0) {
            volume.pixelSizeUm = scanIt->pixelSizeUm;
        }
    }
    sample.segments = parseObjectMap(sampleJson, "segments", parseSegment);
    sample.raw = sampleJson.is_object() ? sampleJson : nlohmann::json::object();
    return sample;
}

std::vector<OpenDataSample> parseSamples(const nlohmann::json& metadataJson)
{
    std::vector<OpenDataSample> samples;
    const auto* items = member(metadataJson, "samples");
    if (!items || !items->is_object()) return samples;
    samples.reserve(items->size());
    for (auto it = items->begin(); it != items->end(); ++it) {
        if (it.value().is_object()) {
            samples.push_back(parseSample(it.key(), it.value()));
        }
    }
    return samples;
}

std::vector<OpenDataModel> parseModels(const nlohmann::json& metadataJson)
{
    std::vector<OpenDataModel> models;
    const auto* items = member(metadataJson, "models");
    if (!items || !items->is_object()) return models;
    models.reserve(items->size());
    for (auto it = items->begin(); it != items->end(); ++it) {
        OpenDataModel model;
        model.raw = it.value().is_object() ? it.value() : nlohmann::json::object();
        model.id = stringValue(model.raw, "id").value_or(it.key());
        model.longId = stringValue(model.raw, "long_id").value_or("");
        model.suffix = stringValue(model.raw, "suffix").value_or("");
        model.creation = parseCreation(objectOrEmpty(model.raw, "creation"));
        model.properties = objectOrEmpty(model.raw, "properties");
        model.task = stringValue(model.properties, "task").value_or("");
        model.architecture = stringValue(model.properties, "architecture").value_or("");
        models.push_back(std::move(model));
    }
    return models;
}

} // namespace

std::vector<OpenDataArtifact> parseCatalogArtifacts(const nlohmann::json& entries)
{
    return parseArtifacts({{"data", entries}});
}

bool OpenDataArtifact::hasResolvedUrl() const noexcept
{
    return !resolvedUrl.empty();
}

std::string OpenDataArtifact::url() const
{
    std::string value = resolvedUrl.empty() ? sourcePath : resolvedUrl;
    while (!value.empty() && value.back() == '/') value.pop_back();
    return value;
}

std::string OpenDataSegment::sourceVolumeId() const
{
    if (creation.derivedFrom && iequals(creation.derivedFrom->type, "volume") && !creation.derivedFrom->id.empty())
        return creation.derivedFrom->id;
    return originalVolumeId;
}

std::string OpenDataSegment::stableId() const
{
    if (!longId.empty()) return longId;
    if (!id.empty()) return id;
    return safePathComponent(suffix);
}

const OpenDataVolumeCoverage* OpenDataSegment::coverage(std::string_view volumeId) const noexcept
{
    const auto it = std::find_if(volumeCoverage.begin(), volumeCoverage.end(),
                                 [&](const auto& c) { return c.volumeId == volumeId; });
    return it == volumeCoverage.end() ? nullptr : &*it;
}

bool OpenDataSegment::hasInkDetection() const
{
    return std::any_of(artifacts.begin(), artifacts.end(), [](const OpenDataArtifact& artifact) {
        return containsInsensitive(artifact.type, "ink");
    });
}

bool OpenDataSegment::hasLayersZarr() const
{
    return std::any_of(artifacts.begin(), artifacts.end(), [](const OpenDataArtifact& artifact) {
        const auto type = lowerCopy(artifact.type);
        return type == "layers_zarr" || type == "layers-zarr" ||
               type == "layers zarr" || (type.find("layers") != std::string::npos &&
                                          type.find("zarr") != std::string::npos);
    });
}

std::size_t OpenDataSample::scanCount() const noexcept
{
    return scans.size();
}

std::size_t OpenDataSample::volumeCount() const noexcept
{
    return volumes.size();
}

std::size_t OpenDataSample::segmentCount() const noexcept
{
    return segments.size();
}

std::size_t OpenDataSample::surfaceSegmentCount() const
{
    return static_cast<std::size_t>(std::count_if(
        segments.begin(), segments.end(), [&](const OpenDataSegment& segment) {
            return !surfaces(segment).empty();
        }));
}

const OpenDataVolume* OpenDataSample::findVolume(std::string_view id) const noexcept
{
    const auto it = std::find_if(volumes.begin(), volumes.end(), [&](const auto& v) { return v.id == id; });
    return it == volumes.end() ? nullptr : &*it;
}

std::vector<OpenDataSurface> OpenDataSample::surfaces(const OpenDataSegment& segment) const
{
    std::vector<OpenDataSurface> result;
    const std::string sourceId = segment.sourceVolumeId();
    if (sourceId.empty() || !findVolume(sourceId)) return result;
    const std::string lineage = segment.stableId();

    std::map<std::string, const OpenDataArtifact*> publishedByTarget;
    for (const auto& artifact : segment.artifacts) {
        if (lowerCopy(artifact.type) != "tifxyz-transformed" || artifact.url().empty() ||
            !artifact.targetVolumeId || !findVolume(*artifact.targetVolumeId))
            continue;
        publishedByTarget.emplace(*artifact.targetVolumeId, &artifact);
    }
    const auto published = [&](const std::string& target, const OpenDataArtifact& artifact) {
        return OpenDataSurface{&artifact, false, target == sourceId, target, 1.0,
                               lineage + "-published-" + safePathComponent(target)};
    };

    if (const auto onSource = publishedByTarget.find(sourceId); onSource != publishedByTarget.end()) {
        result.push_back(published(sourceId, *onSource->second));
    } else if (std::optional<int> level; [&] {
                   for (int l = 0; l <= 5; ++l)
                       if (std::abs(segment.originalVolumeDownscale - std::ldexp(1.0, l)) <=
                           1.0e-9 * std::max(1.0, segment.originalVolumeDownscale)) { level = l; break; }
                   return level.has_value();
               }()) {
        for (const auto* type : {"tifxyz-flattened", "tifxyz", "tifxyz-normalized"}) {
            const auto it = std::find_if(segment.artifacts.begin(), segment.artifacts.end(), [&](const auto& artifact) {
                return lowerCopy(artifact.type) == type && !artifact.url().empty();
            });
            if (it == segment.artifacts.end()) continue;
            result.push_back({&*it, true, true, sourceId, 1.0 / segment.originalVolumeDownscale, lineage + "-authored"});
            break;
        }
    }
    for (const auto& [target, artifact] : publishedByTarget)
        if (target != sourceId) result.push_back(published(target, *artifact));
    return result;
}

std::size_t OpenDataSample::inkDetectionSegmentCount() const
{
    return static_cast<std::size_t>(std::count_if(
        segments.begin(), segments.end(), [](const OpenDataSegment& segment) {
            return segment.hasInkDetection();
        }));
}

std::optional<OpenDataRepresentationKind> classifyDerivedRepresentation(
    const OpenDataArtifact& artifact)
{
    std::string type = lowerCopy(artifact.type);
    std::replace(type.begin(), type.end(), '_', '-');

    if (type.find("normal-grid") != std::string::npos) {
        return OpenDataRepresentationKind::NormalGrids;
    }
    if (type == "lasagna") {
        return OpenDataRepresentationKind::Lasagna;
    }
    const bool prediction =
        type.find("prediction") != std::string::npos ||
        type.starts_with("pred-") || type.ends_with("-pred") ||
        type.find("-pred-") != std::string::npos;
    const bool ink3d = type.find("ink-detection") != std::string::npos &&
                       type.find("3d") != std::string::npos;
    if (prediction || ink3d)
        return OpenDataRepresentationKind::Prediction;
    return std::nullopt;
}

std::vector<OpenDataRepresentationRef> derivedRepresentations(
    const OpenDataSample& sample)
{
    std::vector<OpenDataRepresentationRef> result;
    for (std::size_t volumeIndex = 0;
         volumeIndex < sample.volumes.size(); ++volumeIndex) {
        const auto& volume = sample.volumes[volumeIndex];
        for (std::size_t artifactIndex = 0;
             artifactIndex < volume.artifacts.size(); ++artifactIndex) {
            if (const auto kind =
                    classifyDerivedRepresentation(volume.artifacts[artifactIndex])) {
                result.push_back({volumeIndex, artifactIndex, *kind});
            }
        }
    }
    return result;
}

std::string_view representationKindName(OpenDataRepresentationKind kind) noexcept
{
    switch (kind) {
        case OpenDataRepresentationKind::NormalGrids: return "Normal grids";
        case OpenDataRepresentationKind::Lasagna: return "Lasagna";
        case OpenDataRepresentationKind::Prediction: return "Prediction";
    }
    return "Prediction";
}

const OpenDataSample* OpenDataManifest::findSample(std::string_view id) const noexcept
{
    const auto it = std::find_if(samples.begin(), samples.end(), [&](const OpenDataSample& sample) {
        return sample.id == id;
    });
    return it == samples.end() ? nullptr : &*it;
}

std::optional<OpenDataVolumeView> OpenDataManifest::volumeAt(std::string_view url, int baseLevel) const
{
    const auto trimmed = [](std::string value) {
        while (!value.empty() && value.back() == '/') value.pop_back();
        return value;
    };
    const auto wanted = trimmed(std::string(url));
    if (wanted.empty()) return std::nullopt;
    for (const auto& sample : samples)
        for (const auto& volume : sample.volumes) {
            const auto* source = preferredVolumeArtifact(volume);
            for (const auto& artifact : volume.artifacts) {
                if (artifact.url() != wanted && trimmed(artifact.sourcePath) != wanted) continue;
                if (&artifact == source) return OpenDataVolumeView{sample.id, volume.id, baseLevel, true, &artifact};
                if (!classifyDerivedRepresentation(artifact) || !artifact.sourceCoordinateLevel) continue;
                return OpenDataVolumeView{sample.id, volume.id, baseLevel + *artifact.sourceCoordinateLevel, false, &artifact};
            }
        }
    return std::nullopt;
}

std::optional<OpenDataVolumeMetadata> OpenDataManifest::volumeMetadataAt(const std::string& url) const
{
    const auto view = volumeAt(resolveOpenDataUrl(url));
    if (!view) return std::nullopt;
    const auto* sample = findSample(view->sampleId);
    const auto* volume = sample ? sample->findVolume(view->volumeId) : nullptr;
    if (!volume) return std::nullopt;
    OpenDataVolumeMetadata out{view->sampleId, view->volumeId, volume->scanId, volume->suffix,
                               volume->createdAt, volume->dataFormat, std::nullopt, volume->energyKeV,
                               volume->detectorDistanceMm, volume->raw};
    if (volume->pixelSizeUm) out.voxelSizeUm = std::ldexp(*volume->pixelSizeUm, view->level);
    return out;
}

std::shared_ptr<const OpenDataManifest> cachedManifest(const std::filesystem::path& cacheFile)
{
    struct Entry {
        std::uintmax_t size = 0;
        std::filesystem::file_time_type mtime;
        std::shared_ptr<const OpenDataManifest> manifest;
    };
    static std::mutex mutex;
    static std::map<std::filesystem::path, Entry> entries;
    std::error_code error;
    const auto size = std::filesystem::file_size(cacheFile, error);
    if (error) return nullptr;
    const auto mtime = std::filesystem::last_write_time(cacheFile, error);
    if (error) return nullptr;
    std::lock_guard lock(mutex);
    auto& entry = entries[cacheFile];
    if (!entry.manifest || entry.size != size || entry.mtime != mtime) {
        try {
            entry.manifest = std::make_shared<const OpenDataManifest>(loadOpenDataManifestFile(cacheFile, std::string(kDefaultManifestUrl)));
        } catch (const std::exception&) {
            entry.manifest.reset();
        }
        entry.size = size;
        entry.mtime = mtime;
    }
    return entry.manifest;
}

OpenDataManifest parseOpenDataManifest(std::string_view jsonText, std::string manifestUrl)
{
    auto root = nlohmann::json::parse(jsonText);
    if (!root.is_object()) {
        throw std::runtime_error("Open-data manifest root must be a JSON object");
    }

    OpenDataManifest manifest;
    manifest.manifestUrl = std::move(manifestUrl);
    manifest.samples = parseSamples(root);
    manifest.models = parseModels(root);
    manifest.raw = std::move(root);
    return manifest;
}

OpenDataManifest loadOpenDataManifestFile(const std::filesystem::path& manifestPath,
                                          std::string manifestUrl)
{
    if (manifestUrl.empty()) manifestUrl = manifestPath.string();
    return parseOpenDataManifest(readCatalogFile(manifestPath), std::move(manifestUrl));
}

// The public channels' s3:// and https:// roots, as the scrollprize.org data
// browser rewrites them.
constexpr std::pair<std::string_view, std::string_view> kPublicChannels[] = {
    {"s3://vesuvius-challenge-open-data/", "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"},
    {"s3://vesuvius-challenge/", "https://data.aws.ash2txt.org/samples/"},
};

std::string resolveOpenDataUrl(std::string url)
{
    for (const auto& [s3, https] : kPublicChannels)
        if (startsWith(url, s3)) return std::string(https) + url.substr(s3.size());
    if (startsWith(url, "s3://") || startsWith(url, "s3+")) {
        const auto scheme = url.find("://");
        if (scheme == std::string::npos) return url;
        const auto region = url.starts_with("s3+") ? url.substr(3, scheme - 3) : "us-east-1";
        const auto path = url.substr(scheme + 3);
        const auto slash = path.find('/');
        return "https://" + path.substr(0, slash) + ".s3." + region + ".amazonaws.com" +
               (slash == std::string::npos ? "" : path.substr(slash));
    }
    return url;
}

std::string safePathComponent(std::string value)
{
    for (char& c : value) {
        const auto uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '-' && c != '_' && c != '.') c = '_';
    }
    while (!value.empty() && (value.front() == '.' || value.front() == '_')) value.erase(value.begin());
    return value.empty() ? std::string("unnamed") : value;
}

std::optional<std::string> openDataS3Url(std::string url)
{
    if (startsWith(url, "s3://")) return url;
    for (const auto& [s3, https] : kPublicChannels)
        if (startsWith(url, https)) return std::string(s3) + url.substr(https.size());
    return std::nullopt;
}

std::string joinOpenDataUrl(std::string root, std::string path)
{
    if (path.empty()) {
        return root;
    }
    if (startsWith(path, "s3://") || startsWith(path, "s3+") ||
        startsWith(path, "http://") || startsWith(path, "https://")) {
        return path;
    }
    const auto suffixAt = root.find_first_of("?#");
    const auto suffix = suffixAt == std::string::npos ? std::string{} : root.substr(suffixAt);
    if (suffixAt != std::string::npos) root.resize(suffixAt);
    while (!root.empty() && root.back() == '/') {
        root.pop_back();
    }
    while (!path.empty() && path.front() == '/') {
        path.erase(path.begin());
    }
    return root + "/" + path + suffix;
}

const OpenDataArtifact* findArtifact(const std::vector<OpenDataArtifact>& artifacts,
                                     std::string_view type) noexcept
{
    const auto it = std::find_if(artifacts.begin(), artifacts.end(), [&](const OpenDataArtifact& artifact) {
        return iequals(artifact.type, type);
    });
    return it == artifacts.end() ? nullptr : &*it;
}

std::vector<OpenDataSegmentImage> inkDetectionImages(const OpenDataSegment& segment)
{
    const auto imageExtension = [](std::string url) -> std::string {
        if (const auto pos = url.find_first_of("?#"); pos != std::string::npos) url.erase(pos);
        url = lowerCopy(url);
        for (const auto* ext : {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"})
            if (url.ends_with(ext)) return ext;
        return {};
    };
    const auto isJpeg = [](const std::string& ext) { return ext == ".jpg" || ext == ".jpeg"; };
    std::vector<const OpenDataArtifact*> images;
    bool anyJpeg = false;
    for (const auto& artifact : segment.artifacts) {
        const auto type = lowerCopy(artifact.type);
        if (type.find("ink") == std::string::npos || type.find("zarr") != std::string::npos ||
            type.find("3d") != std::string::npos)
            continue;
        const auto ext = imageExtension(artifact.url());
        if (ext.empty()) continue;
        anyJpeg = anyJpeg || isJpeg(ext);
        images.push_back(&artifact);
    }
    std::vector<OpenDataSegmentImage> result;
    for (const auto* artifact : images) {
        const auto ext = imageExtension(artifact->url());
        if (anyJpeg && !isJpeg(ext)) continue;
        const auto index = result.size();
        std::string base = artifact->type.empty() ? "ink_detection" : artifact->type;
        if (index > 0) base += "_" + std::to_string(index + 1);
        std::string label = (segment.suffix.empty() ? (segment.id.empty() ? std::string("<unnamed segment>") : segment.id)
                                                    : segment.suffix) +
                            " - " + (artifact->type.empty() ? std::string("ink detection") : artifact->type);
        if (index > 0) label += " " + std::to_string(index + 1);
        result.push_back({artifact, artifact->url(), safePathComponent(base) + ext, std::move(label)});
    }
    return result;
}

bool isZarrArtifact(const OpenDataArtifact& artifact)
{
    if (containsInsensitive(artifact.type, "zarr")) return true;
    auto path = artifact.url();
    path.resize(std::min(path.size(), path.find_first_of("?#")));
    while (!path.empty() && path.back() == '/') path.pop_back();
    return lowerCopy(path).ends_with(".zarr");
}

const OpenDataArtifact* preferredVolumeArtifact(const OpenDataVolume& volume) noexcept
{
    for (const auto* type : {"zarr", "zarr3", "volume_zarr", "omezarr", "ome-zarr"}) {
        if (const auto* artifact = findArtifact(volume.artifacts, type)) {
            return artifact;
        }
    }
    const auto it = std::find_if(
        volume.artifacts.begin(), volume.artifacts.end(), [](const OpenDataArtifact& artifact) {
            return artifact.hasResolvedUrl() && isZarrArtifact(artifact);
        });
    if (it != volume.artifacts.end()) {
        return &*it;
    }
    return volume.artifacts.empty() ? nullptr : &volume.artifacts.front();
}

const OpenDataArtifact* preferredPhotoArtifact(const OpenDataSample& sample) noexcept
{
    for (const auto* type : {"photo", "image", "thumbnail"}) {
        if (const auto* artifact = findArtifact(sample.artifacts, type)) {
            return artifact;
        }
    }
    const auto it = std::find_if(
        sample.artifacts.begin(), sample.artifacts.end(), [](const OpenDataArtifact& artifact) {
            const auto type = lowerCopy(artifact.type);
            return type.find("photo") != std::string::npos ||
                   type.find("image") != std::string::npos ||
                   type.find("thumbnail") != std::string::npos;
        });
    return it == sample.artifacts.end() ? nullptr : &*it;
}

} // namespace vcr::catalog
