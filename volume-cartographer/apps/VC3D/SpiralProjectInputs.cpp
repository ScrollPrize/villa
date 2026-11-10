#include "SpiralProjectInputs.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/Manifest.hpp"

#include <vc_object_store/objects/VolumeUmbilicus.hpp>
#include <vc_reader/zarr.hpp>

#include <QCryptographicHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>

#include <cmath>
#include <fstream>
#include <iterator>

namespace vc3d::spiral {

std::string lasagnaIdentity(const std::string& manifestBytes, const std::string& manifestName,
                            const std::string& provenance)
{
    static const QRegularExpression valid(QStringLiteral(R"(\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\z)"));
    if (!provenance.empty()) {
        const QJsonObject record = QJsonDocument::fromJson(QByteArray::fromStdString(provenance)).object();
        const QJsonValue named = record.value(QStringLiteral("manifest"));
        const QString runUuid = record.value(QStringLiteral("run_uuid")).toString();
        if (valid.match(runUuid).hasMatch()
            && (named.isUndefined() || named.isNull()
                || named.toString().toStdString() == manifestName))
            return runUuid.toStdString();
    }
    return "sha256-" + QCryptographicHash::hash(QByteArray::fromStdString(manifestBytes),
                                                QCryptographicHash::Sha256).toHex().toStdString();
}

ProjectLasagna projectLasagna(const vc::lasagna::LasagnaDatasetManifest& manifest,
                              const std::string& location, const std::string& identity)
{
    ProjectLasagna result;
    result.location = location;
    result.identity = identity;
    result.baseShapeZYX = manifest.baseShapeZYX;
    const std::pair<const char*, const char*> inputs[] = {
        {"normal_x", "nx"}, {"normal_y", "ny"}, {"gradient_magnitude", "grad_mag"}};
    std::optional<int> scaledown;
    for (const auto& [input, channel] : inputs) {
        const auto* group = manifest.groupForChannel(channel);
        if (!group) continue;
        if (group->channels.size() != 1) {
            result.error = "Spiral reads single-channel Lasagna arrays; " + location + "'s channel " +
                           channel + " shares group " + group->name;
            return result;
        }
        // The group is an array of an OME-Zarr store: <store>/<array>, a
        // path or a remote location.
        const auto source = vc::lasagna::lasagnaGroupSourceLocation(*group);
        const auto slash = source.find_last_of('/');
        const auto store = source.substr(0, slash);
        const auto array = source.substr(slash + 1);
        if ((!result.group.empty() && array != result.group) || (scaledown && *scaledown != group->scaledown)) {
            result.error = "Spiral reads the Lasagna channels at one array; " + location +
                           " stores them at different ones";
            return result;
        }
        result.stores[input] = store;
        result.group = array;
        scaledown = group->scaledown;
        if (group->isRemote()) result.remoteGroups[input] = *group;
    }
    if (result.stores.empty())
        result.error = location + " has no nx, ny or grad_mag channel";
    else
        result.baseVoxelsPerArrayVoxel = manifest.sourceToBase * std::ldexp(1.0, *scaledown);
    return result;
}

namespace {
std::string fileBytes(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

// A local manifest's provenance record (its "provenance" file, beside it),
// empty when it names none.
std::string provenanceText(const vc::lasagna::LasagnaDatasetManifest& manifest)
{
    const auto name = manifest.raw.value("provenance", std::string{});
    return name.empty() ? std::string{} : fileBytes(manifest.baseDirectory / name);
}
} // namespace

namespace {
// A project dataset selection's location: a remote one as is, a local one
// resolved against the project.
std::string datasetLocation(const VolumePkg& pkg, const std::string& selected)
{
    return vcr::zarr::isRemote(selected)
        ? selected
        : vc::project::resolveLocalPath(selected, pkg.directory()).string();
}

struct OpenedDataset {
    vc::lasagna::LasagnaDatasetManifest manifest;
    std::string identity;  // empty for a remote dataset
};

OpenedDataset openDataset(const std::string& location, const std::filesystem::path& remoteCacheRoot)
{
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.remoteCacheRoot = remoteCacheRoot;
    auto manifest = vc::lasagna::LasagnaDataset::openLocation(location, options).manifest();
    // The service identifies a remote dataset by its URL, the location sent.
    if (manifest.manifestIsRemote) return {std::move(manifest), {}};
    // A local copy of a remote dataset may predate caching its provenance.
    vc::lasagna::cacheLasagnaProvenance(manifest);
    const auto file = std::filesystem::is_directory(manifest.manifestPath)
        ? manifest.manifestPath / ".zattrs" : manifest.manifestPath;
    auto identity = lasagnaIdentity(fileBytes(file), manifest.manifestPath.filename().string(),
                                    provenanceText(manifest));
    return {std::move(manifest), std::move(identity)};
}
} // namespace

std::optional<ProjectLasagna> projectLasagna(const VolumePkg& pkg,
                                             const std::filesystem::path& remoteCacheRoot)
{
    const std::string selected = pkg.selectedLasagnaDataset();
    if (selected.empty()) return std::nullopt;
    const std::string location = datasetLocation(pkg, selected);
    try {
        const auto dataset = openDataset(location, remoteCacheRoot);
        return projectLasagna(dataset.manifest, location, dataset.identity);
    } catch (const std::exception& error) {
        ProjectLasagna unreadable;
        unreadable.location = location;
        unreadable.error = "cannot read " + location + ": " + error.what();
        return unreadable;
    }
}

ProjectLasagna withCachedStores(ProjectLasagna lasagna)
{
    for (const auto& [input, group] : lasagna.remoteGroups) {
        const auto directory = *vc::lasagna::lasagnaChannelArrayCacheDirectory(group);
        if (directory.filename().string() != lasagna.group) {
            lasagna.error = "the cache directory " + directory.string() + " of " + lasagna.stores[input] +
                            " does not end in its array " + lasagna.group;
            return lasagna;
        }
        lasagna.stores[input] = directory.parent_path().string();
    }
    return lasagna;
}

std::vector<vc::lasagna::LasagnaChannelGroup> lasagnaGroupsToCache(
    const ProjectLasagna& lasagna, const std::filesystem::path& spiralCache,
    const std::optional<std::set<std::string>>& kinds)
{
    const std::pair<const char*, std::vector<const char*>> packs[] = {
        {"normals", {"normal_x", "normal_y"}}, {"grad_mag", {"gradient_magnitude"}}};
    std::vector<vc::lasagna::LasagnaChannelGroup> result;
    for (const auto& [kind, inputs] : packs) {
        if (kinds && !kinds->contains(kind)) continue;
        std::error_code ec;
        if (!lasagna.identity.empty()
            && std::filesystem::is_regular_file(spiralCache / "lasagna" / lasagna.identity /
                                                    (std::string(kind) + "_g" + lasagna.group) / "meta.json",
                                                ec))
            continue;
        for (const char* input : inputs)
            if (const auto it = lasagna.remoteGroups.find(input); it != lasagna.remoteGroups.end())
                result.push_back(it->second);
    }
    return result;
}

std::string remoteManifestUrl(const vc::lasagna::LasagnaDatasetManifest& manifest)
{
    if (manifest.manifestIsRemote || manifest.remoteSourceBaseLocation.empty()) return {};
    return vcr::joinLocation(manifest.remoteSourceBaseLocation, manifest.manifestPath.filename().string());
}

std::optional<ProjectDataset> projectFiberInference(const VolumePkg& pkg,
                                                    const std::filesystem::path& remoteCacheRoot)
{
    const std::string selected = pkg.selectedFiberInferenceDataset();
    if (selected.empty()) return std::nullopt;
    ProjectDataset result;
    result.location = datasetLocation(pkg, selected);
    try {
        const auto dataset = openDataset(result.location, remoteCacheRoot);
        result.identity = dataset.identity;
        result.baseShapeZYX = dataset.manifest.baseShapeZYX;
        result.remoteManifest = remoteManifestUrl(dataset.manifest);
    } catch (const std::exception& error) {
        result.error = "cannot read " + result.location + ": " + error.what();
    }
    return result;
}

std::map<std::string, ProjectVolumeFacts> projectVolumeFacts(const VolumePkg& pkg)
{
    std::map<std::string, ProjectVolumeFacts> result;
    for (const auto& id : pkg.volumeIDs()) {
        const auto volume = pkg.volume(id);
        if (!volume) continue;
        const auto annotation = vc::project::annotationVolume(&pkg, *volume);
        ProjectVolumeFacts facts;
        facts.voxelSizeUm = annotation.voxelSizeUm;
        if (const auto view = pkg.catalogView(id)) {
            const auto* sample = pkg.catalog()->findSample(view->sampleId);
            if (const auto* catalogVolume = sample ? sample->findVolume(view->volumeId) : nullptr) {
                facts.baseShapeZYX = catalogVolume->shapeZYX;
                facts.zTopToBottom = catalogVolume->orientation.zTopToBottom;
                facts.leftHandedCoordinates = catalogVolume->orientation.leftHandedCoordinates;
            }
        } else if (annotation.scale == 1) {
            const auto shape = volume->shape();
            facts.baseShapeZYX = std::array<std::size_t, 3>{
                std::size_t(shape[0]), std::size_t(shape[1]), std::size_t(shape[2])};
        }
        // Every working level of one catalog volume states the same facts;
        // the first that knows one gives it.
        auto& known = result[annotation.volume];
        if (!known.voxelSizeUm) known.voxelSizeUm = facts.voxelSizeUm;
        if (!known.baseShapeZYX) known.baseShapeZYX = facts.baseShapeZYX;
        if (!known.zTopToBottom) known.zTopToBottom = facts.zTopToBottom;
        if (!known.leftHandedCoordinates) known.leftHandedCoordinates = facts.leftHandedCoordinates;
    }
    return result;
}

QJsonObject projectBlock(const std::string& name,
                         const std::map<std::string, ProjectVolumeFacts>& volumes,
                         const std::optional<ProjectLasagna>& lasagna,
                         const std::optional<ProjectDataset>& fiberInference,
                         vc::objects::ObjectStoreSession* session,
                         const std::optional<vc::objects::AttachmentIdentity>& umbilicusStore)
{
    const bool readable = lasagna && lasagna->error.empty();
    const bool fibers = fiberInference && fiberInference->error.empty();
    QJsonObject entries;
    for (const auto& [uuid, facts] : volumes) {
        QJsonObject entry;
        if (facts.voxelSizeUm) entry[QStringLiteral("voxel_size_um")] = *facts.voxelSizeUm;
        if (facts.baseShapeZYX)
            entry[QStringLiteral("base_shape_zyx")] = QJsonArray{
                qint64((*facts.baseShapeZYX)[0]), qint64((*facts.baseShapeZYX)[1]),
                qint64((*facts.baseShapeZYX)[2])};
        if (facts.zTopToBottom) entry[QStringLiteral("z_direction_is_top_to_bottom")] = *facts.zTopToBottom;
        if (facts.leftHandedCoordinates)
            entry[QStringLiteral("left_handed_coordinates")] = *facts.leftHandedCoordinates;
        if (readable && lasagna->baseShapeZYX && facts.baseShapeZYX) {
            // An array voxel in this volume's voxels: a whole number, or the
            // volume is not a dyadic level of the dataset's base.
            try {
                const double basePerVolumeVoxel = vc::lasagna::dyadicCoordinateScaleBetweenShapes(
                    *facts.baseShapeZYX, *lasagna->baseShapeZYX);
                const double scale = lasagna->baseVoxelsPerArrayVoxel / basePerVolumeVoxel;
                if (scale >= 1 && scale == std::round(scale))
                    entry[QStringLiteral("lasagna_scale")] = qint64(scale);
            } catch (const std::exception&) {
            }
        }
        if (fibers && fiberInference->baseShapeZYX && facts.baseShapeZYX) {
            // The fiber-direction samples are extracted at this scale.
            try {
                entry[QStringLiteral("fiber_scale")] = vc::lasagna::dyadicCoordinateScaleBetweenShapes(
                    *facts.baseShapeZYX, *fiberInference->baseShapeZYX);
            } catch (const std::exception&) {
            }
        }
        if (session && umbilicusStore) {
            // In the volume's own voxels, as the fit reads it.
            if (const auto umbilicus = vc::objects::volumeUmbilicus(
                    *session, *umbilicusStore, vc::objects::VolumeVoxels{uuid})) {
                QJsonArray points;
                for (const auto& [x, y, z] : umbilicus->points) points.append(QJsonArray{z, y, x});
                entry[QStringLiteral("umbilicus")] = QJsonObject{
                    {QStringLiteral("store"), QString::fromStdString(umbilicus->version.storeUuid)},
                    {QStringLiteral("uuid"), QString::fromStdString(umbilicus->uuid)},
                    {QStringLiteral("generation"), QString::fromStdString(umbilicus->version.generation)},
                    {QStringLiteral("version"), QJsonValue(qint64(umbilicus->version.version))},
                    {QStringLiteral("points_zyx"), points}};
            }
        }
        entries[QString::fromStdString(uuid)] = entry;
    }
    QJsonObject block{{QStringLiteral("name"), QString::fromStdString(name)},
                      {QStringLiteral("volumes"), entries}};
    if (lasagna) {
        QJsonObject value{{QStringLiteral("location"), QString::fromStdString(lasagna->location)}};
        if (!readable) {
            value[QStringLiteral("error")] = QString::fromStdString(lasagna->error);
        } else {
            if (!lasagna->identity.empty())
                value[QStringLiteral("identity")] = QString::fromStdString(lasagna->identity);
            value[QStringLiteral("group")] = QString::fromStdString(lasagna->group);
            for (const auto& [input, store] : lasagna->stores)
                value[QString::fromStdString(input)] = QString::fromStdString(store);
        }
        block[QStringLiteral("lasagna")] = value;
    }
    if (fiberInference) {
        QJsonObject value{{QStringLiteral("location"), QString::fromStdString(fiberInference->location)}};
        if (!fibers)
            value[QStringLiteral("error")] = QString::fromStdString(fiberInference->error);
        else {
            if (!fiberInference->identity.empty())
                value[QStringLiteral("identity")] = QString::fromStdString(fiberInference->identity);
            if (!fiberInference->remoteManifest.empty())
                value[QStringLiteral("remote_manifest")] = QString::fromStdString(fiberInference->remoteManifest);
        }
        block[QStringLiteral("fiber_inference")] = value;
    }
    return block;
}

} // namespace vc3d::spiral
