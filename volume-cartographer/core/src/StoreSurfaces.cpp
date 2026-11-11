#include "vc/core/util/StoreSurfaces.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>

#include <cmath>
#include <stdexcept>

StoreSurfaces::StoreSurfaces(const std::filesystem::path& store, const std::string& tool) : objects_(store, tool) {}

// Loaded surfaces keep the session (and its attachment) alive for their channels;
// the session detaches when the last of them is gone.
StoreSurfaces::~StoreSurfaces() = default;

std::unique_ptr<QuadSurface> StoreSurfaces::load(const std::string& uuid, bool applyValidity,
                                                 const std::optional<vc::objects::VolumeVoxels>& volume)
{
    std::unique_ptr<QuadSurface> surface;
    std::optional<vc::objects::PinnedObject> pin;
    bool identity = true;
    std::optional<vc::objects::CheckVolumeMappingOp> mapping;
    const auto& session = objects_.session();
    session->readStore(objects_.attachment(), [&](const vc::objects::StoreReader& reader, const auto&) {
        const auto activeGeneration = reader.status().generation;
        const auto head = reader.currentObject(vc::objects::util::canonicalUuid(uuid));
        if (!head || !head->geometryRevision) throw std::runtime_error("not a stored surface: " + uuid);
        if (volume) {
            const auto view = vc::objects::GeometrySnapshot(reader).viewTransformedAt(*head, volume->volume, volume->scale);
            identity = view.mapping() == vc::objects::VolumeAffine::identity();
            mapping = view.checkMappingOperation();
            surface = load_quad_from_store(reader, *head, applyValidity, &view);
        } else {
            surface = load_quad_from_store(reader, *head, applyValidity);
        }
        pin = vc::objects::pinOf(objects_.attachment(), *head, activeGeneration);
    });
    objects_.noteRead(*pin, volume, identity, mapping);
    surface->setChannelLoader(store_channel_loader(session, objects_.attachment(), *pin->geometryRevision));
    return surface;
}

std::string StoreSurfaces::create(QuadSurface& surface, const std::string& name, const nlohmann::json& parameters,
                                  const std::vector<std::string>& sources, const std::vector<std::string>& tags,
                                  const std::optional<vc::objects::VolumeVoxels>& volume)
{
    const auto coordinates = objects_.coordinatesFor(sources, volume);
    const auto uuid = vc::objects::util::newUuid();
    const auto& session = objects_.session();
    auto request = session->newRequest(objects_.attachment());
    request.expectedObjects = {{uuid, std::nullopt}};
    request.operations = create_quad_operations(uuid, surface, name, tags, coordinates, {objects_.tool(), parameters, sources});
    session->submit(objects_.attachment(), request);
    surface.id = uuid;
    return uuid;
}

int64_t StoreSurfaces::update(const std::string& uuid, QuadSurface& surface)
{
    const auto canonical = vc::objects::util::canonicalUuid(uuid);
    const auto& points = *surface.rawPointsPtr();
    const auto& session = objects_.session();
    auto request = session->newRequest(objects_.attachment());
    const auto pin = objects_.writable(canonical, request);
    const auto guards = request.operations.size();
    session->readStore(objects_.attachment(), [&](const vc::objects::StoreReader& reader, const auto&) {
        const auto view = vc::objects::GeometrySnapshot(reader).viewStored(canonical, pin.geometryRevision);
        if (auto edit = quad_rect_edit(view, surface, cv::Rect(0, 0, points.cols, points.rows)))
            request.operations.push_back(std::move(*edit));
        vc::objects::expectRelationshipEndpoints(reader, request);
    });
    if (request.operations.size() == guards) return pin.reference().version;
    const auto receipt = session->submit(objects_.attachment(), request);
    return *receipt.objects.at(canonical);
}

bool StoreOutputArgs::parse(const std::string& arg)
{
    auto value = [&](const char* option) -> std::optional<std::string> {
        const std::string prefix = std::string(option) + "=";
        if (arg.rfind(prefix, 0) != 0) return std::nullopt;
        if (arg.size() == prefix.size()) throw std::invalid_argument(std::string(option) + " needs a value");
        return arg.substr(prefix.size());
    };
    if (const auto v = value("--store")) store = *v;
    else if (const auto v = value("--name")) name = *v;
    else if (const auto v = value("--tag")) tags.push_back(*v);
    else if (const auto v = value("--volume-uuid")) {
        if (!volume) volume.emplace();
        volume->volume = *v;
    } else if (const auto v = value("--voxel-scale")) {
        if (!volume) volume.emplace();
        volume->scale = parseVoxelScale(*v);
    } else return false;
    return true;
}

double parseVoxelScale(const std::string& value)
{
    size_t used = 0;
    double scale = 0;
    try {
        scale = std::stod(value, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != value.size() || !std::isfinite(scale) || scale <= 0)
        throw std::invalid_argument("--voxel-scale must be a finite positive number: " + value);
    return scale;
}
