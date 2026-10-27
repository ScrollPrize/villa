#include "vc/core/util/StoreSurfaces.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>

#include <stdexcept>

StoreSurfaces::StoreSurfaces(const std::filesystem::path& store, const std::string& tool) : objects_(store, tool) {}

// Loaded surfaces keep the session (and its attachment) alive for their channels;
// the session detaches when the last of them is gone.
StoreSurfaces::~StoreSurfaces() = default;

std::unique_ptr<QuadSurface> StoreSurfaces::load(const std::string& uuid, bool applyValidity,
                                                 const std::optional<std::string>& volume)
{
    std::unique_ptr<QuadSurface> surface;
    std::string objectRevision, revision;
    bool identity = true;
    const auto& session = objects_.session();
    session->read(objects_.attachment(), [&](const auto& db, const auto& txn, const auto&) {
        if (volume) {
            const auto view = vc::objects::GeometrySnapshot(db, txn).viewTransformedAt(uuid, *volume);
            identity = view.mapping() == vc::objects::VolumeAffine::identity();
            surface = load_quad_from_store(db, txn, uuid, std::nullopt, applyValidity, &view);
        } else {
            surface = load_quad_from_store(db, txn, uuid, std::nullopt, applyValidity);
        }
        revision = db.currentGeometry(txn, surface->id)->uuid;
        objectRevision = db.currentObjectRevision(txn, surface->id)->uuid;
    });
    objects_.noteRead(surface->id, objectRevision, revision, volume, identity);
    surface->setChannelLoader(store_channel_loader(session, objects_.attachment(), revision));
    return surface;
}

std::string StoreSurfaces::create(QuadSurface& surface, const std::string& name, const nlohmann::json& parameters,
                                  const std::vector<std::string>& sources, const std::vector<std::string>& tags,
                                  const std::optional<std::string>& volume)
{
    const auto coordinates = objects_.coordinatesFor(sources, volume);
    const auto uuid = vc::objects::util::newUuid();
    objects_.session()->act(objects_.attachment(), {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
        create_quad_in_store(db, txn, uuid, surface, name, tags, coordinates, {objects_.tool(), parameters, sources});
    });
    objects_.wrote();
    surface.id = uuid;
    return uuid;
}

std::string StoreSurfaces::update(const std::string& uuid, QuadSurface& surface)
{
    const auto canonical = vc::objects::util::canonicalUuid(uuid);
    const auto pins = objects_.writable(canonical);
    const auto& points = *surface.rawPointsPtr();
    std::string revision;
    objects_.session()->editKeepingRelationships(objects_.attachment(), {{canonical, pins.objectRevision}}, {},
        [&](auto& db, auto& txn, const auto& keep) {
            revision = write_quad_rect_to_store(db, txn, canonical, surface, cv::Rect(0, 0, points.cols, points.rows),
                                                pins.geometryRevision, keep);
        });
    objects_.wrote();
    return revision;
}

std::string StoreSurfaces::updateChannels(const std::string& uuid, QuadSurface& surface, const std::vector<std::string>& names)
{
    const auto canonical = vc::objects::util::canonicalUuid(uuid);
    const auto pins = objects_.writable(canonical);
    std::string revision;
    objects_.session()->editKeepingRelationships(objects_.attachment(), {{canonical, pins.objectRevision}}, {},
        [&](auto& db, auto& txn, const auto&) {
            revision = write_quad_channels_to_store(db, txn, canonical, surface, names, {}, pins.geometryRevision);
        });
    objects_.wrote();
    return revision;
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
    else if (const auto v = value("--volume-uuid")) volume = *v;
    else return false;
    return true;
}
