#include "vc/core/util/StoreSurfaces.hpp"

#include <vc_object_store/Uuid.hpp>

#include <cstdlib>
#include <stdexcept>
#include <thread>

StoreSurfaces::StoreSurfaces(const std::filesystem::path& store, const std::string& tool)
    : tool_(tool), session_(std::make_shared<vc::objects::ObjectStoreSession>())
{
    vc::objects::OpenOptions options;
    const char* user = std::getenv("USER");
    options.author = user && *user ? user : tool;
    attachment_ = session_->attach(store, options);
}

// Loaded surfaces keep the session (and its attachment) alive for their channels;
// the session detaches when the last of them is gone.
StoreSurfaces::~StoreSurfaces() = default;

std::unique_ptr<QuadSurface> StoreSurfaces::load(const std::string& uuid, bool applyValidity)
{
    std::unique_ptr<QuadSurface> surface;
    std::string objectRevision, revision;
    session_->read(attachment_, [&](const auto& db, const auto& txn, const auto&) {
        surface = load_quad_from_store(db, txn, uuid, std::nullopt, applyValidity);
        revision = db.currentGeometry(txn, surface->id)->uuid;
        objectRevision = db.currentObjectRevision(txn, surface->id)->uuid;
    });
    {
        std::lock_guard lock(loadedMutex_);
        loaded_[surface->id] = {objectRevision, revision};
    }
    surface->setChannelLoader(store_channel_loader(session_, attachment_, revision));
    return surface;
}

std::string StoreSurfaces::create(QuadSurface& surface, const std::string& name, const nlohmann::json& parameters,
                                  const std::vector<std::string>& sources, const std::vector<std::string>& tags,
                                  const std::optional<nlohmann::json>& coordinates)
{
    auto meta = nlohmann::json::parse(surface.meta.dump());
    if (!meta.is_object()) meta = nlohmann::json::object();
    // Identity, geometry and file layout belong to the object and its geometry.
    for (const auto* key : {"uuid", "name", "format", "type", "bbox", "scale", "tiff_dimensions", "coordinates", "provenance"})
        meta.erase(key);
    if (coordinates) meta["coordinates"] = *coordinates;
    else if (!sources.empty())
        session_->read(attachment_, [&](const auto& db, const auto& txn, const auto&) {
            const auto source = db.objectByUuid(txn, sources.front());
            if (!source) throw std::runtime_error("source object not found: " + sources.front());
            const auto sourceMeta = nlohmann::json::parse(source->meta);
            if (sourceMeta.contains("coordinates")) meta["coordinates"] = sourceMeta.at("coordinates");
        });
    meta["provenance"] = {{"tool", tool_}, {"parameters", parameters}, {"sources", sources}};
    vc::objects::ObjectRecord record;
    record.uuid = vc::objects::util::newUuid();
    record.kind = "surface";
    record.name = name;
    record.tags = tags;
    record.meta = meta.dump();
    const auto geometry = quad_surface_geometry(surface);
    session_->act(attachment_, {{record.uuid, std::nullopt}}, [&](auto& db, auto& txn) {
        db.upsertObject(txn, record);
        db.putGeometry(txn, record.uuid, geometry);
    });
    lastWrite_ = std::chrono::system_clock::now();
    surface.id = record.uuid;
    return record.uuid;
}

std::string StoreSurfaces::update(const std::string& uuid, QuadSurface& surface)
{
    const auto canonical = vc::objects::util::canonicalUuid(uuid);
    const auto pins = loadedRevision(canonical);
    if (!pins) throw std::runtime_error("update a surface this tool loaded: " + uuid);
    const auto& [objectRevision, geometryRevision] = *pins;
    const auto& points = *surface.rawPointsPtr();
    std::string revision;
    session_->editKeepingRelationships(attachment_, {{canonical, objectRevision}}, [&](auto& db, auto& txn, const auto& keep) {
        revision = write_quad_rect_to_store(db, txn, canonical, surface, cv::Rect(0, 0, points.cols, points.rows),
                                            geometryRevision, keep);
    });
    lastWrite_ = std::chrono::system_clock::now();
    return revision;
}

std::string StoreSurfaces::updateChannels(const std::string& uuid, QuadSurface& surface, const std::vector<std::string>& names)
{
    const auto canonical = vc::objects::util::canonicalUuid(uuid);
    const auto pins = loadedRevision(canonical);
    if (!pins) throw std::runtime_error("update a surface this tool loaded: " + uuid);
    const auto& [objectRevision, geometryRevision] = *pins;
    std::string revision;
    session_->editKeepingRelationships(attachment_, {{canonical, objectRevision}}, [&](auto& db, auto& txn, const auto&) {
        revision = write_quad_channels_to_store(db, txn, canonical, surface, names, {}, geometryRevision);
    });
    lastWrite_ = std::chrono::system_clock::now();
    return revision;
}

std::optional<std::pair<std::string, std::string>> StoreSurfaces::loadedRevision(const std::string& uuid)
{
    std::lock_guard lock(loadedMutex_);
    const auto found = loaded_.find(uuid);
    if (found == loaded_.end()) return std::nullopt;
    return found->second;
}

std::filesystem::path StoreSurfaces::artifactDirectory(const std::string& uuid)
{
    return session_->artifactDirectory(attachment_, uuid);
}

void StoreSurfaces::finish(std::chrono::seconds timeout)
{
    session_->finalize();
    if (!lastWrite_ || !session_->hubStatus(attachment_)) return;
    const auto written = std::chrono::duration_cast<std::chrono::microseconds>(lastWrite_->time_since_epoch()).count();
    session_->syncHub(attachment_);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        // Status describes the last completed sync; it must postdate the writes.
        const auto status = session_->hubStatus(attachment_);
        if (status && status->pending.empty() && status->syncedAtUs && *status->syncedAtUs > written) return;
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("writes did not reach the hub" + (status && !status->error.empty() ? ": " + status->error : std::string()));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
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

std::optional<nlohmann::json> StoreOutputArgs::coordinates() const
{
    if (volume.empty()) return std::nullopt;
    return nlohmann::json{{"volume_uuid", volume}, {"scale_factor", 1}};
}
