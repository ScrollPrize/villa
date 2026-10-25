#include "vc/core/util/StoreSurfaces.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/store/GeometryViews.hpp>
#include <vc_object_store/objects/GeometryCoordinates.hpp>

#include <cstdlib>
#include <set>
#include <stdexcept>

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

std::unique_ptr<QuadSurface> StoreSurfaces::load(const std::string& uuid, bool applyValidity,
                                                 const std::optional<std::string>& volume)
{
    std::unique_ptr<QuadSurface> surface;
    std::string objectRevision, revision;
    bool storedUnits = true;
    session_->read(attachment_, [&](const auto& db, const auto& txn, const auto&) {
        if (volume) {
            const auto view = vc::objects::GeometrySnapshot(db, txn).viewTransformedAt(uuid, *volume);
            storedUnits = view.mapping() == vc::objects::VolumeAffine::identity();
            surface = load_quad_from_store(db, txn, uuid, std::nullopt, applyValidity, 1, &view);
        } else {
            surface = load_quad_from_store(db, txn, uuid, std::nullopt, applyValidity);
        }
        revision = db.currentGeometry(txn, surface->id)->uuid;
        objectRevision = db.currentObjectRevision(txn, surface->id)->uuid;
    });
    {
        std::lock_guard lock(loadedMutex_);
        loaded_[surface->id] = {objectRevision, revision, storedUnits, volume};
    }
    surface->setChannelLoader(store_channel_loader(session_, attachment_, revision));
    return surface;
}

std::string StoreSurfaces::create(QuadSurface& surface, const std::string& name, const nlohmann::json& parameters,
                                  const std::vector<std::string>& sources, const std::vector<std::string>& tags,
                                  const std::optional<nlohmann::json>& coordinates)
{
    auto meta = nlohmann::json::object();
    std::optional<std::string> loadedIn;
    {
        std::lock_guard lock(loadedMutex_);
        std::set<std::optional<std::string>> frames;
        for (const auto& source : sources)
            if (const auto found = loaded_.find(source); found != loaded_.end()) frames.insert(found->second.volume);
        if (frames.size() > 1) throw std::runtime_error("sources were loaded in different frames");
        if (!frames.empty()) loadedIn = *frames.begin();
    }
    if (coordinates) meta["coordinates"] = *coordinates;
    else if (loadedIn) meta["coordinates"] = *volumeCoordinates(*loadedIn);
    else if (!sources.empty())
        session_->read(attachment_, [&](const auto& db, const auto& txn, const auto&) {
            const auto source = db.objectByUuid(txn, sources.front());
            if (!source) throw std::runtime_error("source object not found: " + sources.front());
            const auto sourceMeta = nlohmann::json::parse(source->meta);
            if (sourceMeta.contains("coordinates")) meta["coordinates"] = sourceMeta.at("coordinates");
        });
    meta["provenance"] = {{"tool", tool_}, {"parameters", parameters}, {"sources", sources}};
    const auto uuid = vc::objects::util::newUuid();
    session_->act(attachment_, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
        create_quad_in_store(db, txn, uuid, surface, name, tags, meta.dump());
    });
    wrote_ = true;
    surface.id = uuid;
    return uuid;
}

std::string StoreSurfaces::update(const std::string& uuid, QuadSurface& surface)
{
    const auto canonical = vc::objects::util::canonicalUuid(uuid);
    const auto [objectRevision, geometryRevision, storedUnits, volume] = writable(canonical);
    const auto& points = *surface.rawPointsPtr();
    std::string revision;
    session_->editKeepingRelationships(attachment_, {{canonical, objectRevision}}, {}, [&](auto& db, auto& txn, const auto& keep) {
        revision = write_quad_rect_to_store(db, txn, canonical, surface, cv::Rect(0, 0, points.cols, points.rows),
                                            geometryRevision, keep);
    });
    wrote_ = true;
    return revision;
}

std::string StoreSurfaces::updateChannels(const std::string& uuid, QuadSurface& surface, const std::vector<std::string>& names)
{
    const auto canonical = vc::objects::util::canonicalUuid(uuid);
    const auto [objectRevision, geometryRevision, storedUnits, volume] = writable(canonical);
    std::string revision;
    session_->editKeepingRelationships(attachment_, {{canonical, objectRevision}}, {}, [&](auto& db, auto& txn, const auto&) {
        revision = write_quad_channels_to_store(db, txn, canonical, surface, names, {}, geometryRevision);
    });
    wrote_ = true;
    return revision;
}

StoreSurfaces::Loaded StoreSurfaces::writable(const std::string& uuid)
{
    std::lock_guard lock(loadedMutex_);
    const auto found = loaded_.find(uuid);
    if (found == loaded_.end()) throw std::runtime_error("update a surface this tool loaded: " + uuid);
    // Writing a mapped grid back would rewrite every stored coordinate.
    if (!found->second.storedUnits)
        throw std::runtime_error("surface " + uuid + " was loaded through a volume mapping other than the identity");
    return found->second;
}

void StoreSurfaces::finish(std::chrono::seconds timeout)
{
    session_->finalize();
    if (!wrote_) return;
    const auto status = session_->syncHubAndWait(attachment_, timeout);
    if (status && (!status->error.empty() || !status->conflicts.empty()))
        throw std::runtime_error("writes did not reach the hub: " +
                                 (status->error.empty() ? std::string("unresolved hub conflicts") : status->error));
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

std::optional<nlohmann::json> volumeCoordinates(const std::string& volume)
{
    if (volume.empty()) return std::nullopt;
    return vc::objects::geometryCoordinates({{"volume_uuid", volume}, {"scale_factor", 1}});
}

std::optional<nlohmann::json> StoreOutputArgs::coordinates() const
{
    return volumeCoordinates(volume);
}
