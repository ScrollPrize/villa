#include "vc_object_store/objects/ToolObjects.hpp"

#include <cstdlib>
#include <set>
#include <stdexcept>

namespace vc::objects {
ToolObjects::ToolObjects(const std::filesystem::path& store, const std::string& tool)
    : tool_(tool), session_(std::make_shared<ObjectStoreSession>())
{
    OpenOptions options;
    const char* user = std::getenv("USER");
    options.author = user && *user ? user : tool;
    attachment_ = session_->attach(store, options);
}

void ToolObjects::noteRead(const PinnedObject& pin, const std::optional<VolumeVoxels>& volume, bool identity)
{
    std::lock_guard lock(mutex_);
    read_[pin.object.objectUuid] = {pin, volume, !volume || identity};
}

CoordinateJson ToolObjects::coordinatesFor(const std::vector<std::string>& sources, const std::optional<VolumeVoxels>& volume)
{
    if (volume) return volumeCoordinates(*volume);
    std::set<std::optional<VolumeVoxels>> frames;
    {
        std::lock_guard lock(mutex_);
        for (const auto& source : sources)
            if (const auto found = read_.find(source); found != read_.end()) frames.insert(found->second.volume);
    }
    if (frames.size() > 1) throw std::runtime_error("sources were loaded in different frames");
    if (!frames.empty() && *frames.begin()) return volumeCoordinates(**frames.begin());
    auto coordinates = CoordinateJson::object();
    if (!sources.empty())
        session_->readStore(attachment_, [&](const StoreReader& reader, const auto&) {
            const auto source = reader.currentObject(sources.front());
            if (!source) throw std::runtime_error("source object not found: " + sources.front());
            const auto meta = nlohmann::json::parse(source->record.meta);
            if (meta.contains("coordinates")) coordinates = meta.at("coordinates");
        });
    return coordinates;
}

PinnedObject ToolObjects::writable(const std::string& uuid)
{
    std::lock_guard lock(mutex_);
    const auto found = read_.find(uuid);
    if (found == read_.end()) throw std::runtime_error("update a surface this tool loaded: " + uuid);
    if (!found->second.identity)
        throw std::runtime_error("surface " + uuid + " was loaded through a volume mapping other than the identity");
    return found->second.pin;
}

void ToolObjects::finish(std::chrono::seconds timeout)
{
    session_->finalize();
    if (!wrote_) return;
    const auto status = session_->syncHubAndWait(attachment_, timeout);
    if (status && (!status->error.empty() || !status->conflicts.empty()))
        throw std::runtime_error("writes did not reach the hub: " +
                                 (status->error.empty() ? std::string("unresolved hub conflicts") : status->error));
}
}
