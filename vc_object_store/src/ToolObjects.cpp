#include "vc_object_store/objects/ToolObjects.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"

#include <cstdlib>
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

TransactionReceipt ToolObjects::publishArtifacts(const std::string& uuid,
                                                const std::vector<std::pair<std::string, std::filesystem::path>>& files)
{
    std::optional<PinnedObject> pin;
    std::optional<CheckVolumeMappingOp> mapping;
    {
        std::lock_guard lock(mutex_);
        if (const auto read = read_.find(util::canonicalUuid(uuid)); read != read_.end()) {
            pin = read->second.pin; mapping = read->second.mapping;
        }
    }
    if (!pin) {
        const auto current = session_->currentObject(attachment_, uuid);
        if (!current) throw std::runtime_error("no object " + uuid + " to publish artifacts of");
        return session_->publishArtifacts(attachment_, current->reference(), files);
    }
    return session_->publishArtifacts(*pin, files, {}, mapping);
}

void ToolObjects::noteRead(const PinnedObject& pin, const std::optional<VolumeVoxels>& volume, bool identity,
                           const std::optional<CheckVolumeMappingOp>& mapping)
{
    std::lock_guard lock(mutex_);
    if (volume && !mapping) throw std::invalid_argument("a loaded volume view requires its mapping guard");
    read_[util::canonicalUuid(pin.reference().objectUuid)] = {pin, volume, !volume || identity, mapping};
}

CoordinateJson ToolObjects::coordinatesFor(const std::vector<std::string>& sources, const std::optional<VolumeVoxels>& volume)
{
    if (volume) return volumeCoordinates(*volume);
    std::map<std::string, Read> reads;
    {
        std::lock_guard lock(mutex_);
        for (const auto& source : sources)
            if (const auto found = read_.find(util::canonicalUuid(source)); found != read_.end())
                reads.emplace(found->first, found->second);
    }
    std::optional<CoordinateJson> coordinates;
    if (!sources.empty())
        session_->readStore(attachment_, [&](const StoreReader& reader, const auto&) {
            for (const auto& id : sources) {
                const auto read = reads.find(util::canonicalUuid(id));
                CoordinateJson frame;
                if (read != reads.end() && read->second.volume) frame = volumeCoordinates(*read->second.volume);
                else {
                    const auto source = reader.selectedObject(id, {}, {}, read == reads.end() ?
                        std::nullopt : std::optional(read->second.pin.reference()));
                    if (!source) throw std::runtime_error("source object not found: " + id);
                    frame = geometryCoordinates(CoordinateJson::parse(source->record.meta).value("coordinates", CoordinateJson::object()));
                }
                if (coordinates && *coordinates != frame) throw std::runtime_error("sources were loaded in different frames");
                coordinates = std::move(frame);
            }
        });
    return coordinates.value_or(CoordinateJson::object());
}

PinnedObject ToolObjects::writable(const std::string& uuid, TransactionRequest& request)
{
    std::lock_guard lock(mutex_);
    const auto found = read_.find(util::canonicalUuid(uuid));
    if (found == read_.end()) throw std::runtime_error("update a surface this tool loaded: " + uuid);
    if (!found->second.identity)
        throw std::runtime_error("surface " + uuid + " was loaded through a volume mapping other than the identity");
    request.generation = found->second.pin.writeGeneration;
    expectObject(request, found->second.pin.reference());
    if (found->second.mapping) request.operations.push_back(*found->second.mapping);
    return found->second.pin;
}

void ToolObjects::finish()
{
    session_->finalize();
}
}
