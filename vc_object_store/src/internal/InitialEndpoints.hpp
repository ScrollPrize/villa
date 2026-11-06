#pragma once
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "internal/PreparedDependencies.hpp"
#include <map>

namespace vc::objects::detail {
// Build-private directory. No live SQLite connections or coordinate payloads
// cross worker boundaries; after object preparation this is immutable.
struct InitialEndpoint {
    bool retired;
    std::optional<PreparedDependencies> geometry;
};
using InitialEndpointDirectory=std::map<ObjectKey,InitialEndpoint>;
}
