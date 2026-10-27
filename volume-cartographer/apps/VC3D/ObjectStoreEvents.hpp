#pragma once

#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <algorithm>
#include <initializer_list>
#include <string_view>

namespace vc3d::store {

// Attachment and hub events are not record changes; callers handle those
// separately when they need to preserve caches on status-only notifications.
inline bool touches(const vc::objects::StoreSessionEvent& event,
                    std::initializer_list<std::string_view> kinds)
{
    if (event.kind != vc::objects::StoreSessionEvent::Kind::Committed) return true;
    return std::any_of(event.changes.begin(), event.changes.end(), [&](const auto& change) {
        return std::find(kinds.begin(), kinds.end(), change.kind) != kinds.end();
    });
}

} // namespace vc3d::store
