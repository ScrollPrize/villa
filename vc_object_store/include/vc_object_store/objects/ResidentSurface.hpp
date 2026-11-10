#pragma once
#include "vc_object_store/objects/GeometryRegion.hpp"
#include <memory>

namespace vc::geometry { struct Surface; }
namespace vc::objects {
namespace detail { struct SurfaceDraft; }
// Immutable resident geometry, already expressed in the query's coordinate frame.
// Uses the same tile index, validity and intersection kernel as stored and draft
// surfaces, without installing a draft or looking up a current object by UUID.
class ResidentSurface {
public:
    explicit ResidentSurface(const vc::geometry::Surface&);
    RegionResult region(const std::string& identity, const RegionSlice&, const RegionCancel& = {},
                        uint64_t outputByteBudget = 8 * 1024 * 1024) const;
private:
    std::shared_ptr<const detail::SurfaceDraft> data_;
};
}
